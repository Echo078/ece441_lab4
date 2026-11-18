"""
Lab 4: event-related potentials to red circles (targets, button press) vs blue
squares (non-targets), tested with the cluster-based permutation test of
Maris & Oostenveld (2007).

Pipeline (the "Data analysis" section of the manual, in order):

    1. load the OpenBCI .txt file, rename EXG channels and D13 ('Other.2') -> photoresistor
    2. build an MNE RawArray (channels x samples, EEG in volts, photoresistor = 'stim')
    3. photoresistor onsets -> events -> annotations
    4. band-pass 1-50 Hz
    5. crop from 0.5 s after the video start to 1 s after the last video TTL, epoch (90 trials)
    6. split the epochs with circle_mask.csv (circle = 1, square = 0)
    7. z-score every epoch against its own baseline
    8. function 1 + function 2 (lab4_stats.py): cluster statistic and permutation p-value per channel
    9. plot the evoked responses of the channels with p < 0.05

Run from this folder:

    python lab4_analysis.py                    # auto-finds the single OpenBCI-RAW-*.txt in this folder
    python lab4_analysis.py path/to/file.txt   # or name the recording explicitly
    python lab4_analysis.py --demo             # synthetic recording, to try the pipeline without data

Figures go to report/img, every number to report/results.json.

============================ EDIT THIS SECTION ============================
Paths are relative to this file. Use forward slashes "/".
"""

# Recordings to analyse, {tag: path}. Each is analysed on its own (as the manual
# describes) and, if there are several, also pooled. {} = auto-detect every
# OpenBCI GUI .txt recording under this folder.
RECORDING_FILES = {
    "run1": "Data/trial1.txt",
    "run2": "Data/trial2.txt",
}
CIRCLE_MASK_FILE = "circle_mask.csv"
SAVE_DIR = "report/img"
RESULTS_FILE = "report/results.json"

PHOTORESISTOR_COLUMN = "Other.2"  # D13 per the manual; other 'Other'/'Analog' columns are tried if it is not a TTL

BAND_LOW_HZ = 1.0
BAND_HIGH_HZ = 50.0
# The 1-50 Hz band-pass alone leaves a 60 Hz peak (MNE's default transition band
# reaches 62.5 Hz), but the manual asks that no line noise is left: add a notch.
NOTCH_HZ = 60.0  # None = band-pass only

CROP_START_AFTER_VIDEO_START_S = 0.5  # manual: "around 0.5 s after the video start"
CROP_END_AFTER_LAST_TTL_S = 1.0  # manual: "up to 1 second after the last TTL"

EPOCH_TMIN_S = -0.2  # MNE defaults, as the manual asks ("epoch using the default settings")
EPOCH_TMAX_S = 0.5
EXTENDED_TMAX_S = 0.8  # supplementary analysis with the FAQ's -0.2..0.8 s window (late P300)

N_PERMUTATIONS = 1000
P_THRESHOLD = 0.05
RANDOM_SEED = 0

# =========================== END OF EDITABLE SECTION ===========================

import argparse
import inspect
import json
import os
import re
import sys
import warnings
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # figures are only saved, never shown

import mne
import numpy as np
import pandas as pd
from scipy import stats

import lab4_stats

HERE = os.path.dirname(os.path.abspath(__file__))

CHANNEL_NAMES = ["FP1", "FP2", "C3", "C4", "P7", "P8", "O1", "O2"]
STIM_CHANNEL = "photoresistor"
EXG_PREFIX = "EXG Channel"
HEADER_ROWS = 4  # pandas header=4: row 5 holds the column names
MICROVOLT_TO_VOLT = 1e-6
SAMPLE_RATE_PATTERN = re.compile(r"%\s*Sample Rate\s*=\s*([\d.]+)\s*Hz", re.IGNORECASE)
SAMPLE_INDEX_MODULO = 256  # the Cyton packet counter wraps at 255
COUNTER_RESET_WINDOW_SAMPLES = 500  # a counter reset can only happen right after the stream starts
OPENBCI_FIRST_LINE = "%OpenBCI Raw EEG Data"
SYNC_END_MARGIN_S = 0.05  # crop at least this long after the sync TTL has gone low again
SKIP_DIRS = {".venv", "report", "demo", "__pycache__", ".pytest_cache"}

# Timing of Lab4_video.mp4, measured frame by frame (20 fps, every flash is one frame = 50 ms):
# 5 sync flashes 100 ms apart at 5.0-5.4 s, then 90 stimuli every 1.2 s from 6.8 s;
# each stimulus (red circle or blue square on grey) stays on screen for 0.9 s.
VIDEO_SYNC_FLASHES = 5
VIDEO_SYNC_INTERVAL_S = 0.1
VIDEO_FIRST_STIMULUS_AFTER_SYNC_S = 1.8
VIDEO_STIMULUS_INTERVAL_S = 1.2
VIDEO_FLASH_DURATION_S = 0.05
VIDEO_STIMULUS_DURATION_S = 0.9
N_TRIALS = 90

SYNC_MAX_GAP_S = 0.2  # onsets closer than this belong to the same sync burst
SYNC_MIN_FLASHES = 2
TRIAL_MATCH_TOLERANCE_S = 0.15  # a detected onset this close to a predicted trial is that trial
FIRST_TRIAL_SEARCH_S = TRIAL_MATCH_TOLERANCE_S  # must stay < 0.2 s (see match_video_trials)
ANCHOR_CHECK_TRIALS = 3  # a sync candidate must be followed by one of the first 3 trials on time
MIN_TRIALS = 10  # fewer matched trials than this: something is wrong, stop
DRIFT_FIT_TRIALS = 10  # trials used to extrapolate the next onset (absorbs clock drift)
MIN_TTL_EDGES = 10
FLAT_CHANNEL_STD_UV = 0.5


# --------------------------------------------------------------------------
# Steps 1-2: loading and building the MNE object
# --------------------------------------------------------------------------
def parse_sample_rate(path: str) -> float:
    """Sampling rate (Hz) from the '%Sample Rate = 250 Hz' header line."""
    with open(path, "r", encoding="utf-8") as handle:
        for _ in range(HEADER_ROWS):
            match = SAMPLE_RATE_PATTERN.search(handle.readline())
            if match:
                return float(match.group(1))
    raise ValueError("No '%Sample Rate = ... Hz' line in the header of " + path)


def run_lengths(state: np.ndarray) -> np.ndarray:
    """Lengths of the runs of True in a boolean array."""
    edges = np.diff(np.r_[0, state.astype(int), 0])
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def ttl_polarity(values: pd.Series) -> Optional[bool]:
    """True if <values> look like the photoresistor TTL (two levels, brief highs),
    False if they look like an inverted TTL, None if they are not a TTL at all.

    The polarity is decided by the typical (median) run length: the video's
    markers are 50 ms long while the dark gaps last ~1.15 s, and a few long
    bright stretches (desktop before/after the video) do not move the median.
    """
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    levels = np.unique(finite)
    if len(levels) != 2:
        return None
    high = finite == levels[1]
    if int(np.count_nonzero(np.diff(high.astype(int)) == 1)) < MIN_TTL_EDGES:
        return None
    return bool(np.median(run_lengths(high)) <= np.median(run_lengths(~high)))


def binarize_ttl(values: pd.Series, normal_polarity: bool) -> np.ndarray:
    """0/1 version of a two-level TTL column; 1 = bright screen."""
    data = values.to_numpy(dtype=float)
    midpoint = (np.nanmin(data) + np.nanmax(data)) / 2
    bright = data > midpoint if normal_polarity else data < midpoint
    return bright.astype(float)


def choose_photoresistor_column(df: pd.DataFrame, preferred: str) -> Tuple[str, bool]:
    """Column holding D13 and whether it has the expected polarity (bright = 1).

    The manual says 'Other.2'. If that column is not a TTL (e.g. another GUI
    version puts D13 elsewhere) the other 'Other'/'Analog' columns are tried.
    """
    others = [c for c in df.columns if c != preferred and c.startswith(("Other", "Analog Channel"))]
    for column in ([preferred] if preferred in df.columns else []) + others:
        polarity = ttl_polarity(df[column])
        if polarity is not None:
            return column, polarity
    raise ValueError(
        "No photoresistor TTL found: none of the columns "
        + ", ".join([preferred] + others)
        + " toggles between two levels. Was the Digital Read widget switched on?"
    )


def load_recording(path: str, photoresistor_column: str = PHOTORESISTOR_COLUMN) -> Tuple[pd.DataFrame, Dict]:
    """OpenBCI .txt -> DataFrame with columns FP1..O2 (uV) and 'photoresistor' (0/1).

    Also returns a dict describing the file (photoresistor column, lost packets).
    """
    raw_df = pd.read_csv(path, header=HEADER_ROWS, index_col=False, low_memory=False)
    raw_df.columns = [c.strip() for c in raw_df.columns]

    exg = [c for c in raw_df.columns if c.startswith(EXG_PREFIX)]
    if len(exg) != len(CHANNEL_NAMES):
        raise ValueError(f"Expected {len(CHANNEL_NAMES)} '{EXG_PREFIX} *' columns, found {len(exg)}")
    column, polarity = choose_photoresistor_column(raw_df, photoresistor_column)

    df = raw_df[exg].rename(columns=dict(zip(exg, CHANNEL_NAMES))).astype(float)
    df = df.assign(**{STIM_CHANNEL: binarize_ttl(raw_df[column], polarity)})

    info = {"photoresistor_column": column, "photoresistor_inverted": not polarity}
    info.update(count_lost_samples(raw_df.get("Sample Index")))
    return df, info


def count_lost_samples(sample_index: Optional[pd.Series]) -> Dict:
    """Gaps in the Cyton packet counter (0..255): each gap is lost radio packets.

    A jump to 0 from anything but 255 within the first 2 s is the counter being
    reset (the GUI does this once right after the stream starts), not lost data.
    """
    if sample_index is None:
        return {"packet_gaps": None, "lost_samples": None, "counter_resets": None}
    index = sample_index.to_numpy(dtype=float).astype(int)
    steps = np.diff(index) % SAMPLE_INDEX_MODULO
    early = np.arange(len(steps)) < COUNTER_RESET_WINDOW_SAMPLES
    resets = early & (index[1:] == 0) & (index[:-1] != SAMPLE_INDEX_MODULO - 1)
    gaps = steps[(steps != 1) & ~resets]
    lost = int(np.sum((gaps - 1) % SAMPLE_INDEX_MODULO))
    return {"packet_gaps": int(len(gaps)), "lost_samples": lost, "counter_resets": int(resets.sum())}


def build_raw(df: pd.DataFrame, sfreq: float) -> mne.io.RawArray:
    """DataFrame (samples x channels, uV) -> RawArray (channels x samples, V); photoresistor as 'stim'."""
    eeg_volts = df[CHANNEL_NAMES].T.to_numpy(dtype=float) * MICROVOLT_TO_VOLT  # transpose: channels x samples
    ttl = df[[STIM_CHANNEL]].T.to_numpy(dtype=float)  # the photoresistor stays 0/1
    data = np.vstack([eeg_volts, ttl])
    info = mne.create_info(
        ch_names=CHANNEL_NAMES + [STIM_CHANNEL],
        sfreq=sfreq,
        ch_types=["eeg"] * len(CHANNEL_NAMES) + ["stim"],
    )
    return mne.io.RawArray(data, info, verbose="ERROR")


# --------------------------------------------------------------------------
# Step 3: events and annotations from the photoresistor
# --------------------------------------------------------------------------
def find_flash_events(raw: mne.io.BaseRaw) -> np.ndarray:
    """Rising edges (dark -> bright) of the photoresistor channel, MNE events array."""
    return mne.find_events(
        raw, stim_channel=STIM_CHANNEL, shortest_event=1, consecutive="increasing", verbose="ERROR"
    )


def annotate_events(raw: mne.io.BaseRaw, events: np.ndarray) -> mne.io.BaseRaw:
    """Copy of <raw> with one 'flash' annotation per event."""
    annotations = mne.annotations_from_events(
        events, sfreq=raw.info["sfreq"], event_desc={1: "flash"}, first_samp=raw.first_samp
    )
    annotated = raw.copy()
    annotated.set_annotations(annotations, verbose="ERROR")
    return annotated


class SyncBurst(NamedTuple):
    first: int  # index into the onset list
    last: int

    @property
    def n_flashes(self) -> int:
        return self.last - self.first + 1


def flash_runs(onsets_s: np.ndarray, min_flashes: int) -> List[SyncBurst]:
    """Runs of >= <min_flashes> onsets, each less than 0.2 s after the previous one."""
    runs, start = [], 0
    for i in range(1, len(onsets_s) + 1):
        if i == len(onsets_s) or onsets_s[i] - onsets_s[i - 1] > SYNC_MAX_GAP_S:
            if i - start >= min_flashes:
                runs.append(SyncBurst(start, i - 1))
            start = i
    return runs


def find_sync_bursts(onsets_s: np.ndarray) -> List[SyncBurst]:
    """Runs of >= 2 onsets less than 0.2 s apart (the video's start-of-video flashes)."""
    return flash_runs(onsets_s, SYNC_MIN_FLASHES)


def track_trials(onsets_s: np.ndarray, first_trial_s: float) -> np.ndarray:
    """Index (into <onsets_s>) of the onset of each of the 90 trials, -1 if missed.

    Trials are predicted every 1.2 s; once a few are found, the prediction is a
    line fitted to the last matches, which absorbs drift between the EEG clock
    and the video clock and the small shifts caused by lost packets.
    """
    matched = np.full(N_TRIALS, -1)
    found_k: List[int] = []
    found_t: List[float] = []
    for k in range(N_TRIALS):
        if len(found_k) >= 2:
            slope, intercept = np.polyfit(found_k[-DRIFT_FIT_TRIALS:], found_t[-DRIFT_FIT_TRIALS:], 1)
            predicted = intercept + slope * k
        elif found_k:
            predicted = found_t[-1] + VIDEO_STIMULUS_INTERVAL_S * (k - found_k[-1])
        else:
            predicted = first_trial_s + VIDEO_STIMULUS_INTERVAL_S * k
        nearest = int(np.argmin(np.abs(onsets_s - predicted)))
        if abs(onsets_s[nearest] - predicted) <= TRIAL_MATCH_TOLERANCE_S and nearest not in matched:
            matched[k] = nearest
            found_k.append(k)
            found_t.append(float(onsets_s[nearest]))
    return matched


class TrialMatch(NamedTuple):
    trial_onset: np.ndarray  # (90,) index into the onsets, -1 = trial not detected
    burst: Optional[SyncBurst]  # the sync flash(es) used as anchor; None = no anchor found
    video_start_s: float  # time of the first sync flash (estimated if it was not detected)

    @property
    def anchored(self) -> bool:
        """False if no sync flash was found: then a missed trial 0 would shift every label by one."""
        return self.burst is not None

    @property
    def n_matched(self) -> int:
        return int(np.sum(self.trial_onset >= 0))


def _first_trial_guess(onsets_s: np.ndarray, predicted: float) -> float:
    near = np.abs(onsets_s - predicted) <= FIRST_TRIAL_SEARCH_S
    return float(onsets_s[near][np.argmin(np.abs(onsets_s[near] - predicted))]) if near.any() else predicted


def match_video_trials(onsets_s: np.ndarray) -> TrialMatch:
    """Finds the 90 video trials among all photoresistor onsets.

    The recording may also hold flashlight/hand checks, a test playback of the
    first seconds of the video and a bright desktop after the video ends.

    Every run of flashes (< 0.2 s apart, even a single flash, since a slow
    photoresistor can merge or miss sync flashes) is tried as the sync burst;
    its first detected flash may be any of the video's 5 sync flashes, so each
    possibility is tried. Trial 0 is then expected 1.8 s after the first sync
    flash and the candidate followed by the most trials on the 1.2 s grid wins
    (ties: the later candidate, i.e. the real run rather than a test playback).
    A candidate only counts if one of the first 3 trials is found where it is
    predicted (clock drift is negligible that early), and trial 0 must lie
    within 0.15 s of its predicted time, so a trial onset taken for a sync
    flash cannot slide the labels by one or more trials.

    Without any anchor (no sync flash detected at all), every onset is tried as
    trial 0; the result is then flagged as not anchored.
    """
    if len(onsets_s) == 0:
        raise ValueError("The photoresistor never switched on: no events at all")
    best: Optional[TrialMatch] = None
    for run in flash_runs(onsets_s, 1):
        for skipped in range(max(VIDEO_SYNC_FLASHES - run.n_flashes, 0) + 1):  # sync flashes missed before it
            video_start = onsets_s[run.first] - skipped * VIDEO_SYNC_INTERVAL_S
            first = _first_trial_guess(onsets_s, video_start + VIDEO_FIRST_STIMULUS_AFTER_SYNC_S)
            candidate = TrialMatch(track_trials(onsets_s, first), run, float(video_start))
            if not (candidate.trial_onset[:ANCHOR_CHECK_TRIALS] >= 0).any():
                continue  # nothing where the first trials should be: this run is not the video start
            if best is None or candidate.n_matched > best.n_matched or (
                candidate.n_matched == best.n_matched and run.first > best.burst.first
            ):
                best = candidate
    if best is None or best.n_matched < N_TRIALS // 2:
        for onset in onsets_s:
            candidate = TrialMatch(track_trials(onsets_s, onset), None, float(onset - VIDEO_FIRST_STIMULUS_AFTER_SYNC_S))
            if best is None or candidate.n_matched > best.n_matched:
                best = candidate
    if best.n_matched < MIN_TRIALS:
        raise ValueError(f"Only {best.n_matched} photoresistor onsets fit the video's 1.2 s trial grid; check photoresistor.png")
    return best


def trial_times(onsets_s: np.ndarray, match: TrialMatch) -> np.ndarray:
    """Onset time of every trial; missed trials are filled in from a line fit."""
    found = np.flatnonzero(match.trial_onset >= 0)
    slope, intercept = np.polyfit(found, onsets_s[match.trial_onset[found]], 1)
    times = intercept + slope * np.arange(N_TRIALS)
    times[found] = onsets_s[match.trial_onset[found]]
    return times


# --------------------------------------------------------------------------
# Steps 4-6: filter, crop, epoch, split
# --------------------------------------------------------------------------
def bandpass(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Copy of <raw> band-passed 1-50 Hz, plus a 60 Hz notch (EEG channels only;
    the stim channel is untouched)."""
    filtered = raw.copy().filter(BAND_LOW_HZ, BAND_HIGH_HZ, verbose="ERROR")
    if NOTCH_HZ:
        filtered.notch_filter(NOTCH_HZ, verbose="ERROR")
    return filtered


def crop_window(
    onsets_s: np.ndarray, match: TrialMatch, duration_s: float, offsets_s: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """(start, stop) in s: 0.5 s after the video start to 1 s after its last TTL.

    The video start is the first sync flash; if no sync flash was detected it is
    placed 1.8 s before trial 0. The start never falls inside the sync flashes:
    a slow photoresistor can merge all 5 into one long high, so when the falling
    edges (<offsets_s>) are known the crop starts after that high has ended.
    """
    times = trial_times(onsets_s, match)
    video_start = times[0] - VIDEO_FIRST_STIMULUS_AFTER_SYNC_S
    if match.burst is not None:
        video_start = onsets_s[match.burst.first]
    start = video_start + CROP_START_AFTER_VIDEO_START_S
    if match.burst is not None:
        sync_end = onsets_s[match.burst.last] + VIDEO_FLASH_DURATION_S
        if offsets_s is not None:
            later = offsets_s[offsets_s > onsets_s[match.burst.last]]
            sync_end = later[0] if len(later) else sync_end
        start = max(start, sync_end + SYNC_END_MARGIN_S)
    stop = min(times[-1] + CROP_END_AFTER_LAST_TTL_S, duration_s)
    return float(start), float(stop)


def ttl_offsets_s(raw: mne.io.BaseRaw) -> np.ndarray:
    """Times (s) at which the photoresistor goes from bright back to dark."""
    ttl = raw.get_data(picks=STIM_CHANNEL)[0] > 0.5
    return np.flatnonzero(np.diff(ttl.astype(int)) == -1) / raw.info["sfreq"] + raw.times[1]


def trial_events(cropped: mne.io.BaseRaw, match_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Events of the cropped recording that are video trials.

    Returns (events array, trial number of each event, number of events
    find_events reported in the crop). <match_samples> holds the absolute
    sample of every detected trial (-1 = missed).
    """
    cropped_events = find_flash_events(cropped)
    keep = np.isin(cropped_events[:, 0], match_samples[match_samples >= 0])
    events = cropped_events[keep]
    trial_number = np.array([int(np.flatnonzero(match_samples == s)[0]) for s in events[:, 0]], dtype=int)
    return events, trial_number, len(cropped_events)


def make_epochs(cropped: mne.io.BaseRaw, events: np.ndarray, tmax: float = EPOCH_TMAX_S) -> mne.Epochs:
    return mne.Epochs(
        cropped,
        events,
        event_id={"flash": 1},
        tmin=EPOCH_TMIN_S,
        tmax=tmax,
        picks="eeg",
        preload=True,
        verbose="ERROR",
    )


def load_circle_mask(path: str) -> np.ndarray:
    mask = np.loadtxt(path, delimiter=",").astype(int).ravel()
    if len(mask) != N_TRIALS or not set(np.unique(mask)) <= {0, 1}:
        raise ValueError(f"{path} must hold {N_TRIALS} values of 0 or 1")
    return mask


def split_epochs(epochs: mne.Epochs, trial_number: np.ndarray, mask: np.ndarray) -> Tuple[mne.Epochs, mne.Epochs]:
    """(circle epochs, square epochs); epochs MNE dropped are skipped via epochs.selection."""
    is_circle = mask[trial_number[epochs.selection]] == 1
    return epochs[is_circle], epochs[~is_circle]


# --------------------------------------------------------------------------
# Step 7: z-score
# --------------------------------------------------------------------------
def zscore_epochs(data: np.ndarray, times: np.ndarray) -> np.ndarray:
    """z = (x - mean(baseline)) / std(baseline), per epoch and channel; baseline = t <= 0."""
    baseline = data[..., times <= 0]
    mean = baseline.mean(axis=-1, keepdims=True)
    std = baseline.std(axis=-1, keepdims=True)
    safe_std = np.where(std > 0, std, 1.0)
    # an epoch whose baseline is flat (disconnected/saturated channel) becomes all zeros
    return np.where(std > 0, (data - mean) / safe_std, 0.0)


# --------------------------------------------------------------------------
# Step 8: statistics
# --------------------------------------------------------------------------
def cluster_statistics(circles: np.ndarray, squares: np.ndarray, times: np.ndarray, n_perm: int, seed: int) -> Dict:
    """Function 1 + function 2 on z-scored data, plus an independent MNE cross-check."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # t-test of a flat channel
        stat, clusters = lab4_stats.nonparametric_test(circles, squares, return_clusters=True)
        pvals, null = lab4_stats.permutation_test(circles, squares, n_perm=n_perm, seed=seed, return_null=True)
        mne_p = mne_cross_check(circles, squares, n_perm, seed)
    fwer_p = channel_corrected_pvalues(stat, null)
    rows = []
    for ch, name in enumerate(CHANNEL_NAMES):
        cluster = clusters[ch]
        rows.append(
            {
                "channel": name,
                "cluster_stat": float(stat[ch]),
                "cluster_start_ms": None if cluster is None else round(1000 * float(times[cluster[0]])),
                "cluster_end_ms": None if cluster is None else round(1000 * float(times[cluster[1] - 1])),
                "p_value": float(pvals[ch]),
                "p_value_mne": mne_p[ch],
                "p_value_corrected": float(fwer_p[ch]),
                "significant": bool(pvals[ch] < P_THRESHOLD),
            }
        )
    return {"rows": rows, "null": null, "stat": stat, "clusters": clusters, "pvals": pvals}


def channel_corrected_pvalues(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """p-values corrected for testing 8 channels (max-statistic, Maris & Oostenveld section 4.4.3).

    Each channel's |statistic| is compared with the null distribution of the
    largest |statistic| over all channels in the same random partition, which
    controls the family-wise error rate over channels as well as time.
    """
    max_null = np.abs(null).max(axis=1)
    return (max_null[:, None] >= np.abs(observed)[None, :]).mean(axis=0)


def mne_cross_check(circles: np.ndarray, squares: np.ndarray, n_perm: int, seed: int) -> List[Optional[float]]:
    """p-value of each channel's largest cluster from mne.stats.permutation_cluster_test,
    with the same threshold (|t| above the two-sided 0.05 critical value) and statistic.

    MNE is given |t| with tail=1 rather than t with tail=0: with tail=0 MNE builds
    its null distribution from the largest *signed* cluster sum of each
    permutation, which ignores strong negative clusters and makes p too small.
    Summing |t| over a cluster and taking the largest is exactly the two-sided
    max-|cluster| statistic of Maris & Oostenveld. MNE also counts the original
    partition as one of the permutations, so its p is ~1/n_perm larger.
    """
    threshold = stats.t.ppf(1 - lab4_stats.SAMPLE_ALPHA / 2, len(circles) + len(squares) - 2)

    def t_stat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.abs(stats.ttest_ind(a, b, axis=0).statistic)

    # newer MNE renamed seed= to rng=
    seed_kw = "rng" if "rng" in inspect.signature(mne.stats.permutation_cluster_test).parameters else "seed"
    pvalues: List[Optional[float]] = []
    for ch in range(circles.shape[1]):
        t_obs, clusters, cluster_p, _ = mne.stats.permutation_cluster_test(
            [circles[:, ch], squares[:, ch]],
            threshold=threshold,
            n_permutations=n_perm,
            tail=1,
            stat_fun=t_stat,
            out_type="mask",
            verbose="ERROR",
            **{seed_kw: seed},
        )
        if len(clusters) == 0:
            pvalues.append(None)
            continue
        sums = [np.nan_to_num(t_obs)[c].sum() for c in clusters]
        pvalues.append(float(cluster_p[int(np.argmax(sums))]))
    return pvalues




# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def is_openbci_recording(path: str) -> bool:
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            return handle.readline().strip() == OPENBCI_FIRST_LINE
    except OSError:
        return False


def find_recordings(folder: str = HERE) -> Dict[str, str]:
    """Every OpenBCI GUI .txt under <folder> (skipping .venv, report, demo), tagged run1, run2, ..."""
    found = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        found += [os.path.join(root, f) for f in files if f.endswith(".txt") and is_openbci_recording(os.path.join(root, f))]
    if not found:
        sys.exit(f"No OpenBCI recording (.txt starting with '{OPENBCI_FIRST_LINE}') under {folder}.")
    return {f"run{i + 1}": path for i, path in enumerate(sorted(found))}


def condition_summary(circles: np.ndarray, squares: np.ndarray, times: np.ndarray) -> List[Dict]:
    """Peak of the circle-minus-square difference wave (z units) per channel, after 0 s."""
    diff = circles.mean(axis=0) - squares.mean(axis=0)
    post = times > 0
    rows = []
    for ch, name in enumerate(CHANNEL_NAMES):
        i = int(np.argmax(np.abs(diff[ch, post])))
        rows.append({"channel": name, "peak_ms": round(1000 * float(times[post][i])), "peak_diff_z": float(diff[ch, post][i])})
    return rows


class PreparedRun(NamedTuple):
    cropped: mne.io.BaseRaw  # filtered, cropped recording (EEG + photoresistor)
    events: np.ndarray  # events of the video trials in <cropped>
    trial_number: np.ndarray  # trial (0..89) of each event
    info: Dict  # numbers for results.json


def _print_match_warnings(match: TrialMatch, file_info: Dict) -> None:
    detected = match.trial_onset >= 0
    if not match.anchored:
        print("  WARNING: no sync flash found before the trials. If trial 1 was missed, every circle/square label is")
        print("           shifted by one trial. Check the photoresistor figure before trusting the statistics.")
    if detected.sum() < N_TRIALS:
        print(f"  WARNING: trials not detected (1-based): {[int(k) + 1 for k in np.flatnonzero(~detected)]}; they are left out.")
    if file_info["photoresistor_inverted"]:
        print("  WARNING: D13 reads 0 when bright in this file; it was inverted.")
    if file_info["lost_samples"]:
        print(f"  WARNING: {file_info['lost_samples']} samples lost over the radio link.")


def prepare_run(tag: str, recording: str, mask: np.ndarray, save_dir: str) -> PreparedRun:
    """Manual steps 1-5 for one recording, plus its data-quality figures."""
    import lab4_plots

    print(f"[{tag}] {recording}")
    sfreq = parse_sample_rate(recording)
    df, file_info = load_recording(recording)
    raw = build_raw(df, sfreq)
    print(f"  {len(df)} samples at {sfreq:g} Hz ({len(df) / sfreq:.1f} s); D13 read from '{file_info['photoresistor_column']}'")

    events = find_flash_events(raw)
    onsets_s = events[:, 0] / sfreq
    raw = annotate_events(raw, events)
    match = match_video_trials(onsets_s)
    detected = match.trial_onset >= 0
    match_samples = np.where(detected, events[np.maximum(match.trial_onset, 0), 0], -1)
    print(f"  {len(events)} photoresistor onsets, {detected.sum()}/{N_TRIALS} trials matched")
    _print_match_warnings(match, file_info)

    filtered = bandpass(raw)
    crop_start, crop_stop = crop_window(onsets_s, match, raw.times[-1], ttl_offsets_s(raw))
    cropped = filtered.copy().crop(tmin=crop_start, tmax=crop_stop)
    events_crop, trial_number, n_events_in_crop = trial_events(cropped, match_samples)
    print(f"  cropped to {crop_start:.2f}-{crop_stop:.2f} s: find_events gives {n_events_in_crop} events, {len(events_crop)} are trials")

    sync_onsets = 0 if match.burst is None else match.burst.n_flashes
    highs = run_lengths(raw.get_data(picks=STIM_CHANNEL)[0] > 0.5) / sfreq
    std_uv = cropped.get_data(picks="eeg").std(axis=1) / MICROVOLT_TO_VOLT
    info = {
        "recording": os.path.basename(recording),
        "sfreq": sfreq,
        "n_samples": len(df),
        "duration_s": round(len(df) / sfreq, 1),
        **file_info,
        "n_onsets_total": int(len(events)),
        "alignment_anchored": match.anchored,
        "sync_onsets_detected": None if match.burst is None else sync_onsets,
        "median_flash_ms": round(1000 * float(np.median(highs))),
        "video_start_s": round(match.video_start_s, 2),
        "n_trials_matched": int(detected.sum()),
        "missed_trials": [int(k) for k in np.flatnonzero(~detected)],
        "n_unmatched_onsets": int(len(events) - detected.sum() - sync_onsets),
        "crop_start_s": round(crop_start, 2),
        "crop_stop_s": round(crop_stop, 2),
        "n_events_in_crop": n_events_in_crop,
        "channel_std_uv": {n: round(float(s), 1) for n, s in zip(CHANNEL_NAMES, std_uv)},
    }
    info["flat_channels"] = [n for n, s in info["channel_std_uv"].items() if s < FLAT_CHANNEL_STD_UV]

    lab4_plots.plot_photoresistor(raw, onsets_s, match, (crop_start, crop_stop), mask, save_dir, f"{tag}_photoresistor.png")
    lab4_plots.plot_raw_with_annotations(cropped, save_dir, f"{tag}_raw_annotations.png")
    lab4_plots.plot_psd_before_after(raw, filtered, save_dir, f"{tag}_psd_before_after.png")
    return PreparedRun(cropped, events_crop, trial_number, info)


class ConditionData(NamedTuple):
    circles: np.ndarray  # z-scored, (n_circles, n_channels, n_times)
    squares: np.ndarray
    times: np.ndarray
    circle_epochs: mne.Epochs
    square_epochs: mne.Epochs


def condition_data(run: PreparedRun, mask: np.ndarray, tmax: float) -> ConditionData:
    """Steps 5-7: epochs (MNE defaults, tmax as given), split with the mask, z-score."""
    epochs = make_epochs(run.cropped, run.events, tmax=tmax)
    circle_ep, square_ep = split_epochs(epochs, run.trial_number, mask)
    times = epochs.times
    return ConditionData(
        zscore_epochs(circle_ep.get_data(copy=True), times),
        zscore_epochs(square_ep.get_data(copy=True), times),
        times,
        circle_ep,
        square_ep,
    )


def _print_test(tag: str, label: str, block: Dict, rows: List[Dict]) -> None:
    sig = ", ".join(block["significant_channels"]) or "none"
    print(f"  [{tag} {label} {EPOCH_TMIN_S:+.1f}..{block['tmax_s']:.1f} s] {block['n_circles']} circles / {block['n_squares']} squares; p < {P_THRESHOLD}: {sig}")
    for row in rows:
        window = "-" if row["cluster_start_ms"] is None else f"{row['cluster_start_ms']}..{row['cluster_end_ms']} ms"
        mne_p = "  -  " if row["p_value_mne"] is None else f"{row['p_value_mne']:.3f}"
        print(
            f"      {row['channel']:>3}: cluster sum t = {row['cluster_stat']:8.2f}  {window:>16}"
            f"  p = {row['p_value']:.3f}  (MNE {mne_p}, corrected over channels {row['p_value_corrected']:.3f})"
        )


def compare_conditions(tag: str, label: str, data: ConditionData, n_perm: int, seed: int, save_dir: str) -> Dict:
    """Step 8 + figures for one set of circle/square epochs."""
    import lab4_plots

    circles, squares, times = data.circles, data.squares, data.times
    test = cluster_statistics(circles, squares, times, n_perm, seed)
    block = {
        "tmin_s": EPOCH_TMIN_S,
        "tmax_s": round(float(times[-1]), 3),
        "n_epochs": len(circles) + len(squares),
        "n_times": len(times),
        "n_circles": len(circles),
        "n_squares": len(squares),
        "channels": test["rows"],
        "difference_peaks": condition_summary(circles, squares, times),
        "significant_channels": [r["channel"] for r in test["rows"] if r["significant"]],
        **sample_wise_counts(circles, squares),
    }
    add_cluster_microvolts(block["channels"], test["clusters"], data)
    prefix = f"{tag}_{label}"
    lab4_plots.plot_evoked_grid(circles, squares, times, test, f"{prefix}_erp_all.png", save_dir)
    lab4_plots.plot_significant(circles, squares, times, test, f"{prefix}_erp_significant.png", save_dir)
    lab4_plots.plot_null_distributions(test, f"{prefix}_null.png", save_dir)
    _print_test(tag, label, block, test["rows"])
    return block


def sample_wise_counts(circles: np.ndarray, squares: np.ndarray) -> Dict:
    """How many (channel, time) samples a plain t-test calls significant, uncorrected
    and Bonferroni-corrected: the multiple comparisons problem in numbers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p = stats.ttest_ind(circles, squares, axis=0).pvalue
    return {
        "n_tests": int(p.size),
        "uncorrected_samples": int(np.sum(p < lab4_stats.SAMPLE_ALPHA)),
        "bonferroni_samples": int(np.sum(p < lab4_stats.SAMPLE_ALPHA / p.size)),
    }


def add_cluster_microvolts(rows: List[Dict], clusters: Sequence, data: ConditionData) -> None:
    """Adds to every row the circle-minus-square difference (uV) averaged over its largest cluster."""
    circle_uv = data.circle_epochs.get_data(copy=True).mean(axis=0) / MICROVOLT_TO_VOLT
    square_uv = data.square_epochs.get_data(copy=True).mean(axis=0) / MICROVOLT_TO_VOLT
    for ch, (row, cluster) in enumerate(zip(rows, clusters)):
        if cluster is None:
            row["cluster_diff_uv"] = None
            continue
        window = slice(cluster[0], cluster[1])
        row["cluster_diff_uv"] = round(float((circle_uv[ch, window] - square_uv[ch, window]).mean()), 1)


def pool(datasets: Sequence[ConditionData]) -> ConditionData:
    """Epochs of several runs of the same subject, concatenated per condition."""
    return ConditionData(
        np.concatenate([d.circles for d in datasets]),
        np.concatenate([d.squares for d in datasets]),
        datasets[0].times,
        mne.concatenate_epochs([d.circle_epochs for d in datasets], verbose="ERROR"),
        mne.concatenate_epochs([d.square_epochs for d in datasets], verbose="ERROR"),
    )


WINDOWS = (("main", EPOCH_TMAX_S), ("extended", EXTENDED_TMAX_S))


def analyse(
    recordings: Dict[str, str],
    save_dir: str,
    results_path: str,
    n_perm: int = N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
    is_demo: bool = False,
) -> Dict:
    """Every recording on its own and, if there are several, all of them pooled."""
    import lab4_plots

    os.makedirs(save_dir, exist_ok=True)
    mask = load_circle_mask(os.path.join(HERE, CIRCLE_MASK_FILE))
    results: Dict = {"is_demo": is_demo, "n_permutations": n_perm, "seed": seed, "notch_hz": NOTCH_HZ, "runs": {}, "pooled": None}
    per_window: Dict[str, List[ConditionData]] = {label: [] for label, _ in WINDOWS}

    for tag, recording in recordings.items():
        run = prepare_run(tag, recording, mask, save_dir)
        results["runs"][tag] = dict(run.info)
        for label, tmax in WINDOWS:
            data = condition_data(run, mask, tmax)
            per_window[label].append(data)
            results["runs"][tag][label] = compare_conditions(tag, label, data, n_perm, seed, save_dir)
            if label == "main":
                lab4_plots.plot_mne_evoked(data.circle_epochs, data.square_epochs, save_dir, f"{tag}_evoked_microvolts.png")

    if len(recordings) > 1:
        print(f"[pooled] {', '.join(recordings)}")
        results["pooled"] = {}
        for label, _ in WINDOWS:
            data = pool(per_window[label])
            results["pooled"][label] = compare_conditions("pooled", label, data, n_perm, seed, save_dir)
            if label == "main":
                lab4_plots.plot_mne_evoked(data.circle_epochs, data.square_epochs, save_dir, "pooled_evoked_microvolts.png")

    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("Figures in", save_dir, "| numbers in", results_path)
    return results


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="*", help="OpenBCI GUI .txt files (default: RECORDING_FILES, else auto-detect)")
    parser.add_argument("--demo", action="store_true", help="analyse a synthetic recording instead")
    parser.add_argument("--n-perm", type=int, default=N_PERMUTATIONS)
    args = parser.parse_args(argv)

    if args.demo:
        import lab4_synthetic

        path = os.path.join(HERE, "demo", "OpenBCI-RAW-demo.txt")
        lab4_synthetic.write_recording(path)
        recordings = {"run1": path}
    elif args.recordings:
        recordings = {f"run{i + 1}": path for i, path in enumerate(args.recordings)}
    elif RECORDING_FILES:
        recordings = {tag: os.path.join(HERE, path) for tag, path in RECORDING_FILES.items()}
    else:
        recordings = find_recordings()
    return analyse(
        recordings,
        os.path.join(HERE, SAVE_DIR),
        os.path.join(HERE, RESULTS_FILE),
        n_perm=args.n_perm,
        is_demo=args.demo,
    )


if __name__ == "__main__":
    main()
