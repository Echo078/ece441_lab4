"""
Synthetic OpenBCI recording of the Lab 4 experiment, for tests and `--demo`.

It reproduces the file format of the OpenBCI GUI (Cyton, Digital Read mode, D13
in column 'Other.2') and the things that can go wrong in the lab:

* photoresistor checks with a flashlight / hand before the video,
* a test playback of the first seconds of the video (a second sync burst),
* a bright desktop after the video ends (a long TTL after the last trial),
* drift between the EEG clock and the video clock, lost radio packets,
* 60 Hz mains noise, DC offsets and slow drifts on the electrodes.

The EEG is 1/f noise plus alpha, a visual evoked response to every stimulus
and a P300-like positivity that is larger for circles (targets).
"""

import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from lab4_analysis import (
    CHANNEL_NAMES,
    N_TRIALS,
    VIDEO_FIRST_STIMULUS_AFTER_SYNC_S,
    VIDEO_FLASH_DURATION_S,
    VIDEO_STIMULUS_INTERVAL_S,
    VIDEO_SYNC_FLASHES,
    VIDEO_SYNC_INTERVAL_S,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SFREQ = 250.0
STOP_BYTE = 192.0
VIDEO_START_S = 14.0
TEST_PLAYBACK_START_S = 6.0
TEST_PLAYBACK_LENGTH_S = 3.0
FLASHLIGHT_CHECKS_S = ((1.0, 1.6), (2.5, 2.8))
DESKTOP_AFTER_VIDEO_S = (2.0, 3.0)  # (delay after last trial, duration)
TAIL_S = 6.0
FILE_COLUMNS = (
    ["Sample Index"]
    + [f"EXG Channel {i}" for i in range(8)]
    + [f"Accel Channel {i}" for i in range(3)]
    + ["Other"] * 7
    + [f"Analog Channel {i}" for i in range(3)]
    + ["Timestamp", "Other", "Timestamp (Formatted)"]
)
D13_OTHER_INDEX = 2  # 'Other.2'

# Where each response is strongest (weight per channel, order of CHANNEL_NAMES)
VISUAL_WEIGHTS = np.array([0.1, 0.1, 0.3, 0.3, 0.8, 0.8, 1.0, 1.0])
P300_WEIGHTS = np.array([0.3, 0.3, 0.9, 0.9, 1.0, 1.0, 0.7, 0.7])


def _gauss(t: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - centre) / width) ** 2)


def pink_noise(n_rows: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal((n_rows, n_samples)), axis=1)
    freqs = np.fft.rfftfreq(n_samples, 1 / SFREQ)
    freqs[0] = freqs[1]
    noise = np.fft.irfft(spectrum / np.sqrt(freqs), n=n_samples, axis=1)
    return noise / noise.std(axis=1, keepdims=True)


def video_onsets_s(video_start: float) -> np.ndarray:
    """Video-clock onset times of the sync flashes followed by the 90 trials."""
    sync = video_start + VIDEO_SYNC_INTERVAL_S * np.arange(VIDEO_SYNC_FLASHES)
    trials = video_start + VIDEO_FIRST_STIMULUS_AFTER_SYNC_S + VIDEO_STIMULUS_INTERVAL_S * np.arange(N_TRIALS)
    return np.concatenate([sync, trials])


def make_recording(
    seed: int = 0,
    circle_mask: Optional[np.ndarray] = None,
    p300_uv: float = 7.0,
    clock_drift: float = 0.003,
    missed_trials: Iterable[int] = (),
    missed_sync: Iterable[int] = (),
    lost_packets: int = 5,
    with_distractors: bool = True,
) -> pd.DataFrame:
    """DataFrame with the column layout of an OpenBCI GUI file (names without the leading space)."""
    rng = np.random.default_rng(seed)
    mask = circle_mask if circle_mask is not None else np.loadtxt(os.path.join(HERE, "circle_mask.csv"), delimiter=",").astype(int)
    onsets = video_onsets_s(VIDEO_START_S)
    sync_part, trial_part = onsets[:VIDEO_SYNC_FLASHES], onsets[VIDEO_SYNC_FLASHES:]
    sync_part = np.delete(sync_part, list(missed_sync))
    detected_trials = np.delete(trial_part, list(missed_trials))
    flashes = list(np.concatenate([sync_part, detected_trials]))
    long_highs = []
    if with_distractors:
        flashes += [t for t in video_onsets_s(TEST_PLAYBACK_START_S) if t < TEST_PLAYBACK_START_S + TEST_PLAYBACK_LENGTH_S]
        long_highs += list(FLASHLIGHT_CHECKS_S)
        desktop = trial_part[-1] + DESKTOP_AFTER_VIDEO_S[0]
        long_highs.append((desktop, desktop + DESKTOP_AFTER_VIDEO_S[1]))

    duration = trial_part[-1] + TAIL_S
    n = int(duration * SFREQ * (1 + clock_drift))
    t = np.arange(n) / SFREQ  # EEG clock
    to_eeg = 1 + clock_drift  # video second -> EEG seconds

    ttl = np.zeros(n)
    for onset in flashes:
        ttl[(t >= onset * to_eeg) & (t < (onset + VIDEO_FLASH_DURATION_S) * to_eeg)] = 1
    for start, stop in long_highs:
        ttl[(t >= start * to_eeg) & (t < stop * to_eeg)] = 1

    eeg = 10 * pink_noise(8, n, rng) + 3 * rng.standard_normal((8, n))
    eeg[6:] += 6 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 2 * np.pi, (2, 1)))  # occipital alpha
    eeg += 25 * np.sin(2 * np.pi * 60 * t)  # mains
    eeg += rng.uniform(-30000, 30000, (8, 1)) + 200 * np.sin(2 * np.pi * 0.05 * t)  # offsets + drift
    for k, onset in enumerate(trial_part):
        latency = t - onset * to_eeg
        window = (latency > -0.1) & (latency < 1.0)
        lat = latency[window]
        visual = 4 * _gauss(lat, 0.10, 0.02) - 5 * _gauss(lat, 0.17, 0.03)
        p300 = (p300_uv if mask[k] else 0.15 * p300_uv) * _gauss(lat, 0.38, 0.09)
        eeg[:, window] += VISUAL_WEIGHTS[:, None] * visual + P300_WEIGHTS[:, None] * p300

    keep = np.ones(n, dtype=bool)
    for start in rng.integers(int(5 * SFREQ), n - int(5 * SFREQ), lost_packets):
        keep[start : start + 2] = False  # each lost packet = 2 samples gone, timeline shortened
    index = np.arange(n)[keep]
    eeg, ttl = eeg[:, keep], ttl[keep]
    n_kept = len(index)

    columns = {"Sample Index": (index % 256).astype(float)}
    columns.update({f"EXG Channel {i}": eeg[i] for i in range(8)})
    columns.update({f"Accel Channel {i}": np.zeros(n_kept) for i in range(3)})
    for j in range(7):
        name = "Other" if j == 0 else f"Other.{j}"
        columns[name] = np.full(n_kept, STOP_BYTE) if j == 0 else (ttl if j == D13_OTHER_INDEX else np.zeros(n_kept))
    columns.update({f"Analog Channel {i}": np.zeros(n_kept) for i in range(3)})
    columns["Timestamp"] = 1.79e9 + index / SFREQ
    columns["Other.7"] = np.zeros(n_kept)
    columns["Timestamp (Formatted)"] = pd.to_datetime(columns["Timestamp"], unit="s").strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    return pd.DataFrame(columns)


def write_recording(path: str, **kwargs) -> str:
    """Writes make_recording(**kwargs) to <path> in the OpenBCI GUI text format."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df = make_recording(**kwargs)
    header = [
        "%OpenBCI Raw EEG Data",
        "%Number of channels = 8",
        f"%Sample Rate = {SFREQ:g} Hz",
        "%Board = OpenBCI_GUI$BoardCytonSerial",
        ", ".join(FILE_COLUMNS),
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(header) + "\n")
        df.to_csv(handle, header=False, index=False, sep=",", float_format="%.6f", lineterminator="\n")
    return path
