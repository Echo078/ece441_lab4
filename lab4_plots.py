"""Figures for lab4_analysis.py. Every function saves one PNG into <save_dir>."""

import os
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np

from lab4_analysis import (
    CHANNEL_NAMES,
    N_TRIALS,
    P_THRESHOLD,
    STIM_CHANNEL,
    TrialMatch,
    trial_times,
)

FIGURE_DPI = 150
CIRCLE_COLOUR = "#c62828"  # red circles (targets)
SQUARE_COLOUR = "#1f5fa8"  # blue squares (non-targets)
CLUSTER_COLOUR = "#f2c14e"
GRID_SHAPE = (2, 4)
RAW_PLOT_DURATION_S = 10.0
RAW_PLOT_SCALING_V = 40e-6
PSD_FMAX_HZ = 80.0


def filter_title() -> str:
    import lab4_analysis

    notch = f" + {lab4_analysis.NOTCH_HZ:g} Hz notch" if lab4_analysis.NOTCH_HZ else ""
    return f"band-pass {lab4_analysis.BAND_LOW_HZ:g}-{lab4_analysis.BAND_HIGH_HZ:g} Hz{notch}"


def _p_text(p: float, n_perm: int) -> str:
    """'p = 0.012', or 'p < 0.001' when no permutation reached the observed value."""
    return f"p < {1 / n_perm:.3f}" if p == 0 else f"p = {p:.3f}"


def _save(fig: plt.Figure, save_dir: str, name: str) -> None:
    fig.savefig(os.path.join(save_dir, name), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_photoresistor(
    raw: mne.io.BaseRaw,
    onsets_s: np.ndarray,
    match: TrialMatch,
    crop: Tuple[float, float],
    mask: np.ndarray,
    save_dir: str,
    name: str = "photoresistor.png",
) -> None:
    """Whole photoresistor trace with the matched trials, plus a zoom on the video start."""
    ttl = raw.get_data(picks=STIM_CHANNEL)[0]
    times = raw.times
    detected = match.trial_onset >= 0
    trial_t = onsets_s[match.trial_onset[detected]]
    is_circle = mask[detected] == 1
    used = set(match.trial_onset[detected].tolist())
    if match.burst is not None:
        used |= set(range(match.burst.first, match.burst.last + 1))
    other = np.array([t for i, t in enumerate(onsets_s) if i not in used])

    fig, (ax_all, ax_zoom) = plt.subplots(2, 1, figsize=(11, 5.2), gridspec_kw={"height_ratios": [1, 1]})
    ax_all.plot(times, ttl, color="0.35", lw=0.6)
    ax_all.axvspan(*crop, color="#e8f0e0", zorder=0, label="cropped window")
    ax_all.plot(trial_t[is_circle], np.full(is_circle.sum(), 1.15), "o", color=CIRCLE_COLOUR, ms=4, label="circle trial")
    ax_all.plot(trial_t[~is_circle], np.full((~is_circle).sum(), 1.15), "s", color=SQUARE_COLOUR, ms=3, label="square trial")
    if match.burst is not None:
        burst_t = onsets_s[match.burst.first : match.burst.last + 1]
        ax_all.plot(burst_t, np.full(len(burst_t), 1.3), "v", color="k", ms=4, label="sync burst")
    if len(other):
        ax_all.plot(other, np.full(len(other), 1.3), "x", color="#888", ms=5, label="other onset (not a trial)")
    ax_all.set(xlim=(times[0], times[-1]), ylim=(-0.1, 1.45), yticks=[0, 1], ylabel="D13", xlabel="time (s)")
    ax_all.set_title(f"Photoresistor (D13): {detected.sum()}/{N_TRIALS} video trials found", loc="left", fontsize=10)
    ax_all.legend(loc="upper right", fontsize=7, ncol=5, framealpha=0.9)

    start = match.video_start_s - 0.5
    stop = trial_times(onsets_s, match)[2] + 0.3
    window = (times >= start) & (times <= stop)
    ax_zoom.plot(times[window], ttl[window], color="0.2", lw=1)
    ax_zoom.axvline(crop[0], color="g", ls="--", lw=1, label=f"crop start ({crop[0]:.2f} s)")
    ax_zoom.set(xlim=(start, stop), ylim=(-0.1, 1.15), yticks=[0, 1], ylabel="D13", xlabel="time (s)")
    ax_zoom.set_title("Start of the video: sync flashes 100 ms apart, then the first trials every 1.2 s", loc="left", fontsize=10)
    ax_zoom.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    _save(fig, save_dir, name)


def plot_raw_with_annotations(cropped: mne.io.BaseRaw, save_dir: str, name: str = "raw_annotations.png") -> None:
    """MNE browser view of the first 10 s of the cropped, filtered data with the 'flash' annotations."""
    mne.viz.set_browser_backend("matplotlib", verbose="ERROR")
    fig = cropped.plot(
        start=0,
        duration=RAW_PLOT_DURATION_S,
        scalings={"eeg": RAW_PLOT_SCALING_V, "stim": 1},
        show=False,
        show_scrollbars=False,
        verbose="ERROR",
    )
    fig.set_size_inches(11, 5.5)
    _save(fig, save_dir, name)


def plot_psd_before_after(raw: mne.io.BaseRaw, filtered: mne.io.BaseRaw, save_dir: str, name: str = "psd_before_after.png") -> None:
    """PSD of every EEG channel before and after filtering."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, data, title in ((axes[0], raw, "raw"), (axes[1], filtered, filter_title())):
        spectrum = data.compute_psd(picks="eeg", fmax=PSD_FMAX_HZ, verbose="ERROR")
        psd_db = 10 * np.log10(spectrum.get_data() * 1e12)  # V^2/Hz -> dB re 1 uV^2/Hz
        for ch, channel in enumerate(CHANNEL_NAMES):
            ax.plot(spectrum.freqs, psd_db[ch], lw=0.9, label=channel)
        ax.axvline(60, color="0.6", ls=":", lw=1)
        ax.set(title=title, xlabel="frequency (Hz)", xlim=(0, PSD_FMAX_HZ))
    axes[0].set_ylabel("PSD (dB re 1 uV²/Hz)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, save_dir, name)


def _draw_evoked(ax: plt.Axes, circles: np.ndarray, squares: np.ndarray, times: np.ndarray, ch: int, test: Dict) -> None:
    ms = 1000 * times
    for data, colour, label in ((circles, CIRCLE_COLOUR, "circle"), (squares, SQUARE_COLOUR, "square")):
        mean = data[:, ch].mean(axis=0)
        sem = data[:, ch].std(axis=0, ddof=1) / np.sqrt(len(data))
        ax.fill_between(ms, mean - sem, mean + sem, color=colour, alpha=0.2, lw=0)
        ax.plot(ms, mean, color=colour, lw=1.3, label=f"{label} (n={len(data)})")
    cluster = test["clusters"][ch]
    if cluster is not None:
        ax.axvspan(ms[cluster[0]], ms[cluster[1] - 1], color=CLUSTER_COLOUR, alpha=0.35, lw=0, label="largest cluster")
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.4)
    p = test["pvals"][ch]
    ax.set_title(f"{CHANNEL_NAMES[ch]}   {_p_text(p, len(test['null']))}" + ("  *" if p < P_THRESHOLD else ""), fontsize=9.5, fontweight="bold" if p < P_THRESHOLD else "normal")
    ax.set_xlim(ms[0], ms[-1])


def plot_evoked_grid(circles: np.ndarray, squares: np.ndarray, times: np.ndarray, test: Dict, name: str, save_dir: str) -> None:
    """Mean +/- SEM z-scored response of every channel, circles vs squares."""
    fig, axes = plt.subplots(*GRID_SHAPE, figsize=(12, 5.4), sharex=True, sharey=True)
    for ch, ax in enumerate(axes.flat):
        _draw_evoked(ax, circles, squares, times, ch, test)
    for ax in axes[-1]:
        ax.set_xlabel("time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("z-score")
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    _save(fig, save_dir, name)


def plot_significant(circles: np.ndarray, squares: np.ndarray, times: np.ndarray, test: Dict, name: str, save_dir: str) -> None:
    """Manual 'Plot data' step: only channels whose permutation p-value is below 0.05."""
    significant = [ch for ch in range(len(CHANNEL_NAMES)) if test["pvals"][ch] < P_THRESHOLD]
    path = os.path.join(save_dir, name)
    if not significant:
        if os.path.exists(path):
            os.remove(path)  # do not leave a figure from an earlier run behind
        return
    n_cols = min(len(significant), 4)
    n_rows = int(np.ceil(len(significant) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.9 * n_rows), squeeze=False, sharey=True)
    for ax in axes.flat[len(significant):]:
        ax.set_visible(False)
    for ax, ch in zip(axes.flat, significant):
        _draw_evoked(ax, circles, squares, times, ch, test)
        ax.set_xlabel("time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("z-score")
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    _save(fig, save_dir, name)


def plot_null_distributions(test: Dict, name: str, save_dir: str) -> None:
    """Permutation distribution of |largest cluster statistic| per channel, observed value in red."""
    null = np.abs(test["null"])
    fig, axes = plt.subplots(*GRID_SHAPE, figsize=(12, 4.6))
    for ch, ax in enumerate(axes.flat):
        ax.hist(null[:, ch], bins=40, color="0.7")
        ax.axvline(abs(test["stat"][ch]), color=CIRCLE_COLOUR, lw=1.5)
        ax.set_title(f"{CHANNEL_NAMES[ch]}   {_p_text(test['pvals'][ch], len(null))}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("|cluster sum of t|", fontsize=8)
    fig.tight_layout()
    _save(fig, save_dir, name)


def plot_mne_evoked(circle_ep: mne.Epochs, square_ep: mne.Epochs, save_dir: str, name: str = "evoked_microvolts.png") -> None:
    """Evoked responses in microvolts (baseline-corrected, not z-scored), for reference."""
    evokeds = {"circle": circle_ep.average(), "square": square_ep.average()}
    fig, axes = plt.subplots(*GRID_SHAPE, figsize=(12, 5.0), sharex=True, sharey=True)
    for ch, ax in enumerate(axes.flat):
        for label, colour in (("circle", CIRCLE_COLOUR), ("square", SQUARE_COLOUR)):
            evoked = evokeds[label]
            ax.plot(1000 * evoked.times, 1e6 * evoked.data[ch], color=colour, lw=1.2, label=label)
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(CHANNEL_NAMES[ch], fontsize=9.5)
    for ax in axes[-1]:
        ax.set_xlabel("time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("µV")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, save_dir, name)
