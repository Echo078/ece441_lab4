"""
Measures the stimulus timing of Lab4_video.mp4 frame by frame and checks it
against circle_mask.csv. The numbers it prints are the video facts used by
lab4_analysis.py and quoted in the report.

Needs OpenCV, which the analysis itself does not:
    .venv\\Scripts\\python -m pip install opencv-python-headless
    .venv\\Scripts\\python video_check.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "Lab4_video.mp4")
MASK = os.path.join(HERE, "circle_mask.csv")
MARKER_BOX = (slice(985, None), slice(0, 95))  # lower-left white square (1920x1080 video)
STIMULUS_BOX = (slice(260, 820), slice(480, 1440))  # centre of the screen
BRIGHT = 127
MIN_PIXELS = 1000


def frame_features(frame: np.ndarray):
    """(marker brightness, red pixels, blue pixels) of one BGR frame."""
    b, g, r = (frame[STIMULUS_BOX][:, :, k].astype(int) for k in range(3))
    red = int(((r > 150) & (g < 90) & (b < 90)).sum())
    blue = int(((b > 150) & (r < 100)).sum())
    return float(frame[MARKER_BOX].mean()), red, blue


def main() -> None:
    try:
        import cv2
    except ImportError:
        sys.exit("OpenCV is not installed: pip install opencv-python-headless")
    capture = cv2.VideoCapture(VIDEO)
    fps = capture.get(cv2.CAP_PROP_FPS)
    rows = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rows.append(frame_features(frame))
    marker, red, blue = (np.array(col) for col in zip(*rows))
    print(f"{len(rows)} frames at {fps:g} fps ({len(rows) / fps:.1f} s)")

    bright = marker > BRIGHT
    edges = np.diff(np.r_[0, bright.astype(int), 0])
    onsets, offsets = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    gaps = np.diff(onsets) / fps
    n_sync = 1 + int(np.argmax(gaps > 0.5))
    trials = onsets[n_sync:]
    print(f"{len(onsets)} white markers, each {set(np.round((offsets - onsets) / fps, 3))} s long")
    print(f"sync flashes: {n_sync} at {np.round(onsets[:n_sync] / fps, 2)} s")
    print(f"trials: {len(trials)}; first {(trials[0] - onsets[0]) / fps:.2f} s after the first sync flash; "
          f"interval {set(np.round(np.diff(trials) / fps, 2))} s")

    is_circle = np.array([red[k : k + 3].max() > MIN_PIXELS for k in trials]).astype(int)
    mask = np.loadtxt(MASK, delimiter=",").astype(int)
    print(f"circles in the video: {is_circle.sum()}; agree with circle_mask.csv: {bool(np.array_equal(is_circle, mask))}")
    on_screen = (red > MIN_PIXELS) | (blue > MIN_PIXELS)
    runs = np.diff(np.r_[0, on_screen.astype(int), 0])
    durations = (np.flatnonzero(runs == -1) - np.flatnonzero(runs == 1)) / fps
    print(f"stimulus on screen for {set(np.round(durations, 2))} s")
    print(f"area: blue square {blue.max()} px, red circle {red.max()} px (ratio {blue.max() / red.max():.1f})")


if __name__ == "__main__":
    main()
