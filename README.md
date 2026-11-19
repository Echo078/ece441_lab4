# ECE441 Lab 4 · Visual oddball EEG

One participant completed two runs of a visual oddball task, pressing a button for red circles and ignoring blue squares. The analysis compares event-related potentials using the cluster-based permutation test of [Maris and Oostenveld (2007)](https://doi.org/10.1016/j.jneumeth.2007.03.024).

## Report and submission

- [Report PDF](report/lab4_report.pdf) includes the ERP plots, statistical tables, and code appendices.
- [Overleaf project ZIP](report/ece441_lab4_overleaf.zip) contains the LaTeX source and all assets. Upload the ZIP as a new Overleaf project and compile `main.tex`.
- [LaTeX source](report/main.tex) can also be edited in the repo. Compile from inside `report/` so the figure paths resolve.

Complete the two remaining author entries in `main.tex` before submitting. The Overleaf ZIP and the checked-in PDF reflect the current draft.

## Reproduce the analysis

The inputs are [the two OpenBCI recordings](Data/), [the stimulus labels](circle_mask.csv), and [the stimulus video](Lab4_video.mp4) for timing verification. [The analysis script](lab4_analysis.py) loads and aligns the recordings, filters and epochs the EEG, and calls [the two statistical functions](lab4_stats.py). [The saved results](report/results.json) contain the values used in the report. The report source contains the tables, so rerunning the analysis does not update the LaTeX text automatically.

From the repo root, using Python 3.11:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python lab4_analysis.py
python -m pytest test_lab4.py
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1` to activate the environment. The analysis writes `report/results.json` and plots in `report/img/`. To verify the video timing separately, install `opencv-python-headless` and run `python video_check.py`.
