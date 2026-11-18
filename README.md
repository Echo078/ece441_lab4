# Lab 4: ERPs to circles vs squares + cluster-based permutation test

Step-by-step operating guide for both labs: [`Lab4_Lab5_Guide.pdf`](../Lab4_Lab5_Guide.pdf).

One subject watches `Lab4_video.mp4` and presses the PCB button for every red circle (15 of 90 stimuli). A photoresistor on the lower-left corner of the screen drives the Cyton's digital input D13, which marks every stimulus onset. The analysis epochs the EEG around those onsets and tests circles vs squares on each channel with the cluster-based permutation test of Maris & Oostenveld (2007).

- Lab manual: `Lab 4 (1).pdf` (password-protected for printing, but readable). Paper: `Maris&Oostenveld2007.pdf`.
- Deliverables: in-lab demo (2 %) + **group lab report** (4 %): `report/lab4_report.pdf`.

## Files

| File | What it does |
|------|--------------|
| `lab4_analysis.py` | The whole "Data analysis" section of the manual: load → MNE RawArray → photoresistor events/annotations → 1–50 Hz band-pass → crop → epochs → split with `circle_mask.csv` → z-score → statistics → figures. Writes `report/img/*.png` and `report/results.json`, prints a summary. Config block at the top. |
| `lab4_stats.py` | **Function 1** `nonparametric_test` (cluster statistic, paper §3.1.1) and **function 2** `permutation_test` (permutation p-value, paper §2.3). |
| `lab4_plots.py` | Figures. |
| `lab4_synthetic.py` | Writes a fake OpenBCI recording of this experiment (used by the tests and `--demo`). |
| `test_lab4.py` | 38 pytest tests. |
| `report/report_template.html`, `report/build_report.py` | Report source → `report/lab4_report.html` + `.pdf` (numbers and tables are filled in from `results.json`). |
| `circle_mask.csv`, `Lab4_video.mp4` | Course files. The mask was checked against the video frame by frame: all 90 trials agree. |

## Setup (once)

Skip this if `lab4_files/.venv` already exists. Otherwise (Python 3.11, see the [main README](../README.md#2-install-python-311-once)):

```
cd lab4_files
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python lab4_analysis.py --demo      # tries the whole pipeline on synthetic data (~1 min)
```

## In the lab: checklist

1. GUI: start a session, headset on, **Accelerometer widget → Digital Read, switched on**.
2. Monitor at **maximum brightness**. Photoresistor taped on the **lower-left corner**, facing the white square of the video in **full screen**.
3. Check D13: flashlight → 1, hand over it → 0. Play the first seconds of the video and check that D13 toggles.
   (Doing this while the GUI is already recording is fine: the code ignores test flashes and test playbacks.)
4. Start the recording, then play the whole video in full screen without pausing (~1 min 55 s). The subject presses the button for every red circle; avoid touching the jumper cables.
5. Stop the stream after the video ends. **Copy the `OpenBCI-RAW-*.txt` file** (on the lab PCs under `C:\Users\<utorid>\OpenBCI_GUI\Recordings`, possibly in a session subfolder) to a USB stick / email. Write down which file it is.

Good to know, from analysing the video: it starts with **5** sync flashes 100 ms apart (the manual says 3; the photoresistor may miss some), then 90 stimuli every 1.2 s, each on screen for 0.9 s; every white marker lasts only 50 ms (one video frame), which is why the brightness matters.

## Our recordings and results (session of 2026-11-06)

- `Data/trial1.txt`, `Data/trial2.txt`: two runs of the video, same subject (OpenBCI GUI files). `Data/BrainFlow-RAW_*.csv` are the GUI's raw backups (not used).
- In these files D13 is in column `Other.3`, not `Other.2` as the manual says; the loader finds it automatically.
- The slow photoresistor merged the 5 sync flashes into one pulse; run 2 missed trial 24. Both are handled (see the report).
- Report: **`report/lab4_report.pdf`**. Result: a late target negativity (~360–580 ms) at O2 and C4, significant in run 2 and pooled, same direction in run 1, not significant after correcting over the 8 channels.

## Group members: add your name to the report

The author line under the title still has two placeholders, `[Member 2 name] ([student number])` and `[Member 3 name] ([student number])`.

1. `git pull`
2. Open `lab4_files/report/report_template.html`, find the line starting with `<p class="authors">` (near the top, under the comment `GROUP MEMBERS`), and replace your placeholder with your name and student number. Edit only that line.
3. Rebuild the PDF (needs the Lab 4 `.venv`, see Setup, and Microsoft Edge or Google Chrome):
   ```
   cd lab4_files
   .venv\Scripts\python report\build_report.py
   ```
   No need to rerun the analysis: the build only fills the numbers in from `report/results.json`.
4. Check the first page of `report/lab4_report.pdf`, then commit and push:
   ```
   git add report/report_template.html report/lab4_report.html report/lab4_report.pdf
   git commit -m "docs: add <your name> to the Lab 4 report"
   git push
   ```
   If the push is rejected, `git pull` and push again. **Never use `git push --force`.**

The two members adding their names at the same time will both edit that one line; whoever pushes second gets a merge conflict on it. Resolve it by keeping both names (or simply add the names one after the other).

## Analyse (again)

1. Recordings are listed in `RECORDING_FILES` at the top of `lab4_analysis.py` (`{tag: path}`); each is analysed alone and, if there are several, pooled. Set it to `{}` to auto-detect every OpenBCI `.txt` under `lab4_files/`, or pass files on the command line.
2. Run (about 1.5 min; the console output is also kept in `report/analysis_log.txt` if you redirect it there):
   ```
   .venv\Scripts\python lab4_analysis.py > report\analysis_log.txt
   ```
   Check for `90/90 trials matched` (or the listed missing trials) and `find_events gives N events, N are trials`.
3. Build the report: `.venv\Scripts\python report\build_report.py` (needs Edge or Chrome). All numbers and tables are filled in from `report/results.json`; the prose is in `report/report_template.html`.

## Tests

```
.venv\Scripts\python -m pytest test_lab4.py
```

## Robustness built in (beyond the manual)

| Situation | What the code does |
|-----------|--------------------|
| D13 is not in `Other.2` in this GUI version, or reads inverted | Tries the other `Other`/`Analog` columns and both polarities; prints which column was used. |
| Flashlight/hand tests, a test playback, bright desktop after the video | Every burst of flashes is tried as the video start; detected onsets are matched to the 1.2 s trial grid; anything else is ignored and counted. |
| Sync flashes merged or missed (slow photoresistor) | Even a single detected sync flash anchors the trial grid. With no sync flash at all the run is flagged `alignment_anchored: false` and a warning is printed (a missed trial 1 would then shift every label). |
| Photoresistor missed a few flashes | Missing trials are dropped together with their entry in `circle_mask.csv`, so the circle/square labels stay aligned. |
| EEG clock drift / lost radio packets | The trial grid is re-fitted as it goes; lost packets are counted from the sample index. |
| Flat (disconnected) channel | z-score gives 0 instead of NaN, the channel gets p = 1 and is listed as flat. |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No photoresistor TTL found` | The Digital Read widget was off, or D13 never changed. Check the wiring and the brightness. |
| Fewer than 90 trials matched | Some flashes were too dim. The analysis still runs on the matched trials; the report lists the missing ones. |
| `No OpenBCI recording` | The `.txt` must start with `%OpenBCI Raw EEG Data`; list the files in `RECORDING_FILES` or pass them on the command line. |
