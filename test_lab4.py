"""Tests for the Lab 4 pipeline.  Run:  python -m pytest test_lab4.py"""

import json
import os

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter1d

import lab4_analysis as A
import lab4_stats as S
import lab4_synthetic as syn

MASK = np.loadtxt(os.path.join(A.HERE, A.CIRCLE_MASK_FILE), delimiter=",").astype(int)


def smooth_noise(rng: np.random.Generator, n_epochs: int, n_channels: int = 2, n_times: int = 80) -> np.ndarray:
    return gaussian_filter1d(rng.standard_normal((n_epochs, n_channels, n_times)), 3, axis=-1)


# --------------------------------------------------------------------------
# Function 1
# --------------------------------------------------------------------------
def test_function1_finds_the_effect_cluster_and_its_location():
    rng = np.random.default_rng(0)
    a, b = smooth_noise(rng, 20), smooth_noise(rng, 60)
    a[:, 0, 30:50] += 1.0

    stat, clusters = S.nonparametric_test(a, b, return_clusters=True)

    assert stat.shape == (2,)
    assert stat[0] > 20
    start, stop = clusters[0]
    assert 25 <= start <= 32 and 48 <= stop <= 55


def test_function1_sign_follows_the_direction_of_the_difference():
    rng = np.random.default_rng(1)
    a, b = smooth_noise(rng, 20), smooth_noise(rng, 60)
    a[:, 1, 10:30] -= 1.0
    assert S.nonparametric_test(a, b)[1] < -20


def test_function1_returns_zero_when_nothing_is_significant():
    a = np.zeros((5, 1, 10)) + np.arange(5)[:, None, None]
    b = a.copy()
    stat, clusters = S.nonparametric_test(a, b, return_clusters=True)
    assert stat[0] == 0 and clusters[0] is None


def test_function1_never_mixes_positive_and_negative_samples_in_one_cluster():
    t = np.array([3.0, 3.0, -3.0, -3.0, -3.0, 0.0])
    stat, cluster = S._largest_cluster(t, np.abs(t) > 2)
    assert stat == -9.0 and cluster == (2, 5)


def test_function1_matches_a_brute_force_cluster_sum():
    rng = np.random.default_rng(2)
    a, b = smooth_noise(rng, 15, 1), smooth_noise(rng, 40, 1)
    a[:, 0, 20:40] += 0.6
    from scipy import stats

    t, p = stats.ttest_ind(a, b, axis=0)
    best = 0.0
    for sign in (1, -1):
        run = 0.0
        for ti, pi in zip(t[0], p[0]):
            run = run + ti if (pi < 0.05 and np.sign(ti) == sign) else 0.0
            best = run if abs(run) > abs(best) else best
    assert S.nonparametric_test(a, b)[0] == pytest.approx(best)


@pytest.mark.parametrize("shape_a, shape_b", [((5, 2), (5, 2, 10)), ((5, 2, 10), (5, 3, 10)), ((1, 2, 10), (5, 2, 10))])
def test_function1_rejects_bad_shapes(shape_a, shape_b):
    with pytest.raises(ValueError):
        S.nonparametric_test(np.zeros(shape_a), np.zeros(shape_b))


# --------------------------------------------------------------------------
# Function 2
# --------------------------------------------------------------------------
def test_function2_gives_small_p_for_a_real_effect_and_large_p_for_none():
    rng = np.random.default_rng(3)
    a, b = smooth_noise(rng, 15), smooth_noise(rng, 75)
    a[:, 0, 30:50] += 0.8
    p, null = S.permutation_test(a, b, n_perm=300, return_null=True)
    assert p.shape == (2,) and null.shape == (300, 2)
    assert p[0] < 0.01
    assert p[1] > 0.05


def test_function2_is_reproducible_and_does_not_change_its_inputs():
    rng = np.random.default_rng(4)
    a, b = smooth_noise(rng, 10), smooth_noise(rng, 30)
    a_copy, b_copy = a.copy(), b.copy()
    first = S.permutation_test(a, b, n_perm=50, seed=7)
    second = S.permutation_test(a, b, n_perm=50, seed=7)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(a, a_copy)
    np.testing.assert_array_equal(b, b_copy)


def test_function2_false_positive_rate_is_close_to_alpha():
    hits = 0
    n_sim = 60
    for k in range(n_sim):
        rng = np.random.default_rng(100 + k)
        data = smooth_noise(rng, 40, 1, 40)
        hits += S.permutation_test(data[:10], data[10:], n_perm=100, seed=k)[0] < 0.05
    assert hits / n_sim < 0.15  # 5 % expected; generous bound for 60 simulations


def test_function2_rejects_zero_permutations():
    with pytest.raises(ValueError):
        S.permutation_test(np.zeros((3, 1, 5)), np.zeros((3, 1, 5)), n_perm=0)


def test_channel_corrected_pvalues_use_the_max_over_channels():
    null = np.array([[1.0, 5.0], [2.0, 0.5], [0.0, 0.0], [4.0, 1.0]])
    np.testing.assert_allclose(A.channel_corrected_pvalues(np.array([3.0, -1.5]), null), [0.5, 0.75])


def test_mne_cross_check_agrees_with_function2():
    rng = np.random.default_rng(5)
    a, b = smooth_noise(rng, 15), smooth_noise(rng, 75)
    a[:, 0, 30:50] -= 0.6
    ours = S.permutation_test(a, b, n_perm=400, seed=1)
    theirs = A.mne_cross_check(a, b, n_perm=400, seed=1)
    assert theirs[0] == pytest.approx(ours[0], abs=0.03)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    return syn.write_recording(str(tmp_path_factory.mktemp("rec") / "OpenBCI-RAW-test.txt"), seed=1)


def test_loader_renames_channels_and_reads_the_photoresistor(recording):
    df, info = A.load_recording(recording)
    assert list(df.columns) == A.CHANNEL_NAMES + ["photoresistor"]
    assert set(np.unique(df["photoresistor"])) == {0.0, 1.0}
    assert info["photoresistor_column"] == "Other.2"
    assert info["lost_samples"] == 10  # 5 lost packets of 2 samples
    assert A.parse_sample_rate(recording) == 250.0


def test_loader_handles_the_real_gui_spacing(tmp_path):
    header = "%OpenBCI Raw EEG Data\n%Number of channels = 8\n%Sample Rate = 250 Hz\n%Board = x\n"
    columns = ", ".join(syn.FILE_COLUMNS)
    rows = []
    for i in range(40):
        d13 = 1.0 if i % 4 == 1 else 0.0  # 10 rising edges
        values = [float(i % 256)] + [float(i)] * 8 + [0.0] * 3 + [192.0, 0.0, d13, 0.0, 0.0, 0.0, 0.0] + [0.0] * 3
        rows.append(", ".join(str(v) for v in values) + ", 1.79E9, 0.0, 2026-11-13 10:00:00.000")
    path = tmp_path / "gui.txt"
    path.write_text(header + columns + "\n" + "\n".join(rows) + "\n")
    df, info = A.load_recording(str(path))
    assert df["photoresistor"].sum() == 10
    assert info["packet_gaps"] == 0


def test_loader_finds_d13_in_another_column_and_fixes_inverted_polarity():
    ttl = np.zeros(400)
    ttl[::20] = 1
    df = pd.DataFrame({"Other": np.full(400, 192.0), "Other.2": np.zeros(400), "Other.3": 1 - ttl})
    column, normal = A.choose_photoresistor_column(df, "Other.2")
    assert column == "Other.3" and normal is False
    np.testing.assert_array_equal(A.binarize_ttl(df[column], normal), ttl)


def test_loader_explains_a_missing_ttl():
    df = pd.DataFrame({"Other": np.full(100, 192.0), "Other.2": np.zeros(100)})
    with pytest.raises(ValueError, match="Digital Read"):
        A.choose_photoresistor_column(df, "Other.2")


def test_build_raw_converts_to_volts_and_keeps_the_ttl(recording):
    df, _ = A.load_recording(recording)
    raw = A.build_raw(df, 250.0)
    assert raw.get_channel_types() == ["eeg"] * 8 + ["stim"]
    np.testing.assert_allclose(raw.get_data(picks="FP1")[0], df["FP1"].to_numpy() * 1e-6)
    np.testing.assert_array_equal(raw.get_data(picks="photoresistor")[0], df["photoresistor"].to_numpy())


# --------------------------------------------------------------------------
# Trial matching
# --------------------------------------------------------------------------
def onsets_for(**kwargs) -> np.ndarray:
    df = syn.make_recording(**kwargs)
    ttl = df["Other.2"].to_numpy()
    return np.flatnonzero(np.diff(np.r_[0, ttl]) == 1) / syn.SFREQ


def test_matching_picks_the_real_run_over_the_test_playback():
    onsets = onsets_for(seed=0)
    match = A.match_video_trials(onsets)
    assert (match.trial_onset >= 0).all()
    assert match.burst.n_flashes == 5
    assert match.video_start_s == pytest.approx(syn.VIDEO_START_S * 1.003, abs=0.02)
    assert len(A.find_sync_bursts(onsets)) == 2


def test_matching_survives_missed_flashes_and_clock_drift():
    onsets = onsets_for(seed=0, missed_trials=(0, 1, 44), missed_sync=(0, 3), clock_drift=0.01, lost_packets=0)
    match = A.match_video_trials(onsets)
    assert list(np.flatnonzero(match.trial_onset < 0)) == [0, 1, 44]
    times = A.trial_times(onsets, match)
    expected = (syn.VIDEO_START_S + 1.8 + 1.2 * np.arange(90)) * 1.01
    np.testing.assert_allclose(times, expected, atol=0.03)


def test_matching_works_without_any_sync_burst():
    onsets = onsets_for(seed=0, missed_sync=range(5), with_distractors=False)
    match = A.match_video_trials(onsets)
    assert match.burst is None and (match.trial_onset >= 0).all()


@pytest.mark.parametrize("kept_sync", [0, 2, 4])
def test_a_single_sync_flash_anchors_the_labels_even_if_trial_1_is_missed(kept_sync):
    missed_sync = [i for i in range(5) if i != kept_sync]  # also what a slow sensor merging the burst looks like
    onsets = onsets_for(seed=0, missed_trials=(0,), missed_sync=missed_sync, lost_packets=0, clock_drift=0.0)
    match = A.match_video_trials(onsets)
    assert match.anchored
    assert list(np.flatnonzero(match.trial_onset < 0)) == [0]
    expected = syn.VIDEO_START_S + 1.8 + 1.2 * np.arange(90)
    np.testing.assert_allclose(onsets[match.trial_onset[1:]], expected[1:], atol=0.01)


def test_without_any_sync_flash_the_match_is_flagged_as_not_anchored():
    onsets = onsets_for(seed=0, missed_sync=range(5), missed_trials=(0,), with_distractors=False)
    assert not A.match_video_trials(onsets).anchored


def test_matching_needs_events():
    with pytest.raises(ValueError):
        A.match_video_trials(np.array([]))
    with pytest.raises(ValueError, match="trial grid"):
        A.match_video_trials(np.array([1.0, 50.0, 90.0]))


def test_polarity_ignores_long_bright_stretches():
    ttl = np.zeros(3000)
    ttl[100:2000] = 1  # bright desktop for most of the file
    for start in range(2100, 2950, 60):
        ttl[start : start + 12] = 1  # short markers
    assert A.ttl_polarity(pd.Series(ttl)) is True
    assert A.ttl_polarity(pd.Series(1 - ttl)) is False


def test_crop_window_follows_the_manual():
    onsets = onsets_for(seed=0, clock_drift=0.0, lost_packets=0)
    match = A.match_video_trials(onsets)
    start, stop = A.crop_window(onsets, match, duration_s=1e9)
    assert start == pytest.approx(syn.VIDEO_START_S + 0.5, abs=0.01)
    assert stop == pytest.approx(syn.VIDEO_START_S + 1.8 + 1.2 * 89 + 1.0, abs=0.01)


# --------------------------------------------------------------------------
# z-score, mask
# --------------------------------------------------------------------------
def test_zscore_normalises_the_baseline_and_survives_flat_channels():
    rng = np.random.default_rng(6)
    times = np.linspace(-0.2, 0.5, 176)
    data = rng.standard_normal((4, 2, 176)) * 5 + 3
    data[:, 1] = 7.0
    data[0, 1, times > 0.3] = 50.0  # flat baseline, then a jump: must not become inf
    z = A.zscore_epochs(data, times)
    base = z[:, 0, times <= 0]
    np.testing.assert_allclose(base.mean(axis=-1), 0, atol=1e-12)
    np.testing.assert_allclose(base.std(axis=-1), 1)
    assert np.all(z[:, 1] == 0)


def test_circle_mask_has_15_circles_among_90_trials():
    mask = A.load_circle_mask(os.path.join(A.HERE, A.CIRCLE_MASK_FILE))
    assert len(mask) == 90 and mask.sum() == 15


def test_circle_mask_rejects_wrong_lengths(tmp_path):
    path = tmp_path / "mask.csv"
    path.write_text("0, 1, 0")
    with pytest.raises(ValueError):
        A.load_circle_mask(str(path))


def test_find_recordings_uses_the_openbci_header(tmp_path):
    with pytest.raises(SystemExit, match="No OpenBCI"):
        A.find_recordings(str(tmp_path))
    (tmp_path / "Data").mkdir()
    header = "%OpenBCI Raw EEG Data" + chr(10)
    (tmp_path / "Data" / "trial1.txt").write_text(header)
    (tmp_path / "notes.txt").write_text("not a recording")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "x.txt").write_text(header)
    found = A.find_recordings(str(tmp_path))
    assert list(found) == ["run1"] and found["run1"].endswith("trial1.txt")


def test_counter_reset_at_stream_start_is_not_data_loss():
    index = pd.Series(np.r_[np.arange(58, 182), np.arange(0, 256), np.arange(0, 100)] % 256)
    info = A.count_lost_samples(index)
    assert info == {"packet_gaps": 0, "lost_samples": 0, "counter_resets": 1}
    later = pd.Series(np.r_[np.arange(0, 700), np.arange(705, 900)] % 256)
    assert A.count_lost_samples(later)["lost_samples"] == 5  # samples 700-704


def test_bandpass_removes_line_noise():
    sfreq = 250.0
    t = np.arange(int(20 * sfreq)) / sfreq
    data = np.vstack([50e-6 * np.sin(2 * np.pi * 60 * t) + 5e-6 * np.sin(2 * np.pi * 10 * t)] * 8 + [np.zeros_like(t)])
    raw = A.mne.io.RawArray(data, A.mne.create_info(A.CHANNEL_NAMES + ["photoresistor"], sfreq, ["eeg"] * 8 + ["stim"]), verbose="ERROR")
    out = A.bandpass(raw).get_data(picks="FP1")[0][500:-500]
    spectrum = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), 1 / sfreq)
    assert spectrum[np.argmin(np.abs(freqs - 60))] < 0.01 * spectrum[np.argmin(np.abs(freqs - 10))]


def test_crop_starts_after_a_merged_sync_flash():
    onsets = onsets_for(seed=0, clock_drift=0.0, lost_packets=0, missed_sync=(1, 2, 3, 4))
    match = A.match_video_trials(onsets)
    merged_end = syn.VIDEO_START_S + 0.55
    start, _ = A.crop_window(onsets, match, 1e9, offsets_s=np.array([merged_end, 100.0]))
    assert start == pytest.approx(merged_end + A.SYNC_END_MARGIN_S)


# --------------------------------------------------------------------------
# Whole pipeline
# --------------------------------------------------------------------------
def test_pipeline_end_to_end_with_two_runs(tmp_path):
    first = syn.write_recording(str(tmp_path / "a.txt"), seed=2, p300_uv=14.0, missed_trials=(5,))
    second = syn.write_recording(str(tmp_path / "b.txt"), seed=3, p300_uv=14.0)
    results_path = tmp_path / "results.json"
    results = A.analyse({"run1": first, "run2": second}, str(tmp_path / "img"), str(results_path), n_perm=200)

    run1 = results["runs"]["run1"]
    assert run1["n_trials_matched"] == 89 and run1["missed_trials"] == [5]
    assert run1["n_events_in_crop"] == 89 and run1["alignment_anchored"]
    main = run1["main"]
    assert main["n_epochs"] == 89 and main["n_circles"] == 15 and main["n_squares"] == 74
    p = {row["channel"]: row["p_value"] for row in main["channels"]}
    assert p["P7"] < 0.05 and p["C3"] < 0.05  # where the synthetic P300 is largest

    pooled = results["pooled"]["main"]
    assert pooled["n_circles"] == 30 and pooled["n_squares"] == 149
    assert json.loads(results_path.read_text())["runs"]["run2"]["recording"] == "b.txt"
    for name in ("run1_photoresistor.png", "run1_raw_annotations.png", "run1_psd_before_after.png", "run1_main_erp_all.png",
                 "run1_main_erp_significant.png", "run1_main_null.png", "run1_evoked_microvolts.png", "run2_extended_erp_all.png",
                 "pooled_main_erp_all.png", "pooled_evoked_microvolts.png"):
        assert (tmp_path / "img" / name).exists(), name


def test_main_runs_the_demo(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(A, "analyse", lambda recordings, *a, **k: calls.setdefault("rec", recordings))
    monkeypatch.setattr(A, "HERE", str(tmp_path))
    A.main(["--demo"])
    assert list(calls["rec"]) == ["run1"] and os.path.exists(calls["rec"]["run1"])
