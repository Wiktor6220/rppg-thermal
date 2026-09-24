"""Testy walidacji per okno: skok HR śledzony w oknach, okna z utraconym ROI pomijane."""

import numpy as np

from src.validate import compute_mae, compute_rmse, split_into_windows, validate_signal
from tests.synthetic import generate_two_stage_pulse

FS_TEST = 30.0
DURATION_S = 40.0
HR_STAGE1_BPM = 65.0
HR_STAGE2_BPM = 85.0
TOLERANCE_BPM = 5.0


def _two_stage_setup():
    """Para (referencja, estymata) ze skokiem HR w połowie + wektor valid z dziurą 10-20 s."""
    reference_clean, estimated_noisy = generate_two_stage_pulse(
        fs=FS_TEST,
        duration_s=DURATION_S,
        hr_bpm_stage1=HR_STAGE1_BPM,
        hr_bpm_stage2=HR_STAGE2_BPM,
        noise_std=0.2,
        seed=1,
    )
    n_samples = estimated_noisy.shape[0]
    valid_vector = np.ones(n_samples, dtype=bool)
    valid_vector[300:600] = False  # utrata ROI: okno 10-20 s w całości nieważne
    return reference_clean, estimated_noisy, valid_vector


def _idx_at(result, start_s):
    return int(np.where(np.isclose(result["window_start_s"], start_s))[0][0])


def test_invalid_window_is_skipped():
    """Okno 10-20 s (100% nieważnych klatek) jest pomijane i ma NaN."""
    reference, estimated, valid = _two_stage_setup()
    result = validate_signal(estimated, reference, FS_TEST, valid=valid)

    idx_10s = _idx_at(result, 10.0)
    assert not result["window_used"][idx_10s]
    assert np.isnan(result["estimated_hr_bpm"][idx_10s])
    assert result["n_windows_used"] == result["n_windows_total"] - 1


def test_hr_jump_tracked_per_window():
    """Estymaty HR w oknach przed i po skoku odpowiadają odpowiednim etapom."""
    reference, estimated, valid = _two_stage_setup()
    result = validate_signal(estimated, reference, FS_TEST, valid=valid)

    err_stage1 = abs(result["estimated_hr_bpm"][_idx_at(result, 0.0)] - HR_STAGE1_BPM)
    err_stage2 = abs(result["estimated_hr_bpm"][_idx_at(result, 30.0)] - HR_STAGE2_BPM)

    assert err_stage1 <= TOLERANCE_BPM
    assert err_stage2 <= TOLERANCE_BPM


def test_metrics_finite_and_low():
    """MAE i RMSE po użytych oknach są skończone i niewielkie."""
    reference, estimated, valid = _two_stage_setup()
    result = validate_signal(estimated, reference, FS_TEST, valid=valid)

    assert np.isfinite(result["mae_bpm"])
    assert np.isfinite(result["rmse_bpm"])
    assert result["mae_bpm"] <= TOLERANCE_BPM


def test_split_into_windows_shapes():
    """Podział na okna daje fragmenty o oczekiwanej długości i liczbie."""
    signal = np.arange(300, dtype=np.float64)
    windows = split_into_windows(signal, fs=30.0, window_s=5.0, step_s=5.0)
    assert len(windows) == 2
    assert all(w.shape[0] == 150 for w in windows)


def test_compute_mae_rmse_ignore_nan():
    """MAE/RMSE ignorują pary z NaN."""
    est = np.array([60.0, np.nan, 80.0])
    ref = np.array([62.0, 70.0, 84.0])
    assert compute_mae(est, ref) == 3.0
    assert np.isclose(compute_rmse(est, ref), np.sqrt((4 + 16) / 2))


def test_compute_mae_all_nan_returns_nan():
    """Brak par bez NaN -> NaN."""
    est = np.array([np.nan, np.nan])
    ref = np.array([np.nan, np.nan])
    assert np.isnan(compute_mae(est, ref))


def test_delta_snr_near_zero_for_equal_quality_signals():
    """Dwa sygnały o równej jakości → średnie ΔSNR bliskie 0 (brak obciążenia metryki)."""
    from src.validate import validate_against_hr_series

    fs = 30.0
    duration_s = 40.0
    true_hr = 90.0
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    ref_t = np.arange(0.0, duration_s, 1.0)
    ref_hr = np.full(ref_t.shape, true_hr)
    rng = np.random.default_rng(7)
    deltas = []
    for _ in range(200):
        noise_a = rng.standard_normal(n)
        noise_b = rng.standard_normal(n)
        pulse = np.sin(2 * np.pi * (true_hr / 60.0) * t)
        # Identyczna jakość: ten sam SNR (amplituda pulsu / skala szumu).
        sig_a = pulse + 0.5 * noise_a
        sig_b = pulse + 0.5 * noise_b
        va = validate_against_hr_series(sig_a, fs, ref_t, ref_hr)
        vb = validate_against_hr_series(sig_b, fs, ref_t, ref_hr)
        if np.isfinite(va["snr_mean"]) and np.isfinite(vb["snr_mean"]):
            deltas.append(vb["snr_mean"] - va["snr_mean"])
    assert len(deltas) >= 150
    assert abs(float(np.mean(deltas))) < 0.15
