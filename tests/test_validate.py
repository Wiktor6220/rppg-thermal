"""Testy walidacji: MAE/RMSE, równy podział na 4 okna, błędy oktawowe, walidacja vs Polar."""

import numpy as np

from src.validate import (
    compute_mae,
    compute_rmse,
    octave_error_fraction,
    quarter_frame_bounds,
    validate_against_hr_quarters,
)


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


def test_quarter_frame_bounds_equal_split_last_takes_remainder():
    """1889 klatek -> 4 kolejne okna 472/472/472/473, bez luk i nakładania."""
    bounds = quarter_frame_bounds(1889)
    assert [e - s for s, e in bounds] == [472, 472, 472, 473]
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 1889
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))


def test_octave_error_fraction():
    est = np.array([60.0, 120.0, 70.0, 35.0])
    ref = np.array([60.0, 60.0, 70.0, 70.0])
    # 120 = 2×60, 35 = 0.5×70 -> 2/4
    assert abs(octave_error_fraction(est, ref) - 0.5) < 1e-9


def _sine_and_polar(hr_bpm: float = 72.0, fs: float = 30.0, duration_s: float = 40.0):
    t = np.arange(int(duration_s * fs)) / fs
    sig = np.sin(2 * np.pi * (hr_bpm / 60.0) * t)
    ref_t = np.arange(0.0, duration_s, 1.0)
    return sig, fs, duration_s, ref_t, np.full(ref_t.shape, hr_bpm)


def test_quarters_recover_hr_in_every_window():
    sig, fs, dur, ref_t, ref_hr = _sine_and_polar()
    out = validate_against_hr_quarters(sig, fs, ref_t, ref_hr, video_duration_s=dur)
    assert out["n_windows_used"] == 4
    assert out["mae_bpm"] < 3.0
    np.testing.assert_allclose(out["window_t0_s"], [0.0, 10.0, 20.0, 30.0])


def test_quarters_skip_window_with_low_valid_ratio():
    """Okno z valid[] < MIN_VALID_RATIO jest pomijane (NaN), reszta liczona."""
    sig, fs, dur, ref_t, ref_hr = _sine_and_polar()
    valid = np.ones(sig.shape[0], dtype=bool)
    valid[300:600] = False  # całe drugie okno
    out = validate_against_hr_quarters(sig, fs, ref_t, ref_hr, video_duration_s=dur, valid=valid)
    assert np.isnan(out["estimated_hr_bpm"][1])
    assert out["n_windows_used"] == 3
