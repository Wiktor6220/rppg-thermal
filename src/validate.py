"""Walidacja HR w przesuwanych oknach: MAE, RMSE, SNR."""

from collections.abc import Callable

import numpy as np

from src.config import (
    MIN_VALID_RATIO,
    N_QUARTER_WINDOWS,
    QUARTER_FIRST3_SEC,
    VALIDATION_STEP_SEC,
    VALIDATION_WINDOW_SEC,
)
from src.estimate import (
    bandpass_filter,
    detrend_signal,
    estimate_hr_peaks,
    estimate_hr_welch,
    snr_rppg,
)


def _window_bounds(
    n_samples: int, fs: float, window_s: float, step_s: float
) -> list[tuple[int, int]]:
    """Indeksy (start, end) kolejnych okien."""
    window_len = int(round(window_s * fs))
    step_len = int(round(step_s * fs))
    if window_len <= 0 or step_len <= 0:
        raise ValueError("window_s i step_s muszą być dodatnie.")

    bounds = []
    start = 0
    while start + window_len <= n_samples:
        bounds.append((start, start + window_len))
        start += step_len
    return bounds


def split_into_windows(
    signal: np.ndarray, fs: float, window_s: float, step_s: float
) -> list[np.ndarray]:
    """Dzieli sygnał na przesuwane okna."""
    signal = np.asarray(signal, dtype=np.float64)
    return [signal[s:e] for s, e in _window_bounds(signal.shape[0], fs, window_s, step_s)]


def compute_mae(estimated: np.ndarray, reference: np.ndarray) -> float:
    """MAE [BPM]; pary z NaN pomijane."""
    diff = np.abs(
        np.asarray(estimated, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    )
    if not np.any(~np.isnan(diff)):
        return float("nan")
    return float(np.nanmean(diff))


def compute_rmse(estimated: np.ndarray, reference: np.ndarray) -> float:
    """RMSE [BPM]; pary z NaN pomijane."""
    sq = (
        np.asarray(estimated, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    ) ** 2
    if not np.any(~np.isnan(sq)):
        return float("nan")
    return float(np.sqrt(np.nanmean(sq)))


def validate_windows(
    estimated_hr_per_window: np.ndarray, reference_hr_per_window: np.ndarray
) -> dict[str, float]:
    """Agreguje MAE/RMSE i liczbę użytych okien."""
    est = np.asarray(estimated_hr_per_window, dtype=np.float64)
    ref = np.asarray(reference_hr_per_window, dtype=np.float64)
    n_used = int(np.sum(~np.isnan(est) & ~np.isnan(ref)))
    return {
        "mae_bpm": compute_mae(est, ref),
        "rmse_bpm": compute_rmse(est, ref),
        "n_windows_used": n_used,
    }


def _window_valid_ratio(valid: np.ndarray, start: int, end: int) -> float:
    """Udział True w valid[start:end]."""
    segment = valid[start:end]
    return 0.0 if segment.size == 0 else float(np.mean(segment))


def _estimate_window_hr(
    window_signal: np.ndarray,
    fs: float,
    hr_estimator: Callable[[np.ndarray, float], float],
) -> float:
    """Detrend + bandpass + HR; NaN przy błędzie."""
    try:
        cleaned = bandpass_filter(detrend_signal(window_signal), fs)
        return float(hr_estimator(cleaned, fs))
    except (ValueError, np.linalg.LinAlgError):
        return float("nan")


def validate_signal(
    estimated_signal: np.ndarray,
    reference_signal: np.ndarray,
    fs: float,
    valid: np.ndarray | None = None,
    window_s: float = VALIDATION_WINDOW_SEC,
    step_s: float = VALIDATION_STEP_SEC,
    min_valid_ratio: float = MIN_VALID_RATIO,
    hr_estimator: Callable[[np.ndarray, float], float] = estimate_hr_welch,
    reference_hr_estimator: Callable[[np.ndarray, float], float] = estimate_hr_peaks,
) -> dict:
    """Porównuje dwa sygnały czasowe oknami; niski valid[] → pominięcie okna."""
    estimated_signal = np.asarray(estimated_signal, dtype=np.float64)
    reference_signal = np.asarray(reference_signal, dtype=np.float64)
    if estimated_signal.shape[0] != reference_signal.shape[0]:
        raise ValueError("Sygnały muszą mieć tę samą długość.")
    n_samples = estimated_signal.shape[0]

    if valid is None:
        valid = np.ones(n_samples, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape[0] != n_samples:
        raise ValueError("valid[] musi mieć długość sygnału.")

    bounds = _window_bounds(n_samples, fs, window_s, step_s)
    n_windows = len(bounds)
    window_start_s = np.array([start / fs for start, _ in bounds])
    estimated_hr_bpm = np.full(n_windows, np.nan)
    reference_hr_bpm = np.full(n_windows, np.nan)
    window_used = np.zeros(n_windows, dtype=bool)

    for i, (start, end) in enumerate(bounds):
        if _window_valid_ratio(valid, start, end) < min_valid_ratio:
            continue
        est_hr = _estimate_window_hr(estimated_signal[start:end], fs, hr_estimator)
        ref_hr = _estimate_window_hr(reference_signal[start:end], fs, reference_hr_estimator)
        estimated_hr_bpm[i] = est_hr
        reference_hr_bpm[i] = ref_hr
        window_used[i] = not (np.isnan(est_hr) or np.isnan(ref_hr))

    metrics = validate_windows(estimated_hr_bpm, reference_hr_bpm)
    return {
        "window_start_s": window_start_s,
        "estimated_hr_bpm": estimated_hr_bpm,
        "reference_hr_bpm": reference_hr_bpm,
        "error_bpm": np.abs(estimated_hr_bpm - reference_hr_bpm),
        "window_used": window_used,
        "n_windows_total": n_windows,
        "n_windows_used": metrics["n_windows_used"],
        "mae_bpm": metrics["mae_bpm"],
        "rmse_bpm": metrics["rmse_bpm"],
    }


def usable_video_hr_range(
    ref_t_s: np.ndarray,
    video_duration_s: float,
) -> tuple[float, float]:
    """Pokrycie wideo∩HR w sekundach od klatki 0: [max(0, t_HR_min), min(T, t_HR_max)]."""
    ref_t_s = np.asarray(ref_t_s, dtype=np.float64)
    if ref_t_s.size == 0 or not np.isfinite(video_duration_s) or video_duration_s <= 0:
        return 0.0, 0.0
    lo = max(0.0, float(np.nanmin(ref_t_s)))
    hi = min(float(video_duration_s), float(np.nanmax(ref_t_s)))
    if hi <= lo:
        return 0.0, 0.0
    return lo, hi


def quarter_time_bounds(
    t0: float,
    t1: float,
    first3_s: float = QUARTER_FIRST3_SEC,
    n_windows: int = N_QUARTER_WINDOWS,
) -> list[tuple[float, float]]:
    """``n_windows`` równych, kolejnych, nienachodzących okien na [t0, t1].

    ``first3_s`` zachowane w sygnaturze dla kompatybilności — nieużywane
    (podział zawsze równy: duration/n_windows).
    """
    del first3_s  # legacy arg; równe kwartyle
    dur = float(t1) - float(t0)
    if dur <= 0 or n_windows < 1:
        return [(float(t0), float(t0))] * max(1, n_windows)
    w = dur / float(n_windows)
    return [(t0 + i * w, t0 + (i + 1) * w) for i in range(n_windows)]


def quarter_frame_bounds(
    n_samples: int,
    n_windows: int = N_QUARTER_WINDOWS,
) -> list[tuple[int, int]]:
    """Podział klatek: ``n // 4`` na okno; ostatnie absorbuje resztę 1–3 klatek."""
    if n_samples <= 0 or n_windows < 1:
        return [(0, 0)] * max(1, n_windows)
    base = n_samples // n_windows
    if base < 1:
        # Za krótki sygnał — równy podział zdegenerowany.
        return [(0, n_samples)] * n_windows
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(n_windows):
        if i < n_windows - 1:
            end = start + base
        else:
            end = n_samples
        bounds.append((start, end))
        start = end
    return bounds


def validate_against_hr_quarters(
    estimated_signal: np.ndarray,
    fs: float,
    ref_t_s: np.ndarray,
    ref_hr_bpm: np.ndarray,
    video_duration_s: float,
    valid: np.ndarray | None = None,
    min_valid_ratio: float = MIN_VALID_RATIO,
    hr_estimator: Callable[[np.ndarray, float], float] = estimate_hr_welch,
    first3_s: float = QUARTER_FIRST3_SEC,
) -> dict:
    """Walidacja vs Polar w 4 równych oknach na osi wideo [0, duration].

    Podział klatek: ``n // 4``; ostatnie okno bierze resztę. Per okno: Welch,
    mediana Polar w [t_a, t_b). Pominięcie przy valid_ratio < min albo braku ref.
    ``first3_s`` — nieużywane (kompatybilność API).
    """
    del first3_s
    estimated_signal = np.asarray(estimated_signal, dtype=np.float64)
    ref_t_s = np.asarray(ref_t_s, dtype=np.float64)
    ref_hr_bpm = np.asarray(ref_hr_bpm, dtype=np.float64)
    n_samples = estimated_signal.shape[0]

    if valid is None:
        valid = np.ones(n_samples, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape[0] != n_samples:
        raise ValueError("valid[] musi mieć długość sygnału.")

    # Info o pokryciu HR∩wideo (nie wyznacza granic okien).
    t_lo, t_hi = usable_video_hr_range(ref_t_s, video_duration_s)

    frame_bounds = quarter_frame_bounds(n_samples)
    n_windows = len(frame_bounds)
    window_t0 = np.array([s / fs for s, _ in frame_bounds], dtype=np.float64)
    window_t1 = np.array([e / fs for _, e in frame_bounds], dtype=np.float64)
    estimated_hr_bpm = np.full(n_windows, np.nan)
    reference_hr_bpm = np.full(n_windows, np.nan)
    snr_db = np.full(n_windows, np.nan)
    window_used = np.zeros(n_windows, dtype=bool)

    for i, (start, end) in enumerate(frame_bounds):
        if end <= start:
            continue
        ta, tb = float(start / fs), float(end / fs)
        if _window_valid_ratio(valid, start, end) < min_valid_ratio:
            continue
        in_win = (ref_t_s >= ta) & (ref_t_s < tb)
        if not np.any(in_win):
            continue
        ref_hr = float(np.median(ref_hr_bpm[in_win]))
        window_sig = estimated_signal[start:end]
        est_hr = _estimate_window_hr(window_sig, fs, hr_estimator)
        try:
            cleaned = bandpass_filter(detrend_signal(window_sig), fs)
            snr_val = float(snr_rppg(cleaned, fs, ref_hr))
        except (ValueError, np.linalg.LinAlgError):
            snr_val = float("nan")
        estimated_hr_bpm[i] = est_hr
        reference_hr_bpm[i] = ref_hr
        snr_db[i] = snr_val
        window_used[i] = bool(np.isfinite(est_hr) and np.isfinite(ref_hr))

    metrics = validate_windows(estimated_hr_bpm, reference_hr_bpm)
    snr_used = snr_db[window_used & np.isfinite(snr_db)]
    return {
        "usable_t0_s": float(t_lo),
        "usable_t1_s": float(t_hi),
        "window_t0_s": window_t0,
        "window_t1_s": window_t1,
        "estimated_hr_bpm": estimated_hr_bpm,
        "reference_hr_bpm": reference_hr_bpm,
        "error_bpm": np.abs(estimated_hr_bpm - reference_hr_bpm),
        "snr_db": snr_db,
        "snr_mean": float(np.mean(snr_used)) if snr_used.size else float("nan"),
        "octave_error_frac": float(octave_error_fraction(estimated_hr_bpm, reference_hr_bpm)),
        "window_used": window_used,
        "n_windows_total": n_windows,
        "n_windows_used": metrics["n_windows_used"],
        "mae_bpm": metrics["mae_bpm"],
        "rmse_bpm": metrics["rmse_bpm"],
    }


def validate_against_hr_series(
    estimated_signal: np.ndarray,
    fs: float,
    ref_t_s: np.ndarray,
    ref_hr_bpm: np.ndarray,
    valid: np.ndarray | None = None,
    window_s: float = VALIDATION_WINDOW_SEC,
    step_s: float = VALIDATION_STEP_SEC,
    min_valid_ratio: float = MIN_VALID_RATIO,
    hr_estimator: Callable[[np.ndarray, float], float] = estimate_hr_welch,
) -> dict:
    """Walidacja vs seria Polar HR: Welch + mediana ref w oknie; SNR względem ref.

    ``ref_t_s`` to czas [s] od startu wideo (klatka 0). Przy synchronizacji absolutnej
    (creation_time MP4 ↔ Polar) wartości mogą być ujemne przed startem nagrania —
    okna filtrują po ``[start/fs, end/fs)`` jak wcześniej (bez zerowania do pierwszej próbki).
    """
    estimated_signal = np.asarray(estimated_signal, dtype=np.float64)
    ref_t_s = np.asarray(ref_t_s, dtype=np.float64)
    ref_hr_bpm = np.asarray(ref_hr_bpm, dtype=np.float64)
    n_samples = estimated_signal.shape[0]

    if valid is None:
        valid = np.ones(n_samples, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape[0] != n_samples:
        raise ValueError("valid[] musi mieć długość sygnału.")

    bounds = _window_bounds(n_samples, fs, window_s, step_s)
    n_windows = len(bounds)
    window_start_s = np.array([start / fs for start, _ in bounds])
    estimated_hr_bpm = np.full(n_windows, np.nan)
    reference_hr_bpm = np.full(n_windows, np.nan)
    snr_db = np.full(n_windows, np.nan)
    window_used = np.zeros(n_windows, dtype=bool)

    for i, (start, end) in enumerate(bounds):
        if _window_valid_ratio(valid, start, end) < min_valid_ratio:
            continue
        t0, t1 = start / fs, end / fs
        in_win = (ref_t_s >= t0) & (ref_t_s < t1)
        if not np.any(in_win):
            continue

        ref_hr = float(np.median(ref_hr_bpm[in_win]))
        window_sig = estimated_signal[start:end]
        est_hr = _estimate_window_hr(window_sig, fs, hr_estimator)
        try:
            cleaned = bandpass_filter(detrend_signal(window_sig), fs)
            snr_val = float(snr_rppg(cleaned, fs, ref_hr))
        except (ValueError, np.linalg.LinAlgError):
            snr_val = float("nan")

        estimated_hr_bpm[i] = est_hr
        reference_hr_bpm[i] = ref_hr
        snr_db[i] = snr_val
        window_used[i] = not np.isnan(est_hr)

    metrics = validate_windows(estimated_hr_bpm, reference_hr_bpm)
    snr_used = snr_db[window_used & np.isfinite(snr_db)]
    return {
        "window_start_s": window_start_s,
        "estimated_hr_bpm": estimated_hr_bpm,
        "reference_hr_bpm": reference_hr_bpm,
        "error_bpm": np.abs(estimated_hr_bpm - reference_hr_bpm),
        "snr_db": snr_db,
        "snr_mean": float(np.mean(snr_used)) if snr_used.size else float("nan"),
        "octave_error_frac": float(octave_error_fraction(estimated_hr_bpm, reference_hr_bpm)),
        "window_used": window_used,
        "n_windows_total": n_windows,
        "n_windows_used": metrics["n_windows_used"],
        "mae_bpm": metrics["mae_bpm"],
        "rmse_bpm": metrics["rmse_bpm"],
    }


def octave_error_fraction(
    estimated_hr: np.ndarray,
    reference_hr: np.ndarray,
    rel_tol: float = 0.15,
) -> float:
    """Odsetek okien z estymatą ≈ 2× lub ½× referencji."""
    est = np.asarray(estimated_hr, dtype=np.float64)
    ref = np.asarray(reference_hr, dtype=np.float64)
    ok = np.isfinite(est) & np.isfinite(ref) & (ref > 0)
    if not np.any(ok):
        return float("nan")
    ratio = est[ok] / ref[ok]
    is_oct = (np.abs(ratio - 2.0) <= rel_tol) | (np.abs(ratio - 0.5) <= rel_tol)
    return float(np.mean(is_oct))
