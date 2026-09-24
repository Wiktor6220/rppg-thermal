"""Walidacja HR w przesuwanych oknach: MAE, RMSE per okno."""

from collections.abc import Callable

import numpy as np

from src.config import MIN_VALID_RATIO, VALIDATION_STEP_SEC, VALIDATION_WINDOW_SEC
from src.estimate import bandpass_filter, detrend_signal, estimate_hr_peaks, estimate_hr_welch


def _window_bounds(
    n_samples: int, fs: float, window_s: float, step_s: float
) -> list[tuple[int, int]]:
    """Wyznacza indeksy (start, koniec) kolejnych przesuwanych okien próbek."""
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
    """Dzieli sygnał na przesuwane okna czasowe o zadanej długości i kroku.

    Args:
        signal: 1D sygnał wejściowy.
        fs: częstotliwość próbkowania sygnału (Hz).
        window_s: długość okna w sekundach (np. 10 s).
        step_s: krok przesunięcia okna w sekundach.

    Returns:
        Lista fragmentów sygnału (okien), każdy jako 1D tablica.
    """
    signal = np.asarray(signal, dtype=np.float64)
    bounds = _window_bounds(signal.shape[0], fs, window_s, step_s)
    return [signal[start:end] for start, end in bounds]


def compute_mae(estimated: np.ndarray, reference: np.ndarray) -> float:
    """MAE między estymatami HR a referencją (pary z NaN pomijane).

    Args:
        estimated: 1D tablica estymat HR (per okno, bpm), może zawierać NaN.
        reference: 1D tablica referencyjnych wartości HR (per okno, bpm), może zawierać NaN.

    Returns:
        Wartość MAE (bpm), albo NaN, gdy brak par bez NaN.
    """
    estimated = np.asarray(estimated, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    diff = np.abs(estimated - reference)
    if not np.any(~np.isnan(diff)):
        return float("nan")
    return float(np.nanmean(diff))


def compute_rmse(estimated: np.ndarray, reference: np.ndarray) -> float:
    """RMSE między estymatami HR a referencją (pary z NaN pomijane).

    Args:
        estimated: 1D tablica estymat HR (per okno, bpm), może zawierać NaN.
        reference: 1D tablica referencyjnych wartości HR (per okno, bpm), może zawierać NaN.

    Returns:
        Wartość RMSE (bpm), albo NaN, gdy brak par bez NaN.
    """
    estimated = np.asarray(estimated, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    squared_diff = (estimated - reference) ** 2
    if not np.any(~np.isnan(squared_diff)):
        return float("nan")
    return float(np.sqrt(np.nanmean(squared_diff)))


def validate_windows(
    estimated_hr_per_window: np.ndarray, reference_hr_per_window: np.ndarray
) -> dict[str, float]:
    """Agreguje metryki walidacyjne (MAE, RMSE) po wszystkich oknach.

    Args:
        estimated_hr_per_window: 1D tablica estymat HR, po jednej wartości na okno
            (NaN dla okien pominiętych).
        reference_hr_per_window: 1D tablica referencyjnych wartości HR, po jednej na okno
            (NaN dla okien pominiętych).

    Returns:
        Słownik z metrykami: {"mae_bpm", "rmse_bpm", "n_windows_used"}.
    """
    estimated_hr_per_window = np.asarray(estimated_hr_per_window, dtype=np.float64)
    reference_hr_per_window = np.asarray(reference_hr_per_window, dtype=np.float64)
    n_windows_used = int(
        np.sum(~np.isnan(estimated_hr_per_window) & ~np.isnan(reference_hr_per_window))
    )
    return {
        "mae_bpm": compute_mae(estimated_hr_per_window, reference_hr_per_window),
        "rmse_bpm": compute_rmse(estimated_hr_per_window, reference_hr_per_window),
        "n_windows_used": n_windows_used,
    }


def _window_valid_ratio(valid: np.ndarray, start: int, end: int) -> float:
    """Odsetek ważnych klatek (wg `valid[]`) w oknie [start, end)."""
    segment = valid[start:end]
    if segment.size == 0:
        return 0.0
    return float(np.mean(segment))


def _estimate_window_hr(
    window_signal: np.ndarray,
    fs: float,
    hr_estimator: Callable[..., float],
    prev_hr_bpm: float | None = None,
) -> float:
    """Detrend + bandpass + estymacja HR na oknie; NaN przy błędzie."""
    try:
        cleaned = bandpass_filter(detrend_signal(window_signal), fs)
        try:
            return float(hr_estimator(cleaned, fs, prev_hr_bpm=prev_hr_bpm))
        except TypeError:
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
    """HR per okno (detrend + filtr na oknie); okna z niskim valid[] pomijane w metrykach.

    Returns:
        Słownik: window_start_s, estimated_hr_bpm, reference_hr_bpm, error_bpm,
        window_used, n_windows_total, n_windows_used, mae_bpm, rmse_bpm.
    """
    estimated_signal = np.asarray(estimated_signal, dtype=np.float64)
    reference_signal = np.asarray(reference_signal, dtype=np.float64)
    if estimated_signal.shape[0] != reference_signal.shape[0]:
        raise ValueError(
            "Sygnał estymowany i referencyjny muszą mieć tę samą długość (wspólne fs)."
        )
    n_samples = estimated_signal.shape[0]

    if valid is None:
        valid = np.ones(n_samples, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape[0] != n_samples:
        raise ValueError("Wektor valid[] musi mieć tę samą długość co sygnały.")

    bounds = _window_bounds(n_samples, fs, window_s, step_s)
    n_windows = len(bounds)

    window_start_s = np.array([start / fs for start, _ in bounds])
    estimated_hr_bpm = np.full(n_windows, np.nan)
    reference_hr_bpm = np.full(n_windows, np.nan)
    window_used = np.zeros(n_windows, dtype=bool)

    prev_est: float | None = None
    for i, (start, end) in enumerate(bounds):
        if _window_valid_ratio(valid, start, end) < min_valid_ratio:
            continue

        est_hr = _estimate_window_hr(
            estimated_signal[start:end], fs, hr_estimator, prev_hr_bpm=prev_est
        )
        ref_hr = _estimate_window_hr(reference_signal[start:end], fs, reference_hr_estimator)

        estimated_hr_bpm[i] = est_hr
        reference_hr_bpm[i] = ref_hr
        window_used[i] = not (np.isnan(est_hr) or np.isnan(ref_hr))
        if window_used[i]:
            prev_est = est_hr

    error_bpm = np.abs(estimated_hr_bpm - reference_hr_bpm)
    metrics = validate_windows(estimated_hr_bpm, reference_hr_bpm)

    return {
        "window_start_s": window_start_s,
        "estimated_hr_bpm": estimated_hr_bpm,
        "reference_hr_bpm": reference_hr_bpm,
        "error_bpm": error_bpm,
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
    """Waliduje rPPG względem serii HR (EKG/Polar) w oknach 10 s.

    W każdym oknie: HR z Welcha (ciągłość między oknami) oraz mediana referencji
    w ``[t0, t0+window_s)``. Okna bez referencji / z niskim ``valid[]`` → NaN.
    """
    estimated_signal = np.asarray(estimated_signal, dtype=np.float64)
    ref_t_s = np.asarray(ref_t_s, dtype=np.float64)
    ref_hr_bpm = np.asarray(ref_hr_bpm, dtype=np.float64)
    n_samples = estimated_signal.shape[0]

    if valid is None:
        valid = np.ones(n_samples, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape[0] != n_samples:
        raise ValueError("Wektor valid[] musi mieć tę samą długość co sygnał.")

    bounds = _window_bounds(n_samples, fs, window_s, step_s)
    n_windows = len(bounds)
    window_start_s = np.array([start / fs for start, _ in bounds])
    estimated_hr_bpm = np.full(n_windows, np.nan)
    reference_hr_bpm = np.full(n_windows, np.nan)
    window_used = np.zeros(n_windows, dtype=bool)

    prev_hr: float | None = None
    for i, (start, end) in enumerate(bounds):
        if _window_valid_ratio(valid, start, end) < min_valid_ratio:
            continue
        t0 = start / fs
        t1 = end / fs
        in_win = (ref_t_s >= t0) & (ref_t_s < t1)
        if not np.any(in_win):
            continue
        est_hr = _estimate_window_hr(
            estimated_signal[start:end], fs, hr_estimator, prev_hr_bpm=prev_hr
        )
        ref_hr = float(np.median(ref_hr_bpm[in_win]))
        estimated_hr_bpm[i] = est_hr
        reference_hr_bpm[i] = ref_hr
        window_used[i] = not np.isnan(est_hr)
        if window_used[i]:
            prev_hr = est_hr

    error_bpm = np.abs(estimated_hr_bpm - reference_hr_bpm)
    metrics = validate_windows(estimated_hr_bpm, reference_hr_bpm)
    return {
        "window_start_s": window_start_s,
        "estimated_hr_bpm": estimated_hr_bpm,
        "reference_hr_bpm": reference_hr_bpm,
        "error_bpm": error_bpm,
        "window_used": window_used,
        "n_windows_total": n_windows,
        "n_windows_used": metrics["n_windows_used"],
        "mae_bpm": metrics["mae_bpm"],
        "rmse_bpm": metrics["rmse_bpm"],
    }
