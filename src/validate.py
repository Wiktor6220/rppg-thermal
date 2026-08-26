"""Walidacja estymaty HR względem referencji: podział na okna, MAE, RMSE per okno.

Zgodnie z CONTEXT.md walidacja odbywa się w przesuwanych oknach (np. 10 s) —
HR liczone jest osobno w każdym oknie i porównywane z referencją, zamiast
sprowadzać cały nagrany sygnał do jednej liczby. Okna z przewagą nieważnych
klatek (wg wektora `valid[]` z roi.py) są pomijane w metrykach — odrzucane są
OKNA, nigdy pojedyncze klatki (CLAUDE.md).
"""

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
    """Liczy średni błąd bezwzględny (MAE) między estymatami HR a referencją.

    Pary, w których estymata lub referencja to NaN (okno pominięte/nieudane),
    są ignorowane — zgodnie z zasadą odrzucania okien, a nie pojedynczych klatek.

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
    """Liczy pierwiastek błędu średniokwadratowego (RMSE) między estymatami HR a referencją.

    Pary z NaN (okno pominięte/nieudane) są ignorowane, analogicznie do `compute_mae`.

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
    window_signal: np.ndarray, fs: float, hr_estimator: Callable[[np.ndarray, float], float]
) -> float:
    """Detrend + filtracja pasmowa + estymacja HR na pojedynczym oknie; NaN przy błędzie."""
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
    """Waliduje sygnał rPPG względem referencji w przesuwanych oknach czasowych.

    Dla każdego okna: sprawdza odsetek ważnych klatek (`valid[]`) — okna z
    przewagą nieważnych klatek (poniżej `min_valid_ratio`) są pomijane w
    metrykach. Dla pozostałych okien estymuje HR z sygnału (`hr_estimator`) i
    z referencji (`reference_hr_estimator`), po detrendzie i filtracji
    pasmowej każdego okna z osobna. Metryki (MAE, RMSE) liczone są per okno,
    nie jako jedna liczba na cały sygnał.

    Args:
        estimated_signal: 1D sygnał rPPG (wyjście `methods.py`), fs próbek.
        reference_signal: 1D sygnał referencyjny (np. PPG), tej samej długości
            i częstotliwości próbkowania co `estimated_signal`.
        fs: częstotliwość próbkowania obu sygnałów (Hz).
        valid: 1D tablica bool długości sygnału — wynik `roi.track_roi_across_frames`.
            Gdy None, wszystkie klatki uznawane są za ważne.
        window_s: długość okna w sekundach. Domyślnie `config.VALIDATION_WINDOW_SEC`.
        step_s: krok przesunięcia okna w sekundach. Domyślnie `config.VALIDATION_STEP_SEC`.
        min_valid_ratio: minimalny odsetek ważnych klatek wymagany, by okno nie
            zostało pominięte. Domyślnie `config.MIN_VALID_RATIO`.
        hr_estimator: funkcja estymująca HR z okna sygnału estymowanego
            (np. `estimate.estimate_hr_welch` lub `estimate.estimate_hr_peaks`).
        reference_hr_estimator: funkcja estymująca HR z okna sygnału referencyjnego.

    Returns:
        Słownik z wynikami per okno oraz zagregowanymi metrykami:
            "window_start_s": czas początku każdego okna (s),
            "estimated_hr_bpm": estymaty HR per okno (NaN dla pominiętych),
            "reference_hr_bpm": referencyjne HR per okno (NaN dla pominiętych),
            "error_bpm": błąd bezwzględny per okno (NaN dla pominiętych),
            "window_used": maska bool, które okna weszły do metryk,
            "n_windows_total": liczba wszystkich okien,
            "n_windows_used": liczba okien uwzględnionych w metrykach,
            "mae_bpm", "rmse_bpm": zagregowane metryki po użytych oknach.
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

    for i, (start, end) in enumerate(bounds):
        if _window_valid_ratio(valid, start, end) < min_valid_ratio:
            continue  # okno z przewagą nieważnych klatek — pomijamy w metrykach

        est_hr = _estimate_window_hr(estimated_signal[start:end], fs, hr_estimator)
        ref_hr = _estimate_window_hr(reference_signal[start:end], fs, reference_hr_estimator)

        estimated_hr_bpm[i] = est_hr
        reference_hr_bpm[i] = ref_hr
        window_used[i] = not (np.isnan(est_hr) or np.isnan(ref_hr))

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
