"""Walidacja estymaty HR względem referencji: podział na okna, MAE, RMSE per okno."""

import numpy as np


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
    raise NotImplementedError


def compute_mae(estimated: np.ndarray, reference: np.ndarray) -> float:
    """Liczy średni błąd bezwzględny (MAE) między estymatami HR a referencją.

    Args:
        estimated: 1D tablica estymat HR (per okno, bpm).
        reference: 1D tablica referencyjnych wartości HR (per okno, bpm).

    Returns:
        Wartość MAE (bpm).
    """
    raise NotImplementedError


def compute_rmse(estimated: np.ndarray, reference: np.ndarray) -> float:
    """Liczy pierwiastek błędu średniokwadratowego (RMSE) między estymatami HR a referencją.

    Args:
        estimated: 1D tablica estymat HR (per okno, bpm).
        reference: 1D tablica referencyjnych wartości HR (per okno, bpm).

    Returns:
        Wartość RMSE (bpm).
    """
    raise NotImplementedError


def validate_windows(
    estimated_hr_per_window: np.ndarray, reference_hr_per_window: np.ndarray
) -> dict[str, float]:
    """Agreguje metryki walidacyjne (MAE, RMSE) po wszystkich oknach.

    Args:
        estimated_hr_per_window: 1D tablica estymat HR, po jednej wartości na okno.
        reference_hr_per_window: 1D tablica referencyjnych wartości HR, po jednej na okno.

    Returns:
        Słownik z metrykami, np. {"mae": ..., "rmse": ...}.
    """
    raise NotImplementedError
