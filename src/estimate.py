"""Przetwarzanie sygnału rPPG do estymaty HR: detrend, filtracja pasmowa, HR z Welcha/pików."""

import numpy as np


def detrend_signal(signal: np.ndarray) -> np.ndarray:
    """Usuwa trend (dryf) z sygnału.

    Args:
        signal: 1D sygnał wejściowy.

    Returns:
        1D sygnał po usunięciu trendu, tej samej długości co wejście.
    """
    raise NotImplementedError


def bandpass_filter(signal: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Filtruje sygnał pasmowo do zakresu fizjologicznego HR.

    Args:
        signal: 1D sygnał wejściowy.
        fs: częstotliwość próbkowania sygnału (Hz).
        low_hz: dolna granica pasma (Hz).
        high_hz: górna granica pasma (Hz).

    Returns:
        1D sygnał po filtracji pasmowej, tej samej długości co wejście.
    """
    raise NotImplementedError


def estimate_hr_welch(signal: np.ndarray, fs: float) -> float:
    """Estymuje HR na podstawie widma mocy sygnału (metoda Welcha).

    Args:
        signal: 1D sygnał rPPG (po filtracji pasmowej).
        fs: częstotliwość próbkowania sygnału (Hz).

    Returns:
        Estymowana częstość akcji serca w uderzeniach na minutę (bpm).
    """
    raise NotImplementedError


def estimate_hr_peaks(signal: np.ndarray, fs: float) -> float:
    """Estymuje HR na podstawie detekcji pików w dziedzinie czasu.

    Args:
        signal: 1D sygnał rPPG (po filtracji pasmowej).
        fs: częstotliwość próbkowania sygnału (Hz).

    Returns:
        Estymowana częstość akcji serca w uderzeniach na minutę (bpm).
    """
    raise NotImplementedError
