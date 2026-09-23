"""Metody rPPG: GREEN, CHROM, POS, ICA (sygnał→sygnał, bez I/O)."""

import numpy as np
from scipy.signal import periodogram
from sklearn.decomposition import FastICA

from src.config import BAND_HIGH_HZ, BAND_LOW_HZ
from src.estimate import bandpass_filter

_EPS = 1e-8
_WINDOW_SEC = 1.6  # długość okna dla CHROM/POS (de Haan & Jeanne 2013; Wang et al. 2017)


def _validate_rgb_trace(rgb_trace: np.ndarray) -> np.ndarray:
    """Sprawdza kształt wejścia i rzutuje na float64."""
    rgb_trace = np.asarray(rgb_trace, dtype=np.float64)
    if rgb_trace.ndim != 2 or rgb_trace.shape[1] != 3:
        raise ValueError(
            f"Oczekiwano tablicy (N, 3) z kanałami RGB, otrzymano kształt {rgb_trace.shape}"
        )
    return rgb_trace


def green(rgb_trace: np.ndarray, fs: float) -> np.ndarray:
    """Metoda GREEN — surowy sygnał rPPG z kanału zielonego.

    Args:
        rgb_trace: tablica (N, 3) średnich wartości R, G, B w czasie.
        fs: częstotliwość próbkowania sygnału (Hz), nieużywana bezpośrednio
            (zachowana dla spójności interfejsu z pozostałymi metodami).

    Returns:
        1D sygnał długości N (G znormalizowane po osi czasu).
    """
    rgb_trace = _validate_rgb_trace(rgb_trace)
    g = rgb_trace[:, 1]
    return g / (g.mean() + _EPS) - 1.0


def chrom(rgb_trace: np.ndarray, fs: float) -> np.ndarray:
    """CHROM (de Haan & Jeanne, 2013): okna ~1.6 s, filtracja Xs/Ys przed alpha, overlap-add."""
    rgb_trace = _validate_rgb_trace(rgb_trace)
    n_samples = rgb_trace.shape[0]
    window_len = max(2, min(n_samples, int(round(_WINDOW_SEC * fs))))

    signal = np.zeros(n_samples, dtype=np.float64)
    for start in range(0, n_samples - window_len + 1):
        window = rgb_trace[start : start + window_len]
        mean_rgb = window.mean(axis=0) + _EPS
        r_n, g_n, b_n = (window / mean_rgb).T

        x_s = 3.0 * r_n - 2.0 * g_n
        y_s = 1.5 * r_n + g_n - 1.5 * b_n

        # Filtracja pasmowa Xs/Ys przed alpha.
        x_f = bandpass_filter(x_s, fs)
        y_f = bandpass_filter(y_s, fs)

        alpha = np.std(x_f) / (np.std(y_f) + _EPS)
        chrom_window = x_f - alpha * y_f
        chrom_window -= chrom_window.mean()

        signal[start : start + window_len] += chrom_window

    return signal


def pos(rgb_trace: np.ndarray, fs: float) -> np.ndarray:
    """POS (Wang et al., 2017): okna ~1.6 s, overlap-add."""
    rgb_trace = _validate_rgb_trace(rgb_trace)
    n_samples = rgb_trace.shape[0]
    window_len = max(2, min(n_samples, int(round(_WINDOW_SEC * fs))))

    signal = np.zeros(n_samples, dtype=np.float64)
    for start in range(0, n_samples - window_len + 1):
        window = rgb_trace[start : start + window_len]
        mean_rgb = window.mean(axis=0) + _EPS
        r_n, g_n, b_n = (window / mean_rgb).T

        s1 = g_n - b_n
        s2 = g_n + b_n - 2.0 * r_n

        alpha = np.std(s1) / (np.std(s2) + _EPS)
        pos_window = s1 + alpha * s2
        pos_window -= pos_window.mean()

        signal[start : start + window_len] += pos_window

    return signal


def ica_method(rgb_trace: np.ndarray, fs: float) -> np.ndarray:
    """Metoda ICA (Poh et al., 2010) — rozdzielenie źródeł na 3 kanałach RGB.

    Kanały są normalizowane po osi czasu (przez średnią w oknie), a następnie
    rozdzielane niezależnymi składowymi (FastICA). Spośród trzech składowych
    wybierana jest ta, w której największa część mocy widmowej mieści się
    w paśmie fizjologicznym HR (`BAND_LOW_HZ`–`BAND_HIGH_HZ`).

    Args:
        rgb_trace: tablica (N, 3) średnich wartości R, G, B w czasie.
        fs: częstotliwość próbkowania sygnału (Hz).

    Returns:
        1D sygnał rPPG długości N — wybrana składowa niezależna.
    """
    rgb_trace = _validate_rgb_trace(rgb_trace)

    mean_rgb = rgb_trace.mean(axis=0) + _EPS
    normalized = rgb_trace / mean_rgb
    normalized = normalized - normalized.mean(axis=0)

    sources = FastICA(n_components=3, random_state=0, whiten="unit-variance").fit_transform(
        normalized
    )

    best_idx = 0
    best_band_ratio = -np.inf
    for i in range(sources.shape[1]):
        freqs, psd = periodogram(sources[:, i], fs=fs)
        band_mask = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
        band_power = psd[band_mask].sum()
        total_power = psd.sum() + _EPS
        ratio = band_power / total_power
        if ratio > best_band_ratio:
            best_band_ratio = ratio
            best_idx = i

    return sources[:, best_idx]
