"""Przetwarzanie sygnału rPPG: detrend, filtracja pasmowa, estymacja HR."""

import numpy as np
from scipy import sparse
from scipy.signal import butter, find_peaks, periodogram, sosfiltfilt, welch

from src.config import (
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    BUTTERWORTH_ORDER,
    DETREND_LAMBDA,
    WELCH_SEGMENT_SEC,
)

_EPS = 1e-12


def detrend_signal(signal: np.ndarray, lambda_param: float = DETREND_LAMBDA) -> np.ndarray:
    """Detrend smoothness priors (Tarvainen et al., 2002); lambda z config."""
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.shape[0]
    if n < 3:
        return signal - signal.mean()

    identity = sparse.eye(n, format="csc")
    d2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    smoothing_operator = (identity + (lambda_param**2) * (d2.T @ d2)).tocsc()

    trend = sparse.linalg.spsolve(smoothing_operator, signal)
    return signal - trend


def bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """Filtr pasmowo-przepustowy Butterworth (zerofazowy), pasmo z config."""
    signal = np.asarray(signal, dtype=np.float64)
    nyquist_hz = fs / 2.0
    sos = butter(
        BUTTERWORTH_ORDER,
        [BAND_LOW_HZ / nyquist_hz, BAND_HIGH_HZ / nyquist_hz],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, signal)


def estimate_hr_welch(
    signal: np.ndarray,
    fs: float,
    prev_hr_bpm: float | None = None,
    max_jump_bpm: float | None = None,
) -> float:
    """HR = argmax widma Welcha w paśmie HR; zero-padding dla rozdzielczości < 1 BPM.

    ``prev_hr_bpm`` / ``max_jump_bpm`` zachowane dla zgodności interfejsu — ignorowane
    (heurystyki harmoniczne / ciągłość połowiły prawidłowe HR przy szumie 1/f).
    """
    del prev_hr_bpm, max_jump_bpm  # API compat; nie używane
    signal = np.asarray(signal, dtype=np.float64)
    nperseg = min(len(signal), int(round(WELCH_SEGMENT_SEC * fs)))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, nfft=max(nperseg, 2048))
    band = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    if not np.any(band):
        raise ValueError("Brak składowych widma w paśmie fizjologicznym HR.")
    return float(freqs[band][np.argmax(psd[band])] * 60.0)


def estimate_hr_peaks(signal: np.ndarray, fs: float) -> float:
    """HR z odstępów między pikami w dziedzinie czasu."""
    signal = np.asarray(signal, dtype=np.float64)
    min_distance_samples = max(1, int(round(fs / BAND_HIGH_HZ)))

    peaks, _ = find_peaks(signal, distance=min_distance_samples)
    if len(peaks) < 2:
        raise ValueError("Za mało wykrytych pików do estymacji HR.")

    mean_interval_s = np.mean(np.diff(peaks)) / fs
    return 60.0 / mean_interval_s


def snr_rppg(
    signal: np.ndarray,
    fs: float,
    ref_hr_bpm: float,
    n_harmonics: int = 2,
    bin_width_hz: float = 0.2,
) -> float:
    """SNR w paśmie HR: moc w prążkach przy ref_hr vs reszta pasma [dB]."""
    signal = np.asarray(signal, dtype=np.float64)
    freqs, psd = periodogram(signal, fs=fs)

    band_mask = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)

    f0_hz = ref_hr_bpm / 60.0
    signal_mask = np.zeros_like(freqs, dtype=bool)
    for k in range(1, n_harmonics + 1):
        center = k * f0_hz
        if center > BAND_HIGH_HZ:
            break
        signal_mask |= np.abs(freqs - center) <= bin_width_hz
    signal_mask &= band_mask

    noise_mask = band_mask & ~signal_mask

    signal_power = psd[signal_mask].sum()
    noise_power = psd[noise_mask].sum()
    return 10.0 * np.log10((signal_power + _EPS) / (noise_power + _EPS))
