"""Detrend, bandpass i estymacja HR z sygnału rPPG."""

import numpy as np
from scipy import sparse
from scipy.signal import butter, periodogram, sosfiltfilt, welch

from src.config import (
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    BUTTERWORTH_ORDER,
    DETREND_LAMBDA,
    WELCH_SEGMENT_SEC,
)

_EPS = 1e-12


def detrend_signal(signal: np.ndarray, lambda_param: float = DETREND_LAMBDA) -> np.ndarray:
    """Usuwa trend smoothness priors (Tarvainen 2002)."""
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.shape[0]
    if n < 3:
        return signal - signal.mean()

    identity = sparse.eye(n, format="csc")
    d2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    operator = (identity + (lambda_param**2) * (d2.T @ d2)).tocsc()
    trend = sparse.linalg.spsolve(operator, signal)
    return signal - trend


def bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """Butterworth pasmowy, zerofazowy (pasmo z config)."""
    signal = np.asarray(signal, dtype=np.float64)
    nyquist = fs / 2.0
    sos = butter(
        BUTTERWORTH_ORDER,
        [BAND_LOW_HZ / nyquist, BAND_HIGH_HZ / nyquist],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, signal)


def estimate_hr_welch(
    signal: np.ndarray,
    fs: float,
    *,
    segment_sec: float | None = None,
    overlap: float = 0.5,
) -> float:
    """HR = argmax PSD Welcha w paśmie; nfft ≥ 2048.

    ``segment_sec=None`` -> ``WELCH_SEGMENT_SEC`` z config. Krótszy ``segment_sec``
    przy tym samym overlap daje więcej segmentów do uśrednienia i mniejszą wariancję PSD.
    """
    signal = np.asarray(signal, dtype=np.float64)
    seg_s = float(WELCH_SEGMENT_SEC if segment_sec is None else segment_sec)
    nperseg = max(8, min(len(signal), int(round(seg_s * fs))))
    noverlap = int(round(nperseg * float(overlap)))
    noverlap = min(max(0, noverlap), nperseg - 1)
    freqs, psd = welch(
        signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=max(nperseg, 2048)
    )
    band = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    if not np.any(band):
        raise ValueError("Brak składowych widma w paśmie HR.")
    return float(freqs[band][np.argmax(psd[band])] * 60.0)


def snr_rppg(
    signal: np.ndarray,
    fs: float,
    ref_hr_bpm: float,
    n_harmonics: int = 2,
    bin_width_hz: float = 0.2,
) -> float:
    """SNR [dB]: moc wokół f0 i jej harmonicznych vs reszta pasma HR."""
    signal = np.asarray(signal, dtype=np.float64)
    freqs, psd = periodogram(signal, fs=fs)
    band_mask = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)

    f0 = ref_hr_bpm / 60.0
    signal_mask = np.zeros_like(freqs, dtype=bool)
    for k in range(1, n_harmonics + 1):
        center = k * f0
        if center > BAND_HIGH_HZ:
            break
        signal_mask |= np.abs(freqs - center) <= bin_width_hz
    signal_mask &= band_mask
    noise_mask = band_mask & ~signal_mask

    return 10.0 * np.log10(
        (psd[signal_mask].sum() + _EPS) / (psd[noise_mask].sum() + _EPS)
    )
