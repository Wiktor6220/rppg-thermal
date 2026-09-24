"""Przetwarzanie sygnału rPPG: detrend, filtracja pasmowa, estymacja HR."""

import numpy as np
from scipy import sparse
from scipy.signal import butter, find_peaks, periodogram, sosfiltfilt, welch

from src.config import (
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    BUTTERWORTH_ORDER,
    DETREND_LAMBDA,
    HR_MAX_JUMP_BPM,
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


def _welch_band_spectrum(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Zwraca (freqs_hz, psd) ograniczone do pasma HR."""
    signal = np.asarray(signal, dtype=np.float64)
    nperseg = min(len(signal), int(round(WELCH_SEGMENT_SEC * fs)))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    band = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    if not np.any(band):
        raise ValueError("Brak składowych widma w paśmie fizjologicznym HR.")
    return freqs[band], psd[band]


def _prefer_fundamental_hz(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Wybiera częstość z ochroną przed 2× harmoniczną."""
    i_max = int(np.argmax(psd))
    f_max = float(freqs[i_max])

    peak_idx, _ = find_peaks(psd)
    if peak_idx.size == 0:
        peak_idx = np.array([i_max])

    # Jeśli argmax ≈ 2·f0 dla któregoś lokalnego maksimum → fundament.
    for idx in peak_idx:
        f0 = float(freqs[idx])
        if f0 <= 0:
            continue
        if abs(f_max - 2.0 * f0) <= max(0.12, 0.08 * f_max):
            return f0

    f_half = f_max / 2.0
    if f_half >= BAND_LOW_HZ:
        # bin Welcha najbliższy f_half — nawet bez find_peaks
        j = int(np.argmin(np.abs(freqs - f_half)))
        if abs(freqs[j] - f_half) <= 0.15 and psd[j] >= 0.12 * psd[i_max]:
            return float(freqs[j])
    return f_max


def estimate_hr_welch(
    signal: np.ndarray,
    fs: float,
    prev_hr_bpm: float | None = None,
    max_jump_bpm: float = HR_MAX_JUMP_BPM,
) -> float:
    """HR z Welcha: ochrona przed 2× oraz korekta harmoniczna względem poprzedniego okna."""
    freqs, psd = _welch_band_spectrum(signal, fs)
    hr = _prefer_fundamental_hz(freqs, psd) * 60.0

    if prev_hr_bpm is None or not np.isfinite(prev_hr_bpm):
        return float(hr)

    # Tylko warianty harmoniczne bieżącego wyboru — bez blokowania realnej zmiany HR.
    candidates = [hr]
    if BAND_LOW_HZ * 60 <= hr / 2 <= BAND_HIGH_HZ * 60:
        candidates.append(hr / 2)
    if BAND_LOW_HZ * 60 <= hr * 2 <= BAND_HIGH_HZ * 60:
        candidates.append(hr * 2)
    candidates = np.asarray(candidates, dtype=np.float64)
    dist = np.abs(candidates - prev_hr_bpm)
    j = int(np.argmin(dist))
    if dist[j] <= max_jump_bpm:
        return float(candidates[j])
    return float(hr)


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
