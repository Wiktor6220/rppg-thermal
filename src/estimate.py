"""Przetwarzanie sygnału rPPG do estymaty HR: detrend, filtracja pasmowa, HR z Welcha/pików."""

import numpy as np
from scipy import sparse
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

from src.config import (
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    BUTTERWORTH_ORDER,
    DETREND_LAMBDA,
    WELCH_SEGMENT_SEC,
)


def detrend_signal(signal: np.ndarray, lambda_param: float = DETREND_LAMBDA) -> np.ndarray:
    """Usuwa wolnozmienny trend metodą smoothness priors (Tarvainen et al., 2002).

    Trend jest szacowany jako przebieg z minimalną krzywizną, który jednocześnie
    dobrze przybliża sygnał wejściowy: minimalizowana jest suma kwadratów
    odchyleń trendu od sygnału, karana drugą pochodną (krzywizną) trendu ważoną
    parametrem `lambda_param`. Rozwiązanie ma postać zamkniętą
    `z_trend = (I + lambda^2 * D2^T D2)^-1 z`, gdzie `D2` to macierz drugiej
    różnicy. Im większe `lambda_param`, tym silniej tłumione są wolne składowe
    (silniejszy efekt górnoprzepustowy).

    Args:
        signal: 1D sygnał wejściowy.
        lambda_param: parametr regularyzacji. Domyślnie `config.DETREND_LAMBDA`.

    Returns:
        1D sygnał po usunięciu trendu, tej samej długości co wejście.
    """
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
    """Filtruje sygnał pasmowo-przepustowo (Butterworth, filtracja zerofazowa).

    Pasmo i rząd filtru brane są z `config.py` (`BAND_LOW_HZ`, `BAND_HIGH_HZ`,
    `BUTTERWORTH_ORDER`), odpowiadające fizjologicznemu zakresowi HR.

    Args:
        signal: 1D sygnał wejściowy.
        fs: częstotliwość próbkowania sygnału (Hz).

    Returns:
        1D sygnał po filtracji pasmowej, tej samej długości co wejście.
    """
    signal = np.asarray(signal, dtype=np.float64)
    nyquist_hz = fs / 2.0
    sos = butter(
        BUTTERWORTH_ORDER,
        [BAND_LOW_HZ / nyquist_hz, BAND_HIGH_HZ / nyquist_hz],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, signal)


def estimate_hr_welch(signal: np.ndarray, fs: float) -> float:
    """Estymuje HR na podstawie widma mocy sygnału (metoda Welcha) — dominująca częstość.

    Długość segmentu Welcha brana jest z `config.WELCH_SEGMENT_SEC`, a pasmo
    poszukiwania piku widma z `config.BAND_LOW_HZ`/`config.BAND_HIGH_HZ`.

    Args:
        signal: 1D sygnał rPPG (po detrendingu i filtracji pasmowej).
        fs: częstotliwość próbkowania sygnału (Hz).

    Returns:
        Estymowana częstość akcji serca w uderzeniach na minutę (bpm).
    """
    signal = np.asarray(signal, dtype=np.float64)
    nperseg = min(len(signal), int(round(WELCH_SEGMENT_SEC * fs)))

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    band_mask = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    if not np.any(band_mask):
        raise ValueError("Brak składowych widma w paśmie fizjologicznym HR.")

    dominant_freq_hz = freqs[band_mask][np.argmax(psd[band_mask])]
    return dominant_freq_hz * 60.0


def estimate_hr_peaks(signal: np.ndarray, fs: float) -> float:
    """Estymuje HR na podstawie detekcji pików w dziedzinie czasu (`find_peaks`).

    Minimalny odstęp między pikami wyznaczany jest z `config.BAND_HIGH_HZ`
    (najwyższa dopuszczalna częstość HR), aby wykluczyć niefizjologicznie
    bliskie detekcje.

    Args:
        signal: 1D sygnał rPPG (po detrendingu i filtracji pasmowej).
        fs: częstotliwość próbkowania sygnału (Hz).

    Returns:
        Estymowana częstość akcji serca w uderzeniach na minutę (bpm).
    """
    signal = np.asarray(signal, dtype=np.float64)
    min_distance_samples = max(1, int(round(fs / BAND_HIGH_HZ)))

    peaks, _ = find_peaks(signal, distance=min_distance_samples)
    if len(peaks) < 2:
        raise ValueError("Za mało wykrytych pików do estymacji HR.")

    mean_interval_s = np.mean(np.diff(peaks)) / fs
    return 60.0 / mean_interval_s


def _generate_synthetic_pulse(
    fs: float = 30.0,
    duration_s: float = 60.0,
    hr_bpm: float = 72.0,
    seed: int = 0,
) -> np.ndarray:
    """Generuje syntetyczny sygnał pulsacyjny 1D z dryfem i szumem — tylko do testu."""
    rng = np.random.default_rng(seed)
    n_samples = int(round(fs * duration_s))
    t = np.arange(n_samples) / fs

    pulse_freq_hz = hr_bpm / 60.0
    # Druga harmoniczna nadaje przebiegowi ostrzejsze piki, bliższe kształtowi PPG.
    pulse = np.sin(2 * np.pi * pulse_freq_hz * t) + 0.3 * np.sin(2 * np.pi * 2 * pulse_freq_hz * t)

    drift = 5.0 * np.sin(2 * np.pi * 0.02 * t) + 0.05 * t  # wolny dryf: sinusoida + liniowy trend
    noise = rng.normal(0.0, 0.15, size=n_samples)

    return pulse + drift + noise


if __name__ == "__main__":
    FS_TEST = 30.0
    TRUE_HR_BPM = 72.0
    TOLERANCE_BPM = 5.0

    raw_signal = _generate_synthetic_pulse(fs=FS_TEST, duration_s=60.0, hr_bpm=TRUE_HR_BPM)

    detrended = detrend_signal(raw_signal)
    filtered = bandpass_filter(detrended, FS_TEST)

    hr_welch = estimate_hr_welch(filtered, FS_TEST)
    hr_peaks = estimate_hr_peaks(filtered, FS_TEST)

    print(f"Zadana częstość: {TRUE_HR_BPM} BPM (tolerancja ±{TOLERANCE_BPM} BPM)\n")

    results = {"Welch (widmo mocy)": hr_welch, "find_peaks (dziedzina czasu)": hr_peaks}
    all_passed = True
    for name, hr_bpm in results.items():
        error_bpm = abs(hr_bpm - TRUE_HR_BPM)
        passed = error_bpm <= TOLERANCE_BPM
        all_passed &= passed
        status = "OK" if passed else "FAIL"
        print(f"[{status}] {name}: {hr_bpm:.2f} BPM (błąd {error_bpm:.2f} BPM)")

    agreement_bpm = abs(hr_welch - hr_peaks)
    print(f"\nRozbieżność między metodami: {agreement_bpm:.2f} BPM")

    assert all_passed, "Co najmniej jedna metoda estymacji HR przekroczyła tolerancję błędu."
    assert agreement_bpm <= TOLERANCE_BPM, "Metody Welch i find_peaks znacząco się rozjeżdżają."
    print("Obie metody w granicach tolerancji i zgodne ze sobą.")
