"""Przetwarzanie sygnału rPPG do estymaty HR: detrend, filtracja pasmowa, HR z Welcha/pików."""

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


def snr_rppg(
    signal: np.ndarray,
    fs: float,
    ref_hr_bpm: float,
    n_harmonics: int = 2,
    bin_width_hz: float = 0.2,
) -> float:
    """Stosunek sygnału do szumu (SNR) sygnału rPPG względem znanej częstości HR.

    Miara w duchu de Haan & Jeanne (2013): moc skupiona w wąskich prążkach wokół
    częstości podstawowej HR i jej harmonicznych („sygnał tętna") odniesiona do
    mocy w pozostałej części pasma fizjologicznego („reszta"). Im czystszy sygnał
    pulsacyjny (mniej artefaktów ruchu/szumu w paśmie), tym wyższe SNR. Pozwala
    różnicować metody rPPG tam, gdzie sama estymata HR jeszcze się nie rozjeżdża.

    Args:
        signal: 1D sygnał rPPG (wyjście metody z `methods.py`).
        fs: częstotliwość próbkowania sygnału (Hz).
        ref_hr_bpm: referencyjna (znana) częstość HR w bpm, wokół której skupiona
            jest oczekiwana moc pulsacji.
        n_harmonics: liczba uwzględnianych harmonicznych (1 = tylko podstawowa).
        bin_width_hz: połowa szerokości prążka wokół każdej harmonicznej (Hz).

    Returns:
        SNR w decybelach (10*log10(moc_sygnału / moc_reszty_pasma)).
    """
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
