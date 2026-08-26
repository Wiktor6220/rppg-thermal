"""Metody rPPG jako czyste funkcje sygnał→sygnał (GREEN, CHROM, POS, ICA).

Bez wczytywania plików, bez rysowania — dzięki temu testowalne na sygnale
syntetycznym o znanej częstości. ICA/PCA mają sens tylko na wielu kanałach
(3x RGB lub wiele ROI), nigdy na sygnale 1D.

Normalizacja zawsze po osi czasu (np. `x / x.mean()` po całym oknie), nigdy
per pojedyncza klatka — zgodnie z CLAUDE.md.
"""

import numpy as np
from scipy.signal import periodogram
from sklearn.decomposition import FastICA

from src.config import BAND_HIGH_HZ, BAND_LOW_HZ

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
        1D sygnał rPPG długości N — kanał zielony znormalizowany po osi czasu
        (`G / mean(G) - 1`, normalizacja po całym oknie, nigdy per klatka).
    """
    rgb_trace = _validate_rgb_trace(rgb_trace)
    g = rgb_trace[:, 1]
    return g / (g.mean() + _EPS) - 1.0


def chrom(rgb_trace: np.ndarray, fs: float) -> np.ndarray:
    """Metoda CHROM (de Haan & Jeanne, 2013) — sygnał rPPG odporny na ruch i oświetlenie.

    Sygnały chrominancji Xs = 3*Rn - 2*Gn, Ys = 1.5*Rn + Gn - 1.5*Bn liczone są
    w przesuwanym oknie o długości ~1.6 s, gdzie Rn/Gn/Bn to kanały znormalizowane
    przez swoją średnią w obrębie okna (temporal normalization, nigdy per klatka).
    Współczynnik alpha = std(Xs)/std(Ys) usuwa składową tonu skóry, pozostawiając
    pulsację. Okna łączone są metodą overlap-add.

    Args:
        rgb_trace: tablica (N, 3) średnich wartości R, G, B w czasie.
        fs: częstotliwość próbkowania sygnału (Hz).

    Returns:
        1D sygnał rPPG długości N.
    """
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

        alpha = np.std(x_s) / (np.std(y_s) + _EPS)
        chrom_window = x_s - alpha * y_s
        chrom_window -= chrom_window.mean()

        signal[start : start + window_len] += chrom_window

    return signal


def pos(rgb_trace: np.ndarray, fs: float) -> np.ndarray:
    """Metoda POS — Plane-Orthogonal-to-Skin (Wang et al., 2017).

    W przesuwanym oknie o długości ~1.6 s kanały normalizowane są przez swoją
    średnią w obrębie okna (temporal normalization). Projekcja na płaszczyznę
    ortogonalną do tonu skóry: S1 = Gn - Bn, S2 = Gn + Bn - 2*Rn, a sygnał
    pulsacyjny to h = S1 + alpha*S2, gdzie alpha = std(S1)/std(S2). Okna
    łączone są metodą overlap-add.

    Args:
        rgb_trace: tablica (N, 3) średnich wartości R, G, B w czasie.
        fs: częstotliwość próbkowania sygnału (Hz).

    Returns:
        1D sygnał rPPG długości N.
    """
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


def _dominant_hr_bpm(signal: np.ndarray, fs: float) -> float:
    """Pomocnicza estymata HR z piku widma mocy w paśmie fizjologicznym — tylko do testu."""
    freqs, psd = periodogram(signal, fs=fs)
    band_mask = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    band_freqs, band_psd = freqs[band_mask], psd[band_mask]
    peak_freq = band_freqs[np.argmax(band_psd)]
    return peak_freq * 60.0


def _generate_synthetic_rgb(
    fs: float = 30.0,
    duration_s: float = 30.0,
    hr_bpm: float = 72.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generuje syntetyczny sygnał RGB z osadzonym tętnem, szumem i wolnym dryfem."""
    rng = np.random.default_rng(seed)
    n_samples = int(round(fs * duration_s))
    t = np.arange(n_samples) / fs

    dc = np.array([120.0, 135.0, 115.0])  # baza (ton skóry, jednostki dowolne)
    pulse_weights = np.array([0.35, 1.0, 0.55])  # względna amplituda pulsacji per kanał
    pulse_amplitude = 0.02  # ~2% modulacji względem DC

    pulse_freq_hz = hr_bpm / 60.0
    pulse = np.sin(2 * np.pi * pulse_freq_hz * t)
    pulse_component = pulse_amplitude * dc[None, :] * pulse_weights[None, :] * pulse[:, None]

    drift = 0.1 * dc[None, :] * np.sin(2 * np.pi * 0.02 * t)[:, None]  # wolny dryf oświetlenia
    motion = dc[None, :] * rng.normal(0.0, 0.01, size=(n_samples, 1))  # artefakt ruchu (wspólny)
    noise = rng.normal(0.0, 0.3, size=(n_samples, 3))  # szum pomiarowy per kanał

    rgb_trace = dc[None, :] + pulse_component + drift + motion + noise
    return rgb_trace, t


if __name__ == "__main__":
    FS_TEST = 30.0
    TRUE_HR_BPM = 72.0
    TOLERANCE_BPM = 5.0

    rgb_signal, _ = _generate_synthetic_rgb(fs=FS_TEST, duration_s=30.0, hr_bpm=TRUE_HR_BPM)

    methods_under_test = {
        "GREEN": green,
        "CHROM": chrom,
        "POS": pos,
        "ICA": ica_method,
    }

    print(f"Zadana częstość: {TRUE_HR_BPM} BPM (tolerancja ±{TOLERANCE_BPM} BPM)\n")

    all_passed = True
    for name, method_fn in methods_under_test.items():
        pulse_signal = method_fn(rgb_signal, FS_TEST)
        estimated_bpm = _dominant_hr_bpm(pulse_signal, FS_TEST)
        error_bpm = abs(estimated_bpm - TRUE_HR_BPM)
        passed = error_bpm <= TOLERANCE_BPM
        all_passed &= passed
        status = "OK" if passed else "FAIL"
        print(f"[{status}] {name}: estymacja {estimated_bpm:.2f} BPM (błąd {error_bpm:.2f} BPM)")

    assert all_passed, "Co najmniej jedna metoda przekroczyła tolerancję błędu HR."
    print("\nWszystkie metody w granicach tolerancji.")
