"""Generatory sygnałów i klatek syntetycznych dla testów.

Nie jest to moduł testowy (brak prefiksu `test_`) — pytest go nie zbiera.
Trzyma logikę generującą dane, przeniesioną z bloków `__main__` modułów `src/`,
aby testy mogły ją współdzielić bez powielania.
"""

import numpy as np
from scipy.signal import periodogram

from src.config import BAND_HIGH_HZ, BAND_LOW_HZ


def dominant_hr_bpm(signal: np.ndarray, fs: float) -> float:
    """Estymata HR z piku widma mocy w paśmie fizjologicznym — pomocnicza do testów."""
    freqs, psd = periodogram(signal, fs=fs)
    band_mask = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    band_freqs, band_psd = freqs[band_mask], psd[band_mask]
    peak_freq = band_freqs[np.argmax(band_psd)]
    return peak_freq * 60.0


def generate_synthetic_rgb(
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


def generate_synthetic_pulse(
    fs: float = 30.0,
    duration_s: float = 60.0,
    hr_bpm: float = 72.0,
    seed: int = 0,
) -> np.ndarray:
    """Generuje syntetyczny sygnał pulsacyjny 1D z dryfem i szumem — do testów estimate."""
    rng = np.random.default_rng(seed)
    n_samples = int(round(fs * duration_s))
    t = np.arange(n_samples) / fs

    pulse_freq_hz = hr_bpm / 60.0
    # Druga harmoniczna nadaje przebiegowi ostrzejsze piki, bliższe kształtowi PPG.
    pulse = np.sin(2 * np.pi * pulse_freq_hz * t) + 0.3 * np.sin(2 * np.pi * 2 * pulse_freq_hz * t)

    drift = 5.0 * np.sin(2 * np.pi * 0.02 * t) + 0.05 * t  # wolny dryf: sinusoida + liniowy trend
    noise = rng.normal(0.0, 0.15, size=n_samples)

    return pulse + drift + noise


def generate_hard_synthetic_rgb(
    fs: float = 30.0,
    duration_s: float = 30.0,
    hr_bpm: float = 72.0,
    hrv_amplitude_bpm: float = 6.0,
    resp_freq_hz: float = 0.25,
    motion_amplitude: float = 0.05,
    interference_amplitude: float = 0.02,
    interference_freq_hz: float = 1.6,
    pulse_amplitude: float = 0.01,
    noise_std: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Trudny syntetyczny RGB: zmienne HR (HRV), wspólny artefakt ruchu i regulowany SNR.

    Model bliższy realnemu niż czysty sinus:
      - HRV: chwilowa częstość HR faluje wokół `hr_bpm` (arytmia zatokowa oddechowa),
      - artefakt ruchu/oświetlenia: WSPÓLNA multiplikatywna zmiana jasności wszystkich
        kanałów — składowa intensywności, którą CHROM/POS z założenia tłumią (kanały
        znoszą się w projekcji chrominancji), a GREEN przepuszcza w całości. Ma część
        wolną (`motion_amplitude`, ~0.11 Hz + błądzenie) oraz — co kluczowe —
        składową W PAŚMIE tętna (`interference_amplitude` przy `interference_freq_hz`,
        np. 1.6 Hz ≈ 96 bpm), która psuje SNR GREEN, ale nie CHROM/POS,
      - pulsacja o sygnaturze chrominancji (różne wagi per kanał, zielony najsilniejszy)
        — dzięki różnym wagom przeżywa projekcję CHROM/POS,
      - regulowany SNR przez `pulse_amplitude` (siła pulsu) i `noise_std` (szum).

    Zwraca (rgb_trace (N, 3), t). Pulsacja jest multiplikatywna względem DC, więc
    normalizacja po osi czasu (a nie per klatka) zachowuje kształt tętna.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(round(fs * duration_s))
    t = np.arange(n_samples) / fs

    dc = np.array([120.0, 135.0, 115.0])  # ton skóry (jednostki dowolne)
    pulse_weights = np.array([0.30, 1.0, 0.55])  # sygnatura chrominancji pulsu per kanał

    # HRV: chwilowa częstość waha się sinusoidalnie wokół częstości podstawowej.
    f0_hz = hr_bpm / 60.0
    hrv_hz = (hrv_amplitude_bpm / 60.0) * np.sin(2 * np.pi * resp_freq_hz * t)
    inst_freq_hz = f0_hz + hrv_hz
    phase = 2 * np.pi * np.cumsum(inst_freq_hz) / fs
    pulse = np.sin(phase)  # (N,)

    # Wspólny artefakt jasności (ten sam dla RGB): część wolna + składowa w paśmie tętna.
    random_walk = rng.standard_normal(n_samples).cumsum() / np.sqrt(n_samples)
    motion = motion_amplitude * (np.sin(2 * np.pi * 0.11 * t) + 0.5 * random_walk)
    interference = interference_amplitude * np.sin(2 * np.pi * interference_freq_hz * t)
    intensity = 1.0 + motion + interference  # (N,) — identyczna dla wszystkich kanałów

    pulse_factor = 1.0 + pulse_amplitude * pulse_weights[None, :] * pulse[:, None]  # (N, 3)
    noise = rng.normal(0.0, noise_std, size=(n_samples, 3))

    rgb_trace = dc[None, :] * intensity[:, None] * pulse_factor + noise
    return rgb_trace, t


def generate_synthetic_frames(
    fs: float = 30.0,
    duration_s: float = 20.0,
    hr_bpm: float = 72.0,
    height: int = 24,
    width: int = 24,
    pulse_amplitude: float = 0.03,
    noise_std: float = 1.5,
    seed: int = 0,
) -> dict:
    """Generuje syntetyczne KLATKI: RGB + skorejestrowaną termikę z łatą perfuzji.

    Konstrukcja odzwierciedla tezę pracy: istnieje przestrzenna łata (patch) o
    podwyższonej temperaturze (wysoka perfuzja), skorejestrowana między kamerą RGB
    i termiczną. TYLKO piksele łaty niosą silny sygnał pulsacyjny; reszta ROI to
    prawie sam szum. Dzięki temu ekstrakcja bramkowana termiką (łata) daje czystszy
    sygnał niż uśrednianie po całym ROI.

    Termika ma wartości radiometryczne (temperatura bezwzględna, °C), bez normalizacji.

    Returns:
        Słownik z kluczami:
            "rgb_frames": (N, H, W, 3) float,
            "thermal_frames": (N, H, W) float — temperatura bezwzględna,
            "roi_mask": (H, W) bool — cały analizowany obszar,
            "roi_positions": lista długości N tej samej maski ROI (ROI stałe w czasie),
            "valid": (N,) bool (wszystkie True),
            "patch_mask": (H, W) bool — prawdziwa łata wysokiej perfuzji,
            "hr_bpm": zadane HR, "t": (N,) oś czasu.
    """
    rng = np.random.default_rng(seed)
    n_frames = int(round(fs * duration_s))
    t = np.arange(n_frames) / fs

    # ROI: prostokąt w centrum kadru. Łata perfuzji: mniejszy prostokąt w rogu ROI.
    roi_mask = np.zeros((height, width), dtype=bool)
    roi_mask[4 : height - 4, 4 : width - 4] = True

    patch_mask = np.zeros((height, width), dtype=bool)
    patch_mask[6:12, 6:12] = True  # ~36 pikseli, wyraźnie wewnątrz ROI

    # Termika: baza 34.0 °C w ROI, łata cieplejsza o +1.5 °C (wysoka perfuzja).
    thermal_base = 34.0
    patch_temp_delta = 1.5
    thermal_static = np.full((height, width), thermal_base, dtype=np.float64)
    thermal_static[patch_mask] += patch_temp_delta

    pulse = np.sin(2 * np.pi * (hr_bpm / 60.0) * t)  # (N,)

    dc = np.array([120.0, 135.0, 115.0])  # ton skóry
    pulse_weights = np.array([0.35, 1.0, 0.55])

    rgb_frames = np.empty((n_frames, height, width, 3), dtype=np.float64)
    thermal_frames = np.empty((n_frames, height, width), dtype=np.float64)

    for i in range(n_frames):
        frame = np.broadcast_to(dc, (height, width, 3)).astype(np.float64).copy()
        # Modulacja pulsem (o sygnaturze per-kanał) TYLKO w łacie; poza łatą tylko DC + szum.
        frame[patch_mask] *= 1.0 + pulse_amplitude * pulse[i] * pulse_weights[None, :]
        frame += rng.normal(0.0, noise_std, size=frame.shape)
        rgb_frames[i] = frame

        # Termika prawie stała w czasie + drobny szum radiometryczny (bez pulsu).
        thermal_frames[i] = thermal_static + rng.normal(0.0, 0.05, size=(height, width))

    return {
        "rgb_frames": rgb_frames,
        "thermal_frames": thermal_frames,
        "roi_mask": roi_mask,
        "roi_positions": [roi_mask] * n_frames,
        "valid": np.ones(n_frames, dtype=bool),
        "patch_mask": patch_mask,
        "hr_bpm": hr_bpm,
        "t": t,
    }


def generate_two_stage_pulse(
    fs: float,
    duration_s: float,
    hr_bpm_stage1: float,
    hr_bpm_stage2: float,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generuje parę (czysty, zaszumiony) sygnał pulsacyjny ze skokową zmianą HR w połowie."""
    rng = np.random.default_rng(seed)
    n_samples = int(round(fs * duration_s))
    half = n_samples // 2

    inst_freq_hz = np.concatenate(
        [np.full(half, hr_bpm_stage1 / 60.0), np.full(n_samples - half, hr_bpm_stage2 / 60.0)]
    )
    phase = 2 * np.pi * np.cumsum(inst_freq_hz) / fs
    clean = np.sin(phase) + 0.3 * np.sin(2 * phase)  # harmoniczna -> ostrzejsze piki, jak w PPG
    noisy = clean + rng.normal(0.0, noise_std, size=n_samples)
    return clean, noisy
