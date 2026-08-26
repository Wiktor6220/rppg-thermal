"""Walidacja estymaty HR względem referencji: podział na okna, MAE, RMSE per okno.

Zgodnie z CONTEXT.md walidacja odbywa się w przesuwanych oknach (np. 10 s) —
HR liczone jest osobno w każdym oknie i porównywane z referencją, zamiast
sprowadzać cały nagrany sygnał do jednej liczby. Okna z przewagą nieważnych
klatek (wg wektora `valid[]` z roi.py) są pomijane w metrykach — odrzucane są
OKNA, nigdy pojedyncze klatki (CLAUDE.md).
"""

from collections.abc import Callable

import numpy as np

from src.config import MIN_VALID_RATIO, VALIDATION_STEP_SEC, VALIDATION_WINDOW_SEC
from src.estimate import bandpass_filter, detrend_signal, estimate_hr_peaks, estimate_hr_welch


def _window_bounds(n_samples: int, fs: float, window_s: float, step_s: float) -> list[tuple[int, int]]:
    """Wyznacza indeksy (start, koniec) kolejnych przesuwanych okien próbek."""
    window_len = int(round(window_s * fs))
    step_len = int(round(step_s * fs))
    if window_len <= 0 or step_len <= 0:
        raise ValueError("window_s i step_s muszą być dodatnie.")

    bounds = []
    start = 0
    while start + window_len <= n_samples:
        bounds.append((start, start + window_len))
        start += step_len
    return bounds


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
    signal = np.asarray(signal, dtype=np.float64)
    return [signal[start:end] for start, end in _window_bounds(signal.shape[0], fs, window_s, step_s)]


def compute_mae(estimated: np.ndarray, reference: np.ndarray) -> float:
    """Liczy średni błąd bezwzględny (MAE) między estymatami HR a referencją.

    Pary, w których estymata lub referencja to NaN (okno pominięte/nieudane),
    są ignorowane — zgodnie z zasadą odrzucania okien, a nie pojedynczych klatek.

    Args:
        estimated: 1D tablica estymat HR (per okno, bpm), może zawierać NaN.
        reference: 1D tablica referencyjnych wartości HR (per okno, bpm), może zawierać NaN.

    Returns:
        Wartość MAE (bpm), albo NaN, gdy brak par bez NaN.
    """
    estimated = np.asarray(estimated, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    diff = np.abs(estimated - reference)
    if not np.any(~np.isnan(diff)):
        return float("nan")
    return float(np.nanmean(diff))


def compute_rmse(estimated: np.ndarray, reference: np.ndarray) -> float:
    """Liczy pierwiastek błędu średniokwadratowego (RMSE) między estymatami HR a referencją.

    Pary z NaN (okno pominięte/nieudane) są ignorowane, analogicznie do `compute_mae`.

    Args:
        estimated: 1D tablica estymat HR (per okno, bpm), może zawierać NaN.
        reference: 1D tablica referencyjnych wartości HR (per okno, bpm), może zawierać NaN.

    Returns:
        Wartość RMSE (bpm), albo NaN, gdy brak par bez NaN.
    """
    estimated = np.asarray(estimated, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    squared_diff = (estimated - reference) ** 2
    if not np.any(~np.isnan(squared_diff)):
        return float("nan")
    return float(np.sqrt(np.nanmean(squared_diff)))


def validate_windows(
    estimated_hr_per_window: np.ndarray, reference_hr_per_window: np.ndarray
) -> dict[str, float]:
    """Agreguje metryki walidacyjne (MAE, RMSE) po wszystkich oknach.

    Args:
        estimated_hr_per_window: 1D tablica estymat HR, po jednej wartości na okno
            (NaN dla okien pominiętych).
        reference_hr_per_window: 1D tablica referencyjnych wartości HR, po jednej na okno
            (NaN dla okien pominiętych).

    Returns:
        Słownik z metrykami: {"mae_bpm", "rmse_bpm", "n_windows_used"}.
    """
    estimated_hr_per_window = np.asarray(estimated_hr_per_window, dtype=np.float64)
    reference_hr_per_window = np.asarray(reference_hr_per_window, dtype=np.float64)
    n_windows_used = int(
        np.sum(~np.isnan(estimated_hr_per_window) & ~np.isnan(reference_hr_per_window))
    )
    return {
        "mae_bpm": compute_mae(estimated_hr_per_window, reference_hr_per_window),
        "rmse_bpm": compute_rmse(estimated_hr_per_window, reference_hr_per_window),
        "n_windows_used": n_windows_used,
    }


def _window_valid_ratio(valid: np.ndarray, start: int, end: int) -> float:
    """Odsetek ważnych klatek (wg `valid[]`) w oknie [start, end)."""
    segment = valid[start:end]
    if segment.size == 0:
        return 0.0
    return float(np.mean(segment))


def _estimate_window_hr(
    window_signal: np.ndarray, fs: float, hr_estimator: Callable[[np.ndarray, float], float]
) -> float:
    """Detrend + filtracja pasmowa + estymacja HR na pojedynczym oknie; NaN przy błędzie."""
    try:
        cleaned = bandpass_filter(detrend_signal(window_signal), fs)
        return float(hr_estimator(cleaned, fs))
    except (ValueError, np.linalg.LinAlgError):
        return float("nan")


def validate_signal(
    estimated_signal: np.ndarray,
    reference_signal: np.ndarray,
    fs: float,
    valid: np.ndarray | None = None,
    window_s: float = VALIDATION_WINDOW_SEC,
    step_s: float = VALIDATION_STEP_SEC,
    min_valid_ratio: float = MIN_VALID_RATIO,
    hr_estimator: Callable[[np.ndarray, float], float] = estimate_hr_welch,
    reference_hr_estimator: Callable[[np.ndarray, float], float] = estimate_hr_peaks,
) -> dict:
    """Waliduje sygnał rPPG względem referencji w przesuwanych oknach czasowych.

    Dla każdego okna: sprawdza odsetek ważnych klatek (`valid[]`) — okna z
    przewagą nieważnych klatek (poniżej `min_valid_ratio`) są pomijane w
    metrykach. Dla pozostałych okien estymuje HR z sygnału (`hr_estimator`) i
    z referencji (`reference_hr_estimator`), po detrendzie i filtracji
    pasmowej każdego okna z osobna. Metryki (MAE, RMSE) liczone są per okno,
    nie jako jedna liczba na cały sygnał.

    Args:
        estimated_signal: 1D sygnał rPPG (wyjście `methods.py`), fs próbek.
        reference_signal: 1D sygnał referencyjny (np. PPG), tej samej długości
            i częstotliwości próbkowania co `estimated_signal`.
        fs: częstotliwość próbkowania obu sygnałów (Hz).
        valid: 1D tablica bool długości sygnału — wynik `roi.track_roi_across_frames`.
            Gdy None, wszystkie klatki uznawane są za ważne.
        window_s: długość okna w sekundach. Domyślnie `config.VALIDATION_WINDOW_SEC`.
        step_s: krok przesunięcia okna w sekundach. Domyślnie `config.VALIDATION_STEP_SEC`.
        min_valid_ratio: minimalny odsetek ważnych klatek wymagany, by okno nie
            zostało pominięte. Domyślnie `config.MIN_VALID_RATIO`.
        hr_estimator: funkcja estymująca HR z okna sygnału estymowanego
            (np. `estimate.estimate_hr_welch` lub `estimate.estimate_hr_peaks`).
        reference_hr_estimator: funkcja estymująca HR z okna sygnału referencyjnego.

    Returns:
        Słownik z wynikami per okno oraz zagregowanymi metrykami:
            "window_start_s": czas początku każdego okna (s),
            "estimated_hr_bpm": estymaty HR per okno (NaN dla pominiętych),
            "reference_hr_bpm": referencyjne HR per okno (NaN dla pominiętych),
            "error_bpm": błąd bezwzględny per okno (NaN dla pominiętych),
            "window_used": maska bool, które okna weszły do metryk,
            "n_windows_total": liczba wszystkich okien,
            "n_windows_used": liczba okien uwzględnionych w metrykach,
            "mae_bpm", "rmse_bpm": zagregowane metryki po użytych oknach.
    """
    estimated_signal = np.asarray(estimated_signal, dtype=np.float64)
    reference_signal = np.asarray(reference_signal, dtype=np.float64)
    if estimated_signal.shape[0] != reference_signal.shape[0]:
        raise ValueError(
            "Sygnał estymowany i referencyjny muszą mieć tę samą długość (wspólne fs)."
        )
    n_samples = estimated_signal.shape[0]

    if valid is None:
        valid = np.ones(n_samples, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape[0] != n_samples:
        raise ValueError("Wektor valid[] musi mieć tę samą długość co sygnały.")

    bounds = _window_bounds(n_samples, fs, window_s, step_s)
    n_windows = len(bounds)

    window_start_s = np.array([start / fs for start, _ in bounds])
    estimated_hr_bpm = np.full(n_windows, np.nan)
    reference_hr_bpm = np.full(n_windows, np.nan)
    window_used = np.zeros(n_windows, dtype=bool)

    for i, (start, end) in enumerate(bounds):
        if _window_valid_ratio(valid, start, end) < min_valid_ratio:
            continue  # okno z przewagą nieważnych klatek — pomijamy w metrykach

        est_hr = _estimate_window_hr(estimated_signal[start:end], fs, hr_estimator)
        ref_hr = _estimate_window_hr(reference_signal[start:end], fs, reference_hr_estimator)

        estimated_hr_bpm[i] = est_hr
        reference_hr_bpm[i] = ref_hr
        window_used[i] = not (np.isnan(est_hr) or np.isnan(ref_hr))

    error_bpm = np.abs(estimated_hr_bpm - reference_hr_bpm)
    metrics = validate_windows(estimated_hr_bpm, reference_hr_bpm)

    return {
        "window_start_s": window_start_s,
        "estimated_hr_bpm": estimated_hr_bpm,
        "reference_hr_bpm": reference_hr_bpm,
        "error_bpm": error_bpm,
        "window_used": window_used,
        "n_windows_total": n_windows,
        "n_windows_used": metrics["n_windows_used"],
        "mae_bpm": metrics["mae_bpm"],
        "rmse_bpm": metrics["rmse_bpm"],
    }


def _generate_two_stage_pulse(
    fs: float, duration_s: float, hr_bpm_stage1: float, hr_bpm_stage2: float, noise_std: float, seed: int
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


if __name__ == "__main__":
    FS_TEST = 30.0
    DURATION_S = 40.0
    HR_STAGE1_BPM = 65.0
    HR_STAGE2_BPM = 85.0
    TOLERANCE_BPM = 5.0

    reference_clean, estimated_noisy = _generate_two_stage_pulse(
        fs=FS_TEST,
        duration_s=DURATION_S,
        hr_bpm_stage1=HR_STAGE1_BPM,
        hr_bpm_stage2=HR_STAGE2_BPM,
        noise_std=0.2,
        seed=1,
    )
    n_samples = estimated_noisy.shape[0]

    # Symulacja utraty ROI (np. obrót głowy) przez 10 s: okno 10-20 s w całości nieważne.
    valid_vector = np.ones(n_samples, dtype=bool)
    valid_vector[300:600] = False

    result = validate_signal(estimated_noisy, reference_clean, FS_TEST, valid=valid_vector)

    print(f"Skok HR: {HR_STAGE1_BPM} -> {HR_STAGE2_BPM} BPM w połowie nagrania (t=20s)")
    print(f"Okna: {result['n_windows_total']} łącznie, {result['n_windows_used']} użytych w metrykach\n")

    print(f"{'start [s]':>10} {'est [bpm]':>10} {'ref [bpm]':>10} {'błąd':>8} {'użyte':>7}")
    for start_s, est, ref, err, used in zip(
        result["window_start_s"],
        result["estimated_hr_bpm"],
        result["reference_hr_bpm"],
        result["error_bpm"],
        result["window_used"],
    ):
        print(f"{start_s:>10.1f} {est:>10.2f} {ref:>10.2f} {err:>8.2f} {str(used):>7}")

    print(f"\nMAE = {result['mae_bpm']:.2f} bpm, RMSE = {result['rmse_bpm']:.2f} bpm")

    idx_10s = int(np.where(np.isclose(result["window_start_s"], 10.0))[0][0])
    assert not result["window_used"][idx_10s], "Okno 10-20s powinno zostać pominięte (100% nieważne)."
    assert np.isnan(result["estimated_hr_bpm"][idx_10s]), "Pominięte okno powinno mieć NaN."

    idx_0s = int(np.where(np.isclose(result["window_start_s"], 0.0))[0][0])
    idx_30s = int(np.where(np.isclose(result["window_start_s"], 30.0))[0][0])
    err_stage1 = abs(result["estimated_hr_bpm"][idx_0s] - HR_STAGE1_BPM)
    err_stage2 = abs(result["estimated_hr_bpm"][idx_30s] - HR_STAGE2_BPM)

    assert err_stage1 <= TOLERANCE_BPM, f"Okno 0-10s: błąd {err_stage1:.2f} bpm przekracza tolerancję."
    assert err_stage2 <= TOLERANCE_BPM, f"Okno 30-40s: błąd {err_stage2:.2f} bpm przekracza tolerancję."
    assert result["n_windows_used"] == result["n_windows_total"] - 1, "Powinno zostać pominięte 1 okno."

    print("\nTest zaliczony: skok HR śledzony per okno, okno z utraconym ROI poprawnie pominięte.")
