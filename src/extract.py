"""Ekstrakcja surowego sygnału rPPG z ROI oraz maskowanie termiczne wg perfuzji.

Rdzeń tezy pracy: mapa termiczna (temperatura bezwzględna, bez normalizacji per
klatka) wskazuje piksele o najwyższej perfuzji i bramkuje z nich ekstrakcję
sygnału RGB — termika jest przestrzennym selektorem ROI, nie równoległym pomiarem.

Zgodnie z CLAUDE.md ekstrakcja liczy średnią po pikselach ROI dla KAŻDEJ klatki
(nigdy nie usuwamy klatki — oś czasu ma stały krok). Wektor `valid[]` służy tylko
do późniejszego odrzucania OKIEN w walidacji.
"""

import numpy as np

from src.config import PERFUSION_TEMP_STD_FACTOR

_EPS = 1e-8


def _roi_to_mask(roi_position: np.ndarray, height: int, width: int) -> np.ndarray:
    """Sprowadza ROI (maska binarna (H, W) lub bbox [y0, x0, y1, x1]) do maski (H, W)."""
    roi_position = np.asarray(roi_position)
    if roi_position.ndim == 2 and roi_position.shape == (height, width):
        return roi_position.astype(bool)
    if roi_position.shape == (4,):
        y0, x0, y1, x1 = (int(v) for v in roi_position)
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask
    raise ValueError(
        f"ROI musi być maską (H, W)=({height}, {width}) lub bboxem (4,), "
        f"otrzymano kształt {roi_position.shape}"
    )


def _mean_rgb_in_mask(rgb_frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Średnia R, G, B po pikselach maski; przy pustej masce — średnia po całej klatce.

    Nigdy nie zwraca NaN i nie „gubi" klatki — pusta maska degraduje się do średniej
    z całej klatki, aby zachować stały krok osi czasu (wymóg FFT/Welcha).
    """
    if mask.any():
        return rgb_frame[mask].mean(axis=0)
    return rgb_frame.reshape(-1, rgb_frame.shape[-1]).mean(axis=0)


def _check_lengths(n_frames: int, roi_positions: list, valid: np.ndarray) -> None:
    """Waliduje spójność długości sekwencji klatek, pozycji ROI i wektora valid."""
    if len(roi_positions) != n_frames:
        raise ValueError("Liczba pozycji ROI musi odpowiadać liczbie klatek.")
    if np.asarray(valid).shape[0] != n_frames:
        raise ValueError("Długość wektora valid[] musi odpowiadać liczbie klatek.")


def extract_rgb_trace(
    rgb_frames: np.ndarray, roi_positions: list[np.ndarray], valid: np.ndarray
) -> np.ndarray:
    """Liczy średnie wartości kanałów RGB w obrębie ROI dla każdej klatki.

    Args:
        rgb_frames: sekwencja klatek RGB o kształcie (N, H, W, 3).
        roi_positions: lista długości N z maską (H, W) lub bboxem ROI dla każdej klatki
            (wynik `roi.track_roi_across_frames`).
        valid: 1D tablica bool długości N oznaczająca klatki z faktyczną detekcją ROI.
            Nie służy do usuwania klatek — ekstrakcja liczona jest dla wszystkich.

    Returns:
        Tablica (N, 3) średnich wartości R, G, B w ROI dla każdej klatki.
    """
    rgb_frames = np.asarray(rgb_frames, dtype=np.float64)
    n_frames, height, width, _ = rgb_frames.shape
    _check_lengths(n_frames, roi_positions, valid)

    trace = np.empty((n_frames, 3), dtype=np.float64)
    for i in range(n_frames):
        mask = _roi_to_mask(roi_positions[i], height, width)
        trace[i] = _mean_rgb_in_mask(rgb_frames[i], mask)
    return trace


def compute_perfusion_mask(thermal_frame: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Wyznacza maskę pikseli o podwyższonej perfuzji na podstawie mapy termicznej.

    Działa na wartościach radiometrycznych (temperatura bezwzględna), bez
    normalizacji per klatka. Piksel jest „wysokiej perfuzji", gdy jego temperatura
    przekracza średnią ROI o co najmniej `PERFUSION_TEMP_STD_FACTOR` odchyleń
    standardowych temperatury w ROI. Próg jest względny wobec rozkładu temperatury
    w ROI (nie skalujemy ani nie normalizujemy wartości pikseli).

    Args:
        thermal_frame: pojedyncza klatka termiczna (H, W), wartości radiometryczne.
        roi_mask: maska binarna ROI (H, W) ograniczająca obszar analizy.

    Returns:
        Maska binarna (H, W) pikseli o podwyższonej perfuzji w obrębie ROI.
    """
    thermal_frame = np.asarray(thermal_frame, dtype=np.float64)
    roi_mask = np.asarray(roi_mask, dtype=bool)
    if thermal_frame.shape != roi_mask.shape:
        raise ValueError("Klatka termiczna i maska ROI muszą mieć ten sam kształt (H, W).")

    roi_values = thermal_frame[roi_mask]
    if roi_values.size == 0:
        return np.zeros_like(roi_mask, dtype=bool)

    threshold = roi_values.mean() + PERFUSION_TEMP_STD_FACTOR * roi_values.std()
    return roi_mask & (thermal_frame >= threshold)


def extract_rgb_trace_thermal_gated(
    rgb_frames: np.ndarray,
    thermal_frames: np.ndarray,
    roi_positions: list[np.ndarray],
    valid: np.ndarray,
) -> np.ndarray:
    """Liczy średnie RGB w ROI ograniczonym dodatkowo maską perfuzji z termiki.

    Args:
        rgb_frames: sekwencja klatek RGB o kształcie (N, H, W, 3).
        thermal_frames: sekwencja klatek termicznych (N, H, W), po korekcji
            paralaksy względem kamery RGB (warping wykonany wcześniej).
        roi_positions: lista długości N z maską (H, W) lub bboxem ROI dla każdej klatki.
        valid: 1D tablica bool długości N oznaczająca klatki z faktyczną detekcją ROI.

    Returns:
        Tablica (N, 3) średnich wartości R, G, B w ROI zbramkowanym mapą perfuzji.
        Gdy maska perfuzji jest pusta, degraduje się do średniej z ROI (klatka nie ginie).
    """
    rgb_frames = np.asarray(rgb_frames, dtype=np.float64)
    thermal_frames = np.asarray(thermal_frames, dtype=np.float64)
    n_frames, height, width, _ = rgb_frames.shape
    _check_lengths(n_frames, roi_positions, valid)
    if thermal_frames.shape[0] != n_frames:
        raise ValueError("Liczba klatek termicznych musi odpowiadać liczbie klatek RGB.")

    trace = np.empty((n_frames, 3), dtype=np.float64)
    for i in range(n_frames):
        roi_mask = _roi_to_mask(roi_positions[i], height, width)
        perfusion_mask = compute_perfusion_mask(thermal_frames[i], roi_mask)
        # Przy pustej masce perfuzji wracamy do pełnego ROI — nie gubimy klatki.
        gated_mask = perfusion_mask if perfusion_mask.any() else roi_mask
        trace[i] = _mean_rgb_in_mask(rgb_frames[i], gated_mask)
    return trace
