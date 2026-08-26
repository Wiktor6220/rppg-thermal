"""Ekstrakcja surowego sygnału rPPG z ROI oraz maskowanie termiczne wg perfuzji.

Rdzeń tezy pracy: mapa termiczna (temperatura bezwzględna, bez normalizacji per
klatka) wskazuje piksele o najwyższej perfuzji i bramkuje z nich ekstrakcję
sygnału RGB — termika jest przestrzennym selektorem ROI, nie równoległym pomiarem.
"""

import numpy as np


def extract_rgb_trace(
    rgb_frames: np.ndarray, roi_positions: list[np.ndarray], valid: np.ndarray
) -> np.ndarray:
    """Liczy średnie wartości kanałów RGB w obrębie ROI dla każdej klatki.

    Args:
        rgb_frames: sekwencja klatek RGB o kształcie (N, H, W, 3).
        roi_positions: lista długości N z maską/bboxem ROI dla każdej klatki
            (wynik `roi.track_roi_across_frames`).
        valid: 1D tablica bool długości N oznaczająca klatki z faktyczną detekcją ROI.

    Returns:
        Tablica (N, 3) średnich wartości R, G, B w ROI dla każdej klatki.
    """
    raise NotImplementedError


def compute_perfusion_mask(thermal_frame: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Wyznacza maskę pikseli o podwyższonej perfuzji na podstawie mapy termicznej.

    Działa na wartościach radiometrycznych (temperatura bezwzględna), bez
    normalizacji per klatka — zgodnie z zasadami projektu.

    Args:
        thermal_frame: pojedyncza klatka termiczna (H, W), wartości radiometryczne.
        roi_mask: maska binarna ROI (H, W) ograniczająca obszar analizy.

    Returns:
        Maska binarna (H, W) pikseli o podwyższonej perfuzji w obrębie ROI.
    """
    raise NotImplementedError


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
        roi_positions: lista długości N z maską/bboxem ROI dla każdej klatki.
        valid: 1D tablica bool długości N oznaczająca klatki z faktyczną detekcją ROI.

    Returns:
        Tablica (N, 3) średnich wartości R, G, B w ROI zbramkowanym mapą perfuzji.
    """
    raise NotImplementedError
