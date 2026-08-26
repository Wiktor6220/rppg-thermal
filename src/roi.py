"""Detekcja twarzy (MediaPipe Face Mesh), wybór i śledzenie ROI między klatkami.

Zasada „nigdy nie usuwaj klatki": gdy ROI nie zostanie znalezione, pozycja jest
przytrzymywana z poprzedniej klatki lub interpolowana, a klatka oznaczana jako
nieważna w wektorze `valid[]`. Odrzucane są okna, nie pojedyncze klatki.
"""

import numpy as np


def detect_face_landmarks(frame: np.ndarray) -> np.ndarray | None:
    """Wykrywa punkty charakterystyczne twarzy (Face Mesh) na pojedynczej klatce RGB.

    Args:
        frame: pojedyncza klatka obrazu RGB o kształcie (H, W, 3).

    Returns:
        Tablica punktów charakterystycznych (N, 2) w pikselach, albo None, gdy twarz
        nie została wykryta na klatce.
    """
    raise NotImplementedError


def select_roi_from_landmarks(landmarks: np.ndarray, region: str) -> np.ndarray:
    """Wyznacza maskę lub bounding box ROI na podstawie punktów charakterystycznych.

    Args:
        landmarks: punkty charakterystyczne twarzy (N, 2), wynik `detect_face_landmarks`.
        region: nazwa obszaru ROI (np. "forehead", "cheeks") zdefiniowana w config.py.

    Returns:
        Maska binarna ROI o kształcie (H, W) lub współrzędne bounding boxa.
    """
    raise NotImplementedError


def track_roi_across_frames(
    frames: np.ndarray,
) -> tuple[list[np.ndarray | None], np.ndarray]:
    """Śledzi ROI w sekwencji klatek, stosując detekcję co N klatek i śledzenie między nimi.

    Przy braku detekcji ROI na danej klatce: przytrzymuje ostatnią znaną pozycję lub
    interpoluje, nigdy nie usuwa klatki z sekwencji.

    Args:
        frames: sekwencja klatek RGB o kształcie (N, H, W, 3).

    Returns:
        Krotka (roi_positions, valid):
            roi_positions: lista długości N z maską/bboxem ROI dla każdej klatki
                (nigdy None po zastosowaniu przytrzymania/interpolacji).
            valid: 1D tablica bool długości N — True, gdy ROI pochodzi z faktycznej
                detekcji, False, gdy zostało przytrzymane/interpolowane.
    """
    raise NotImplementedError
