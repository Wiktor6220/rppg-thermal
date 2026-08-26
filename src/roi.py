"""Detekcja twarzy (MediaPipe Face Mesh), wybór i śledzenie ROI między klatkami.

Zasada „nigdy nie usuwaj klatki": gdy ROI nie zostanie znalezione, pozycja jest
przytrzymywana z poprzedniej klatki lub interpolowana, a klatka oznaczana jako
nieważna w wektorze `valid[]`. Odrzucane są okna, nie pojedyncze klatki.
"""

from collections.abc import Callable

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


def _fill_missing_roi(
    raw_roi: list[np.ndarray | None], valid: np.ndarray
) -> list[np.ndarray | None]:
    """Wypełnia luki w ROI: przytrzymanie ostatniej pozycji, a luki wiodące — pierwszą znaną.

    Zasada „nigdy nie usuwaj klatki": klatka bez detekcji dostaje ostatnią znaną
    pozycję ROI (hold). Klatki przed pierwszą detekcją nie mają czego przytrzymać —
    są uzupełniane wstecznie pierwszą wykrytą pozycją. Gdy nie ma ŻADNEJ detekcji,
    pozycje pozostają None (nie ma czym wypełnić).
    """
    n = len(raw_roi)
    filled: list[np.ndarray | None] = list(raw_roi)

    last_known: np.ndarray | None = None
    for i in range(n):
        if valid[i]:
            last_known = raw_roi[i]
        elif last_known is not None:
            filled[i] = last_known  # przytrzymanie ostatniej znanej pozycji

    first_known = next((raw_roi[i] for i in range(n) if valid[i]), None)
    if first_known is not None:
        for i in range(n):
            if filled[i] is None:  # luki wiodące (przed pierwszą detekcją)
                filled[i] = first_known
    return filled


def track_roi_across_frames(
    frames: np.ndarray,
    detector: Callable[[np.ndarray], np.ndarray | None] = detect_face_landmarks,
    roi_builder: Callable[[np.ndarray, str], np.ndarray] = select_roi_from_landmarks,
    region: str = "forehead",
) -> tuple[list[np.ndarray | None], np.ndarray]:
    """Śledzi ROI w sekwencji klatek, stosując detekcję landmarków i śledzenie między nimi.

    Sama logika śledzenia jest niezależna od konkretnego detektora — `detector`
    i `roi_builder` są wstrzykiwane (domyślnie MediaPipe Face Mesh). Dzięki temu
    logikę przytrzymania/`valid[]` można testować na atrapie detektora, bez
    uruchamiania MediaPipe. Przy braku detekcji ROI na danej klatce: przytrzymuje
    ostatnią znaną pozycję (luki wiodące — pierwsza znana), nigdy nie usuwa klatki.

    Args:
        frames: sekwencja klatek RGB o kształcie (N, H, W, 3).
        detector: funkcja klatka -> landmarki (N, 2) lub None przy braku detekcji.
        roi_builder: funkcja (landmarki, region) -> maska/bbox ROI.
        region: nazwa obszaru ROI przekazywana do `roi_builder`.

    Returns:
        Krotka (roi_positions, valid):
            roi_positions: lista długości N z maską/bboxem ROI dla każdej klatki
                (None tylko, gdy w całej sekwencji nie było ani jednej detekcji).
            valid: 1D tablica bool długości N — True, gdy ROI pochodzi z faktycznej
                detekcji, False, gdy zostało przytrzymane/uzupełnione.
    """
    n_frames = len(frames)
    raw_roi: list[np.ndarray | None] = [None] * n_frames
    valid = np.zeros(n_frames, dtype=bool)

    for i in range(n_frames):
        landmarks = detector(frames[i])
        if landmarks is not None:
            raw_roi[i] = roi_builder(landmarks, region)
            valid[i] = True

    roi_positions = _fill_missing_roi(raw_roi, valid)
    return roi_positions, valid
