"""Detekcja twarzy (MediaPipe Face Mesh), wybór i śledzenie ROI między klatkami.

Zasada „nigdy nie usuwaj klatki": gdy ROI nie zostanie znalezione, pozycja jest
przytrzymywana z poprzedniej klatki lub interpolowana, a klatka oznaczana jako
nieważna w wektorze `valid[]`. Odrzucane są okna, nie pojedyncze klatki.

MediaPipe importowany jest leniwie (dopiero przy pierwszej realnej detekcji), aby
import modułu i testy jednostkowe (atrapa detektora) pozostały szybkie.
"""

from collections.abc import Callable, Iterable
from contextlib import contextmanager
import os
import sys

import cv2
import numpy as np

from src.config import FACE_LANDMARKER_MODEL_PATH, FACE_MESH_LANDMARK_INDICES

_FACE_LANDMARKER = None  # singleton MediaPipe Tasks FaceLandmarker (tworzony leniwie)


def _quiet_native_logs() -> None:
    """Tłumi spam C++ z MediaPipe / TFLite / glog (INFO/WARNING/ERROR telemetry)."""
    os.environ["GLOG_minloglevel"] = "3"  # tylko FATAL
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")


@contextmanager
def _suppress_stderr():
    """Przekierowuje fd=2 na /dev/null (logi natywne omijają logging Pythona)."""
    devnull = open(os.devnull, "w")
    stderr_fd = sys.stderr.fileno()
    saved = os.dup(stderr_fd)
    try:
        os.dup2(devnull.fileno(), stderr_fd)
        yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)
        devnull.close()


def _get_face_landmarker():
    """Zwraca współdzieloną instancję FaceLandmarker (Tasks API), tworzoną przy 1. użyciu.

    Ten build mediapipe udostępnia tylko API Tasks (brak `solutions.face_mesh`), więc
    detekcja wymaga pliku modelu `.task` (patrz `config.FACE_LANDMARKER_MODEL_PATH`).
    Tryb IMAGE = każda klatka niezależnie (uczciwa miara pokrycia detekcji).
    """
    global _FACE_LANDMARKER
    if _FACE_LANDMARKER is None:
        if not FACE_LANDMARKER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Brak modelu FaceLandmarker: {FACE_LANDMARKER_MODEL_PATH}. "
                "Pobierz face_landmarker.task do models/ (patrz komentarz w config.py)."
            )
        _quiet_native_logs()
        from mediapipe.tasks import python as mp_python  # leniwy import — ciężka biblioteka
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        # Init MediaPipe sypie I/W/E na stderr mimo GLOG_minloglevel.
        with _suppress_stderr():
            _FACE_LANDMARKER = vision.FaceLandmarker.create_from_options(options)
    return _FACE_LANDMARKER


def detect_face_landmarks(frame: np.ndarray) -> np.ndarray | None:
    """Wykrywa punkty charakterystyczne twarzy (FaceLandmarker) na pojedynczej klatce RGB.

    Args:
        frame: pojedyncza klatka obrazu RGB (H, W, 3), uint8 (ciągła w pamięci).

    Returns:
        Tablica (K, 2) punktów [x, y] w pikselach danej klatki (K=478 dla tego modelu;
        indeksy ROI z config są <468), albo None, gdy twarz nie została wykryta.
    """
    import mediapipe as mp

    landmarker = _get_face_landmarker()
    rgb = np.ascontiguousarray(frame, dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    height, width = frame.shape[:2]
    landmarks = result.face_landmarks[0]
    return np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float64)


def make_facemesh_detector(
    detection_width: int = 640,
) -> Callable[[np.ndarray], np.ndarray | None]:
    """Buduje detektor, który wykrywa na pomniejszonej klatce, ale zwraca punkty w oryginale.

    Klatki 4K są duże — detekcja na zmniejszonej kopii (`detection_width`) jest znacznie
    szybsza, a landmarki są przeskalowywane z powrotem do współrzędnych ORYGINAŁU, więc
    ROI liczone jest na pełnej rozdzielczości.

    Args:
        detection_width: docelowa szerokość klatki do detekcji (0 = bez zmniejszania).

    Returns:
        Funkcja klatka -> landmarki (468, 2) w oryginalnych współrzędnych albo None.
    """

    def detector(frame: np.ndarray) -> np.ndarray | None:
        height, width = frame.shape[:2]
        if detection_width and width > detection_width:
            scale = detection_width / width
            small = cv2.resize(
                frame,
                (detection_width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            points = detect_face_landmarks(small)
            return None if points is None else points / scale  # -> współrzędne oryginału
        return detect_face_landmarks(frame)

    return detector


def make_cropping_detector(
    crop_sizes: tuple[int, ...] = (1000, 1300, 700, 1600),
    reacquire_from_center: bool = True,
) -> Callable[[np.ndarray], np.ndarray | None]:
    """Buduje detektor „detekcja na wykadrowanym obszarze twarzy", odporny na małą twarz.

    Na nagraniach z drona twarz zajmuje kilka % szerokości kadru 4K; wewnętrzny detektor
    FaceLandmarker skaluje CAŁY obraz do ~192 px, więc tak mała twarz ginie i detekcja
    na pełnej klatce zawodzi. Rozwiązanie: utrzymuj środek ostatnio znalezionej twarzy
    i wykrywaj na kwadratowym wycinku wokół niego (twarz staje się dużo większą częścią
    kadru), a punkty mapuj z powrotem do współrzędnych ORYGINAŁU.

    Rozmiar twarzy zależy od dystansu (osoba dalej → mniejsza twarz → potrzebny CIAŚNIEJSZY
    wycinek). Dlatego próbujemy kilku rozmiarów `crop_sizes` w kolejności, zaczynając od
    ostatnio skutecznego; pierwszy z detekcją wygrywa. Detektor jest stanowy (pamięta
    środek i skalę między klatkami) — twórz osobną instancję na nagranie. Przy całkowitej
    utracie środek jest resetowany do centrum kadru (reakwizycja).

    Args:
        crop_sizes: boki kwadratowych wycinków (px oryginału) próbowane w kolejności.
        reacquire_from_center: przy braku detekcji wróć do środka kadru na następną próbę.

    Returns:
        Funkcja klatka -> landmarki (K, 2) we współrzędnych oryginału albo None.
    """
    state: dict[str, float | int | None] = {"cx": None, "cy": None, "size_idx": 0}

    def detector(frame: np.ndarray) -> np.ndarray | None:
        height, width = frame.shape[:2]
        center_x = state["cx"] if state["cx"] is not None else width / 2.0
        center_y = state["cy"] if state["cy"] is not None else height / 2.0

        last_idx = int(state["size_idx"] or 0)
        order = [last_idx] + [k for k in range(len(crop_sizes)) if k != last_idx]
        for k in order:
            size = crop_sizes[k]
            half = size // 2
            x0 = int(np.clip(center_x - half, 0, max(0, width - size)))
            y0 = int(np.clip(center_y - half, 0, max(0, height - size)))
            crop = np.ascontiguousarray(frame[y0 : y0 + size, x0 : x0 + size])

            points = detect_face_landmarks(crop)
            if points is not None:
                points_full = points + np.array([x0, y0], dtype=np.float64)  # -> oryginał
                state["cx"] = float(points_full[:, 0].mean())
                state["cy"] = float(points_full[:, 1].mean())
                state["size_idx"] = k
                return points_full

        if reacquire_from_center:
            state["cx"], state["cy"] = None, None
        return None

    return detector


def select_roi_from_landmarks(landmarks: np.ndarray, region: str) -> np.ndarray:
    """Wyznacza bounding box ROI dla danego regionu na podstawie punktów charakterystycznych.

    Region to jeden z kluczy `config.FACE_MESH_LANDMARK_INDICES` (np. "forehead",
    "left_cheek", "right_cheek"). ROI zwracane jest jako bbox `[y0, x0, y1, x1]`
    (spójnie z `extract._roi_to_mask`), obejmujący klaster punktów regionu.

    Args:
        landmarks: punkty charakterystyczne twarzy (468, 2) [x, y], wynik detekcji.
        region: nazwa regionu ROI z `config.FACE_MESH_LANDMARK_INDICES`.

    Returns:
        Bounding box `[y0, x0, y1, x1]` (int), z dolną granicą przyciętą do 0.

    Raises:
        ValueError: gdy `region` nie występuje w konfiguracji.
    """
    if region not in FACE_MESH_LANDMARK_INDICES:
        raise ValueError(
            f"Nieznany region ROI: {region!r}. Dostępne: {list(FACE_MESH_LANDMARK_INDICES)}"
        )
    points = landmarks[FACE_MESH_LANDMARK_INDICES[region]]
    x0 = max(0, int(np.floor(points[:, 0].min())))
    y0 = max(0, int(np.floor(points[:, 1].min())))
    x1 = int(np.ceil(points[:, 0].max()))
    y1 = int(np.ceil(points[:, 1].max()))
    return np.array([y0, x0, y1, x1], dtype=int)


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
    frames: Iterable[np.ndarray],
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

    Klatki są konsumowane leniwie (iterowalne), więc można podać generator z io_layer
    i nie materializować całego (dużego, 4K) wideo w pamięci.

    Args:
        frames: iterowalne klatek RGB (H, W, 3) — np. generator z `io_layer`.
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
    raw_roi: list[np.ndarray | None] = []
    valid_list: list[bool] = []
    for frame in frames:
        landmarks = detector(frame)
        if landmarks is not None:
            raw_roi.append(roi_builder(landmarks, region))
            valid_list.append(True)
        else:
            raw_roi.append(None)
            valid_list.append(False)

    valid = np.array(valid_list, dtype=bool)
    roi_positions = _fill_missing_roi(raw_roi, valid)
    return roi_positions, valid
