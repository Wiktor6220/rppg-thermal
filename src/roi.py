"""Detekcja twarzy (MediaPipe FaceLandmarker na wycinku) i ROI bbox z landmarków."""

import os
import sys
from collections.abc import Callable
from contextlib import contextmanager

import numpy as np

from src.config import FACE_LANDMARKER_MODEL_PATH, FACE_MESH_LANDMARK_INDICES

_FACE_LANDMARKER = None  # singleton FaceLandmarker


def _quiet_native_logs() -> None:
    """Tłumi logi C++ MediaPipe/glog."""
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")


@contextmanager
def _suppress_stderr():
    """Tymczasowo przekierowuje stderr (fd=2) na /dev/null."""
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
    """Singleton FaceLandmarker (Tasks API, model .task w config)."""
    global _FACE_LANDMARKER
    if _FACE_LANDMARKER is None:
        if not FACE_LANDMARKER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Brak modelu FaceLandmarker: {FACE_LANDMARKER_MODEL_PATH}"
            )
        _quiet_native_logs()
        # Leniwy import: mediapipe ładuje natywne biblioteki i loguje na stderr,
        # więc wczytujemy go dopiero po wyciszeniu logów i tylko gdy detektor jest potrzebny.
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        with _suppress_stderr():
            _FACE_LANDMARKER = vision.FaceLandmarker.create_from_options(options)
    return _FACE_LANDMARKER


def detect_face_landmarks(frame: np.ndarray) -> np.ndarray | None:
    """Wykrywa punkty charakterystyczne twarzy (FaceLandmarker) na pojedynczej klatce RGB."""
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


def make_cropping_detector(
    crop_sizes: tuple[int, ...] = (1000, 1300, 700, 1600),
    reacquire_from_center: bool = True,
) -> Callable[[np.ndarray], np.ndarray | None]:
    """Detekcja na kwadratowym wycinku wokół ostatniej twarzy (4K / mała twarz)."""
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
                points_full = points + np.array([x0, y0], dtype=np.float64)
                state["cx"] = float(points_full[:, 0].mean())
                state["cy"] = float(points_full[:, 1].mean())
                state["size_idx"] = k
                return points_full

        if reacquire_from_center:
            state["cx"], state["cy"] = None, None
        return None

    return detector


def select_roi_from_landmarks(landmarks: np.ndarray, region: str) -> np.ndarray:
    """Bbox ROI [y0, x0, y1, x1] z landmarków regionu."""
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


def roi_mask_from_landmarks(
    landmarks: np.ndarray,
    region: str,
    shape: tuple[int, int],
) -> np.ndarray:
    """Maska bool ROI (prostokąt bbox z landmarków regionu)."""
    height, width = int(shape[0]), int(shape[1])
    y0, x0, y1, x1 = (int(v) for v in select_roi_from_landmarks(landmarks, region))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask
