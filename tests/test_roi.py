"""Testy ROI bbox z landmarków MediaPipe."""

import numpy as np

from src.config import FACE_MESH_LANDMARK_INDICES
from src.roi import roi_mask_from_landmarks, select_roi_from_landmarks


def _forehead_landmarks() -> np.ndarray:
    """468 landmarków; punkty czoła rozłożone w prostokącie x w [100, 200], y w [50, 90]."""
    landmarks = np.full((468, 2), 300.0)
    idx = FACE_MESH_LANDMARK_INDICES["forehead"]
    xs = np.linspace(100.0, 200.0, len(idx))
    ys = np.linspace(50.0, 90.0, len(idx))
    landmarks[idx] = np.column_stack([xs, ys])
    return landmarks


def test_forehead_bbox_is_full_landmark_extent():
    """Czoło = pełny bbox landmarków regionu (bez obcinania góry)."""
    y0, x0, y1, x1 = select_roi_from_landmarks(_forehead_landmarks(), "forehead")
    assert (y0, x0, y1, x1) == (50, 100, 90, 200)


def test_roi_mask_matches_bbox():
    """Maska ROI to dokładnie prostokąt bboxa."""
    mask = roi_mask_from_landmarks(_forehead_landmarks(), "forehead", (120, 240))
    assert mask[50:90, 100:200].all()
    assert mask.sum() == 40 * 100
