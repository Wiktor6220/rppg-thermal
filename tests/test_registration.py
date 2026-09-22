"""Testy korejestracji na syntetycznych maskach (bez nagrań)."""

import numpy as np

from src.registration import (
    affine_from_contours,
    affine_from_mask_moments,
    apply_affine,
    compose_affine,
    contour_sample_points,
    rgb_face_mask,
    warp_thermal_to_rgb,
)


def _ellipse_mask(shape, center, axes, angle_deg=0.0) -> np.ndarray:
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(center[0]), int(center[1])),
        (int(axes[0]), int(axes[1])),
        angle_deg,
        0,
        360,
        1,
        -1,
    )
    return mask.astype(bool)


def test_compose_and_apply_affine_identity():
    eye = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pts = np.array([[10.0, 20.0], [3.0, 4.0]])
    assert np.allclose(apply_affine(eye, pts), pts)
    assert np.allclose(compose_affine(eye, eye), eye)


def test_moments_recover_similarity():
    th = _ellipse_mask((200, 240), (80, 100), (30, 40), angle_deg=10)
    rgb = _ellipse_mask((400, 500), (80 * 2 + 50, 100 * 2 + 30), (60, 80), angle_deg=10)
    affine = affine_from_mask_moments(th, rgb)
    assert affine is not None
    pred = apply_affine(affine, np.array([[80.0, 100.0]]))[0]
    assert np.linalg.norm(pred - np.array([210.0, 230.0])) < 5.0


def test_contour_affine_maps_scaled_ellipse():
    """Kontury odzyskują skalę+translację między elipsami."""
    th = _ellipse_mask((200, 240), (80, 100), (30, 40))
    rgb = _ellipse_mask((400, 500), (200, 220), (60, 80))
    affine = affine_from_contours(th, rgb)
    assert affine is not None
    pred = apply_affine(affine, np.array([[80.0, 100.0]]))[0]
    assert np.linalg.norm(pred - np.array([200.0, 220.0])) < 8.0


def test_contour_sample_count():
    mask = _ellipse_mask((100, 100), (50, 50), (30, 25))
    pts = contour_sample_points(mask, n_points=24)
    assert pts is not None
    assert pts.shape == (24, 2)


def test_rgb_face_mask_from_square_landmarks():
    landmarks = np.array([[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]])
    mask = rgb_face_mask(landmarks, (60, 60))
    assert mask[30, 30]
    assert not mask[0, 0]
    assert mask.sum() > 100


def test_thermal_eye_line_in_band():
    """Najciemniejszy wiersz w pasie oczu, nie na górze maski."""
    from src.registration import thermal_eye_line

    gray = np.full((100, 80), 200, dtype=np.uint8)
    mask = np.zeros((100, 80), dtype=bool)
    mask[10:90, 20:60] = True
    gray[10, 20:60] = 50  # ciemna góra (włosy) — poza pasem 0.35–0.55
    gray[50, 20:60] = 80  # ciemny pas oczu w środku
    eye = thermal_eye_line(gray, mask, band_top=0.35, band_bottom=0.55)
    assert eye is not None
    assert eye[1] == 50.0


def test_refine_affine_eye_y_shifts_ty():
    from src.registration import refine_affine_eye_y

    gray = np.full((100, 80), 200, dtype=np.uint8)
    mask = np.zeros((100, 80), dtype=bool)
    mask[10:90, 20:60] = True
    gray[50, 20:60] = 80
    landmarks = np.zeros((300, 2), dtype=np.float64)
    landmarks[33] = [100.0, 200.0]
    landmarks[263] = [140.0, 200.0]
    # Affine identity: thermal (40,50) → (40,50); RGB eye y=200 → dy=+150
    eye = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    refined, info = refine_affine_eye_y(eye, gray, mask, landmarks)
    assert info is not None
    assert abs(info["dy"] - 150.0) < 1.0
    assert abs(refined[1, 2] - 150.0) < 1.0


def test_warp_thermal_to_rgb_shape():
    thermal = np.full((40, 50), 180, dtype=np.uint8)
    affine = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, 5.0]], dtype=np.float64)
    warped = warp_thermal_to_rgb(thermal, affine, (80, 100))
    assert warped.shape == (80, 100)
    assert warped.dtype == np.float64
