"""Testy korejestracji na syntetycznych maskach (bez nagrań)."""

import numpy as np

from src.registration import (
    affine_from_contours,
    affine_from_mask_moments,
    apply_affine,
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


def test_apply_affine_identity():
    eye = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pts = np.array([[10.0, 20.0], [3.0, 4.0]])
    assert np.allclose(apply_affine(eye, pts), pts)


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
    # Affine identity: thermal (40,50) -> (40,50); RGB eye y=200 -> dy=+150
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


def test_cut_neck_pinch_removes_shoulders():
    """Profil głowa -> szyja -> barki: cięcie w przewężeniu, nie poniżej najszerszego (barków)."""
    from src.registration import _cut_neck_width_profile

    h, w = 200, 120
    comp = np.zeros((h, w), dtype=np.uint8)
    # głowa (wiersze 20–80, szer. 40), szyja (80–100, szer. 18), barki (100–180, szer. 90)
    for r in range(20, 80):
        comp[r, 40:80] = 1
    for r in range(80, 100):
        comp[r, 51:69] = 1
    for r in range(100, 180):
        comp[r, 15:105] = 1

    out, info = _cut_neck_width_profile(comp)
    assert info["neck_method"] in ("pinch_shoulders", "pinch_deep", "head_frac_fallback")
    assert info["cut_row"] is not None
    assert 75 <= info["cut_row"] <= 105
    assert out[50, 60] == 1  # głowa zostaje
    assert out[150, 60] == 0  # barki odcięte
    # najszerszy wiersz to barki — cięcie „poniżej najszerszego wiersza” zostawiłoby barki
    assert int(comp.sum(axis=1).argmax()) >= 100


def test_constrain_thermal_mask_to_rgb_face():
    from src.registration import constrain_thermal_mask_to_rgb_face

    th = np.zeros((100, 80), dtype=bool)
    th[10:90, 10:70] = True  # duży blob
    rgb = np.zeros((200, 160), dtype=bool)
    rgb[40:120, 40:100] = True
    # scale 2, offset 0: th (x,y) -> rgb (2x, 2y)
    affine = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    constrained = constrain_thermal_mask_to_rgb_face(th, rgb, affine)
    assert constrained.any()
    assert constrained.sum() < th.sum()
    # punkt poza hull RGB (w termice nisko) powinien zniknąć
    assert not constrained[85, 40]


def _head_and_torso_thermal() -> np.ndarray:
    """Mała jasna elipsa (głowa) nad dużym jasnym prostokątem (tułów), rozdzielone ciemnym pasem."""
    import cv2

    gray = np.full((400, 300), 30, dtype=np.uint8)
    cv2.ellipse(gray, (150, 80), (30, 40), 0, 0, 360, 220, -1)  # głowa: bbox ~60×80
    gray[150:390, 40:260] = 220  # tułów: 220×240, 30 px ciemnego pasa pod głową
    return gray


def test_segment_prefers_head_with_expected_size():
    from src.registration import segment_thermal_face

    gray = _head_and_torso_thermal()
    window = (0, 0, gray.shape[1], gray.shape[0])

    mask, info = segment_thermal_face(gray, window, expected_wh=(60.0, 80.0))
    assert mask is not None and isinstance(info, dict)
    assert mask[80, 150] and not mask[270, 150]
    assert info["n_candidates"] == 2
    assert info["picked_rank"] == 1

    mask_big, info_big = segment_thermal_face(gray, window)
    assert mask_big is not None and isinstance(info_big, dict)
    assert mask_big[270, 150] and not mask_big[80, 150]
    assert info_big["picked_rank"] == 0


def test_calibration_rejects_implausible_scale():
    """W oknie priora tylko duży blob (skala > 1.0) -> ValueError."""
    import pytest

    from src.config import REG_CALIB_WINDOW_PAD, REG_PRIOR_OFFSET, REG_PRIOR_SCALE
    from src.registration import calibrate_nominal_registration, thermal_window_from_mapping

    # Twarz RGB 300×400 px w centrum klatki 4K.
    landmarks = np.array([[1770.0, 880.0], [2070.0, 880.0], [2070.0, 1280.0], [1770.0, 1280.0]])
    thermal = np.full((1024, 1280), 30, dtype=np.uint8)
    x0, y0, x1, y1 = thermal_window_from_mapping(
        landmarks, thermal.shape, REG_PRIOR_SCALE, REG_PRIOR_OFFSET, REG_CALIB_WINDOW_PAD
    )
    # Blob 400×550 px w termice (skala ~1.35), cały wewnątrz okna.
    assert x0 < 400 and x1 > 800 and y0 < 250 and y1 > 800
    thermal[250:800, 400:800] = 220
    with pytest.raises(ValueError, match="nieprawdopodobna skala"):
        calibrate_nominal_registration(np.zeros((8, 8, 3), np.uint8), thermal, landmarks)


def _glasses_head_neck_shoulders() -> np.ndarray:
    """Głowa 90 wierszy: okulary na ~40%, szyja na ~90% oczekiwanej wysokości 100, barki."""
    comp = np.zeros((180, 140), dtype=np.uint8)
    comp[10:100, 40:100] = 1  # głowa, szer. 60
    comp[46:53, 40:100] = 0
    comp[46:53, 55:85] = 1  # okulary: szer. 30 na wierszach 36–42 od góry segmentu
    comp[100:110, 58:82] = 1  # szyja, szer. 24 (90–99 od góry)
    comp[110:170, 5:135] = 1  # barki, szer. 130
    return comp


def test_neck_cut_skips_glasses_with_min_cut_rows():
    from src.registration import _cut_neck_width_profile

    comp = _glasses_head_neck_shoulders()
    _, info_free = _cut_neck_width_profile(comp, min_cut_rows=0)
    assert info_free["cut_row"] is not None
    assert 40 <= info_free["cut_row"] <= 60  # tnie na okularach

    out, info = _cut_neck_width_profile(comp, min_cut_rows=int(0.75 * 100))
    assert info["cut_row"] is not None
    assert 95 <= info["cut_row"] <= 112  # tnie na szyi
    assert out[70, 70] == 1  # dolna część twarzy zostaje
    assert out[140, 70] == 0  # barki odcięte


def test_affine_is_plausible_scale_and_anisotropy():
    from src.registration import affine_is_plausible

    def sim(s: float) -> np.ndarray:
        return np.array([[s, 0.0, 10.0], [0.0, s, 20.0]])

    assert affine_is_plausible(sim(1.9))
    assert not affine_is_plausible(sim(3.4))
    assert not affine_is_plausible(sim(0.6))
    assert not affine_is_plausible(np.array([[1.9, 0.0, 0.0], [0.0, 1.9 * 1.25, 0.0]]))


def test_calibration_rejects_wide_but_low_segment():
    """Segment ucięty na oczach: szer. ~0.62, wys. ~0.39 twarzy RGB -> ValueError."""
    import pytest

    from src.config import REG_CALIB_WINDOW_PAD, REG_PRIOR_OFFSET, REG_PRIOR_SCALE
    from src.registration import calibrate_nominal_registration, thermal_window_from_mapping

    landmarks = np.array([[1770.0, 880.0], [2070.0, 880.0], [2070.0, 1280.0], [1770.0, 1280.0]])
    thermal = np.full((1024, 1280), 30, dtype=np.uint8)
    x0, y0, x1, y1 = thermal_window_from_mapping(
        landmarks, thermal.shape, REG_PRIOR_SCALE, REG_PRIOR_OFFSET, REG_CALIB_WINDOW_PAD
    )
    # Blob 186×156 px (bbox max-min) = 0.62 × 300 i 0.39 × 400, wewnątrz okna.
    assert x0 < 503 and x1 > 690 and y0 < 426 and y1 > 583
    thermal[426:583, 503:690] = 220
    with pytest.raises(ValueError, match=r"szer\. 0\.62, wys\. 0\.39"):
        calibrate_nominal_registration(np.zeros((8, 8, 3), np.uint8), thermal, landmarks)
