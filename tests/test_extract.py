"""Testy ekstrakcji RGB z ROI i maski perfuzji (dane syntetyczne)."""

import numpy as np
import pytest

from src.estimate import snr_rppg
from src.extract import (
    compute_perfusion_mask,
    extract_rgb_trace,
    gated_means_per_frame_from_samples,
    gated_means_per_window_from_samples,
)
from src.methods import green
from tests.synthetic import dominant_hr_bpm, generate_synthetic_frames

FS_TEST = 30.0
TRUE_HR_BPM = 72.0
TOLERANCE_BPM = 5.0
SEEDS = [0, 1, 2, 3]


@pytest.mark.parametrize("seed", SEEDS)
def test_perfusion_mask_matches_patch(seed):
    """Maska perfuzji pokrywa się z łatą i mieści się w ROI."""
    d = generate_synthetic_frames(fs=FS_TEST, hr_bpm=TRUE_HR_BPM, seed=seed)
    mask = compute_perfusion_mask(d["thermal_frames"][0], d["roi_mask"])
    truth = d["patch_mask"]

    assert (mask & ~d["roi_mask"]).sum() == 0
    jaccard = (mask & truth).sum() / (mask | truth).sum()
    assert jaccard >= 0.9


def test_perfusion_mask_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_perfusion_mask(np.zeros((10, 10)), np.ones((8, 8), dtype=bool))


@pytest.mark.parametrize("seed", SEEDS)
def test_extract_rgb_trace_shape_and_hr(seed):
    """Ekstrakcja z ROI → (N, 3) i odzyskany HR."""
    d = generate_synthetic_frames(fs=FS_TEST, hr_bpm=TRUE_HR_BPM, seed=seed)
    trace = extract_rgb_trace(d["rgb_frames"], d["roi_positions"], d["valid"])

    assert trace.shape == (d["rgb_frames"].shape[0], 3)
    hr = dominant_hr_bpm(green(trace, FS_TEST), FS_TEST)
    assert abs(hr - TRUE_HR_BPM) <= TOLERANCE_BPM


@pytest.mark.parametrize("seed", SEEDS)
def test_thermal_gating_improves_snr(seed):
    """Bramkowanie (ścieżka potoku: próbki → per-frame) daje wyższe SNR niż plain ROI."""
    d = generate_synthetic_frames(fs=FS_TEST, hr_bpm=TRUE_HR_BPM, seed=seed)
    rgb = d["rgb_frames"]
    thermal = d["thermal_frames"]
    n, height, width, _ = rgb.shape

    plain = extract_rgb_trace(rgb, d["roi_positions"], d["valid"])
    rgb_pix: list[np.ndarray | None] = []
    temps: list[np.ndarray | None] = []
    for i in range(n):
        roi = np.asarray(d["roi_positions"][i], dtype=bool)
        if roi.shape != (height, width):
            y0, x0, y1, x1 = (int(v) for v in roi)
            roi = np.zeros((height, width), dtype=bool)
            roi[y0:y1, x0:x1] = True
        ys, xs = np.where(roi)
        rgb_pix.append(rgb[i][ys, xs])
        temps.append(thermal[i][ys, xs].astype(np.float64))

    gated, _ = gated_means_per_frame_from_samples(rgb_pix, temps, plain)
    snr_plain = snr_rppg(green(plain, FS_TEST), FS_TEST, TRUE_HR_BPM)
    snr_gated = snr_rppg(green(gated, FS_TEST), FS_TEST, TRUE_HR_BPM)

    assert gated.shape == plain.shape
    assert snr_gated > snr_plain


def test_extract_rejects_length_mismatch():
    d = generate_synthetic_frames(fs=FS_TEST, duration_s=2.0, seed=0)
    with pytest.raises(ValueError):
        extract_rgb_trace(d["rgb_frames"], d["roi_positions"][:-1], d["valid"])


def test_extract_supports_bbox_roi():
    d = generate_synthetic_frames(fs=FS_TEST, duration_s=2.0, seed=0)
    n = d["rgb_frames"].shape[0]
    bbox = np.array([4, 4, d["rgb_frames"].shape[1] - 4, d["rgb_frames"].shape[2] - 4])

    trace_bbox = extract_rgb_trace(d["rgb_frames"], [bbox] * n, d["valid"])
    trace_mask = extract_rgb_trace(d["rgb_frames"], d["roi_positions"], d["valid"])
    np.testing.assert_allclose(trace_bbox, trace_mask)


def test_gated_per_window_one_threshold():
    """Per-window: jeden próg na okno; stała decyzja fallback wewnątrz okna."""
    n = 30
    plain = np.tile(np.array([0.1, 0.2, 0.3]), (n, 1))
    temps = []
    rgb_pix = []
    for _ in range(n):
        t = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        r = np.array(
            [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9]]
        )
        temps.append(t)
        rgb_pix.append(r)
    gated, fb = gated_means_per_window_from_samples(
        rgb_pix, temps, plain, fs=30.0, window_s=1.0, min_roi_frac=0.10
    )
    assert gated.shape == (n, 3)
    assert bool(np.all(fb == fb[0]))


def test_consensus_median_affine_and_point_dispersion():
    from src.extract import affine_point_dispersion_px, consensus_median_affine

    mats = [
        np.array([[1.0, 0.0, float(tx)], [0.0, 1.0, 0.0]]) for tx in (0.0, 10.0, 20.0)
    ]
    pts = np.array([[100.0, 50.0], [200.0, 50.0], [150.0, 150.0], [120.0, 200.0]])
    aff, resid = consensus_median_affine(mats, pts)
    assert aff is not None
    assert abs(aff[0, 2] - 10.0) < 1.0
    disp = affine_point_dispersion_px(mats, pts)
    assert disp["median_abs_dev"] > 0
    assert disp["iqr_radial"] > 0
