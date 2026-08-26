"""Testy ekstrakcji RGB z ROI i maski perfuzji na syntetycznych KLATKACH (bez danych)."""

import numpy as np
import pytest

from src.estimate import snr_rppg
from src.extract import (
    compute_perfusion_mask,
    extract_rgb_trace,
    extract_rgb_trace_thermal_gated,
)
from src.methods import green
from tests.synthetic import dominant_hr_bpm, generate_synthetic_frames

FS_TEST = 30.0
TRUE_HR_BPM = 72.0
TOLERANCE_BPM = 5.0
SEEDS = [0, 1, 2, 3]


@pytest.mark.parametrize("seed", SEEDS)
def test_perfusion_mask_matches_patch(seed):
    """Maska perfuzji pokrywa się z prawdziwą łatą i mieści się w ROI."""
    d = generate_synthetic_frames(fs=FS_TEST, hr_bpm=TRUE_HR_BPM, seed=seed)
    mask = compute_perfusion_mask(d["thermal_frames"][0], d["roi_mask"])
    truth = d["patch_mask"]

    assert (mask & ~d["roi_mask"]).sum() == 0  # maska nie wychodzi poza ROI
    jaccard = (mask & truth).sum() / (mask | truth).sum()
    assert jaccard >= 0.9


def test_perfusion_mask_shape_mismatch_raises():
    """Niezgodny kształt klatki i maski ROI -> ValueError."""
    with pytest.raises(ValueError):
        compute_perfusion_mask(np.zeros((10, 10)), np.ones((8, 8), dtype=bool))


@pytest.mark.parametrize("seed", SEEDS)
def test_extract_rgb_trace_shape_and_hr(seed):
    """Ekstrakcja z ROI zwraca (N, 3) i pozwala odzyskać HR."""
    d = generate_synthetic_frames(fs=FS_TEST, hr_bpm=TRUE_HR_BPM, seed=seed)
    trace = extract_rgb_trace(d["rgb_frames"], d["roi_positions"], d["valid"])

    assert trace.shape == (d["rgb_frames"].shape[0], 3)
    hr = dominant_hr_bpm(green(trace, FS_TEST), FS_TEST)
    assert abs(hr - TRUE_HR_BPM) <= TOLERANCE_BPM


@pytest.mark.parametrize("seed", SEEDS)
def test_thermal_gating_improves_snr(seed):
    """Bramkowanie termiką (łata perfuzji) daje wyższe SNR niż uśrednianie po całym ROI."""
    d = generate_synthetic_frames(fs=FS_TEST, hr_bpm=TRUE_HR_BPM, seed=seed)

    plain = extract_rgb_trace(d["rgb_frames"], d["roi_positions"], d["valid"])
    gated = extract_rgb_trace_thermal_gated(
        d["rgb_frames"], d["thermal_frames"], d["roi_positions"], d["valid"]
    )

    snr_plain = snr_rppg(green(plain, FS_TEST), FS_TEST, TRUE_HR_BPM)
    snr_gated = snr_rppg(green(gated, FS_TEST), FS_TEST, TRUE_HR_BPM)

    assert gated.shape == plain.shape
    assert snr_gated > snr_plain


def test_extract_rejects_length_mismatch():
    """Niespójna długość roi_positions/valid względem liczby klatek -> ValueError."""
    d = generate_synthetic_frames(fs=FS_TEST, duration_s=2.0, seed=0)
    with pytest.raises(ValueError):
        extract_rgb_trace(d["rgb_frames"], d["roi_positions"][:-1], d["valid"])


def test_extract_supports_bbox_roi():
    """ROI podane jako bbox [y0, x0, y1, x1] działa tak jak maska prostokątna."""
    d = generate_synthetic_frames(fs=FS_TEST, duration_s=2.0, seed=0)
    n = d["rgb_frames"].shape[0]
    bbox = np.array([4, 4, d["rgb_frames"].shape[1] - 4, d["rgb_frames"].shape[2] - 4])

    trace_bbox = extract_rgb_trace(d["rgb_frames"], [bbox] * n, d["valid"])
    trace_mask = extract_rgb_trace(d["rgb_frames"], d["roi_positions"], d["valid"])

    np.testing.assert_allclose(trace_bbox, trace_mask)
