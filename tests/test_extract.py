"""Testy ekstrakcji RGB z ROI i bramkowania termiką (dane syntetyczne)."""

import numpy as np
import pytest

import src.extract as extract
from src.estimate import snr_rppg
from src.extract import (
    gated_means_per_frame_from_samples,
    mean_rgb_in_mask,
    sample_roi_temps_via_affine,
)
from src.methods import green
from tests.synthetic import generate_synthetic_frames

FS_TEST = 30.0
TRUE_HR_BPM = 72.0
SEEDS = [0, 1, 2, 3]


@pytest.mark.parametrize("seed", SEEDS)
def test_thermal_gating_improves_snr(seed):
    """Bramkowanie (ścieżka potoku: próbki -> per-frame) daje wyższe SNR niż plain ROI."""
    d = generate_synthetic_frames(fs=FS_TEST, hr_bpm=TRUE_HR_BPM, seed=seed)
    rgb = d["rgb_frames"]
    thermal = d["thermal_frames"]
    roi = d["roi_mask"]
    ys, xs = np.where(roi)

    plain = np.stack([mean_rgb_in_mask(frame, roi) for frame in rgb])
    rgb_pix: list[np.ndarray | None] = [frame[ys, xs] for frame in rgb]
    temps: list[np.ndarray | None] = [th[ys, xs].astype(np.float64) for th in thermal]

    gated, _ = gated_means_per_frame_from_samples(rgb_pix, temps, plain)
    snr_plain = snr_rppg(green(plain, FS_TEST), FS_TEST, TRUE_HR_BPM)
    snr_gated = snr_rppg(green(gated, FS_TEST), FS_TEST, TRUE_HR_BPM)

    assert gated.shape == plain.shape
    assert snr_gated > snr_plain


def test_mean_rgb_in_empty_mask_falls_back_to_frame_mean():
    frame = np.arange(2 * 2 * 3, dtype=np.float64).reshape(2, 2, 3)
    out = mean_rgb_in_mask(frame, np.zeros((2, 2), dtype=bool))
    np.testing.assert_allclose(out, frame.reshape(-1, 3).mean(axis=0))


def _thermal_and_roi():
    rng = np.random.default_rng(0)
    thermal = rng.uniform(0, 255, size=(60, 80)).astype(np.float32)
    roi = np.zeros((120, 160), dtype=bool)
    roi[20:100, 30:130] = True
    affine = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])  # termika (x, y) -> RGB (2x, 2y)
    return thermal, affine, roi


def test_sample_roi_temps_chunked_equals_single_remap(monkeypatch):
    """Porcjowanie remap nie zmienia wyniku względem jednego wywołania."""
    thermal, affine, roi = _thermal_and_roi()
    ys1, xs1, t1 = sample_roi_temps_via_affine(thermal, affine, roi)
    monkeypatch.setattr(extract, "_REMAP_CHUNK", 777)
    ys2, xs2, t2 = sample_roi_temps_via_affine(thermal, affine, roi)
    np.testing.assert_array_equal(ys1, ys2)
    np.testing.assert_array_equal(xs1, xs2)
    np.testing.assert_array_equal(t1, t2)


def test_sample_roi_temps_maps_through_affine():
    """Piksel RGB (2x, 2y) dostaje temperaturę termiki z (x, y)."""
    thermal, affine, roi = _thermal_and_roi()
    ys, xs, temps = sample_roi_temps_via_affine(thermal, affine, roi)
    k = int(np.where((ys == 40) & (xs == 60))[0][0])
    assert temps[k] == pytest.approx(float(thermal[20, 30]), abs=1e-3)


def test_sample_roi_temps_empty_roi():
    thermal, affine, _ = _thermal_and_roi()
    ys, xs, temps = sample_roi_temps_via_affine(thermal, affine, np.zeros((120, 160), bool))
    assert ys.size == xs.size == temps.size == 0
