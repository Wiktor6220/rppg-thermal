"""Testy czystych funkcji potoku: bramkowanie affine, jakość sceny, ref_suspect."""

import numpy as np

import src.pipeline as pipeline
from src.pipeline import _accept_affine, _scene_registration_quality, is_ref_suspect

THR = 18.0


def test_accept_better_affine_below_threshold():
    """Kandydat poniżej progu jest przyjmowany, także gdy bieżący już jest trustworthy."""
    assert _accept_affine(10.0, current_rms=30.0, current_trusted=False, threshold=THR)
    assert _accept_affine(12.0, current_rms=8.0, current_trusted=True, threshold=THR)


def test_reject_worse_affine_above_threshold():
    """Kandydat powyżej progu nie zastępuje affine, który przeszedł próg."""
    assert not _accept_affine(25.0, current_rms=10.0, current_trusted=True, threshold=THR)
    assert not _accept_affine(float("nan"), current_rms=10.0, current_trusted=True, threshold=THR)


def test_best_so_far_before_first_trustworthy():
    """Dopóki nic nie przeszło progu: pierwszy zawsze, potem tylko lepszy od bieżącego."""
    assert _accept_affine(40.0, current_rms=None, current_trusted=False, threshold=THR)
    assert _accept_affine(30.0, current_rms=40.0, current_trusted=False, threshold=THR)
    assert not _accept_affine(35.0, current_rms=30.0, current_trusted=False, threshold=THR)


def test_scene_quality_uses_median_and_good_fraction():
    first = {"contour_rms_px": 40.0, "iou": 0.6}
    q = _scene_registration_quality(
        [40.0, 10.0, 12.0, float("nan")], [0.6, 0.9, 0.8, float("nan")], first, threshold=THR
    )
    assert q.contour_rms_px == 12.0
    assert np.isclose(q.iou, 0.8)
    assert q.rms_first_px == 40.0
    assert q.n_attempts == 4
    assert np.isclose(q.good_frac, 0.5)
    assert q.trustworthy


def test_ref_suspect_comes_from_config(monkeypatch):
    """Flaga ref_suspect wynika z REF_SUSPECT_SUBJECTS, nie ze stałej w kodzie."""
    monkeypatch.setattr(pipeline, "REF_SUSPECT_SUBJECTS", ("subject07",))
    assert is_ref_suspect("subject07")
    assert not is_ref_suspect("subject03")
