"""Testy logiki śledzenia ROI i wektora valid[] — z atrapą detektora, bez MediaPipe.

Sprawdzamy wyłącznie logikę przytrzymania/uzupełniania pozycji ROI przy sekwencji
z dziurami (klatki bez detekcji). Prawdziwa detekcja (MediaPipe) nie jest tu
wywoływana ani implementowana.
"""

import numpy as np

from src.roi import track_roi_across_frames


class DummyDetector:
    """Atrapa detektora: dla kolejnych klatek zwraca landmarki albo None wg podanej sekwencji."""

    def __init__(self, sequence):
        # sequence: lista wartości None (brak detekcji) lub identyfikatora detekcji (int).
        self.sequence = list(sequence)
        self.calls = 0

    def __call__(self, frame):
        value = self.sequence[self.calls]
        self.calls += 1
        if value is None:
            return None
        # Landmarki jako (1, 2) niosące identyfikator — pozwala rozpoznać, z której detekcji.
        return np.array([[float(value), float(value)]])


def _roi_builder(landmarks, region):
    """Atrapa buildera ROI: bbox wokół landmarka, nadal niosący jego identyfikator."""
    ident = landmarks[0, 0]
    return np.array([ident, ident, ident + 1, ident + 1])


def _track(sequence):
    frames = np.zeros((len(sequence), 2, 2, 3), dtype=np.float64)
    return track_roi_across_frames(
        frames, detector=DummyDetector(sequence), roi_builder=_roi_builder, region="forehead"
    )


def test_valid_marks_detected_frames():
    """valid[] jest True dokładnie tam, gdzie była detekcja."""
    roi_positions, valid = _track([1, 2, None, 4])
    np.testing.assert_array_equal(valid, [True, True, False, True])
    assert len(roi_positions) == 4


def test_hole_holds_last_known_position():
    """Dziura w środku: ROI przytrzymane z poprzedniej detekcji (hold-last)."""
    roi_positions, valid = _track([1, 2, None, 4])
    # Klatka 2 (dziura) trzyma pozycję z klatki 1 (detekcja id=2).
    np.testing.assert_array_equal(roi_positions[2], roi_positions[1])
    assert not valid[2]


def test_trailing_holes_hold_last():
    """Dziury na końcu: przytrzymana ostatnia znana pozycja."""
    roi_positions, valid = _track([5, None, None])
    np.testing.assert_array_equal(valid, [True, False, False])
    np.testing.assert_array_equal(roi_positions[1], roi_positions[0])
    np.testing.assert_array_equal(roi_positions[2], roi_positions[0])


def test_leading_holes_backfilled_with_first_detection():
    """Dziury na początku: uzupełnione wstecznie pierwszą wykrytą pozycją."""
    roi_positions, valid = _track([None, None, 3])
    np.testing.assert_array_equal(valid, [False, False, True])
    np.testing.assert_array_equal(roi_positions[0], roi_positions[2])
    np.testing.assert_array_equal(roi_positions[1], roi_positions[2])


def test_no_none_positions_when_any_detection():
    """Gdy jest choć jedna detekcja, żadna pozycja ROI nie jest None."""
    roi_positions, _ = _track([None, 2, None, None, 5, None])
    assert all(pos is not None for pos in roi_positions)


def test_all_missing_leaves_none_and_valid_false():
    """Brak jakiejkolwiek detekcji: valid[] w całości False, pozycje pozostają None."""
    roi_positions, valid = _track([None, None, None])
    assert not valid.any()
    assert all(pos is None for pos in roi_positions)


def test_detector_called_once_per_frame():
    """Detektor wołany dokładnie raz na klatkę (śledzenie nie gubi ani nie dubluje klatek)."""
    detector = DummyDetector([1, None, 3])
    frames = np.zeros((3, 2, 2, 3), dtype=np.float64)
    track_roi_across_frames(frames, detector=detector, roi_builder=_roi_builder)
    assert detector.calls == 3
