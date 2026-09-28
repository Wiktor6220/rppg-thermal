"""Testy referencji Polar HR.csv (jedyna ścieżka walidacji) — subject02."""

from pathlib import Path

import numpy as np
import pytest

from src.config import EVAL_SUBJECT, POLAR_HR_SKIP_SAMPLES
from src.io_layer import (
    list_eval_recordings,
    load_polar_hr,
    load_reference_hr,
)

DATA = Path(__file__).resolve().parent.parent / "data"
HAS_S02 = (DATA / EVAL_SUBJECT / "s1_rest_rest").is_dir()


@pytest.mark.skipif(not HAS_S02, reason=f"brak data/{EVAL_SUBJECT}/s1_rest_rest")
def test_list_eval_recordings_only_subject02():
    recs = list_eval_recordings()
    assert recs
    assert all(r.subject == EVAL_SUBJECT for r in recs)
    assert not any(r.subject == "subject01" for r in recs)


@pytest.mark.skipif(not HAS_S02, reason=f"brak data/{EVAL_SUBJECT}/s1_rest_rest")
def test_load_reference_hr_is_hr_csv_only():
    series, src = load_reference_hr(EVAL_SUBJECT, "s1_rest_rest")
    assert series is not None
    assert src == "hr_csv"
    assert series.hr_bpm.size >= 10
    # Bez skipu: pierwsza próbka ≈ start wideo (t_s ≈ 0)
    assert POLAR_HR_SKIP_SAMPLES == 0
    assert float(series.t_s.min()) < 1.0


@pytest.mark.skipif(not HAS_S02, reason=f"brak data/{EVAL_SUBJECT}/s1_rest_rest")
def test_load_polar_hr_s02_s1_resting_range():
    series = load_polar_hr(EVAL_SUBJECT, "s1_rest_rest")
    assert series is not None
    med = float(np.median(series.hr_bpm))
    assert 50.0 <= med <= 100.0


def test_load_polar_hr_no_skip_starts_at_zero(tmp_path, monkeypatch):
    """t0 = pierwszy wiersz; przy skip=0 pierwsza próbka ma t_s == 0."""
    session = tmp_path / "subject99" / "s1_rest_rest"
    session.mkdir(parents=True)
    hr_path = session / "subject99_s1_HR.csv"
    lines = ["Phone timestamp,sensor timestamp [ns],HR [bpm],extra"]
    for i in range(10):
        lines.append(f"12:00:{i:02d}.000000,0,0,{60 + i},x")
    hr_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "src.io_layer.find_polar_hr_path",
        lambda subject, scenario, data_dir=None: hr_path,
    )
    series = load_polar_hr("subject99", "s1_rest_rest", data_dir=tmp_path, skip_samples=0)
    assert series is not None
    assert series.t_s[0] == 0.0
    assert len(series.hr_bpm) == 10
