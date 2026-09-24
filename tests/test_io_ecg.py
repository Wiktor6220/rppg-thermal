"""Testy wczytywania EKG Polara i referencji HR."""

from pathlib import Path

import numpy as np
import pytest

from src.io_layer import (
    find_polar_ecg_path,
    load_polar_ecg_hr,
    load_reference_hr,
    parse_polar_ecg_samples,
)

DATA = Path(__file__).resolve().parent.parent / "data"
HAS_S01 = (DATA / "subject01" / "s1_rest_rest").is_dir()


@pytest.mark.skipif(not HAS_S01, reason="brak data/subject01/s1_rest_rest")
def test_parse_polar_ecg_has_samples():
    path = find_polar_ecg_path("subject01", "s1_rest_rest")
    assert path is not None
    samples = parse_polar_ecg_samples(path)
    assert samples.size > 1000


@pytest.mark.skipif(not HAS_S01, reason="brak data/subject01/s1_rest_rest")
def test_load_polar_ecg_hr_median_near_resting():
    series = load_polar_ecg_hr("subject01", "s1_rest_rest")
    assert series is not None
    assert series.hr_bpm.size >= 10
    med = float(np.median(series.hr_bpm))
    # Spoczynek s01 ≈ 70 BPM (Polar HR ~70.7); tolerancja szeroka na detekcję R.
    assert 50.0 <= med <= 95.0
    assert series.t_s.min() >= 9.0  # po skip 10 s


@pytest.mark.skipif(not HAS_S01, reason="brak data/subject01/s1_rest_rest")
def test_load_reference_prefers_ecg():
    series, src = load_reference_hr("subject01", "s1_rest_rest")
    assert series is not None
    assert src == "ecg"


@pytest.mark.skipif(not HAS_S01, reason="brak data/subject01/s2_person_move")
def test_parse_dr_marker_ecg():
    """Niektóre sesje mają marker 68,82,8,0 zamiast 67,82,8,0."""
    path = find_polar_ecg_path("subject01", "s2_person_move")
    assert path is not None
    samples = parse_polar_ecg_samples(path)
    assert samples.size > 1000


@pytest.mark.skipif(not HAS_S01, reason="brak data/subject01/s2_person_move")
def test_reference_rejects_bad_ecg_vs_hr_csv():
    """Gdy mediana EKG mocno odbiega od Polar HR → fallback do hr_csv."""
    series, src = load_reference_hr("subject01", "s2_person_move")
    assert series is not None
    # s2: detekcja R zawyża (~95) vs Polar (~66) → hr_csv
    assert src == "hr_csv"
    med = float(np.median(series.hr_bpm))
    assert 50.0 <= med <= 85.0
    # t_s względem startu wideo (po skip 5 próbek ~1 Hz → ≈5 s)
    assert float(series.t_s.min()) >= 4.0


def test_load_polar_hr_time_axis_starts_after_skip(tmp_path, monkeypatch):
    """t0 = pierwszy wiersz danych; po skip=5 pierwsza próbka ma t_s == 5.0."""
    from src.io_layer import load_polar_hr

    session = tmp_path / "subject99" / "s1_rest_rest"
    session.mkdir(parents=True)
    hr_path = session / "subject99_s1_HR.csv"
    lines = ["Phone timestamp,sensor timestamp [ns],HR [bpm],extra"]
    for i in range(10):
        # kolumna 3 (0-based) = BPM — w pliku Polar: idx 2 często; config POLAR_HR_COLUMN=3
        # Format: time, a, b, HR, ...
        lines.append(f"12:00:{i:02d}.000000,0,0,{60 + i},x")
    hr_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        "src.io_layer.find_polar_hr_path",
        lambda subject, scenario, data_dir=None: hr_path,
    )
    series = load_polar_hr("subject99", "s1_rest_rest", data_dir=tmp_path, skip_samples=5)
    assert series is not None
    assert series.t_s[0] == 5.0
    assert len(series.hr_bpm) == 5
