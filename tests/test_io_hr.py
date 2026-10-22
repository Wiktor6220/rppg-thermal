"""Testy wczytywania Polar HR.csv na syntetycznym pliku (bez danych z data/)."""

from datetime import UTC, datetime

import numpy as np
import pytest

import src.io_layer as io_layer
from src.io_layer import load_polar_hr_csv

# Nagłówek: 2026-10-10 10:26:30 czasu lokalnego (Europe/Warsaw, CEST = UTC+2).
HEADER = "TIMESTAMP,10,10,2026,10,26,30,0"
HR_START_UTC = datetime(2026, 10, 10, 8, 26, 30, tzinfo=UTC)


def _write_hr_csv(path, header: str, clock: list[str], hrs: list[float]) -> None:
    rows = [header] + [f"{c},0,0,{h}" for c, h in zip(clock, hrs, strict=True)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.fixture
def hr_csv(tmp_path):
    """10 próbek co 1 s od 10:26:30, HR 60..69."""
    path = tmp_path / "subject99_s1_HR.csv"
    clock = [f"10:26:{30 + i:02d}.000000" for i in range(10)]
    _write_hr_csv(path, HEADER, clock, [60.0 + i for i in range(10)])
    return path


def _fake_creation_time(monkeypatch, value: datetime | None) -> None:
    monkeypatch.setattr(io_layer, "probe_mp4_creation_time_utc", lambda path: value)


def test_parses_hr_column_and_relative_axis_without_video(hr_csv):
    """Bez video_path: oś od pierwszej próbki, HR z kolumny POLAR_HR_COLUMN."""
    series = load_polar_hr_csv(hr_csv, skip_samples=0)
    assert series is not None
    np.testing.assert_allclose(series.hr_bpm, np.arange(60.0, 70.0))
    np.testing.assert_allclose(series.t_s, np.arange(10.0))
    assert series.sync_mode == "relative_zero"
    assert series.sync_offset_s == 0.0


def test_skip_samples_drops_leading_rows(hr_csv):
    series = load_polar_hr_csv(hr_csv, skip_samples=3)
    assert series is not None
    assert series.hr_bpm[0] == 63.0
    assert series.t_s[0] == 0.0
    assert len(series.hr_bpm) == 7


def test_absolute_axis_relative_to_video_start(hr_csv, tmp_path, monkeypatch):
    """Wideo startuje 2 s po pierwszej próbce HR -> t_s = -2, -1, 0, ..."""
    _fake_creation_time(monkeypatch, datetime(2026, 10, 10, 8, 26, 32, tzinfo=UTC))
    series = load_polar_hr_csv(
        hr_csv, skip_samples=0, video_path=tmp_path / "v.mp4", video_duration_s=6.0
    )
    assert series is not None
    assert series.sync_mode == "absolute_utc"
    assert series.sync_offset_s == pytest.approx(2.0)
    np.testing.assert_allclose(series.t_s, np.arange(10.0) - 2.0)
    assert "SANITY_FAIL" not in series.sync_note
    assert HR_START_UTC.isoformat() in series.sync_note


def test_absolute_axis_flags_missing_overlap(hr_csv, tmp_path, monkeypatch):
    """Wideo godzinę po HR: seria zwrócona, ale sync_note z SANITY_FAIL."""
    _fake_creation_time(monkeypatch, datetime(2026, 10, 10, 9, 26, 30, tzinfo=UTC))
    series = load_polar_hr_csv(
        hr_csv, skip_samples=0, video_path=tmp_path / "v.mp4", video_duration_s=60.0
    )
    assert series is not None
    assert series.sync_mode == "absolute_utc"
    assert "SANITY_FAIL" in series.sync_note


def test_fallback_relative_zero_without_creation_time(hr_csv, tmp_path, monkeypatch):
    _fake_creation_time(monkeypatch, None)
    series = load_polar_hr_csv(hr_csv, skip_samples=0, video_path=tmp_path / "v.mp4")
    assert series is not None
    assert series.sync_mode == "relative_zero"
    assert np.isnan(series.sync_offset_s)
    assert series.t_s[0] == 0.0
    assert "brak creation_time" in series.sync_note


def test_fallback_relative_zero_without_header_date(tmp_path, monkeypatch):
    path = tmp_path / "hr.csv"
    _write_hr_csv(path, "Phone timestamp,a,b,HR", ["12:00:00.000000", "12:00:01.000000"], [70, 71])
    _fake_creation_time(monkeypatch, datetime(2026, 10, 10, 10, 0, 0, tzinfo=UTC))
    series = load_polar_hr_csv(path, skip_samples=0, video_path=tmp_path / "v.mp4")
    assert series is not None
    assert series.sync_mode == "relative_zero"
    assert "brak daty w nagłówku HR" in series.sync_note


def test_absolute_axis_handles_midnight_rollover(tmp_path, monkeypatch):
    """Próbki przechodzą przez północ: oś czasu rośnie dalej (dzień + 1)."""
    path = tmp_path / "hr.csv"
    clock = ["23:59:59.000000", "00:00:00.000000", "00:00:01.000000"]
    _write_hr_csv(path, "TIMESTAMP,10,10,2026,23,59,59,0", clock, [60, 61, 62])
    _fake_creation_time(monkeypatch, datetime(2026, 10, 10, 21, 59, 59, tzinfo=UTC))
    series = load_polar_hr_csv(path, skip_samples=0, video_path=tmp_path / "v.mp4")
    assert series is not None
    np.testing.assert_allclose(series.t_s, [0.0, 1.0, 2.0])


def test_header_only_returns_none(tmp_path):
    path = tmp_path / "hr.csv"
    path.write_text(HEADER + "\n", encoding="utf-8")
    assert load_polar_hr_csv(path) is None
