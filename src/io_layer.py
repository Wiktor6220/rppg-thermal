"""Wczytywanie nagrań RGB + termika (podgląd mp4), metadane, generatory klatek, Polar HR."""

import csv
import math
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from src.config import POLAR_HR_COLUMN, POLAR_HR_SKIP_SAMPLES


@dataclass(frozen=True)
class VideoMeta:
    """Metadane strumienia wideo."""

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        """Długość [s]."""
        return self.frame_count / self.fps if self.fps > 0 else math.nan

    @property
    def resolution(self) -> tuple[int, int]:
        """(width, height)."""
        return (self.width, self.height)


@dataclass(frozen=True)
class Recording:
    """Para ścieżek RGB + termika jednej sesji."""

    subject: str
    scenario: str  # np. s1_rest_rest
    rgb_path: Path
    thermal_path: Path


@dataclass(frozen=True)
class LoadedRecording:
    """Nagranie z metadanymi obu strumieni."""

    recording: Recording
    rgb_meta: VideoMeta
    thermal_meta: VideoMeta

    def synced_pairs(
        self,
        reference: str = "rgb",
        match_resolution: str | None = None,
        to_rgb: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, float]]:
        """Generator par (rgb_frame, thermal_frame, t_seconds) zsynchronizowanych w CZASIE.

        Cienki wrapper na `iter_time_synced_pairs` — patrz tam po pełny opis parametrów
        i semantykę odniesienia czasu.
        """
        return iter_time_synced_pairs(
            self, reference=reference, match_resolution=match_resolution, to_rgb=to_rgb
        )


def probe_video(path: Path) -> VideoMeta:
    """Odczytuje metadane wideo (fps, liczba klatek, rozdzielczość) przez OpenCV."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Nie można otworzyć wideo: {path}")
    try:
        meta = VideoMeta(
            path=path,
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()
    return meta


def iter_video_frames(path: Path, to_rgb: bool = True) -> Iterator[np.ndarray]:
    """Generator klatek wideo (H, W, 3), uint8; leniwie, po jednej klatce."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Nie można otworzyć wideo: {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if to_rgb else frame
    finally:
        cap.release()


def resize_to(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """cv2.resize do (width, height); to nie jest korejestracja geometryczna."""
    target_w, target_h = size
    src_h, src_w = frame.shape[:2]
    if (src_w, src_h) == (target_w, target_h):
        return frame
    # Zmniejszanie -> INTER_AREA (mniej aliasingu); powiększanie -> INTER_LINEAR.
    interpolation = cv2.INTER_AREA if target_w * target_h < src_w * src_h else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_w, target_h), interpolation=interpolation)


def iter_time_synced_pairs(
    loaded: LoadedRecording,
    reference: str = "rgb",
    match_resolution: str | None = None,
    to_rgb: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray, float]]:
    """Pary (rgb, thermal, t) dopasowane po czasie; reference wyznacza oś t."""
    if reference not in ("rgb", "thermal"):
        raise ValueError(f"reference musi być 'rgb' albo 'thermal', otrzymano {reference!r}")
    rgb_meta, thermal_meta = loaded.rgb_meta, loaded.thermal_meta
    if rgb_meta.fps <= 0 or thermal_meta.fps <= 0:
        raise ValueError("Nieznane fps (≤0) — nie można synchronizować w czasie.")
    target_size = _resolve_target_size(match_resolution, rgb_meta, thermal_meta)

    if reference == "rgb":
        ref_meta, other_meta = rgb_meta, thermal_meta
    else:
        ref_meta, other_meta = thermal_meta, rgb_meta
    ref_fps, other_fps = ref_meta.fps, other_meta.fps

    ref_gen = iter_video_frames(ref_meta.path, to_rgb=to_rgb)
    other_gen = iter_video_frames(other_meta.path, to_rgb=to_rgb)
    try:
        try:
            current_other = next(other_gen)
        except StopIteration:
            return  # drugi strumień pusty — nie ma czego parować
        j = 0
        for i, ref_frame in enumerate(ref_gen):
            t_ref = i / ref_fps
            # Przesuwaj drugi strumień do przodu, dopóki NASTĘPNA klatka jest bliżej w czasie.
            while abs((j + 1) / other_fps - t_ref) <= abs(j / other_fps - t_ref):
                try:
                    current_other = next(other_gen)
                except StopIteration:
                    break
                j += 1

            if reference == "rgb":
                rgb_frame, thermal_frame = ref_frame, current_other
            else:
                rgb_frame, thermal_frame = current_other, ref_frame

            if match_resolution == "rgb":
                thermal_frame = resize_to(thermal_frame, target_size)  # termika -> rozmiar RGB
            elif match_resolution == "thermal":
                rgb_frame = resize_to(rgb_frame, target_size)  # RGB -> rozmiar termiki

            yield rgb_frame, thermal_frame, t_ref
    finally:
        # Zwolnij oba VideoCapture od razu (nie czekaj na GC). iter_video_frames zwraca
        # generator, więc ma .close() — sięgamy przez getattr, bo typ zwrotny to Iterator.
        for gen in (ref_gen, other_gen):
            close = getattr(gen, "close", None)
            if callable(close):
                close()


def _resolve_target_size(
    match_resolution: str | None, rgb_meta: VideoMeta, thermal_meta: VideoMeta
) -> tuple[int, int]:
    """Zwraca docelowy rozmiar (width, height) dla `match_resolution` (lub (0,0), gdy None)."""
    if match_resolution is None:
        return (0, 0)  # nieużywane
    if match_resolution == "rgb":
        return rgb_meta.resolution
    if match_resolution == "thermal":
        return thermal_meta.resolution
    raise ValueError(
        f"match_resolution musi być None, 'rgb' albo 'thermal', otrzymano {match_resolution!r}"
    )


# --- Referencja Polar H10 ---


@dataclass(frozen=True)
class PolarHrSeries:
    """Seria Polar HR.csv (t_s względem startu wideo)."""

    t_s: np.ndarray  # [s] od startu wideo (może być ujemne przed startem)
    hr_bpm: np.ndarray
    sync_mode: str = "relative_zero"  # "absolute_utc" | "relative_zero"
    sync_offset_s: float = float("nan")  # start_wideo - start_HR [s]
    sync_note: str = ""


def probe_mp4_creation_time_utc(path: Path) -> datetime | None:
    """Odczytuje ``creation_time`` z tagów MP4 przez ffprobe (UTC). NIE używa mtime."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format_tags=creation_time",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    raw = (proc.stdout or "").strip()
    if not raw:
        return None
    # np. 2026-10-10T08:26:32.000000Z
    try:
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except ValueError:
        return None


def _parse_polar_hr_header_local(header: list[str]) -> datetime | None:
    """Nagłówek ``TIMESTAMP,DD,MM,YYYY,HH,MM,SS,us`` -> datetime lokalny (naive) lub None."""
    if len(header) < 8 or header[0].strip().upper() != "TIMESTAMP":
        return None
    try:
        day, month, year = int(header[1]), int(header[2]), int(header[3])
        hour, minute, sec = int(header[4]), int(header[5]), int(header[6])
        us = int(header[7])
        if us < 1_000_000:
            micro = us
        elif us < 1_000_000_000:
            micro = us // 1000  # nanosekundy -> µs (przybliżenie)
        else:
            micro = us % 1_000_000
        return datetime(year, month, day, hour, minute, sec, micro)
    except (ValueError, IndexError):
        return None


def hr_video_time_overlap_ok(
    series: PolarHrSeries,
    video_duration_s: float,
    min_overlap_s: float = 5.0,
) -> tuple[bool, str]:
    """Czy okno wideo [0, duration] ma sensowne pokrycie z osią HR ``t_s``."""
    if series.t_s.size == 0 or not np.isfinite(video_duration_s) or video_duration_s <= 0:
        return False, "pusta seria HR albo nieznany czas trwania wideo"
    hr_lo, hr_hi = float(series.t_s.min()), float(series.t_s.max())
    overlap_lo = max(0.0, hr_lo)
    overlap_hi = min(float(video_duration_s), hr_hi)
    overlap = overlap_hi - overlap_lo
    if overlap < min_overlap_s:
        return (
            False,
            f"brak pokrycia wideo∩HR: HR t∈[{hr_lo:.1f},{hr_hi:.1f}] s, "
            f"wideo [0,{video_duration_s:.1f}] s (overlap={overlap:.1f} s)",
        )
    if abs(float(series.sync_offset_s)) > 3600.0 and series.sync_mode == "absolute_utc":
        return (
            False,
            f"podejrzany offset sync={series.sync_offset_s:.1f} s (>1 h) — błąd strefy?",
        )
    return True, f"overlap={overlap:.1f} s"


def _relative_zero_series(
    times_clock: list[datetime],
    hrs_arr: np.ndarray,
    sync_offset_s: float,
    note: str,
) -> PolarHrSeries:
    """Seria z osią czasu od pierwszej próbki HR (t_s[0] = 0)."""
    t0 = times_clock[0]
    t_s = np.array([(t - t0).total_seconds() for t in times_clock], dtype=np.float64)
    return PolarHrSeries(
        t_s=t_s,
        hr_bpm=hrs_arr,
        sync_mode="relative_zero",
        sync_offset_s=sync_offset_s,
        sync_note=note,
    )


def _local_sample_times(
    header_local: datetime, times_clock: list[datetime], tz: tzinfo
) -> list[datetime]:
    """Absolutny czas lokalny każdej próbki: data z nagłówka + godzina z wiersza.

    Rollover po północy: gdy godzina próbki jest mniejsza niż poprzedniej, dzień + 1.
    """
    base_date = header_local.date()
    abs_local: list[datetime] = []
    day_offset = 0
    prev_tod = None
    for tc in times_clock:
        tod = tc.time()
        if prev_tod is not None and tod < prev_tod:
            day_offset += 1
        prev_tod = tod
        d = base_date + timedelta(days=day_offset)
        naive = datetime.combine(d, tod)
        abs_local.append(naive.replace(tzinfo=tz))
    return abs_local


def _absolute_utc_series(
    path: Path,
    header_local: datetime | None,
    times_clock: list[datetime],
    hrs_arr: np.ndarray,
    video_path: Path,
    video_duration_s: float | None,
) -> PolarHrSeries:
    """t_s = t_abs_HR_UTC - creation_time_wideo_UTC; przy braku danych relative_zero."""
    video_utc = probe_mp4_creation_time_utc(video_path)
    if video_utc is None or header_local is None:
        reason = "brak creation_time w MP4" if video_utc is None else "brak daty w nagłówku HR"
        note = f"sync fallback relative_zero: {reason}"
        print(f"[warn] {note} ({path.name})")
        return _relative_zero_series(times_clock, hrs_arr, float("nan"), note)

    try:
        warsaw = ZoneInfo("Europe/Warsaw")
    except Exception as exc:  # noqa: BLE001
        note = f"sync fallback relative_zero: zoneinfo ({exc})"
        print(f"[warn] {note}")
        return _relative_zero_series(times_clock, hrs_arr, float("nan"), note)

    abs_local = _local_sample_times(header_local, times_clock, warsaw)
    hr_start_utc = header_local.replace(tzinfo=warsaw).astimezone(UTC)
    sync_offset = (video_utc - hr_start_utc).total_seconds()
    t_s = np.array(
        [(t.astimezone(UTC) - video_utc).total_seconds() for t in abs_local],
        dtype=np.float64,
    )
    note = (
        f"absolute_utc: video={video_utc.isoformat()} "
        f"HR_header={hr_start_utc.isoformat()} offset={sync_offset:.3f}s"
    )
    series = PolarHrSeries(
        t_s=t_s,
        hr_bpm=hrs_arr,
        sync_mode="absolute_utc",
        sync_offset_s=float(sync_offset),
        sync_note=note,
    )
    if video_duration_s is not None:
        ok, msg = hr_video_time_overlap_ok(series, float(video_duration_s))
        if not ok:
            # Zwracamy serię, ale sync_note zawiera ostrzeżenie; caller może pominąć.
            return PolarHrSeries(
                t_s=series.t_s,
                hr_bpm=series.hr_bpm,
                sync_mode=series.sync_mode,
                sync_offset_s=series.sync_offset_s,
                sync_note=f"{note}; SANITY_FAIL: {msg}",
            )
    return series


def load_polar_hr_csv(
    path: Path,
    skip_samples: int = POLAR_HR_SKIP_SAMPLES,
    hr_column: int = POLAR_HR_COLUMN,
    video_path: Path | None = None,
    video_duration_s: float | None = None,
) -> PolarHrSeries | None:
    """Wczytuje Polar HR.csv.

    Bez ``video_path``: t_s względem pierwszej próbki HR.
    Z ``video_path``: t_s = t_abs_HR_UTC - creation_time_wideo_UTC (zachowany offset ułamkowy).
    """
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return None

    header_local = _parse_polar_hr_header_local(rows[0])

    times_clock: list[datetime] = []
    hrs: list[float] = []
    for row in rows[1:]:
        if len(row) <= hr_column:
            continue
        try:
            times_clock.append(datetime.strptime(row[0].strip(), "%H:%M:%S.%f"))
            hrs.append(float(row[hr_column]))
        except (ValueError, IndexError):
            continue
    if len(hrs) <= skip_samples:
        return None

    hrs_arr = np.asarray(hrs[skip_samples:], dtype=np.float64)
    times_clock = times_clock[skip_samples:]

    if video_path is not None:
        return _absolute_utc_series(
            path, header_local, times_clock, hrs_arr, video_path, video_duration_s
        )
    # Bez video_path: oś czasu od pierwszej próbki.
    return _relative_zero_series(times_clock, hrs_arr, 0.0, "relative_zero (brak video_path)")
