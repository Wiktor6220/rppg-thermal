"""Wczytywanie nagrań RGB + termika (podgląd mp4), metadane i generatory klatek."""

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import (
    DATA_DIR,
    ECG_FRAME_MARKERS,
    ECG_FS_HZ,
    ECG_HR_MAX_BPM,
    ECG_HR_MIN_BPM,
    ECG_SKIP_SEC,
    POLAR_HR_COLUMN,
    POLAR_HR_SKIP_SAMPLES,
)

# Konwencja nazw strumieni w folderze sesji (porównania case-insensitive).
_RGB_SUFFIX = "rgb"
_THERMAL_SUFFIX = "thermal"
_VIDEO_EXT = ".mp4"
_SESSION_RE = re.compile(r"^s\d+", re.IGNORECASE)  # token sesji, np. "s1" z "s1_rest_rest"


@dataclass(frozen=True)
class VideoMeta:
    """Metadane pojedynczego strumienia wideo (bez wczytywania klatek do pamięci)."""

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        """Długość nagrania w sekundach (NaN, gdy fps nieznane)."""
        return self.frame_count / self.fps if self.fps > 0 else math.nan

    @property
    def resolution(self) -> tuple[int, int]:
        """Rozdzielczość (szerokość, wysokość) w pikselach."""
        return (self.width, self.height)


@dataclass(frozen=True)
class Recording:
    """Jedno nagranie sesji: para ścieżek RGB + termika (bez wczytanych klatek)."""

    subject: str  # np. "subject01"
    scenario: str  # pełna nazwa folderu sesji, np. "s1_rest_rest"
    rgb_path: Path
    thermal_path: Path

    @property
    def scenario_code(self) -> str:
        """Krótki kod sesji (token sN), np. "s1" z "s1_rest_rest"."""
        match = _SESSION_RE.match(self.scenario)
        return match.group(0).lower() if match else self.scenario


@dataclass(frozen=True)
class LoadedRecording:
    """Nagranie z odczytanymi metadanymi obu strumieni i leniwym dostępem do klatek."""

    recording: Recording
    rgb_meta: VideoMeta
    thermal_meta: VideoMeta

    def rgb_frames(self, to_rgb: bool = True) -> Iterator[np.ndarray]:
        """Generator klatek strumienia RGB (H, W, 3), uint8."""
        return iter_video_frames(self.recording.rgb_path, to_rgb=to_rgb)

    def thermal_frames(self, to_rgb: bool = True) -> Iterator[np.ndarray]:
        """Generator klatek strumienia termicznego (podgląd wizualny) (H, W, 3), uint8."""
        return iter_video_frames(self.recording.thermal_path, to_rgb=to_rgb)

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


def _find_stream(session_dir: Path, suffix: str) -> Path | None:
    """Znajduje w folderze sesji plik wideo o zadanym sufiksie (np. `_rgb`, `_thermal`).

    Dopasowanie po sufiksie i rozszerzeniu jest case-insensitive, dzięki czemu pliki
    Polara (.csv) i `.DS_Store` są automatycznie pomijane.
    """
    for path in sorted(session_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() == _VIDEO_EXT and path.stem.lower().endswith(f"_{suffix}"):
            return path
    return None


def list_recordings(data_dir: Path = DATA_DIR) -> list[Recording]:
    """Nagrania z oboma strumieniami wideo (RGB + termika) w data/."""
    recordings: list[Recording] = []
    if not data_dir.is_dir():
        return recordings

    for subject_dir in sorted(data_dir.iterdir()):
        if not subject_dir.is_dir() or not subject_dir.name.startswith("subject"):
            continue
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir() or not _SESSION_RE.match(session_dir.name):
                continue
            rgb_path = _find_stream(session_dir, _RGB_SUFFIX)
            thermal_path = _find_stream(session_dir, _THERMAL_SUFFIX)
            if rgb_path is not None and thermal_path is not None:
                recordings.append(
                    Recording(subject_dir.name, session_dir.name, rgb_path, thermal_path)
                )
    return recordings


def _normalize_subject(subject: str) -> str:
    """Sprowadza identyfikator osoby do formy `subjectXX` (akceptuje też np. "1", "01")."""
    value = subject.strip()
    if value.lower().startswith("subject"):
        return value
    if value.isdigit():
        return f"subject{int(value):02d}"
    return value


def find_recording(subject: str, scenario: str, data_dir: Path = DATA_DIR) -> Recording:
    """Znajduje kompletne nagranie dla danej osoby i scenariusza.

    Args:
        subject: identyfikator osoby, np. "subject01", "01" lub "1".
        scenario: nazwa folderu sesji ("s1_rest_rest"), kod ("s1") lub jego prefiks.
        data_dir: katalog główny danych. Domyślnie `config.DATA_DIR`.

    Returns:
        Pasujący `Recording`.

    Raises:
        FileNotFoundError: gdy nie ma kompletnego nagrania dla (subject, scenario).
    """
    subj = _normalize_subject(subject).lower()
    scen = scenario.strip().lower()
    for rec in list_recordings(data_dir):
        if rec.subject.lower() != subj:
            continue
        scenario_lower = rec.scenario.lower()
        if scen in (scenario_lower, rec.scenario_code.lower()) or scenario_lower.startswith(scen):
            return rec
    raise FileNotFoundError(
        f"Nie znaleziono kompletnego nagrania: subject={subject!r}, scenario={scenario!r}"
    )


def probe_video(path: Path) -> VideoMeta:
    """Odczytuje metadane wideo (fps, liczba klatek, rozdzielczość) przez OpenCV.

    Nie dekoduje klatek — czyta tylko właściwości strumienia.

    Raises:
        OSError: gdy pliku nie da się otworzyć.
    """
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
    """Generator klatek wideo (H, W, 3), uint8; leniwie, po jednej klatce.

    Args:
        path: ścieżka do pliku wideo.
        to_rgb: gdy True, konwertuje klatki z BGR (OpenCV) na RGB. Gdy False, zwraca
            surowe klatki BGR.

    Yields:
        Kolejne klatki jako tablice (H, W, 3) uint8.

    Raises:
        OSError: gdy pliku nie da się otworzyć (przy rozpoczęciu iteracji).
    """
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


def load_recording(subject: str, scenario: str, data_dir: Path = DATA_DIR) -> LoadedRecording:
    """Metadane obu strumieni + generatory klatek (bez pełnego wczytania do RAM)."""
    rec = find_recording(subject, scenario, data_dir)
    return LoadedRecording(
        recording=rec,
        rgb_meta=probe_video(rec.rgb_path),
        thermal_meta=probe_video(rec.thermal_path),
    )


def resize_to(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """cv2.resize do (width, height); to nie jest korejestracja geometryczna."""
    target_w, target_h = size
    src_h, src_w = frame.shape[:2]
    if (src_w, src_h) == (target_w, target_h):
        return frame
    # Zmniejszanie → INTER_AREA (mniej aliasingu); powiększanie → INTER_LINEAR.
    interpolation = cv2.INTER_AREA if target_w * target_h < src_w * src_h else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_w, target_h), interpolation=interpolation)


def iter_time_synced_pairs(
    loaded: LoadedRecording,
    reference: str = "rgb",
    match_resolution: str | None = None,
    to_rgb: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray, float]]:
    """Pary (rgb, thermal, t) dopasowane po czasie; reference wyznacza oś t.

    Yields:
        rgb_frame, thermal_frame, t_seconds (czas klatki reference).
    """
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
                thermal_frame = resize_to(thermal_frame, target_size)  # termika → rozmiar RGB
            elif match_resolution == "thermal":
                rgb_frame = resize_to(rgb_frame, target_size)  # RGB → rozmiar termiki

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
    """Seria HR z pliku Polar ``*_HR.csv`` (czas względem pierwszej próbki po skipie)."""

    t_s: np.ndarray  # sekundy od startu serii (≈ start wideo)
    hr_bpm: np.ndarray  # BPM


def find_polar_hr_path(subject: str, scenario: str, data_dir: Path = DATA_DIR) -> Path | None:
    """Zwraca ścieżkę do ``*_HR.csv`` w folderze sesji albo None, gdy brak pliku."""
    rec = find_recording(subject, scenario, data_dir)
    session_dir = rec.rgb_path.parent
    matches = sorted(session_dir.glob("*_HR.csv")) + sorted(session_dir.glob("*HR.csv"))
    # Unikaj duplikatów przy pokrywających się globach.
    unique = list(dict.fromkeys(matches))
    return unique[0] if unique else None


def load_polar_hr(
    subject: str,
    scenario: str,
    data_dir: Path = DATA_DIR,
    skip_samples: int = POLAR_HR_SKIP_SAMPLES,
    hr_column: int = POLAR_HR_COLUMN,
) -> PolarHrSeries | None:
    """Wczytuje HR z Polara H10: kolumna BPM, pomija pierwsze ``skip_samples`` po nagłówku.

    Czas ``t_s`` jest względem **pierwszego wiersza danych** (start wideo), nie względem
    pierwszej zachowanej próbki po skipie — dzięki temu t ≈ skip_samples sekund, a nie 0.
    Zwraca None, gdy brak pliku (np. subject01/s5).
    """
    path = find_polar_hr_path(subject, scenario, data_dir)
    if path is None:
        return None

    import csv
    from datetime import datetime

    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return None

    times: list[datetime] = []
    hrs: list[float] = []
    for row in rows[1:]:
        if len(row) <= hr_column:
            continue
        try:
            times.append(datetime.strptime(row[0].strip(), "%H:%M:%S.%f"))
            hrs.append(float(row[hr_column]))
        except (ValueError, IndexError):
            continue
    if len(hrs) <= skip_samples:
        return None

    t0 = times[0]
    t_s = np.array([(t - t0).total_seconds() for t in times], dtype=np.float64)
    return PolarHrSeries(
        t_s=t_s[skip_samples:].astype(np.float64),
        hr_bpm=np.asarray(hrs[skip_samples:], dtype=np.float64),
    )


def find_polar_ecg_path(subject: str, scenario: str, data_dir: Path = DATA_DIR) -> Path | None:
    """Ścieżka do ``*_ECG.csv`` w folderze sesji albo None."""
    rec = find_recording(subject, scenario, data_dir)
    session_dir = rec.rgb_path.parent
    matches = sorted(session_dir.glob("*_ECG.csv")) + sorted(session_dir.glob("*ECG.csv"))
    unique = list(dict.fromkeys(matches))
    return unique[0] if unique else None


def parse_polar_ecg_samples(
    path: Path,
    markers: tuple[tuple[int, int, int, int], ...] = ECG_FRAME_MARKERS,
) -> np.ndarray:
    """Dekoduje surowe próbki EKG z CSV Polara (markery CR/DR + 3-bajtowe LE signed)."""
    import csv

    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:
        return np.array([], dtype=np.float64)

    marker_lists = [list(m) for m in markers]
    samples: list[int] = []
    for row in rows[1:]:
        try:
            vals = [int(x) for x in row]
        except ValueError:
            continue
        start = None
        for i in range(len(vals) - 3):
            chunk = vals[i : i + 4]
            if any(chunk == m for m in marker_lists):
                start = i + 4
                break
        if start is None:
            continue
        payload = vals[start:]
        for j in range(0, len(payload) - 2, 3):
            b0, b1, b2 = payload[j : j + 3]
            v = b0 | (b1 << 8) | (b2 << 16)
            if v & 0x800000:
                v -= 0x1000000
            samples.append(v)
    return np.asarray(samples, dtype=np.float64)


def load_polar_ecg_hr(
    subject: str,
    scenario: str,
    data_dir: Path = DATA_DIR,
    fs_hz: float = ECG_FS_HZ,
    skip_sec: float = ECG_SKIP_SEC,
    hr_min: float = ECG_HR_MIN_BPM,
    hr_max: float = ECG_HR_MAX_BPM,
) -> PolarHrSeries | None:
    """HR z EKG Polara (neurokit2): odcięcie ``skip_sec``, seria (t_s, hr) od R-R.

    Czas ``t_s`` jest względem początku pliku EKG (= założony start wideo).
    Pierwsze ``skip_sec`` sekund sygnału są pomijane przed detekcją załamków R.
    """
    path = find_polar_ecg_path(subject, scenario, data_dir)
    if path is None:
        return None

    import neurokit2 as nk

    raw = parse_polar_ecg_samples(path)
    if raw.size < int(fs_hz * (skip_sec + 5)):
        return None

    skip = int(round(skip_sec * fs_hz))
    seg = raw[skip:]
    clean = nk.ecg_clean(seg, sampling_rate=fs_hz)
    _, info = nk.ecg_peaks(clean, sampling_rate=fs_hz, method="pantompkins1985")
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    if peaks.size < 3:
        return None

    rr_s = np.diff(peaks) / fs_hz
    hr = 60.0 / rr_s
    t_mid = (peaks[:-1] + peaks[1:]) / (2.0 * fs_hz) + skip_sec
    ok = (hr >= hr_min) & (hr <= hr_max) & np.isfinite(hr)
    if not np.any(ok):
        return None
    return PolarHrSeries(t_s=t_mid[ok].astype(np.float64), hr_bpm=hr[ok].astype(np.float64))


def load_reference_hr(
    subject: str,
    scenario: str,
    data_dir: Path = DATA_DIR,
    max_median_diff_bpm: float = 20.0,
) -> tuple[PolarHrSeries | None, str]:
    """Preferuje EKG (neurokit2); fallback do pliku HR.

    Jeśli EKG i plik HR się mocno rozjeżdżają (|Δ mediana| > ``max_median_diff_bpm``),
    uznajemy detekcję R za zawodną i wracamy do pliku HR (po skip).
    """
    ecg = load_polar_ecg_hr(subject, scenario, data_dir)
    hr = load_polar_hr(subject, scenario, data_dir)

    if ecg is not None and ecg.hr_bpm.size >= 3:
        if hr is not None and hr.hr_bpm.size >= 3:
            ecg_med = float(np.median(ecg.hr_bpm))
            hr_med = float(np.median(hr.hr_bpm))
            if abs(ecg_med - hr_med) > max_median_diff_bpm:
                return hr, "hr_csv"
        return ecg, "ecg"
    if hr is not None:
        return hr, "hr_csv"
    return None, "none"


# Placeholdery: UBFC / iBVP (później)


def load_ubfc_subject(subject_dir: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Wczytuje jedną sesję ze zbioru UBFC-rPPG (RGB + PPG). Do implementacji później."""
    raise NotImplementedError


def load_ibvp_subject(
    subject_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Wczytuje jedną sesję ze zbioru iBVP (RGB + termika radiometryczna + PPG). Później."""
    raise NotImplementedError


def load_ppg_reference(csv_path: Path) -> tuple[np.ndarray, float]:
    """Wczytuje referencyjny sygnał PPG z pliku CSV. Do implementacji później."""
    raise NotImplementedError
