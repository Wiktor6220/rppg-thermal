"""Wczytywanie nagrań wideo z realnego zbioru (dron: RGB + termika podglądowa).

Zbiór bieżący: `data/subjectXX/sN_opis/subjectXX_sN_{rgb,thermal}.MP4` (+ pliki Polara
HR/ECG, ignorowane na tym etapie). Termika to **wizualny podgląd mp4, NIE radiometryka**.

Moduł tylko odczytuje klatki i metadane — nie synchronizuje, nie skaluje, nie przetwarza
sygnału. RGB i termika mogą mieć różną rozdzielczość, fps i liczbę klatek; zwracamy te
informacje osobno dla każdego strumienia i NIE zakładamy, że są równe.

Klatki wideo są duże (RGB bywa 4K), więc domyślnie zwracamy je jako **generator**
(leniwie, klatka po klatce), a nie jako jedną tablicę w pamięci.
"""

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import DATA_DIR

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
    """Listuje kompletne nagrania (RGB + termika) w `data/`, pomijając niekompletne.

    „Kompletne" na tym etapie oznacza obecność OBU strumieni wideo (RGB i termika) —
    pliki Polara są tu ignorowane (synchronizacja później). Sesja bez któregoś strumienia
    jest pomijana.

    Args:
        data_dir: katalog główny danych. Domyślnie `config.DATA_DIR`.

    Returns:
        Lista `Recording` posortowana po (subject, scenario).
    """
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
    """Wczytuje pojedyncze nagranie (RGB + termika) dla danej osoby i scenariusza.

    Zwraca metadane obu strumieni od razu (szybki `probe_video`) oraz leniwy dostęp do
    klatek (`.rgb_frames()`, `.thermal_frames()`). NIE wczytuje wszystkich klatek do
    pamięci i NIE synchronizuje strumieni.

    Args:
        subject: identyfikator osoby, np. "subject01", "01".
        scenario: nazwa/kod scenariusza, np. "s1_rest_rest" albo "s1".
        data_dir: katalog główny danych. Domyślnie `config.DATA_DIR`.

    Returns:
        `LoadedRecording` z metadanymi RGB i termiki oraz generatorami klatek.
    """
    rec = find_recording(subject, scenario, data_dir)
    return LoadedRecording(
        recording=rec,
        rgb_meta=probe_video(rec.rgb_path),
        thermal_meta=probe_video(rec.thermal_path),
    )


# --- Placeholdery dla publicznych zbiorów referencyjnych (do implementacji później) ---
# Zbiór bieżący (dron) obsługują funkcje powyżej. Poniższe są rezerwą pod UBFC/iBVP.


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
