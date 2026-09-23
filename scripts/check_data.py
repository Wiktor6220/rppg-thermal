"""Raport spójności plików w data/ (wideo, Polar, braki, metadane OpenCV)."""

import re
import sys
from datetime import datetime
from pathlib import Path

# Root repo na sys.path (import src.config).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402 - po ustawieniu sys.path

from src.config import DATA_DIR  # noqa: E402

# Rola -> (sufiks kanoniczny w nazwie, rozszerzenie kanoniczne).
ROLES: dict[str, tuple[str, str]] = {
    "rgb": ("rgb", "MP4"),
    "thermal": ("thermal", "MP4"),
    "HR": ("HR", "csv"),
    "ECG": ("ECG", "csv"),
}
VIDEO_ROLES = ("rgb", "thermal")
POLAR_ROLES = ("HR", "ECG")

# Surowa nazwa Polara, np. dataHR_H10_18_9_2026_11_36_26.csv
RAW_POLAR_RE = re.compile(
    r"^data(?P<kind>HR|ECG)_H10_(?P<d>\d+)_(?P<m>\d+)_(?P<Y>\d+)_(?P<H>\d+)_(?P<M>\d+)_(?P<S>\d+)",
    re.IGNORECASE,
)
SESSION_RE = re.compile(r"^(s\d+)", re.IGNORECASE)


def _fmt_ts(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "—"


def timestamp_from_polar_header(path: Path) -> datetime | None:
    """Zwraca czas startu z wiersza nagłówka `TIMESTAMP,d,m,Y,H,M,S,...` pliku Polara."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline().strip()
    except OSError:
        return None
    parts = first_line.split(",")
    if len(parts) < 7 or parts[0].strip().upper() != "TIMESTAMP":
        return None
    try:
        d, mo, y, h, mi, s = (int(parts[i]) for i in range(1, 7))
        return datetime(y, mo, d, h, mi, s)
    except (ValueError, IndexError):
        return None


def timestamp_from_polar_name(name: str) -> datetime | None:
    """Wyłuskuje czas startu z surowej nazwy Polara (dataHR_/dataECG_...), jeśli pasuje."""
    match = RAW_POLAR_RE.match(name)
    if not match:
        return None
    g = match.groupdict()
    try:
        return datetime(
            int(g["Y"]), int(g["m"]), int(g["d"]), int(g["H"]), int(g["M"]), int(g["S"])
        )
    except ValueError:
        return None


def probe_video(path: Path) -> str:
    """Zwraca opis wideo (klatki, FPS, rozdzielczość, długość) przez OpenCV lub komunikat błędu."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return "NIE DA SIĘ OTWORZYĆ (OpenCV)"
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps > 0:
        seconds = frames / fps
        length = f"{seconds:6.1f} s ({int(seconds // 60)}:{int(seconds % 60):02d})"
    else:
        length = "n/d (FPS=0)"
    flag = "  [!] FPS/klatki podejrzane" if (fps <= 0 or frames <= 0) else ""
    return f"{frames} klatek, {fps:.3f} FPS, {width}x{height}, {length}{flag}"


def classify_extra_file(name: str, prefix: str) -> str:
    """Opis, dlaczego plik jest niespójny (surowa nazwa Polara / literówka / śmieć / nieznany)."""
    if name == ".DS_Store":
        return ".DS_Store (śmieć systemu macOS — do usunięcia)"
    if RAW_POLAR_RE.match(name):
        return "surowa nazwa Polara (do przemianowania na *_HR.csv / *_ECG.csv)"

    lower = name.lower()
    remainder = name[len(prefix):] if name.startswith(prefix) else name
    stem = remainder.rsplit(".", 1)[0].lower()
    ext = remainder.rsplit(".", 1)[1] if "." in remainder else ""

    for role, (suffix, canon_ext) in ROLES.items():
        # Ta sama rola, ale inna wielkość liter w sufiksie lub rozszerzeniu (literówka).
        if stem == suffix.lower() and (remainder != f"{suffix}.{canon_ext}"):
            canon = f"{prefix}{suffix}.{canon_ext}"
            return f"literówka/wielkość liter — chyba rola '{role}' (kanon: {canon})"
    for role, (suffix, canon_ext) in ROLES.items():
        if suffix.lower() in lower:
            canon = f"{prefix}{suffix}.{canon_ext}"
            return f"nietypowa nazwa — być może rola '{role}' (kanon: {canon}); ext='{ext}'"
    return "plik nieoczekiwany (nierozpoznana rola)"


def check_session(subject: str, session_dir: Path) -> dict:
    """Sprawdza jedną sesję: kompletność ról + niespójne pliki. Zwraca podsumowanie."""
    session_match = SESSION_RE.match(session_dir.name)
    session = session_match.group(1) if session_match else session_dir.name
    prefix = f"{subject}_{session}_"

    entries = sorted(p.name for p in session_dir.iterdir())
    canonical = {role: f"{prefix}{suffix}.{ext}" for role, (suffix, ext) in ROLES.items()}
    present = {role: (canonical[role] in entries) for role in ROLES}
    consumed = {canonical[role] for role in ROLES if present[role]}
    extras = [name for name in entries if name not in consumed]

    rel = f"{subject}/{session_dir.name}"
    print(f"\n[{rel}]")
    for role in ROLES:
        canon = canonical[role]
        if present[role]:
            path = session_dir / canon
            if role in VIDEO_ROLES:
                info = probe_video(path)
            else:
                ts_head = timestamp_from_polar_header(path)
                info = f"start {_fmt_ts(ts_head)} (z nagłówka)"
            print(f"  {role:<8}: OK   {canon}  —  {info}")
        else:
            print(f"  {role:<8}: BRAK ({canon})")

    if extras:
        print("  niespójne / nieoczekiwane pliki:")
        for name in extras:
            print(f"    - {name}: {classify_extra_file(name, prefix)}")

    missing = [role for role in ROLES if not present[role]]
    return {"rel": rel, "missing": missing, "extras": extras, "complete": not missing}


def report_polar_pool(polar_dir: Path) -> None:
    """Listuje surowe pliki Polara w data/polar/ z czasami startu (do ręcznego przypisania)."""
    if not polar_dir.is_dir():
        return
    files = sorted(p for p in polar_dir.iterdir() if p.is_file() and p.name != ".DS_Store")
    print(f"\n=== Surowe pliki Polara w {polar_dir.relative_to(DATA_DIR.parent)}/ "
          f"({len(files)}) — do przypisania do sesji ===")
    for path in files:
        ts_name = timestamp_from_polar_name(path.name)
        ts_head = timestamp_from_polar_header(path)
        print(f"  {path.name:<44}  nazwa: {_fmt_ts(ts_name)}   nagłówek: {_fmt_ts(ts_head)}")


def find_ds_store(root: Path) -> list[Path]:
    """Znajduje wszystkie pliki .DS_Store pod danym katalogiem."""
    return sorted(root.rglob(".DS_Store"))


def main() -> None:
    if not DATA_DIR.is_dir():
        print(f"Brak katalogu danych: {DATA_DIR}")
        return

    print(f"=== Raport spójności danych: {DATA_DIR} ===")
    subjects = sorted(p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("subject"))

    results = []
    for subject_dir in subjects:
        sessions = sorted(
            p for p in subject_dir.iterdir() if p.is_dir() and SESSION_RE.match(p.name)
        )
        for session_dir in sessions:
            results.append(check_session(subject_dir.name, session_dir))

    report_polar_pool(DATA_DIR / "polar")

    ds_files = find_ds_store(DATA_DIR)
    print(f"\n=== Pliki .DS_Store pod data/ ({len(ds_files)}) ===")
    for path in ds_files:
        print(f"  - {path.relative_to(DATA_DIR.parent)}")

    total = len(results)
    complete = sum(r["complete"] for r in results)
    print("\n=== Podsumowanie ===")
    print(f"  sesje: {total},  kompletne (4/4): {complete},  niekompletne: {total - complete}")
    for r in results:
        if not r["complete"]:
            print(f"    - {r['rel']}: brakuje {', '.join(r['missing'])}")
    sessions_with_extras = [r for r in results if r["extras"]]
    if sessions_with_extras:
        print("  sesje z niespójnymi/nieoczekiwanymi plikami:")
        for r in sessions_with_extras:
            print(f"    - {r['rel']}: {', '.join(r['extras'])}")


if __name__ == "__main__":
    main()
