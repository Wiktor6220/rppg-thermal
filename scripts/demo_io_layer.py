"""scripts/demo_io_layer.py — szybki podgląd io_layer na realnym nagraniu (tylko odczyt).

Wczytuje subject01/s1_rest_rest i wypisuje dla RGB i termiki: fps, liczbę klatek,
rozdzielczość oraz długość w sekundach. Sprawdza też kształt pierwszej klatki obu
strumieni (bez dekodowania całości).

Uruchomienie:  uv run python scripts/demo_io_layer.py
"""

import sys
from pathlib import Path

# Root repo na ścieżkę, by zaimportować pakiet src.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_layer import VideoMeta, list_recordings, load_recording  # noqa: E402


def _print_meta(label: str, meta: VideoMeta) -> None:
    width, height = meta.resolution
    print(
        f"  {label:<8}: fps={meta.fps:.3f}, klatki={meta.frame_count}, "
        f"rozdzielczość={width}x{height}, długość={meta.duration_s:.2f} s"
    )


def _first_frame_shape(frames_iter) -> tuple[int, ...] | None:
    for frame in frames_iter:
        return frame.shape
    return None


def main() -> None:
    print(f"Dostępne kompletne nagrania (RGB+termika): {len(list_recordings())}")

    loaded = load_recording("subject01", "s1_rest_rest")
    rec = loaded.recording
    print(f"\nNagranie: {rec.subject} / {rec.scenario} (kod {rec.scenario_code})")
    print(f"  rgb    : {rec.rgb_path.name}")
    print(f"  thermal: {rec.thermal_path.name}")

    print("\nMetadane strumieni (osobno, bez zakładania równości):")
    _print_meta("RGB", loaded.rgb_meta)
    _print_meta("THERMAL", loaded.thermal_meta)

    print("\nKontrola pierwszej klatki (kształt H, W, 3):")
    print(f"  RGB    : {_first_frame_shape(loaded.rgb_frames())}")
    print(f"  THERMAL: {_first_frame_shape(loaded.thermal_frames())}")

    rgb, thermal = loaded.rgb_meta, loaded.thermal_meta
    print("\nRóżnice RGB vs termika:")
    print(f"  rozdzielczość: {rgb.resolution} vs {thermal.resolution}")
    print(f"  fps          : {rgb.fps:.3f} vs {thermal.fps:.3f}")
    print(f"  klatki       : {rgb.frame_count} vs {thermal.frame_count} "
          f"(różnica {abs(rgb.frame_count - thermal.frame_count)})")


if __name__ == "__main__":
    main()
