"""scripts/demo_roi.py — sprawdzenie detekcji twarzy i stabilności ROI na realnym RGB.

Przetwarza JEDNO nagranie RGB (domyślnie subject01/s1_rest_rest — baseline), używa
przetestowanej logiki śledzenia z `roi.track_roi_across_frames` (detekcja + `valid[]` +
przytrzymanie) z detektorem „detekcja na wycinku twarzy" (`make_cropping_detector`),
odpornym na małą twarz w kadrze 4K z drona. Zapisuje:
  (a) kilkanaście przykładowych klatek z narysowanym ROI (czoło + policzki)
      do results/roi_preview/,
  (b) statystykę pokrycia detekcji: % klatek z wykrytą twarzą, histogram długości
      przerw, liczbę klatek valid vs invalid.

ROI liczone jest na współrzędnych ORYGINAŁU; podgląd zapisujemy pomniejszony.
NIE robi ekstrakcji sygnału ani maski termicznej — to kolejne kroki.

Uruchomienie:  uv run python scripts/demo_roi.py
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")  # wycisz logi C++ MediaPipe/absl

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402
from src.io_layer import load_recording  # noqa: E402
from src.roi import (  # noqa: E402
    make_cropping_detector,
    select_roi_from_landmarks,
    track_roi_across_frames,
)

# Kolory (RGB) regionów ROI na podglądzie; kolejność = kolejność wierszy w roi_builder.
REGION_COLORS = {
    "forehead": (0, 255, 0),
    "left_cheek": (0, 160, 255),
    "right_cheek": (255, 160, 0),
}
REGIONS = list(REGION_COLORS)
PREVIEW_WIDTH = 480  # szerokość zapisywanego, wykadrowanego na twarz podglądu
CROP_PAD = 110  # margines wokół ROI przy kadrowaniu podglądu [px oryginału]


def _gap_lengths(valid: np.ndarray) -> list[int]:
    """Zwraca długości kolejnych przerw (ciągów nieważnych klatek) w `valid`."""
    gaps: list[int] = []
    run = 0
    for is_valid in valid:
        if not is_valid:
            run += 1
        elif run > 0:
            gaps.append(run)
            run = 0
    if run > 0:
        gaps.append(run)
    return gaps


def _print_coverage(valid: np.ndarray, fps: float) -> None:
    """Wypisuje statystykę pokrycia detekcji: valid/invalid, przerwy, histogram."""
    total = int(valid.size)
    detected = int(valid.sum())
    invalid = total - detected
    pct = 100.0 * detected / total if total else 0.0

    print("\n=== Pokrycie detekcji ===")
    print(f"  klatki łącznie       : {total}  ({total / fps:.1f} s @ {fps:.3f} fps)")
    print(f"  wykryta twarz (valid): {detected}  ({pct:.1f}%)")
    print(f"  bez detekcji (hold)  : {invalid}  ({100.0 - pct:.1f}%)")

    gaps = _gap_lengths(valid)
    print(f"  liczba przerw        : {len(gaps)}")
    if gaps:
        longest = max(gaps)
        print(f"  najdłuższa przerwa   : {longest} klatek ({longest / fps:.2f} s)")
        print("  histogram długości przerw (długość [klatki] : liczba):")
        for length, count in sorted(Counter(gaps).items()):
            print(f"    {length:>4} : {count}")
    else:
        print("  brak przerw — twarz wykryta na każdej klatce")


def _face_preview(frame_rgb: np.ndarray, roi_boxes, frame_index: int, fps: float, valid_flag: bool):
    """Buduje wykadrowany na twarz podgląd (RGB) z narysowanymi ROI i adnotacją.

    Kadruje oryginalną klatkę wokół unii bboxów ROI (z marginesem), rysuje regiony na
    pełnej rozdzielczości, a dopiero potem pomniejsza do `PREVIEW_WIDTH` — dzięki temu
    ROI są czytelne, a współrzędne liczone na oryginale.
    """
    height, width = frame_rgb.shape[:2]
    if roi_boxes is not None:
        ys = [int(b[0]) for b in roi_boxes] + [int(b[2]) for b in roi_boxes]
        xs = [int(b[1]) for b in roi_boxes] + [int(b[3]) for b in roi_boxes]
        y0 = max(0, min(ys) - CROP_PAD)
        y1 = min(height, max(ys) + CROP_PAD)
        x0 = max(0, min(xs) - CROP_PAD)
        x1 = min(width, max(xs) + CROP_PAD)
        crop = frame_rgb[y0:y1, x0:x1].copy()
        for color, box in zip(REGION_COLORS.values(), roi_boxes, strict=False):
            by0, bx0, by1, bx1 = (int(v) for v in box)
            cv2.rectangle(crop, (bx0 - x0, by0 - y0), (bx1 - x0, by1 - y0), color, 2)
    else:
        crop = frame_rgb.copy()

    ch, cw = crop.shape[:2]
    preview_height = max(1, int(round(ch * PREVIEW_WIDTH / cw)))
    crop = cv2.resize(crop, (PREVIEW_WIDTH, preview_height), interpolation=cv2.INTER_AREA)
    label = f"frame {frame_index}  t={frame_index / fps:.1f}s  valid={valid_flag}"
    cv2.putText(
        crop, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    return crop


def _multi_roi_builder(landmarks: np.ndarray, region: str) -> np.ndarray:
    """Zwraca bboxy wszystkich regionów naraz jako tablicę (len(REGIONS), 4)."""
    return np.stack([select_roi_from_landmarks(landmarks, r) for r in REGIONS])


def main() -> None:
    parser = argparse.ArgumentParser(description="Detekcja twarzy i ROI na jednym nagraniu RGB.")
    parser.add_argument("--subject", default="subject01")
    parser.add_argument("--scenario", default="s1_rest_rest")
    parser.add_argument("--num-preview", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=1600, help="bok wycinka detekcji [px]")
    args = parser.parse_args()

    loaded = load_recording(args.subject, args.scenario)
    rec, meta = loaded.recording, loaded.rgb_meta
    fps, n_frames = meta.fps, meta.frame_count
    orig_width = meta.resolution[0]
    print(f"Nagranie: {rec.subject}/{rec.scenario}  ({rec.rgb_path.name})")
    print(f"  {n_frames} klatek, {fps:.3f} fps, {orig_width}x{meta.resolution[1]}")
    print(f"  regiony ROI: {REGIONS};  wycinek detekcji: {args.crop_size}px")

    sample_indices = set(np.linspace(0, max(0, n_frames - 1), args.num_preview, dtype=int).tolist())
    stash: dict[int, np.ndarray] = {}

    def tee(frame_iter):
        """Przepuszcza klatki do detektora, po drodze zapisując pełne kopie klatek-próbek."""
        for i, frame in enumerate(frame_iter):
            if i in sample_indices:
                stash[i] = frame.copy()  # pełna rozdzielczość — kadr na twarz zrobimy po detekcji
            yield frame

    detector = make_cropping_detector(crop_size=args.crop_size)
    print("\nDetekcja + śledzenie (może chwilę potrwać na pełnym nagraniu)...")
    roi_positions, valid = track_roi_across_frames(
        tee(loaded.rgb_frames(to_rgb=True)),
        detector=detector,
        roi_builder=_multi_roi_builder,
        region=REGIONS[0],
    )

    _print_coverage(valid, fps)

    out_dir = RESULTS_DIR / "roi_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    print(f"\n=== Zapisane klatki z ROI ({len(stash)}) → {out_dir} ===")
    saved = []
    for i in sorted(stash):
        preview = _face_preview(stash[i], roi_positions[i], i, fps, bool(valid[i]))
        out_path = out_dir / f"frame_{i:05d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))  # RGB->BGR do zapisu
        saved.append(out_path.name)
        print(f"  {out_path.name}  (valid={bool(valid[i])})")

    print(f"\nZapisano {len(saved)} klatek podglądu do {out_dir}")


if __name__ == "__main__":
    main()
