"""scripts/roi_coverage_report.py — pokrycie detekcji ROI na WSZYSTKICH nagraniach.

Dla każdego nagrania (subject01/02 × s1–s5) uruchamia przetestowaną logikę
`roi.track_roi_across_frames` z detektorem „detekcja na wycinku twarzy" i zbiera:
  - % klatek valid (faktyczna detekcja), liczbę i łączną długość przerw (hold),
    najdłuższą przerwę,
do jednej tabeli results/roi_coverage.md (+ wydruk).

Dla każdego nagrania zapisuje kilka klatek z narysowanym ROI do
results/roi_preview/<subject>_<scenario>/ — równomiernie w czasie ORAZ z okolic
najdłuższych przerw w detekcji (klatka przed/pośrodku/po przerwie).

NIE robi ekstrakcji ani maski termicznej. Uruchomienie: uv run python scripts/roi_coverage_report.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402
from src.io_layer import list_recordings, load_recording  # noqa: E402
from src.roi import (  # noqa: E402
    make_cropping_detector,
    select_roi_from_landmarks,
    track_roi_across_frames,
)

REGION_COLORS = {"forehead": (0, 255, 0), "left_cheek": (0, 160, 255), "right_cheek": (255, 160, 0)}
REGIONS = list(REGION_COLORS)
PREVIEW_WIDTH = 480
CROP_PAD = 110


def _multi_roi_builder(landmarks: np.ndarray, region: str) -> np.ndarray:
    return np.stack([select_roi_from_landmarks(landmarks, r) for r in REGIONS])


def _gaps(valid: np.ndarray) -> list[tuple[int, int]]:
    """Zwraca listę przerw jako (start, end) — domknięte przedziały nieważnych klatek."""
    spans: list[tuple[int, int]] = []
    n = len(valid)
    i = 0
    while i < n:
        if not valid[i]:
            j = i
            while j + 1 < n and not valid[j + 1]:
                j += 1
            spans.append((i, j))
            i = j + 1
        else:
            i += 1
    return spans


def _preview_indices(valid: np.ndarray, n_even: int = 6, max_gaps: int = 3) -> list[int]:
    """Indeksy podglądu: równomiernie w czasie + okolice najdłuższych przerw."""
    n = len(valid)
    idx: set[int] = set(np.linspace(0, max(0, n - 1), n_even, dtype=int).tolist())
    for start, end in sorted(_gaps(valid), key=lambda s: s[1] - s[0], reverse=True)[:max_gaps]:
        for c in (start - 1, (start + end) // 2, end + 1):
            idx.add(int(np.clip(c, 0, n - 1)))
    return sorted(idx)


def _face_preview(frame_rgb: np.ndarray, roi_boxes, label: str) -> np.ndarray:
    """Wykadrowany na twarz podgląd (BGR) z ROI i adnotacją; kadr liczony na oryginale."""
    height, width = frame_rgb.shape[:2]
    if roi_boxes is not None:
        ys = [int(b[0]) for b in roi_boxes] + [int(b[2]) for b in roi_boxes]
        xs = [int(b[1]) for b in roi_boxes] + [int(b[3]) for b in roi_boxes]
        y0, y1 = max(0, min(ys) - CROP_PAD), min(height, max(ys) + CROP_PAD)
        x0, x1 = max(0, min(xs) - CROP_PAD), min(width, max(xs) + CROP_PAD)
        crop = frame_rgb[y0:y1, x0:x1].copy()
        for color, box in zip(REGION_COLORS.values(), roi_boxes, strict=False):
            by0, bx0, by1, bx1 = (int(v) for v in box)
            cv2.rectangle(crop, (bx0 - x0, by0 - y0), (bx1 - x0, by1 - y0), color, 2)
    else:
        crop = frame_rgb.copy()
    ch, cw = crop.shape[:2]
    crop = cv2.resize(crop, (PREVIEW_WIDTH, max(1, int(round(ch * PREVIEW_WIDTH / cw)))))
    cv2.putText(
        crop, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )
    return cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)


def _process(recording) -> dict:
    """Detekcja+śledzenie na jednym nagraniu; zwraca statystyki i zapisuje podglądy."""
    loaded = load_recording(recording.subject, recording.scenario)
    fps = loaded.rgb_meta.fps

    detector = make_cropping_detector()  # wielkoskalowy adaptacyjny wycinek
    roi_positions, valid = track_roi_across_frames(
        loaded.rgb_frames(to_rgb=True), detector=detector, roi_builder=_multi_roi_builder,
        region=REGIONS[0],
    )

    detected = int(valid.sum())
    total = int(valid.size)
    spans = _gaps(valid)
    longest = max((e - s + 1 for s, e in spans), default=0)

    # Podgląd: druga (tylko dekodująca) przebieżka po potrzebnych indeksach.
    wanted = _preview_indices(valid)
    max_i = wanted[-1] if wanted else -1
    out_dir = RESULTS_DIR / "roi_preview" / f"{recording.subject}_{recording.scenario}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    wanted_set = set(wanted)
    for i, frame in enumerate(loaded.rgb_frames(to_rgb=True)):
        if i in wanted_set:
            label = f"f{i} t={i / fps:.1f}s valid={bool(valid[i])}"
            preview = _face_preview(frame, roi_positions[i], label)
            cv2.imwrite(str(out_dir / f"frame_{i:05d}.png"), preview)
        if i >= max_i:
            break

    return {
        "subject": recording.subject,
        "scenario": recording.scenario,
        "frames": total,
        "fps": fps,
        "valid_pct": 100.0 * detected / total if total else 0.0,
        "n_gaps": len(spans),
        "hold_frames": total - detected,
        "longest_gap": longest,
        "longest_gap_s": longest / fps if fps else 0.0,
        "n_previews": len(wanted),
    }


def main() -> None:
    recordings = list_recordings()
    print(f"Nagrania do przetworzenia: {len(recordings)}")
    rows = []
    for rec in recordings:
        print(f"  -> {rec.subject}/{rec.scenario} ...", flush=True)
        rows.append(_process(rec))

    header = (
        f"{'subject':<9} {'scenario':<14} {'frames':>6} {'valid%':>7} "
        f"{'gaps':>5} {'hold':>6} {'maxgap(f/s)':>13}"
    )
    lines = ["=== Pokrycie detekcji ROI — wszystkie nagrania ===", header, "-" * len(header)]
    for r in rows:
        maxgap = f"{r['longest_gap']}/{r['longest_gap_s']:.1f}s"
        lines.append(
            f"{r['subject']:<9} {r['scenario']:<14} {r['frames']:>6} {r['valid_pct']:>6.1f}% "
            f"{r['n_gaps']:>5} {r['hold_frames']:>6} {maxgap:>13}"
        )
    table = "\n".join(lines)
    print("\n" + table)

    md = [
        "# Pokrycie detekcji ROI (wszystkie nagrania)",
        "",
        "| subject | scenario | klatki | fps | valid % | przerwy | hold [kl.] | "
        "najdłuższa przerwa |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['subject']} | {r['scenario']} | {r['frames']} | {r['fps']:.3f} | "
            f"{r['valid_pct']:.1f}% | {r['n_gaps']} | {r['hold_frames']} | "
            f"{r['longest_gap']} kl. ({r['longest_gap_s']:.1f} s) |"
        )
    md.append("")
    (RESULTS_DIR / "roi_coverage.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nTabela zapisana do: {RESULTS_DIR / 'roi_coverage.md'}")
    print(f"Podglądy: {RESULTS_DIR / 'roi_preview'}/<subject>_<scenario>/")


if __name__ == "__main__":
    main()
