"""Porównanie affine auto vs ręcznej (kliknięcia na termice)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Przed OpenCV/MediaPipe — mniej śmieci na stderr.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402 — GUI (ginput); NIE Agg
import numpy as np  # noqa: E402

from src.config import REG_GO_RMS_PX, RESULTS_DIR  # noqa: E402
from src.io_layer import load_recording  # noqa: E402
from src.registration import (  # noqa: E402
    apply_affine,
    estimate_affine_thermal_to_rgb,
    warp_thermal_to_rgb,
)
from src.roi import make_cropping_detector  # noqa: E402

# (landmark RGB, etykieta, czy rejon perfuzji)
CONTROL_POINTS = [
    (127, "skroń L", True),
    (356, "skroń P", True),
    (9, "czoło", True),
    (205, "policzek L", True),
    (425, "policzek P", True),
    (61, "usta L", False),
    (291, "usta P", False),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto vs ręczna korejestracja — klikanie GT + metryki na jednym nagraniu."
    )
    p.add_argument("--subject", required=True, help="np. subject01")
    p.add_argument("--scenario", required=True, help="np. s1_rest_rest")
    p.add_argument(
        "--every",
        type=int,
        default=0,
        help="Ponowne klikanie co N klatek (0 = tylko pierwsza klatka). Dla s5 np. 300.",
    )
    p.add_argument(
        "--max-checks",
        type=int,
        default=5,
        help="Maks. liczba sesji klikania przy --every > 0 (domyślnie 5).",
    )
    return p.parse_args()


def _pick_thermal_points(thermal_rgb: np.ndarray, title_prefix: str) -> np.ndarray:
    """Interaktywne klikanie 7 punktów na termice; zwraca (7, 2) float64."""
    height, width = thermal_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(max(8.0, width / 140.0), max(6.0, height / 140.0)))
    ax.imshow(thermal_rgb, interpolation="nearest")
    ax.set_xlabel("x [px termiki]")
    ax.set_ylabel("y [px termiki]")

    picked: list[tuple[float, float]] = []
    for i, (_, label, _) in enumerate(CONTROL_POINTS):
        ax.set_title(f"{title_prefix}\nKliknij: {label}  ({i + 1}/{len(CONTROL_POINTS)})")
        fig.canvas.draw()
        clicks = plt.ginput(1, timeout=0)
        if not clicks:
            plt.close(fig)
            raise SystemExit("Przerwano — brak kliknięcia.")
        x, y = float(clicks[0][0]), float(clicks[0][1])
        picked.append((x, y))
        ax.plot(x, y, "g+", markersize=14, markeredgewidth=2)
        ax.annotate(str(i + 1), (x, y), color="lime", fontsize=9)
        fig.canvas.draw()
    plt.close(fig)
    return np.asarray(picked, dtype=np.float64)


def _fit_manual_affine(thermal_xy: np.ndarray, rgb_xy: np.ndarray) -> np.ndarray:
    """Affine LS termika→RGB z klikniętych par punktów."""
    matrix, _ = cv2.estimateAffine2D(
        thermal_xy.astype(np.float32),
        rgb_xy.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=1e6,
    )
    if matrix is None:
        raise RuntimeError("estimateAffine2D nie zwróciło macierzy (za mało punktów?).")
    return matrix.astype(np.float64)


def _residuals(
    affine: np.ndarray, thermal_xy: np.ndarray, rgb_xy: np.ndarray
) -> np.ndarray:
    pred = apply_affine(affine, thermal_xy)
    return np.linalg.norm(pred - rgb_xy, axis=1)


def _print_resid_table(title: str, resid: np.ndarray) -> dict[str, float]:
    print(f"\n=== {title} ===")
    for (_, label, _), r in zip(CONTROL_POINTS, resid, strict=True):
        print(f"  {label:<12} {r:7.1f} px")
    rms = float(np.sqrt(np.mean(resid**2)))
    perf = np.array([p[2] for p in CONTROL_POINTS], dtype=bool)
    # czoło = indeks 2; policzki = 3,4
    forehead = float(resid[2])
    cheeks = float(np.sqrt(np.mean(resid[3:5] ** 2)))
    rms_perf = float(np.sqrt(np.mean(resid[perf] ** 2)))
    print(f"  {'RMS':<12} {rms:7.1f} px   max {resid.max():7.1f}")
    print(f"  {'RMS perfuzja':<12} {rms_perf:7.1f} px")
    print(f"  {'czoło':<12} {forehead:7.1f} px")
    print(f"  {'policzki RMS':<12} {cheeks:7.1f} px")
    return {
        "rms": rms,
        "rms_perf": rms_perf,
        "forehead": forehead,
        "cheeks_rms": cheeks,
        "max": float(resid.max()),
    }


def _save_overlay(
    rgb: np.ndarray,
    thermal: np.ndarray,
    affine: np.ndarray,
    rgb_xy: np.ndarray,
    pred_xy: np.ndarray,
    out_path: Path,
    title_note: str,
) -> None:
    warped = warp_thermal_to_rgb(thermal, affine, rgb.shape[:2])
    red = np.zeros_like(rgb)
    red[..., 0] = np.clip(warped, 0, 255).astype(np.uint8)
    blended = cv2.addWeighted(rgb, 1.0, red, 0.5, 0)
    for x, y in rgb_xy:
        cv2.circle(blended, (int(x), int(y)), 10, (0, 255, 0), 2)  # GT RGB
    for x, y in pred_xy:
        cv2.circle(blended, (int(x), int(y)), 6, (255, 255, 0), 2)  # pred z affine
    cv2.putText(
        blended,
        title_note,
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    # crop twarzy
    xs, ys = rgb_xy[:, 0], rgb_xy[:, 1]
    x0, x1 = int(xs.min()) - 80, int(xs.max()) + 80
    y0, y1 = int(ys.min()) - 80, int(ys.max()) + 80
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(rgb.shape[1], x1), min(rgb.shape[0], y1)
    crop_path = out_path.with_name(out_path.stem + "_face_crop.png")
    cv2.imwrite(str(crop_path), cv2.cvtColor(blended[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))


def _evaluate_frame(
    rgb: np.ndarray,
    thermal: np.ndarray,
    landmarks: np.ndarray,
    thermal_xy: np.ndarray,
    frame_idx: int,
    out_dir: Path,
    tag: str,
) -> dict:
    rgb_xy = np.array([landmarks[idx] for idx, _, _ in CONTROL_POINTS], dtype=np.float64)

    manual = _fit_manual_affine(thermal_xy, rgb_xy)
    resid_m = _residuals(manual, thermal_xy, rgb_xy)
    stats_m = _print_resid_table(f"RĘCZNA affine  (klatka {frame_idx})", resid_m)

    auto, info = estimate_affine_thermal_to_rgb(rgb, thermal, landmarks)
    if auto is None:
        print(f"[NO-GO] auto affine: {info}")
        stats_a = None
        resid_a = None
    else:
        resid_a = _residuals(auto, thermal_xy, rgb_xy)
        stats_a = _print_resid_table(
            f"AUTO affine ({info.get('method', '?')})  (klatka {frame_idx})", resid_a
        )
        if info.get("eye_refine"):
            er = info["eye_refine"]
            print(f"  refine Y oczu: dy={er['dy']:.1f} px")

    # Porównanie czoło / policzki
    if stats_a is not None:
        print("\n=== AUTO vs RĘCZNA (różnica RMS) ===")
        print(
            f"  czoło:     auto {stats_a['forehead']:.1f}  vs  ręczna {stats_m['forehead']:.1f}  "
            f"(Δ {stats_a['forehead'] - stats_m['forehead']:+.1f})"
        )
        print(
            f"  policzki:  auto {stats_a['cheeks_rms']:.1f}  vs  ręczna {stats_m['cheeks_rms']:.1f}  "
            f"(Δ {stats_a['cheeks_rms'] - stats_m['cheeks_rms']:+.1f})"
        )
        print(
            f"  RMS all:   auto {stats_a['rms']:.1f}  vs  ręczna {stats_m['rms']:.1f}  "
            f"(Δ {stats_a['rms'] - stats_m['rms']:+.1f})"
        )
        go = stats_a["rms"] <= REG_GO_RMS_PX
        print(
            f"\n[{'GO' if go else 'NO-GO'}] auto RMS {stats_a['rms']:.1f} "
            f"{'≤' if go else '>'} próg {REG_GO_RMS_PX:.1f} px"
        )

    _save_overlay(
        rgb,
        thermal,
        manual,
        rgb_xy,
        apply_affine(manual, thermal_xy),
        out_dir / f"{tag}_manual_overlay.png",
        f"manual f={frame_idx}",
    )
    if auto is not None:
        _save_overlay(
            rgb,
            thermal,
            auto,
            rgb_xy,
            apply_affine(auto, thermal_xy),
            out_dir / f"{tag}_auto_overlay.png",
            f"auto f={frame_idx}",
        )
        print(f"\nNakładki: {out_dir}/{tag}_manual_overlay*.png  oraz  {tag}_auto_overlay*.png")

    return {
        "frame_idx": frame_idx,
        "thermal_xy": thermal_xy.tolist(),
        "rgb_xy": rgb_xy.tolist(),
        "manual": stats_m,
        "auto": stats_a,
        "manual_affine": manual.tolist(),
        "auto_affine": None if auto is None else auto.tolist(),
    }


def main() -> None:
    args = _parse_args()
    loaded = load_recording(args.subject, args.scenario)
    fs = loaded.rgb_meta.fps
    n_frames = loaded.rgb_meta.frame_count
    print(f"Nagranie: {args.subject}/{args.scenario}  {n_frames} klatek @ {fs:.3f} fps")
    if args.every <= 0:
        print("Tryb: klikanie TYLKO na pierwszej klatce (zalecane dla s1–s4).")
    else:
        print(
            f"Tryb: klikanie co {args.every} klatek "
            f"(max {args.max_checks} sesji) — np. drift / s5."
        )

    detector = make_cropping_detector()
    out_dir = RESULTS_DIR / "registration_probe" / f"{args.subject}_{args.scenario}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    check_indices = {0}
    if args.every > 0:
        i = args.every
        while i < n_frames and len(check_indices) < args.max_checks:
            check_indices.add(i)
            i += args.every
    check_indices = sorted(check_indices)

    for rgb, thermal, t_sec in loaded.synced_pairs(reference="rgb"):
        # synced_pairs nie daje indeksu — liczymy po czasie
        frame_idx = int(round(t_sec * fs))
        # dopasuj do najbliższego checkpointu (tolerancja 1 klatka)
        matched = None
        for c in check_indices:
            if abs(frame_idx - c) <= 1 and not any(r["frame_idx"] == c for r in results):
                matched = c
                break
        if matched is None:
            if frame_idx > max(check_indices) + 1:
                break
            continue

        landmarks = detector(rgb)
        if landmarks is None:
            print(f"[skip] klatka {matched}: brak twarzy RGB")
            continue

        print(f"\n{'=' * 60}")
        print(f"Sesja klikania: klatka ~{matched}  t={t_sec:.2f}s")
        print("Kolejność: " + ", ".join(lab for _, lab, _ in CONTROL_POINTS))
        thermal_xy = _pick_thermal_points(
            thermal if thermal.ndim == 3 else cv2.cvtColor(thermal, cv2.COLOR_GRAY2RGB),
            title_prefix=f"{args.subject}/{args.scenario}  f≈{matched}",
        )

        tag = f"f{matched:05d}"
        rec = _evaluate_frame(rgb, thermal, landmarks, thermal_xy, matched, out_dir, tag)
        results.append(rec)

        # zapis cząstkowy GT
        gt_path = out_dir / f"{tag}_gt_points.json"
        gt_path.write_text(
            json.dumps(
                {
                    "subject": args.subject,
                    "scenario": args.scenario,
                    "frame_idx": matched,
                    "t_sec": t_sec,
                    "labels": [lab for _, lab, _ in CONTROL_POINTS],
                    "landmark_idx": [idx for idx, _, _ in CONTROL_POINTS],
                    "thermal_xy": thermal_xy.tolist(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Zapisano GT: {gt_path}")

    summary_path = out_dir / "auto_vs_manual_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nPodsumowanie: {summary_path}")
    print("Zielone kółka = landmarki RGB; żółte = predikcja z affine (termika→RGB).")


if __name__ == "__main__":
    main()
