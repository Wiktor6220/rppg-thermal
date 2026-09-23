"""Go/no-go auto-korejestracji (src.registration) vs ręczny GT offline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.config import REG_GO_RMS_PX, RESULTS_DIR  # noqa: E402
from src.io_layer import load_recording  # noqa: E402
from src.registration import (  # noqa: E402
    apply_affine,
    estimate_affine_thermal_to_rgb,
    warp_thermal_to_rgb,
)
from src.roi import make_cropping_detector  # noqa: E402

SUBJECT, SCENARIO = "subject01", "s1_rest_rest"

# Ręczny GT (subject01/s1, t=0) — tylko do metryki; runtime go nie widzi.
GT_POINTS = [
    (127, "skroń L", (667.1, 491.6), True),
    (356, "skroń P", (789.4, 491.6), True),
    (9, "czoło", (721.6, 453.4), True),
    (205, "policzek L", (681.1, 540.7), True),
    (425, "policzek P", (767.6, 544.6), True),
    (61, "usta L", (701.4, 554.0), False),
    (291, "usta P", (735.7, 552.4), False),
]


def main() -> None:
    loaded = load_recording(SUBJECT, SCENARIO)
    rgb, thermal, t_sec = next(loaded.synced_pairs(reference="rgb"))
    print(f"{SUBJECT}/{SCENARIO}  t={t_sec:.3f}s  RGB {rgb.shape}  termika {thermal.shape}")

    detector = make_cropping_detector()
    landmarks = detector(rgb)
    if landmarks is None:
        print("[NO-GO] brak detekcji twarzy RGB")
        return

    affine, info = estimate_affine_thermal_to_rgb(rgb, thermal, landmarks)
    if affine is None:
        print(f"[NO-GO] estymacja affine: {info}")
        return

    gt_th = np.array([p[2] for p in GT_POINTS], dtype=np.float64)
    gt_rgb = np.array([landmarks[p[0]] for p in GT_POINTS], dtype=np.float64)
    pred = apply_affine(affine, gt_th)
    resid = np.linalg.norm(pred - gt_rgb, axis=1)
    rms = float(np.sqrt(np.mean(resid**2)))
    perf = np.array([p[3] for p in GT_POINTS])
    rms_perf = float(np.sqrt(np.mean(resid[perf] ** 2)))

    print(f"\nMetoda: {info['method']}")
    if info.get("eye_refine"):
        er = info["eye_refine"]
        print(f"  refine Y oczu: dy={er['dy']:.1f} px  (thermal eye row={er['eye_thermal'][1]:.0f})")
    # RMS samego konturu (bez refine) dla porównania.
    resid_init = np.linalg.norm(apply_affine(info["init_affine"], gt_th) - gt_rgb, axis=1)
    rms_init = float(np.sqrt(np.mean(resid_init**2)))
    print("=== Residuum AUTO vs GT [px RGB] ===")
    for (_, label, _, _), r in zip(GT_POINTS, resid, strict=True):
        print(f"  {label:<11} {r:7.1f}")
    print(f"  {'RMS':<11} {rms:7.1f}   max {resid.max():7.1f}")
    print(f"  {'RMS perfuzja':<11} {rms_perf:7.1f}")
    print(f"  RMS sam kontur (bez eye_y): {rms_init:.1f}")
    print(f"  próg GO:    {REG_GO_RMS_PX:.1f} px")
    print("  odniesienie: ręczna affine ~10.5 px; stary auto ~53 px")

    out_dir = RESULTS_DIR / "registration_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    warped = warp_thermal_to_rgb(thermal, affine, rgb.shape[:2])
    thermal_red = np.zeros_like(rgb)
    thermal_red[..., 0] = np.clip(warped, 0, 255).astype(np.uint8)
    blended = cv2.addWeighted(rgb, 1.0, thermal_red, 0.5, 0)
    for x, y in gt_rgb:
        cv2.circle(blended, (int(x), int(y)), 10, (0, 255, 0), 2)
    for x, y in pred:
        cv2.circle(blended, (int(x), int(y)), 6, (255, 255, 0), 2)
    full = out_dir / "overlay_mask_ecc.png"
    cv2.imwrite(str(full), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    ph = int(round(blended.shape[0] * 1600 / blended.shape[1]))
    cv2.imwrite(
        str(out_dir / "overlay_mask_ecc_preview.png"),
        cv2.cvtColor(cv2.resize(blended, (1600, ph), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR),
    )
    xs, ys = gt_rgb[:, 0], gt_rgb[:, 1]
    x0, x1 = int(xs.min()) - 80, int(xs.max()) + 80
    y0, y1 = int(ys.min()) - 80, int(ys.max()) + 80
    cv2.imwrite(
        str(out_dir / "overlay_mask_ecc_face_crop.png"),
        cv2.cvtColor(blended[y0:y1, x0:x1], cv2.COLOR_RGB2BGR),
    )
    print(f"\nNakładka: {full.name} (+ _preview, _face_crop) w {out_dir}")

    if rms <= REG_GO_RMS_PX:
        print(f"\n[GO] RMS {rms:.1f} ≤ {REG_GO_RMS_PX:.1f} px")
    else:
        print(f"\n[NO-GO] RMS {rms:.1f} > {REG_GO_RMS_PX:.1f} px")


if __name__ == "__main__":
    main()
