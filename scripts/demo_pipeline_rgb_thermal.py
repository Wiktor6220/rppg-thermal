"""scripts/demo_pipeline_rgb_thermal.py — porównanie RGB vs RGB+termika.

Bez referencji EKG/Polar. Dla każdego regionu ROI:
  plain  = średnie RGB w bboxie,
  gated  = średnie RGB w bboxie ∩ masce perfuzji (termika przez odwrotną affine).

Wyniki per nagranie: ``results/pipeline_rgb_thermal/<subject>/<scenario>/``
Zbiorczo (``--all``): ``results/pipeline_rgb_thermal/summary_all.md`` (+ .csv).

Uruchomienie:
    uv run python scripts/demo_pipeline_rgb_thermal.py
    uv run python scripts/demo_pipeline_rgb_thermal.py --subject subject02 --scenario s3_drone_move
    uv run python scripts/demo_pipeline_rgb_thermal.py --all
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Przed importem OpenCV/MediaPipe — inaczej logi C++ i tak się wyleją.
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import welch  # noqa: E402

from src import estimate, methods  # noqa: E402
from src.config import (  # noqa: E402
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    PERFUSION_TEMP_STD_FACTOR,
    RESULTS_DIR,
)
from src.io_layer import list_recordings, load_recording  # noqa: E402
from src.registration import (  # noqa: E402
    apply_affine,
    estimate_affine_thermal_to_rgb,
)
from src.roi import make_cropping_detector, select_roi_from_landmarks  # noqa: E402

REGIONS = ["forehead", "left_cheek", "right_cheek"]
METHODS = {"CHROM": methods.chrom, "POS": methods.pos}
AFFINE_EVERY = 30  # ~1 s przy 30 fps
OUT_ROOT = RESULTS_DIR / "pipeline_rgb_thermal"


def _mute_native_stderr() -> None:
    """Wycisza logi C++ (MediaPipe/TFLite/clearcut) na fd=2; Python traceback zostaje.

    Natywne biblioteki piszą wprost na deskryptor 2, omijając ``sys.stderr``.
    Podmieniamy fd=2 na /dev/null, a ``sys.stderr`` kierujemy na kopię oryginału.
    """
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
    devnull = open(os.devnull, "w")
    sys.stderr = os.fdopen(os.dup(2), "w", buffering=1)
    os.dup2(devnull.fileno(), 2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Porównanie RGB vs RGB+termika (jedno nagranie lub --all)."
    )
    parser.add_argument("--subject", default="subject01", help="np. subject01, subject02")
    parser.add_argument(
        "--scenario",
        default="s1_rest_rest",
        help="np. s1_rest_rest, s2_person_move, …",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Przelicz wszystkie kompletne nagrania w data/ i zapisz tabelę zbiorczą.",
    )
    return parser.parse_args()


def _bbox_to_mask(bbox: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    y0, x0, y1, x1 = (int(v) for v in bbox)
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _cheeks_union_bbox(landmarks: np.ndarray) -> np.ndarray:
    left = select_roi_from_landmarks(landmarks, "left_cheek")
    right = select_roi_from_landmarks(landmarks, "right_cheek")
    return np.array(
        [
            min(int(left[0]), int(right[0])),
            min(int(left[1]), int(right[1])),
            max(int(left[2]), int(right[2])),
            max(int(left[3]), int(right[3])),
        ],
        dtype=int,
    )


def _region_boxes(landmarks: np.ndarray) -> dict[str, np.ndarray]:
    boxes = {r: select_roi_from_landmarks(landmarks, r) for r in REGIONS}
    boxes["cheeks"] = _cheeks_union_bbox(landmarks)
    return boxes


def _mean_rgb(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.any():
        return frame[mask].mean(axis=0).astype(np.float64)
    return frame.reshape(-1, 3).mean(axis=0).astype(np.float64)


def _thermal_gray(thermal: np.ndarray) -> np.ndarray:
    if thermal.ndim == 2:
        return thermal.astype(np.float32)
    return cv2.cvtColor(thermal, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _gated_mean_rgb(
    rgb: np.ndarray,
    thermal_gray: np.ndarray,
    affine_th_to_rgb: np.ndarray,
    roi_mask: np.ndarray,
) -> np.ndarray:
    ys, xs = np.where(roi_mask)
    if ys.size == 0:
        return _mean_rgb(rgb, roi_mask)

    inv = cv2.invertAffineTransform(affine_th_to_rgb.astype(np.float32)).astype(np.float64)
    pts_th = apply_affine(inv, np.column_stack([xs.astype(np.float64), ys.astype(np.float64)]))
    th_h, th_w = thermal_gray.shape[:2]
    inside = (
        (pts_th[:, 0] >= 0)
        & (pts_th[:, 0] < th_w - 1)
        & (pts_th[:, 1] >= 0)
        & (pts_th[:, 1] < th_h - 1)
    )
    if not np.any(inside):
        return _mean_rgb(rgb, roi_mask)

    map_x = pts_th[inside, 0].astype(np.float32).reshape(1, -1)
    map_y = pts_th[inside, 1].astype(np.float32).reshape(1, -1)
    temps = cv2.remap(
        thermal_gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).ravel()
    thr = float(temps.mean() + PERFUSION_TEMP_STD_FACTOR * temps.std())
    keep = temps >= thr
    if not np.any(keep):
        return _mean_rgb(rgb, roi_mask)
    return rgb[ys[inside][keep], xs[inside][keep]].mean(axis=0).astype(np.float64)


def _process_signal(rgb_trace: np.ndarray, fs: float, method_fn) -> tuple[np.ndarray, float]:
    sig = method_fn(rgb_trace, fs)
    cleaned = estimate.bandpass_filter(estimate.detrend_signal(sig), fs)
    hr = estimate.estimate_hr_welch(cleaned, fs)
    return cleaned, hr


def _out_dir(subject: str, scenario: str) -> Path:
    return OUT_ROOT / subject / scenario


def run_one(subject: str, scenario: str) -> list[dict]:
    """Przelicza jedno nagranie; zwraca wiersze wyników i zapisuje je lokalnie."""
    loaded = load_recording(subject, scenario)
    fs = loaded.rgb_meta.fps
    n_frames = loaded.rgb_meta.frame_count
    print(f"\n{'=' * 60}")
    print(f"Nagranie: {subject}/{scenario}  {n_frames} klatek @ {fs:.3f} fps")
    print(f"Affine odświeżana co {AFFINE_EVERY} klatek")

    detector = make_cropping_detector()
    all_regions = REGIONS + ["cheeks"]
    plain_lists: dict[str, list] = {r: [] for r in all_regions}
    gated_lists: dict[str, list] = {r: [] for r in all_regions}
    valid_list: list[bool] = []

    affine = None
    last_boxes: dict[str, np.ndarray] | None = None
    n_affine_ok = 0
    n_affine_fail = 0

    print("Przebieg synced RGB+termika...", flush=True)
    for i, (rgb, thermal, _) in enumerate(loaded.synced_pairs(reference="rgb")):
        landmarks = detector(rgb)
        if landmarks is not None:
            boxes = _region_boxes(landmarks)
            last_boxes = boxes
            valid_list.append(True)
            if affine is None or (i % AFFINE_EVERY == 0):
                new_affine, info = estimate_affine_thermal_to_rgb(rgb, thermal, landmarks)
                if new_affine is not None:
                    affine = new_affine
                    n_affine_ok += 1
                else:
                    n_affine_fail += 1
                    if i == 0:
                        print(f"  [warn] affine na klatce 0: {info}")
        else:
            valid_list.append(False)
            boxes = last_boxes

        if boxes is None:
            full = rgb.reshape(-1, 3).mean(axis=0)
            for region in all_regions:
                plain_lists[region].append(full)
                gated_lists[region].append(full)
            continue

        h, w = rgb.shape[:2]
        th_gray = _thermal_gray(thermal)
        for region in all_regions:
            roi_mask = _bbox_to_mask(boxes[region], (h, w))
            plain_lists[region].append(_mean_rgb(rgb, roi_mask))
            if affine is None:
                gated_lists[region].append(plain_lists[region][-1])
            else:
                gated_lists[region].append(
                    _gated_mean_rgb(rgb, th_gray, affine, roi_mask)
                )

        if (i + 1) % 200 == 0 or i + 1 == n_frames:
            print(f"  klatka {i + 1}/{n_frames}", flush=True)

    valid = np.array(valid_list, dtype=bool)
    plain = {r: np.asarray(v, dtype=np.float64) for r, v in plain_lists.items()}
    gated = {r: np.asarray(v, dtype=np.float64) for r, v in gated_lists.items()}
    print(
        f"Pokrycie detekcji: {100.0 * valid.mean():.1f}%  "
        f"({int(valid.sum())}/{valid.size}); affine OK/fail: {n_affine_ok}/{n_affine_fail}"
    )

    rows: list[dict] = []
    cleaned_store: dict[tuple[str, str, str], np.ndarray] = {}
    header = (
        f"{'region':<12} {'metoda':<6} "
        f"{'HR plain':>10} {'HR gated':>10} {'ΔHR':>8} "
        f"{'SNR plain':>10} {'SNR gated':>10} {'ΔSNR':>8}"
    )
    print(header)
    print("-" * len(header))

    for region in all_regions:
        for name, method_fn in METHODS.items():
            clean_p, hr_p = _process_signal(plain[region], fs, method_fn)
            clean_g, hr_g = _process_signal(gated[region], fs, method_fn)
            snr_p = estimate.snr_rppg(clean_p, fs, hr_p)
            snr_g = estimate.snr_rppg(clean_g, fs, hr_p)
            row = {
                "subject": subject,
                "scenario": scenario,
                "region": region,
                "method": name,
                "hr_plain": float(hr_p),
                "hr_gated": float(hr_g),
                "d_hr": float(hr_g - hr_p),
                "snr_plain": float(snr_p),
                "snr_gated": float(snr_g),
                "d_snr": float(snr_g - snr_p),
                "valid_pct": float(100.0 * valid.mean()),
                "affine_ok": n_affine_ok,
                "affine_fail": n_affine_fail,
            }
            rows.append(row)
            cleaned_store[(region, name, "plain")] = clean_p
            cleaned_store[(region, name, "gated")] = clean_g
            print(
                f"{region:<12} {name:<6} "
                f"{hr_p:>10.2f} {hr_g:>10.2f} {hr_g - hr_p:>+8.2f} "
                f"{snr_p:>10.2f} {snr_g:>10.2f} {snr_g - snr_p:>+8.2f}"
            )

    out_dir = _out_dir(subject, scenario)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [
        f"# RGB vs RGB+termika — {subject}/{scenario}",
        "",
        "Bez referencji EKG. SNR względem HR z wariantu **plain**.",
        "",
        "| region | metoda | HR plain | HR gated | ΔHR | SNR plain [dB] | SNR gated [dB] | ΔSNR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['region']} | {row['method']} | {row['hr_plain']:.2f} | "
            f"{row['hr_gated']:.2f} | {row['d_hr']:+.2f} | "
            f"{row['snr_plain']:.2f} | {row['snr_gated']:.2f} | {row['d_snr']:+.2f} |"
        )
    md.append("")
    (out_dir / "comparison_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11, 3.5))
    for kind, color in [("plain", "tab:blue"), ("gated", "tab:orange")]:
        sig = cleaned_store[("forehead", "CHROM", kind)]
        freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), int(10 * fs)))
        ax.plot(freqs * 60.0, psd, color=color, label=f"CHROM forehead {kind}")
    ax.set_xlim(BAND_LOW_HZ * 60, BAND_HIGH_HZ * 60)
    ax.set_xlabel("częstość [BPM]")
    ax.set_ylabel("gęstość mocy [j.u.]")
    ax.set_title(f"Widmo CHROM forehead — plain vs gated ({subject}/{scenario})")
    ax.legend(loc="upper right")
    fig.savefig(out_dir / "spectrum_forehead_chrom.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Zapisano: {out_dir}")
    return rows


def _write_summary(all_rows: list[dict]) -> Path:
    """Zbiorcza tabela markdown + CSV dla wszystkich przebiegów."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / "summary_all.csv"
    fieldnames = [
        "subject",
        "scenario",
        "region",
        "method",
        "hr_plain",
        "hr_gated",
        "d_hr",
        "snr_plain",
        "snr_gated",
        "d_snr",
        "valid_pct",
        "affine_ok",
        "affine_fail",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row[k] for k in fieldnames})

    md_path = OUT_ROOT / "summary_all.md"
    lines = [
        "# Zbiorcze porównanie RGB vs RGB+termika (wszystkie nagrania)",
        "",
        "Bez referencji EKG. SNR względem HR plain. ΔSNR > 0 ⇒ termika poprawia czystość sygnału.",
        "",
        "| subject | scenario | region | metoda | HR plain | HR gated | ΔHR | SNR plain | SNR gated | ΔSNR |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        lines.append(
            f"| {row['subject']} | {row['scenario']} | {row['region']} | {row['method']} | "
            f"{row['hr_plain']:.2f} | {row['hr_gated']:.2f} | {row['d_hr']:+.2f} | "
            f"{row['snr_plain']:.2f} | {row['snr_gated']:.2f} | {row['d_snr']:+.2f} |"
        )

    # Skrót: średnie ΔSNR per region/metoda
    lines.extend(["", "## Średnie ΔSNR [dB] (gated − plain)", ""])
    lines.append("| region | metoda | średnie ΔSNR | n nagrań |")
    lines.append("|---|---|---:|---:|")
    for region in REGIONS + ["cheeks"]:
        for method in METHODS:
            vals = [
                r["d_snr"]
                for r in all_rows
                if r["region"] == region and r["method"] == method
            ]
            if vals:
                lines.append(
                    f"| {region} | {method} | {float(np.mean(vals)):+.2f} | {len(vals)} |"
                )
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> None:
    args = _parse_args()
    _mute_native_stderr()

    if args.all:
        recordings = list_recordings()
        if not recordings:
            raise SystemExit("Brak kompletnych nagrań w data/.")
        print(f"--all: {len(recordings)} nagrań")
        all_rows: list[dict] = []
        failures: list[str] = []
        for rec in recordings:
            try:
                all_rows.extend(run_one(rec.subject, rec.scenario))
            except Exception as exc:  # noqa: BLE001 — kontynuuj pozostałe nagrania
                msg = f"{rec.subject}/{rec.scenario}: {exc}"
                print(f"[FAIL] {msg}")
                failures.append(msg)
        summary = _write_summary(all_rows)
        print(f"\n=== Zbiorczo: {len(all_rows)} wierszy → {summary} ===")
        print(f"CSV: {OUT_ROOT / 'summary_all.csv'}")
        if failures:
            print(f"Nieudane ({len(failures)}):")
            for msg in failures:
                print(f"  - {msg}")
        return

    rows = run_one(args.subject, args.scenario)
    # Przy pojedynczym nagraniu też dopisz/odśwież mini-summary tylko tego runu? Nie —
    # zbiorcza powstaje wyłącznie przy --all. Jedno nagranie ma swój comparison_table.md.
    _ = rows


if __name__ == "__main__":
    main()
