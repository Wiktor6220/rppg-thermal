"""Porównanie potoku RGB (plain) vs RGB+bramkowanie termiką (gated); opcjonalnie Polar HR.

Wyniki: results/pipeline_rgb_thermal/. Uruchomienie: uv run python scripts/demo_pipeline_rgb_thermal.py [--all]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Przed OpenCV/MediaPipe — mniej logów C++ na stderr.
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

from src import estimate, methods, validate  # noqa: E402
from src.config import (  # noqa: E402
    AFFINE_EVERY_BY_SCENARIO,
    AFFINE_EVERY_DEFAULT,
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    PERFUSION_MIN_ROI_FRAC,
    PERFUSION_TEMP_STD_FACTOR,
    RESULTS_DIR,
)
from src.io_layer import list_recordings, load_recording, load_reference_hr  # noqa: E402
from src.registration import (  # noqa: E402
    apply_affine,
    estimate_affine_thermal_to_rgb,
)
from src.roi import make_cropping_detector, select_roi_from_landmarks  # noqa: E402

REGIONS_EXTRACT = ["forehead", "left_cheek", "right_cheek"]
REGIONS_REPORT = ["forehead", "cheeks"]  # L/R tylko wewnętrznie → średnia „cheeks”
METHODS = {"CHROM": methods.chrom, "POS": methods.pos}
OUT_ROOT = RESULTS_DIR / "pipeline_rgb_thermal"


def _affine_every(scenario: str) -> int:
    """Interwał odświeżania affine: częściej przy ruchu / zmiennym dystansie."""
    return AFFINE_EVERY_BY_SCENARIO.get(scenario, AFFINE_EVERY_DEFAULT)


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
    parser.add_argument(
        "--manual-gt",
        default=None,
        metavar="PATH|auto",
        help=(
            "Stała ręczna affine z pliku GT (json z auto_manual_registration). "
            "Wartość 'auto' = results/registration_probe/<subject>_<scenario>/f00000_gt_points.json. "
            "Wyniki trafiają do …/<scenario>_manual/ (nie nadpisują przebiegu auto)."
        ),
    )
    return parser.parse_args()


def _resolve_manual_gt(subject: str, scenario: str, manual_gt: str) -> Path:
    """Zwraca ścieżkę do f*.json z klikniętymi punktami termicznymi."""
    if manual_gt.strip().lower() == "auto":
        path = (
            RESULTS_DIR
            / "registration_probe"
            / f"{subject}_{scenario}"
            / "f00000_gt_points.json"
        )
    else:
        path = Path(manual_gt).expanduser()
        if not path.is_absolute():
            path = (ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Brak pliku GT: {path}\n"
            f"Najpierw: uv run python scripts/auto_manual_registration.py "
            f"--subject {subject} --scenario {scenario}"
        )
    return path


def _fit_affine_from_gt(landmarks: np.ndarray, gt: dict) -> np.ndarray:
    """Affine LS termika→RGB z klikniętych punktów i landmarków na tej klatce."""
    idxs = [int(i) for i in gt["landmark_idx"]]
    thermal_xy = np.asarray(gt["thermal_xy"], dtype=np.float64)
    rgb_xy = np.asarray([landmarks[i] for i in idxs], dtype=np.float64)
    if thermal_xy.shape != rgb_xy.shape:
        raise ValueError(
            f"Niezgodna liczba punktów GT: thermal {thermal_xy.shape} vs rgb {rgb_xy.shape}"
        )
    matrix, _ = cv2.estimateAffine2D(
        thermal_xy.astype(np.float32),
        rgb_xy.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=1e6,
    )
    if matrix is None:
        raise RuntimeError("estimateAffine2D nie zwróciło macierzy z GT.")
    resid = np.linalg.norm(apply_affine(matrix.astype(np.float64), thermal_xy) - rgb_xy, axis=1)
    rms = float(np.sqrt(np.mean(resid**2)))
    # czoło = landmark 9, zwykle 3. punkt w CONTROL_POINTS
    forehead_i = idxs.index(9) if 9 in idxs else 2
    print(
        f"  Ręczna affine z GT: RMS={rms:.1f} px, "
        f"czoło={resid[forehead_i]:.1f} px  (stała na całe nagranie)"
    )
    return matrix.astype(np.float64)

def _bbox_to_mask(bbox: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    y0, x0, y1, x1 = (int(v) for v in bbox)
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _region_boxes(landmarks: np.ndarray) -> dict[str, np.ndarray]:
    """Bboxy ROI do ekstrakcji: czoło + lewy/prawy policzek (bez nosa/ust)."""
    return {r: select_roi_from_landmarks(landmarks, r) for r in REGIONS_EXTRACT}


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
    n_roi = int(inside.sum())
    n_keep = int(np.count_nonzero(keep))
    if n_roi == 0 or n_keep < PERFUSION_MIN_ROI_FRAC * n_roi:
        return _mean_rgb(rgb, roi_mask)
    return rgb[ys[inside][keep], xs[inside][keep]].mean(axis=0).astype(np.float64)


def _process_signal(rgb_trace: np.ndarray, fs: float, method_fn) -> tuple[np.ndarray, float]:
    sig = method_fn(rgb_trace, fs)
    cleaned = estimate.bandpass_filter(estimate.detrend_signal(sig), fs)
    hr = estimate.estimate_hr_welch(cleaned, fs)
    return cleaned, hr


def _out_dir(subject: str, scenario: str, manual: bool = False) -> Path:
    suffix = "_manual" if manual else ""
    return OUT_ROOT / subject / f"{scenario}{suffix}"


def run_one(
    subject: str,
    scenario: str,
    manual_gt: Path | None = None,
) -> list[dict]:
    """Przelicza jedno nagranie; zwraca wiersze wyników i zapisuje je lokalnie.

    ``manual_gt``: jeśli podane, gated używa **stałej** affine z klikniętych punktów
    (bez odświeżania auto) — eksperyment rozdzielający warping vs maskę.
    """
    loaded = load_recording(subject, scenario)
    fs = loaded.rgb_meta.fps
    n_frames = loaded.rgb_meta.frame_count
    use_manual = manual_gt is not None
    affine_every = _affine_every(scenario)
    print(f"\n{'=' * 60}")
    print(f"Nagranie: {subject}/{scenario}  {n_frames} klatek @ {fs:.3f} fps")
    if use_manual:
        print(f"Affine: RĘCZNA stała z GT  ({manual_gt})")
        gt = json.loads(manual_gt.read_text(encoding="utf-8"))
    else:
        print(f"Affine: AUTO, odświeżana co {affine_every} klatek")
        gt = None

    detector = make_cropping_detector()
    # Ekstrakcja: forehead + L/R. Raport tylko: forehead + cheeks (= średnia L+R).
    plain_lists: dict[str, list] = {r: [] for r in REGIONS_EXTRACT}
    gated_lists: dict[str, list] = {r: [] for r in REGIONS_EXTRACT}
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
            if use_manual:
                if affine is None:
                    try:
                        affine = _fit_affine_from_gt(landmarks, gt)
                        n_affine_ok = 1
                    except Exception as exc:  # noqa: BLE001
                        n_affine_fail += 1
                        if i == 0:
                            print(f"  [warn] ręczna affine: {exc}")
            elif affine is None or (i % affine_every == 0):
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
            for region in REGIONS_EXTRACT:
                plain_lists[region].append(full)
                gated_lists[region].append(full)
            continue

        h, w = rgb.shape[:2]
        th_gray = _thermal_gray(thermal)
        for region in REGIONS_EXTRACT:
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
    plain_ex = {r: np.asarray(v, dtype=np.float64) for r, v in plain_lists.items()}
    gated_ex = {r: np.asarray(v, dtype=np.float64) for r, v in gated_lists.items()}
    plain = {
        "forehead": plain_ex["forehead"],
        "cheeks": 0.5 * (plain_ex["left_cheek"] + plain_ex["right_cheek"]),
    }
    gated = {
        "forehead": gated_ex["forehead"],
        "cheeks": 0.5 * (gated_ex["left_cheek"] + gated_ex["right_cheek"]),
    }
    print(
        f"Pokrycie detekcji: {100.0 * valid.mean():.1f}%  "
        f"({int(valid.sum())}/{valid.size}); affine OK/fail: {n_affine_ok}/{n_affine_fail}"
    )

    polar, ref_src = load_reference_hr(subject, scenario)
    if polar is None:
        print("Referencja HR: brak EKG i pliku HR — pomijam MAE/RMSE okienne")
    else:
        src_lbl = "EKG (neurokit2, skip 10 s)" if ref_src == "ecg" else "plik HR (skip 5)"
        print(
            f"Referencja: {src_lbl} — {len(polar.hr_bpm)} próbek, "
            f"mediana {float(np.median(polar.hr_bpm)):.1f} BPM, "
            f"średnia {float(polar.hr_bpm.mean()):.1f} BPM"
        )

    rows: list[dict] = []
    cleaned_store: dict[tuple[str, str, str], np.ndarray] = {}
    header = (
        f"{'region':<12} {'metoda':<6} "
        f"{'HR plain':>10} {'HR gated':>10} {'ΔHR':>8} "
        f"{'SNR plain':>10} {'SNR gated':>10} {'ΔSNR':>8}"
    )
    if polar is not None:
        header += f" {'MAE_p':>8} {'MAE_g':>8}"
    print(header)
    print("-" * len(header))

    for region in REGIONS_REPORT:
        for name, method_fn in METHODS.items():
            sig_p = method_fn(plain[region], fs)
            sig_g = method_fn(gated[region], fs)
            clean_p = estimate.bandpass_filter(estimate.detrend_signal(sig_p), fs)
            clean_g = estimate.bandpass_filter(estimate.detrend_signal(sig_g), fs)
            hr_p = estimate.estimate_hr_welch(clean_p, fs)
            hr_g = estimate.estimate_hr_welch(clean_g, fs)
            # ΔSNR tylko vs referencja (okna); bez ref → NaN (nie vs estymaty plain).
            snr_p = float("nan")
            snr_g = float("nan")
            row = {
                "subject": subject,
                "scenario": scenario,
                "affine_mode": "manual" if use_manual else "auto",
                "region": region,
                "method": name,
                "hr_plain": float(hr_p),
                "hr_gated": float(hr_g),
                "d_hr": float(hr_g - hr_p),
                "snr_plain": snr_p,
                "snr_gated": snr_g,
                "d_snr": float("nan"),
                "mae_plain": float("nan"),
                "mae_gated": float("nan"),
                "rmse_plain": float("nan"),
                "rmse_gated": float("nan"),
                "n_windows": 0,
                "ref_src": ref_src if polar is not None else "none",
                "valid_pct": float(100.0 * valid.mean()),
                "affine_ok": n_affine_ok,
                "affine_fail": n_affine_fail,
            }
            if polar is not None:
                # Surowy sygnał metody — detrend/bandpass per okno wewnątrz validate.
                vp = validate.validate_against_hr_series(
                    sig_p, fs, polar.t_s, polar.hr_bpm, valid=valid
                )
                vg = validate.validate_against_hr_series(
                    sig_g, fs, polar.t_s, polar.hr_bpm, valid=valid
                )
                row["mae_plain"] = float(vp["mae_bpm"])
                row["mae_gated"] = float(vg["mae_bpm"])
                row["rmse_plain"] = float(vp["rmse_bpm"])
                row["rmse_gated"] = float(vg["rmse_bpm"])
                row["n_windows"] = int(vp["n_windows_used"])
                row["snr_plain"] = float(vp["snr_mean"])
                row["snr_gated"] = float(vg["snr_mean"])
                row["d_snr"] = float(vg["snr_mean"] - vp["snr_mean"])
                snr_p = row["snr_plain"]
                snr_g = row["snr_gated"]
            rows.append(row)
            cleaned_store[(region, name, "plain")] = clean_p
            cleaned_store[(region, name, "gated")] = clean_g
            snr_p_s = f"{snr_p:>10.2f}" if np.isfinite(snr_p) else f"{'—':>10}"
            snr_g_s = f"{snr_g:>10.2f}" if np.isfinite(snr_g) else f"{'—':>10}"
            d_snr = row["d_snr"]
            d_snr_s = f"{d_snr:>+8.2f}" if np.isfinite(d_snr) else f"{'—':>8}"
            line = (
                f"{region:<12} {name:<6} "
                f"{hr_p:>10.2f} {hr_g:>10.2f} {hr_g - hr_p:>+8.2f} "
                f"{snr_p_s} {snr_g_s} {d_snr_s}"
            )
            if polar is not None:
                line += f" {row['mae_plain']:>8.2f} {row['mae_gated']:>8.2f}"
            print(line)

    out_dir = _out_dir(subject, scenario, manual=use_manual)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_note = (
        f"Affine: **ręczna stała** z `{manual_gt.name}`."
        if use_manual
        else "Affine: **auto** (kontury + eye_y)."
    )
    md = [
        f"# RGB vs RGB+termika — {subject}/{scenario}",
        "",
        mode_note,
        "SNR w oknach 10 s **względem referencji** (EKG/HR), nie względem estymaty. "
        + (
            "MAE: okna 10 s vs ta sama referencja."
            if polar is not None
            else "Brak referencji HR/EKG — ΔSNR = NaN."
        ),
        "",
        "| region | metoda | HR plain | HR gated | ΔHR | SNR plain | SNR gated | ΔSNR | MAE plain | MAE gated |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        mae_p = f"{row['mae_plain']:.2f}" if not np.isnan(row["mae_plain"]) else "—"
        mae_g = f"{row['mae_gated']:.2f}" if not np.isnan(row["mae_gated"]) else "—"
        snr_p = f"{row['snr_plain']:.2f}" if np.isfinite(row["snr_plain"]) else "—"
        snr_g = f"{row['snr_gated']:.2f}" if np.isfinite(row["snr_gated"]) else "—"
        d_snr = f"{row['d_snr']:+.2f}" if np.isfinite(row["d_snr"]) else "—"
        md.append(
            f"| {row['region']} | {row['method']} | {row['hr_plain']:.2f} | "
            f"{row['hr_gated']:.2f} | {row['d_hr']:+.2f} | "
            f"{snr_p} | {snr_g} | {d_snr} | "
            f"{mae_p} | {mae_g} |"
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
    mode_lbl = "manual affine" if use_manual else "auto affine"
    ax.set_title(
        f"Widmo CHROM forehead — plain vs gated ({subject}/{scenario}, {mode_lbl})"
    )
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
        "mae_plain",
        "mae_gated",
        "rmse_plain",
        "rmse_gated",
        "n_windows",
        "ref_src",
        "valid_pct",
        "affine_ok",
        "affine_fail",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    md_path = OUT_ROOT / "summary_all.md"
    lines = [
        "# Zbiorcze porównanie RGB vs RGB+termika (wszystkie nagrania)",
        "",
        "ΔSNR = średnia SNR okienna gated − plain, **względem referencji HR** (nie względem estymaty). "
        "MAE: okna 10 s vs ta sama referencja. ΔMAE = gated − plain.",
        "",
        "| subject | scenario | region | metoda | HR plain | HR gated | ΔSNR | MAE plain | MAE gated | ΔMAE |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        mae_p = row.get("mae_plain", float("nan"))
        mae_g = row.get("mae_gated", float("nan"))
        d_snr = row.get("d_snr", float("nan"))
        if np.isnan(mae_p) or np.isnan(mae_g):
            mae_p_s, mae_g_s, d_mae_s = "—", "—", "—"
        else:
            mae_p_s = f"{mae_p:.2f}"
            mae_g_s = f"{mae_g:.2f}"
            d_mae_s = f"{mae_g - mae_p:+.2f}"
        d_snr_s = f"{d_snr:+.2f}" if np.isfinite(d_snr) else "—"
        lines.append(
            f"| {row['subject']} | {row['scenario']} | {row['region']} | {row['method']} | "
            f"{row['hr_plain']:.2f} | {row['hr_gated']:.2f} | {d_snr_s} | "
            f"{mae_p_s} | {mae_g_s} | {d_mae_s} |"
        )

    lines.extend(["", "## Średnie ΔSNR [dB] (gated − plain, tylko nagrania z referencją)", ""])
    lines.append("| region | metoda | średnie ΔSNR | n |")
    lines.append("|---|---|---:|---:|")
    for region in REGIONS_REPORT:
        for method in METHODS:
            vals = [
                r["d_snr"]
                for r in all_rows
                if r["region"] == region
                and r["method"] == method
                and np.isfinite(r.get("d_snr", float("nan")))
            ]
            if vals:
                lines.append(
                    f"| {region} | {method} | {float(np.mean(vals)):+.2f} | {len(vals)} |"
                )

    lines.extend(["", "## Średnie MAE vs referencja [BPM] (okna 10 s)", ""])
    lines.append("| region | metoda | MAE plain | MAE gated | ΔMAE | n |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for region in REGIONS_REPORT:
        for method in METHODS:
            subset = [
                r
                for r in all_rows
                if r["region"] == region
                and r["method"] == method
                and not np.isnan(r.get("mae_plain", float("nan")))
            ]
            if not subset:
                continue
            mp = float(np.mean([r["mae_plain"] for r in subset]))
            mg = float(np.mean([r["mae_gated"] for r in subset]))
            lines.append(
                f"| {region} | {method} | {mp:.2f} | {mg:.2f} | {mg - mp:+.2f} | {len(subset)} |"
            )
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main() -> None:
    args = _parse_args()
    _mute_native_stderr()

    if args.all and args.manual_gt:
        raise SystemExit(
            "--all z --manual-gt nie jest wspierane (każde nagranie ma własny GT). "
            "Odpalaj pojedynczo, np.:\n"
            "  uv run python scripts/demo_pipeline_rgb_thermal.py "
            "--subject subject01 --scenario s1_rest_rest --manual-gt auto"
        )

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

    gt_path: Path | None = None
    if args.manual_gt:
        gt_path = _resolve_manual_gt(args.subject, args.scenario, args.manual_gt)
        print(f"Tryb ręcznej affine: {gt_path}")

    run_one(args.subject, args.scenario, manual_gt=gt_path)


if __name__ == "__main__":
    main()
