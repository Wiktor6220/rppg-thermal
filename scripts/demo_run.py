"""Demo end-to-end: demo_data/ (nazwy DJI) → demo_results/ bez ręcznych kroków.

Mapuje samo: *_V.MP4→RGB, *_T.MP4→termika, dataHR_*.csv→HR (bez trimu).
Kalibracja nominalna: demo_data/reg_calib.json (auto, jeśli brak).
config.py / EVAL_SUBJECT / subject02 — nietknięte.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

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
    AFFINE_EVERY_DEFAULT,
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    POLAR_HR_SKIP_SAMPLES,
    REG_GO_RMS_PX,
    VALIDATION_WINDOW_SEC,
)
from src.extract import (  # noqa: E402
    gated_means_per_frame_from_samples,
    mean_rgb_in_mask,
    sample_roi_temps_via_affine,
)
from src.io_layer import (  # noqa: E402
    LoadedRecording,
    Recording,
    load_polar_hr_csv,
    probe_video,
)
from src.registration import (  # noqa: E402
    _to_gray,
    apply_nominal_calibration,
    calibrate_nominal_registration,
    estimate_affine_thermal_to_rgb,
    registration_quality,
    rgb_face_mask,
    warp_thermal_to_rgb,
)
from src.roi import make_cropping_detector, select_roi_from_landmarks  # noqa: E402

DEMO_DATA = ROOT / "demo_data"
DEMO_RESULTS = ROOT / "demo_results"
CALIB_PATH = DEMO_DATA / "reg_calib.json"

REGIONS_EXTRACT = ["forehead", "left_cheek", "right_cheek"]
REGIONS_REPORT = ["forehead", "cheeks"]
METHODS = {"CHROM": methods.chrom, "POS": methods.pos}
FIG_DPI = 160


# ---------------------------------------------------------------------------
# Odkrywanie nagrań DJI (bez io_layer._find_stream)
# ---------------------------------------------------------------------------


def _pick_one(paths: list[Path], label: str, session: Path) -> Path:
    if not paths:
        raise FileNotFoundError(f"Brak {label} w {session}")
    return sorted(paths)[0]


def discover_demo_sessions(demo_data: Path = DEMO_DATA) -> list[tuple[str, str, Path]]:
    """Lista (subject, scenario, session_dir) z demo_data/<subject>/<scenario>/."""
    out: list[tuple[str, str, Path]] = []
    if not demo_data.is_dir():
        return out
    for subject_dir in sorted(p for p in demo_data.iterdir() if p.is_dir() and p.name.startswith("subject")):
        for session in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            out.append((subject_dir.name, session.name, session))
    return out


def resolve_dji_streams(session_dir: Path) -> tuple[Path, Path, Path]:
    """*_V.MP4 → RGB, *_T.MP4 → termika, dataHR_*.csv → HR (preferuj surowy dataHR)."""
    rgb = _pick_one(
        list(session_dir.glob("*_V.MP4")) + list(session_dir.glob("*_V.mp4")),
        "RGB (*_V.MP4)",
        session_dir,
    )
    thermal = _pick_one(
        list(session_dir.glob("*_T.MP4")) + list(session_dir.glob("*_T.mp4")),
        "termika (*_T.MP4)",
        session_dir,
    )
    hr_candidates = sorted(session_dir.glob("dataHR_*.csv"))
    if not hr_candidates:
        hr_candidates = sorted(session_dir.glob("*_HR.csv"))
    hr = _pick_one(hr_candidates, "HR (dataHR_*.csv)", session_dir)
    return rgb, thermal, hr


def load_demo_recording(session_dir: Path, subject: str, scenario: str) -> tuple[LoadedRecording, Path]:
    rgb_path, th_path, hr_path = resolve_dji_streams(session_dir)
    rec = Recording(
        subject=subject,
        scenario=scenario,
        rgb_path=rgb_path,
        thermal_path=th_path,
    )
    loaded = LoadedRecording(
        recording=rec,
        rgb_meta=probe_video(rgb_path),
        thermal_meta=probe_video(th_path),
    )
    return loaded, hr_path


# ---------------------------------------------------------------------------
# Kalibracja
# ---------------------------------------------------------------------------


def ensure_calibration(
    demo_data: Path = DEMO_DATA,
    force: bool = False,
) -> dict:
    """Ładuje reg_calib.json albo liczy z pierwszego nagrania i zapisuje."""
    if CALIB_PATH.is_file() and not force:
        calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
        apply_nominal_calibration(calib)
        print(f"Kalibracja z {CALIB_PATH}")
        return calib

    sessions = discover_demo_sessions(demo_data)
    if not sessions:
        raise FileNotFoundError(f"Brak sesji w {demo_data}")
    subject, scenario, session_dir = sessions[0]
    loaded, _ = load_demo_recording(session_dir, subject, scenario)
    rgb, thermal, _ = next(loaded.synced_pairs(reference="rgb"))
    lm = make_cropping_detector()(rgb)
    if lm is None:
        raise RuntimeError(f"Kalibracja: brak twarzy w {subject}/{scenario}")
    calib = calibrate_nominal_registration(rgb, thermal, lm)
    calib["source"] = f"{subject}/{scenario}"
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIB_PATH.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    apply_nominal_calibration(calib)
    print(
        f"Zapisano kalibrację → {CALIB_PATH}\n"
        f"  scale={calib['REG_NOMINAL_SCALE']:.4f} "
        f"offset={calib['REG_NOMINAL_OFFSET']} pad={calib['REG_WINDOW_PAD']}"
    )
    return calib


# ---------------------------------------------------------------------------
# Potok jednego nagrania
# ---------------------------------------------------------------------------


def _bbox_to_mask(bbox: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    y0, x0, y1, x1 = (int(v) for v in bbox)
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _region_boxes(landmarks: np.ndarray) -> dict[str, np.ndarray]:
    return {r: select_roi_from_landmarks(landmarks, r) for r in REGIONS_EXTRACT}


def run_session(
    subject: str,
    scenario: str,
    session_dir: Path,
    out_dir: Path,
) -> dict:
    loaded, hr_path = load_demo_recording(session_dir, subject, scenario)
    fs = loaded.rgb_meta.fps
    n_frames = loaded.rgb_meta.frame_count
    polar = load_polar_hr_csv(hr_path, skip_samples=POLAR_HR_SKIP_SAMPLES)
    if polar is None:
        raise RuntimeError(f"Nie udało się wczytać HR: {hr_path}")

    print(f"\n{'=' * 60}")
    print(f"{subject}/{scenario}  {n_frames} kl. @ {fs:.3f} fps  HR={hr_path.name} (n={len(polar.hr_bpm)})")

    detector = make_cropping_detector()
    affine_every = AFFINE_EVERY_DEFAULT
    affine = None
    last_boxes = None
    pending: list[tuple[np.ndarray, np.ndarray]] = []
    n_affine_ok = n_affine_fail = 0

    plain_lists: dict[str, list] = {r: [] for r in REGIONS_EXTRACT}
    rgb_pix: dict[str, list] = {r: [] for r in REGIONS_EXTRACT}
    temps_lists: dict[str, list] = {r: [] for r in REGIONS_EXTRACT}
    valid_list: list[bool] = []

    # Pierwsza udana detekcja+affine → figury i quality (nie sztywno i==0).
    frame0_rgb = frame0_th = frame0_lm = None
    reg_info0 = None
    quality = None
    quality_frame_idx: int | None = None

    def _append(rgb, thermal, boxes, detected: bool) -> None:
        nonlocal affine, n_affine_ok, n_affine_fail
        valid_list.append(detected)
        h, w = rgb.shape[:2]
        th_gray = _to_gray(thermal).astype(np.float32)
        for region in REGIONS_EXTRACT:
            roi_mask = _bbox_to_mask(boxes[region], (h, w))
            plain_lists[region].append(mean_rgb_in_mask(rgb, roi_mask))
            if affine is None:
                rgb_pix[region].append(None)
                temps_lists[region].append(None)
                continue
            ys, xs, tvals = sample_roi_temps_via_affine(th_gray, affine, roi_mask)
            if tvals.size == 0:
                rgb_pix[region].append(None)
                temps_lists[region].append(None)
            else:
                rgb_pix[region].append(rgb[ys, xs].astype(np.float64))
                temps_lists[region].append(tvals)

    print("Przebieg synced RGB+termika...", flush=True)
    for i, (rgb, thermal, _) in enumerate(loaded.synced_pairs(reference="rgb")):
        landmarks = detector(rgb)
        if landmarks is not None:
            boxes = _region_boxes(landmarks)
            if affine is None or (i % affine_every == 0):
                new_aff, info = estimate_affine_thermal_to_rgb(rgb, thermal, landmarks)
                if new_aff is not None:
                    affine = new_aff.astype(np.float64)
                    n_affine_ok += 1
                    if quality is None:
                        frame0_rgb, frame0_th, frame0_lm = rgb.copy(), thermal.copy(), landmarks
                        reg_info0 = info
                        quality = registration_quality(
                            affine, info["thermal_mask"], info["rgb_mask"]
                        )
                        quality_frame_idx = i
                        if i > 0:
                            print(
                                f"  [info] pierwsza udana rejestracja na klatce {i} "
                                f"(wcześniejsze bez twarzy/affine)",
                                flush=True,
                            )
                else:
                    n_affine_fail += 1
                    if quality is None:
                        print(f"  [warn] affine (klatka {i}): {info}")
            if last_boxes is None and pending:
                for pr, pt in pending:
                    _append(pr, pt, boxes, False)
                pending.clear()
            last_boxes = boxes
            _append(rgb, thermal, boxes, True)
        else:
            if last_boxes is None:
                pending.append((rgb, thermal))
            else:
                _append(rgb, thermal, last_boxes, False)
        if (i + 1) % 200 == 0 or i + 1 == n_frames:
            print(f"  klatka {i + 1}/{n_frames}", flush=True)

    if quality is None:
        raise RuntimeError(
            f"{subject}/{scenario}: brak udanej detekcji twarzy + affine "
            f"w całym nagraniu (affine OK/fail={n_affine_ok}/{n_affine_fail}) — "
            f"nie można policzyć quality/figur."
        )

    valid = np.array(valid_list, dtype=bool)
    plain_ex = {r: np.asarray(v, dtype=np.float64) for r, v in plain_lists.items()}
    gated_ex: dict[str, np.ndarray] = {}
    fallback_flags: dict[str, np.ndarray] = {}
    for region in REGIONS_EXTRACT:
        g, fb = gated_means_per_frame_from_samples(
            rgb_pix[region], temps_lists[region], plain_ex[region]
        )
        gated_ex[region] = g
        fallback_flags[region] = fb

    plain = {
        "forehead": plain_ex["forehead"],
        "cheeks": 0.5 * (plain_ex["left_cheek"] + plain_ex["right_cheek"]),
    }
    gated = {
        "forehead": gated_ex["forehead"],
        "cheeks": 0.5 * (gated_ex["left_cheek"] + gated_ex["right_cheek"]),
    }
    fb_fore = 100.0 * float(fallback_flags["forehead"].mean())
    fb_cheek = 100.0 * float(
        0.5
        * (fallback_flags["left_cheek"].astype(float) + fallback_flags["right_cheek"].astype(float)).mean()
    )

    hr_ref = float(np.median(polar.hr_bpm))
    rows: list[dict] = []
    cleaned: dict[tuple[str, str, str], np.ndarray] = {}
    window_hrs: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}

    trustworthy = bool(quality and quality["trustworthy"])
    gate_label = "trustworthy" if trustworthy else "NO-GO"

    print(
        f"Rejestracja: RMS={quality['contour_rms_px']:.1f} px  "
        f"IoU={quality['iou']:.3f}  → gated {gate_label}  "
        f"(próg {REG_GO_RMS_PX:.0f} px); affine OK/fail={n_affine_ok}/{n_affine_fail}"
        f"; quality@klatka={quality_frame_idx}"
    )

    for region in REGIONS_REPORT:
        for name, fn in METHODS.items():
            sig_p = fn(plain[region], fs)
            sig_g = fn(gated[region], fs)
            clean_p = estimate.bandpass_filter(estimate.detrend_signal(sig_p), fs)
            clean_g = estimate.bandpass_filter(estimate.detrend_signal(sig_g), fs)
            hr_p = estimate.estimate_hr_welch(clean_p, fs)
            hr_g = estimate.estimate_hr_welch(clean_g, fs)
            vp = validate.validate_against_hr_series(
                sig_p, fs, polar.t_s, polar.hr_bpm, valid=valid
            )
            vg = validate.validate_against_hr_series(
                sig_g, fs, polar.t_s, polar.hr_bpm, valid=valid
            )
            fb = fb_fore if region == "forehead" else fb_cheek
            row = {
                "subject": subject,
                "scenario": scenario,
                "region": region,
                "method": name,
                "hr_ref": hr_ref,
                "hr_plain": float(hr_p),
                "hr_gated": float(hr_g),
                "snr_plain": float(vp["snr_mean"]),
                "snr_gated": float(vg["snr_mean"]),
                "delta_snr": float(vg["snr_mean"] - vp["snr_mean"]),
                "mae_plain": float(vp["mae_bpm"]),
                "mae_gated": float(vg["mae_bpm"]),
                "rmse_plain": float(vp["rmse_bpm"]),
                "rmse_gated": float(vg["rmse_bpm"]),
                "octave_frac_plain": float(vp["octave_error_frac"]),
                "octave_frac_gated": float(vg["octave_error_frac"]),
                "fallback_pct": float(fb),
                "n_windows": int(vp["n_windows_used"]),
                "reg_rms_px": float(quality["contour_rms_px"]),
                "reg_iou": float(quality["iou"]),
                "gated_status": gate_label,
            }
            rows.append(row)
            cleaned[(region, name, "plain")] = clean_p
            cleaned[(region, name, "gated")] = clean_g
            # HR per okno 10 s
            win = max(1, int(round(VALIDATION_WINDOW_SEC * fs)))
            step = max(1, win // 2)
            t_centers = []
            hr_series_p = []
            hr_series_g = []
            for start in range(0, len(clean_p) - win + 1, step):
                sl = slice(start, start + win)
                if valid[sl].mean() < 0.5:
                    continue
                t_centers.append((start + win / 2) / fs)
                try:
                    hr_series_p.append(estimate.estimate_hr_welch(clean_p[sl], fs))
                    hr_series_g.append(estimate.estimate_hr_welch(clean_g[sl], fs))
                except Exception:  # noqa: BLE001
                    hr_series_p.append(float("nan"))
                    hr_series_g.append(float("nan"))
            window_hrs[(region, name, "plain")] = (
                np.asarray(t_centers),
                np.asarray(hr_series_p),
            )
            window_hrs[(region, name, "gated")] = (
                np.asarray(t_centers),
                np.asarray(hr_series_g),
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_tables(out_dir, rows, quality, gate_label)
    if frame0_rgb is not None and reg_info0 is not None and affine is not None:
        _save_figures(
            out_dir,
            frame0_rgb,
            frame0_th,
            frame0_lm,
            reg_info0,
            affine,
            quality,
            cleaned,
            window_hrs,
            polar,
            fs,
            rows,
        )

    (out_dir / "reg_quality.json").write_text(
        json.dumps(
            {
                **quality,
                "method": reg_info0["method"] if reg_info0 else None,
                "gated_status": gate_label,
                "affine_ok": n_affine_ok,
                "affine_fail": n_affine_fail,
                "valid_pct": float(100.0 * valid.mean()),
                "quality_frame_idx": quality_frame_idx,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"rows": rows, "quality": quality, "gated_status": gate_label}


def _write_tables(out_dir: Path, rows: list[dict], quality: dict, gate_label: str) -> None:
    fields = [
        "region",
        "method",
        "hr_ref",
        "hr_plain",
        "hr_gated",
        "delta_snr",
        "mae_plain",
        "mae_gated",
        "rmse_plain",
        "rmse_gated",
        "octave_frac_plain",
        "octave_frac_gated",
        "fallback_pct",
        "gated_status",
        "reg_rms_px",
        "reg_iou",
    ]
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = [
        f"# Metryki — rejestracja RMS={quality['contour_rms_px']:.1f} px, "
        f"IoU={quality['iou']:.3f}, gated **{gate_label}** (próg {REG_GO_RMS_PX:.0f} px)",
        "",
        "| region | metoda | HR ref | HR plain | HR gated | ΔSNR | MAE plain | MAE gated | "
        "RMSE plain | RMSE gated | % oktaw plain | % oktaw gated | fallback% |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['region']} | {r['method']} | {r['hr_ref']:.1f} | {r['hr_plain']:.2f} | "
            f"{r['hr_gated']:.2f} | {r['delta_snr']:+.2f} | {r['mae_plain']:.2f} | "
            f"{r['mae_gated']:.2f} | {r['rmse_plain']:.2f} | {r['rmse_gated']:.2f} | "
            f"{100 * r['octave_frac_plain']:.0f}% | {100 * r['octave_frac_gated']:.0f}% | "
            f"{r['fallback_pct']:.1f} |"
        )
    (out_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabela → {csv_path}")


def _save_figures(
    out_dir: Path,
    rgb: np.ndarray,
    thermal: np.ndarray,
    landmarks: np.ndarray,
    info: dict,
    affine: np.ndarray,
    quality: dict,
    cleaned: dict,
    window_hrs: dict,
    polar,
    fs: float,
    rows: list[dict],
) -> None:
    # 1) RGB + ROI
    vis = rgb.copy()
    for region, color in [
        ("forehead", (0, 255, 0)),
        ("left_cheek", (255, 200, 0)),
        ("right_cheek", (255, 200, 0)),
    ]:
        box = select_roi_from_landmarks(landmarks, region)
        y0, x0, y1, x1 = (int(v) for v in box)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 3)
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    x0, x1 = int(xs.min()) - 40, int(xs.max()) + 40
    y0, y1 = int(ys.min()) - 40, int(ys.max()) + 40
    crop = vis[max(0, y0) : min(vis.shape[0], y1), max(0, x0) : min(vis.shape[1], x1)]
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.imshow(crop)
    ax.set_title("RGB — ROI (czoło / policzki)")
    ax.axis("off")
    fig.savefig(out_dir / "fig_rgb_roi.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # 2) IR + segment twarzy + maska perfuzji (względem ROI czoła w termice)
    th_gray = _to_gray(thermal)
    th_vis = cv2.cvtColor(th_gray, cv2.COLOR_GRAY2RGB)
    seg_mask = info["thermal_mask"]
    overlay = th_vis.copy()
    overlay[seg_mask] = (0.45 * th_vis[seg_mask] + np.array([180, 40, 40])).astype(np.uint8)
    # perfuzja: względna jasność w segmencie
    vals = th_gray[seg_mask].astype(np.float64)
    if vals.size:
        thr = vals.mean() + 0.5 * vals.std()
        perf = seg_mask & (th_gray.astype(np.float64) >= thr)
        overlay[perf] = (0.35 * th_vis[perf] + np.array([40, 200, 80])).astype(np.uint8)
    win = info["window"]
    cv2.rectangle(overlay, (win[0], win[1]), (win[2], win[3]), (0, 255, 255), 2)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(overlay)
    ax.set_title("Termika — segment twarzy (czerwony) + perfuzja (zielony)")
    ax.axis("off")
    fig.savefig(out_dir / "fig_thermal_seg.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # 3) Nakładka warpowana + residuum
    warped = warp_thermal_to_rgb(thermal, affine, rgb.shape[:2])
    # poprawny remap maski na RGB
    from src.registration import invert_affine, apply_affine

    rgb_h, rgb_w = rgb.shape[:2]
    ys_r, xs_r = np.mgrid[0:rgb_h, 0:rgb_w]
    inv = invert_affine(affine)
    pts = apply_affine(
        inv,
        np.column_stack([xs_r.ravel().astype(np.float64), ys_r.ravel().astype(np.float64)]),
    )
    map_x = pts[:, 0].reshape(rgb_h, rgb_w).astype(np.float32)
    map_y = pts[:, 1].reshape(rgb_h, rgb_w).astype(np.float32)
    wm = cv2.remap(
        info["thermal_mask"].astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    blend = rgb.copy()
    red = np.zeros_like(rgb)
    red[..., 0] = np.clip(warped, 0, 255).astype(np.uint8)
    blend[wm] = (0.4 * rgb[wm] + 0.6 * red[wm]).astype(np.uint8)
    cv2.drawContours(
        blend,
        cv2.findContours(wm.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
        -1,
        (0, 255, 255),
        2,
    )
    face_crop = blend[max(0, y0) : min(blend.shape[0], y1), max(0, x0) : min(blend.shape[1], x1)]
    status = "trustworthy" if quality["trustworthy"] else "NO-GO"
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.imshow(face_crop)
    ax.set_title(
        f"Nakładka termika→RGB  |  RMS={quality['contour_rms_px']:.1f} px  "
        f"IoU={quality['iou']:.2f}  ({status})"
    )
    ax.axis("off")
    fig.savefig(out_dir / "fig_overlay_reg.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # 4) CHROM vs POS — HR okna + fala
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    t_ref = polar.t_s
    hr_ref = polar.hr_bpm
    for col, method in enumerate(["CHROM", "POS"]):
        ax_hr = axes[0, col]
        ax_hr.plot(t_ref, hr_ref, "k-", lw=1.2, label="Polar HR")
        for kind, style in [("plain", "-"), ("gated", "--")]:
            key = ("forehead", method, kind)
            if key not in window_hrs:
                continue
            tc, hrs = window_hrs[key]
            ax_hr.plot(tc, hrs, style, lw=1.0, label=f"{kind}")
        ax_hr.set_ylabel("HR [BPM]")
        ax_hr.set_title(f"{method} — HR w oknach 10 s (czoło)")
        ax_hr.legend(fontsize=8, loc="best")
        ax_hr.grid(True, alpha=0.3)

        ax_w = axes[1, col]
        n_show = min(len(cleaned[("forehead", method, "plain")]), int(15 * fs))
        t = np.arange(n_show) / fs
        for kind, color in [("plain", "tab:blue"), ("gated", "tab:orange")]:
            sig = cleaned[("forehead", method, kind)][:n_show]
            sig = (sig - sig.mean()) / (sig.std() + 1e-9)
            ax_w.plot(t, sig, color=color, lw=0.8, label=kind)
        ax_w.set_xlabel("czas [s]")
        ax_w.set_ylabel("rPPG (znorm.)")
        ax_w.set_title(f"{method} — fragment fali")
        ax_w.legend(fontsize=8)
        ax_w.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_hr_windows.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # 5) Widmo Welcha czoło — plain vs gated, oś BPM
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, method in zip(axes, ["CHROM", "POS"], strict=True):
        for kind, color in [("plain", "tab:blue"), ("gated", "tab:orange")]:
            sig = cleaned[("forehead", method, kind)]
            nperseg = min(len(sig), int(10 * fs))
            freqs, psd = welch(sig, fs=fs, nperseg=nperseg, nfft=max(nperseg, 2048))
            band = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
            bpm = freqs[band] * 60.0
            ax.plot(bpm, psd[band], color=color, lw=1.2, label=kind)
            peak_i = int(np.argmax(psd[band]))
            ax.axvline(bpm[peak_i], color=color, ls=":", alpha=0.7)
            ax.annotate(
                f"{bpm[peak_i]:.0f}",
                (bpm[peak_i], psd[band][peak_i]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color=color,
            )
        # HR z wiersza tabeli
        for r in rows:
            if r["region"] == "forehead" and r["method"] == method:
                ax.axvline(r["hr_ref"], color="k", ls="--", alpha=0.5, label="ref")
                break
        ax.set_xlabel("częstotliwość [BPM]")
        ax.set_title(f"Widmo Welcha — czoło / {method}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("PSD")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_spectrum.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figury → {out_dir}/fig_*.png")


def _write_summary(all_rows: list[dict], out_root: Path) -> None:
    path = out_root / "summary.md"
    lines = [
        "# Podsumowanie demo_run",
        "",
        "| subject | scenario | region | metoda | MAE plain | MAE gated | ΔSNR | "
        "% oktaw plain | gated | RMS px |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in all_rows:
        lines.append(
            f"| {r['subject']} | {r['scenario']} | {r['region']} | {r['method']} | "
            f"{r['mae_plain']:.2f} | {r['mae_gated']:.2f} | {r['delta_snr']:+.2f} | "
            f"{100 * r['octave_frac_plain']:.0f}% | {r['gated_status']} | {r['reg_rms_px']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Zbiorcza → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo RGB+termika z demo_data/ (nazwy DJI).")
    parser.add_argument("--subject", default=None, help="np. subject00 (domyślnie wszystkie)")
    parser.add_argument("--scenario", default=None, help="np. s01_1 (domyślnie wszystkie)")
    parser.add_argument("--recalibrate", action="store_true", help="Przelicz reg_calib.json")
    parser.add_argument(
        "--demo-data",
        type=Path,
        default=DEMO_DATA,
        help="Katalog wejściowy (domyślnie demo_data/)",
    )
    args = parser.parse_args()

    demo_data = args.demo_data.resolve()
    global CALIB_PATH
    CALIB_PATH = demo_data / "reg_calib.json"

    ensure_calibration(demo_data, force=args.recalibrate)

    sessions = discover_demo_sessions(demo_data)
    if args.subject:
        sessions = [s for s in sessions if s[0] == args.subject]
    if args.scenario:
        sessions = [s for s in sessions if s[1] == args.scenario]
    if not sessions:
        raise SystemExit("Brak sesji do przetworzenia.")

    all_rows: list[dict] = []
    for subject, scenario, session_dir in sessions:
        out_dir = DEMO_RESULTS / subject / scenario
        result = run_session(subject, scenario, session_dir, out_dir)
        all_rows.extend(result["rows"])
        print(
            f"\n→ {subject}/{scenario}: RMS={result['quality']['contour_rms_px']:.1f} px "
            f"({result['gated_status']})"
        )
        for r in result["rows"]:
            if r["region"] == "forehead":
                print(
                    f"   {r['method']}: MAE_p={r['mae_plain']:.2f} MAE_g={r['mae_gated']:.2f} "
                    f"ΔSNR={r['delta_snr']:+.2f} oktawy_p={100 * r['octave_frac_plain']:.0f}%"
                )

    _write_summary(all_rows, DEMO_RESULTS)


if __name__ == "__main__":
    main()
