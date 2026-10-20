"""Demo end-to-end: demo_data/ lub data/ → demo_results/ bez ręcznych kroków.

Mapuje samo: *_RGB.MP4/*_V.MP4→RGB, *_T.MP4→termika, dataHR_*.csv/*_HR.csv→HR (bez trimu).
Kalibracja nominalna: <demo-data>/reg_calib.json (auto, jeśli brak).
config.py / EVAL_SUBJECT / subject02 — nietknięte.
"""

from __future__ import annotations

import argparse
import contextlib
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
    HEADLINE_REGION,
    PEAK_SNR_CONF_HIGH_DB,
    PEAK_SNR_CONF_MED_DB,
    POLAR_HR_SKIP_SAMPLES,
    REG_GO_RMS_PX,
)
from src.extract import (  # noqa: E402
    gated_means_per_frame_from_samples,
    mean_rgb_in_mask,
    sample_roi_temps_via_affine as _sample_roi_temps_via_affine,
)
from src.io_layer import (  # noqa: E402
    LoadedRecording,
    Recording,
    hr_video_time_overlap_ok,
    load_polar_hr_csv,
    probe_video,
)
from src.quality_roi import (  # noqa: E402
    peak_snr_db,
    skin_fraction_in_mask,
)
from src.registration import (  # noqa: E402
    _to_gray,
    apply_affine,
    apply_nominal_calibration,
    calibrate_nominal_registration,
    estimate_affine_thermal_to_rgb,
    registration_quality,
    rgb_face_mask,
    warp_thermal_to_rgb,
)
from src.roi import (  # noqa: E402
    make_cropping_detector,
    roi_mask_from_landmarks,
    select_roi_from_landmarks,
)

# OpenCV remap: dst.cols musi być < SHRT_MAX (~32k). Duże ROI (Zoom) → chunki.
_REMAP_CHUNK = 16_384


def sample_roi_temps_via_affine(
    thermal_gray: np.ndarray,
    affine_th_to_rgb: np.ndarray,
    roi_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jak ``src.extract.sample_roi_temps_via_affine``, ale chunkuje remap przy dużym ROI."""
    ys, xs = np.where(np.asarray(roi_mask, dtype=bool))
    if ys.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float64)
    if ys.size <= _REMAP_CHUNK:
        return _sample_roi_temps_via_affine(thermal_gray, affine_th_to_rgb, roi_mask)

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
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float64)

    pts_in = pts_th[inside]
    th = np.asarray(thermal_gray, dtype=np.float32)
    chunks: list[np.ndarray] = []
    for start in range(0, pts_in.shape[0], _REMAP_CHUNK):
        chunk = pts_in[start : start + _REMAP_CHUNK]
        map_x = chunk[:, 0].astype(np.float32).reshape(1, -1)
        map_y = chunk[:, 1].astype(np.float32).reshape(1, -1)
        chunks.append(
            cv2.remap(
                th,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).ravel()
        )
    temps = np.concatenate(chunks).astype(np.float64)
    return ys[inside], xs[inside], temps

DEMO_DATA = ROOT / "demo_data"
DEMO_RESULTS = ROOT / "demo_results"
CALIB_PATH = DEMO_DATA / "reg_calib.json"

REGIONS_EXTRACT = ["forehead", "left_cheek", "right_cheek"]
REGIONS_REPORT = ["forehead", "cheeks"]
FIG_DPI = 160


def _chrom_fn(*, window_bandpass: bool):
    """CHROM legacy (bandpass per-okno) albo fixed (bez BP + Hanning OLA)."""
    if window_bandpass:
        return lambda rgb, fs: methods.chrom(rgb, fs, window_bandpass=True, hann_ola=False)
    return lambda rgb, fs: methods.chrom(rgb, fs, window_bandpass=False, hann_ola=True)


def _method_fns(*, chrom_window_bandpass: bool) -> dict:
    """POS = główny pipeline; CHROM = porównawczy (domyślnie legacy)."""
    return {
        "POS": methods.pos,
        "CHROM": _chrom_fn(window_bandpass=chrom_window_bandpass),
    }


def _variant_name(roi_mask_mode: str, chrom_window_bandpass: bool) -> str:
    chrom_tag = "chrom_legacy" if chrom_window_bandpass else "chrom_fixed"
    return f"{roi_mask_mode}_{chrom_tag}"


@contextlib.contextmanager
def _redirect_stderr_to(log_path: Path):
    """Przekierowuje fd=2 (stderr) do pliku na cały blok; stdout (pasek postępu) zostaje.

    MediaPipe telemetria (portable_clearcut_uploader) leci z wątku w tle na fd=2
    poza jednorazowym create FaceLandmarker — stąd zakres sesji, nie /dev/null.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    log_f = open(log_path, "w", encoding="utf-8", buffering=1)  # noqa: SIM115
    try:
        os.dup2(log_f.fileno(), stderr_fd)
        yield log_path
    except Exception:
        os.dup2(saved_fd, stderr_fd)
        print(f"[error] Native/MediaPipe stderr → {log_path}", flush=True)
        raise
    finally:
        try:
            os.dup2(saved_fd, stderr_fd)
        except OSError:
            pass
        os.close(saved_fd)
        log_f.close()


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
    """*_RGB.MP4/*_V.MP4 → RGB, *_T.MP4 → termika, dataHR_*.csv/*_HR.csv → HR."""
    rgb = _pick_one(
        list(session_dir.glob("*_RGB.MP4"))
        + list(session_dir.glob("*_RGB.mp4"))
        + list(session_dir.glob("*_V.MP4"))
        + list(session_dir.glob("*_V.mp4")),
        "RGB (*_RGB.MP4/*_V.MP4)",
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
    hr = _pick_one(hr_candidates, "HR (dataHR_*.csv/*_HR.csv)", session_dir)
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


def _try_calibrate_from_session(
    session_dir: Path,
    subject: str,
    scenario: str,
    max_frames: int = 90,
) -> dict | None:
    """Kalibracja z pierwszej klatki z twarzą w sesji; None gdy brak detekcji."""
    loaded, _ = load_demo_recording(session_dir, subject, scenario)
    detector = make_cropping_detector()
    for i, (rgb, thermal, _) in enumerate(loaded.synced_pairs(reference="rgb")):
        if i >= max_frames:
            break
        lm = detector(rgb)
        if lm is None:
            continue
        calib = calibrate_nominal_registration(rgb, thermal, lm)
        calib["source"] = f"{subject}/{scenario}#frame{i}"
        return calib
    return None


def _best_sibling_calib(subject: str, exclude_scenario: str) -> tuple[dict, str] | None:
    """Fallback: scena tego badanego z najniższym RMS w reg_quality.json."""
    subj_root = DEMO_RESULTS / subject
    if not subj_root.is_dir():
        return None
    best: tuple[float, Path, str] | None = None
    for scen_dir in sorted(p for p in subj_root.iterdir() if p.is_dir()):
        if scen_dir.name == exclude_scenario:
            continue
        qpath = scen_dir / "reg_quality.json"
        cpath = scen_dir / "reg_calib.json"
        if not (qpath.is_file() and cpath.is_file()):
            continue
        try:
            q = json.loads(qpath.read_text(encoding="utf-8"))
            rms = float(q.get("contour_rms_px", float("inf")))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if best is None or rms < best[0]:
            best = (rms, cpath, scen_dir.name)
    if best is None:
        return None
    calib = json.loads(best[1].read_text(encoding="utf-8"))
    return calib, f"fallback:{best[2]}"


def ensure_scene_calibration(
    demo_data: Path,
    subject: str,
    scenario: str,
    session_dir: Path,
    force: bool = False,
) -> tuple[dict, str]:
    """Kalibracja **per scena** → ``demo_results/<subject>/<scena>/reg_calib.json``.

    Zwraca ``(calib, reg_calib_source)`` gdzie source ∈
    ``self`` | ``fallback:<scena>`` | ``fallback:subject`` | ``fallback:shared``.
    """
    out_calib = DEMO_RESULTS / subject / scenario / "reg_calib.json"
    if out_calib.is_file() and not force:
        calib = json.loads(out_calib.read_text(encoding="utf-8"))
        apply_nominal_calibration(calib)
        src = str(calib.get("calib_source_tag", "self"))
        print(f"Kalibracja {subject}/{scenario} z {out_calib} ({src})")
        return calib, src

    calib = _try_calibrate_from_session(session_dir, subject, scenario)
    source_tag = "self"
    if calib is None:
        sib = _best_sibling_calib(subject, exclude_scenario=scenario)
        if sib is not None:
            calib, source_tag = sib
        else:
            subj_calib = DEMO_RESULTS / subject / "reg_calib.json"
            shared = demo_data / "reg_calib.json"
            if subj_calib.is_file():
                calib = json.loads(subj_calib.read_text(encoding="utf-8"))
                source_tag = "fallback:subject"
            elif shared.is_file():
                calib = json.loads(shared.read_text(encoding="utf-8"))
                source_tag = "fallback:shared"
            else:
                raise RuntimeError(
                    f"Kalibracja {subject}/{scenario}: brak twarzy i brak fallbacku"
                )

    calib = dict(calib)
    calib["calib_source_tag"] = source_tag
    out_calib.parent.mkdir(parents=True, exist_ok=True)
    out_calib.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    apply_nominal_calibration(calib)
    print(
        f"Zapisano kalibrację {subject}/{scenario} → {out_calib} [{source_tag}]\n"
        f"  źródło={calib.get('source')} scale={calib.get('REG_NOMINAL_SCALE')}"
    )
    return calib, source_tag


def ensure_subject_calibration(
    demo_data: Path,
    subject: str,
    force: bool = False,
) -> dict:
    """Kompatybilność: kalibracja z pierwszej sceny badanego (preferuj ensure_scene_*)."""
    sessions = [s for s in discover_demo_sessions(demo_data) if s[0] == subject]
    if not sessions:
        raise FileNotFoundError(f"Brak sesji dla {subject}")
    _, scenario, session_dir = sessions[0]
    calib, _ = ensure_scene_calibration(
        demo_data, subject, scenario, session_dir, force=force
    )
    return calib


def ensure_calibration(
    demo_data: Path = DEMO_DATA,
    force: bool = False,
) -> dict:
    """Kompatybilność wstecz."""
    sessions = discover_demo_sessions(demo_data)
    if not sessions:
        raise FileNotFoundError(f"Brak sesji w {demo_data}")
    return ensure_subject_calibration(demo_data, sessions[0][0], force=force)


# ---------------------------------------------------------------------------
# Potok jednego nagrania
# ---------------------------------------------------------------------------


def run_session(
    subject: str,
    scenario: str,
    session_dir: Path,
    out_dir: Path,
    *,
    roi_mask_mode: str = "bbox",
    chrom_window_bandpass: bool | None = True,
    ab_sweep: bool = False,
    reg_calib_source: str = "self",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stderr_log = out_dir / "run_stderr.log"
    with _redirect_stderr_to(stderr_log):
        return _run_session_body(
            subject,
            scenario,
            session_dir,
            out_dir,
            stderr_log,
            roi_mask_mode=roi_mask_mode,
            chrom_window_bandpass=chrom_window_bandpass,
            ab_sweep=ab_sweep,
            reg_calib_source=reg_calib_source,
        )


def _run_session_body(
    subject: str,
    scenario: str,
    session_dir: Path,
    out_dir: Path,
    stderr_log: Path,
    *,
    roi_mask_mode: str = "bbox",
    chrom_window_bandpass: bool | None = True,
    ab_sweep: bool = False,
    reg_calib_source: str = "self",
) -> dict:
    loaded, hr_path = load_demo_recording(session_dir, subject, scenario)
    fs = loaded.rgb_meta.fps
    n_frames = loaded.rgb_meta.frame_count
    duration_s = float(loaded.rgb_meta.duration_s)
    rgb_path = loaded.recording.rgb_path
    polar = load_polar_hr_csv(
        hr_path,
        skip_samples=POLAR_HR_SKIP_SAMPLES,
        video_path=rgb_path,
        video_duration_s=duration_s,
    )
    if polar is None:
        raise RuntimeError(f"Nie udało się wczytać HR: {hr_path}")

    ok_overlap, overlap_msg = hr_video_time_overlap_ok(polar, duration_s)
    if (not ok_overlap) or ("SANITY_FAIL" in (polar.sync_note or "")):
        msg = overlap_msg if not ok_overlap else polar.sync_note
        print(
            f"\n{'=' * 60}\n"
            f"[SKIP] {subject}/{scenario}: brak sensownego pokrycia HR↔wideo — {msg}",
            flush=True,
        )
        return {"rows": [], "quality": None, "gated_status": "SKIP", "skipped": True}

    print(f"\n{'=' * 60}")
    offset_s = float(polar.sync_offset_s)
    offset_str = f"{offset_s:.3f} s" if np.isfinite(offset_s) else "n/a"
    print(
        f"{subject}/{scenario}  {n_frames} kl. @ {fs:.3f} fps  HR={hr_path.name} "
        f"(n={len(polar.hr_bpm)})  sync={polar.sync_mode}  "
        f"offset(start_wideo−start_HR)={offset_str}"
    )
    if polar.sync_note:
        print(f"  {polar.sync_note}")
    print(f"  native stderr → {stderr_log}")

    # Mediana HR tylko w oknie wideo [0, duration] (nie całe nagranie Polara).
    in_vid = (polar.t_s >= 0.0) & (polar.t_s < duration_s)
    hr_ref = (
        float(np.median(polar.hr_bpm[in_vid]))
        if np.any(in_vid)
        else float(np.median(polar.hr_bpm))
    )

    def _extract_traces(mask_mode: str) -> dict:
        """Jeden przebieg dekodowania: plain/gated RGB + quality (ROI bbox|hull)."""
        detector = make_cropping_detector()
        affine_every = AFFINE_EVERY_DEFAULT
        affine = None
        # Hold-last landmarków: klatki bez detekcji mają valid=False (nie wliczane
        # do progu 50% okna), ale próbka RGB z zamrożonego ROI nadal wchodzi w sygnał.
        last_lm = None
        pending: list[tuple[np.ndarray, np.ndarray]] = []
        n_affine_ok = n_affine_fail = 0
        plain_lists: dict[str, list] = {r: [] for r in REGIONS_EXTRACT}
        rgb_pix: dict[str, list] = {r: [] for r in REGIONS_EXTRACT}
        temps_lists: dict[str, list] = {r: [] for r in REGIONS_EXTRACT}
        skin_lists: dict[str, list] = {r: [] for r in REGIONS_EXTRACT}
        valid_list: list[bool] = []
        frame0_rgb = frame0_th = frame0_lm = None
        reg_info0 = None
        quality = None
        quality_frame_idx: int | None = None

        def _append(rgb, thermal, landmarks, detected: bool) -> None:
            nonlocal affine
            valid_list.append(detected)
            h, w = rgb.shape[:2]
            th_gray = _to_gray(thermal).astype(np.float32)
            for region in REGIONS_EXTRACT:
                roi_mask = roi_mask_from_landmarks(
                    landmarks, region, (h, w), mode=mask_mode
                )
                plain_lists[region].append(mean_rgb_in_mask(rgb, roi_mask))
                if detected:
                    skin_lists[region].append(skin_fraction_in_mask(rgb, roi_mask))
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

        print(
            f"Przebieg synced RGB+termika (roi={mask_mode})...",
            flush=True,
        )
        for i, (rgb, thermal, _) in enumerate(loaded.synced_pairs(reference="rgb")):
            landmarks = detector(rgb)
            if landmarks is not None:
                if affine is None or (i % affine_every == 0):
                    new_aff, info = estimate_affine_thermal_to_rgb(
                        rgb, thermal, landmarks
                    )
                    if new_aff is not None:
                        affine = new_aff.astype(np.float64)
                        n_affine_ok += 1
                        if quality is None:
                            frame0_rgb = rgb.copy()
                            frame0_th = thermal.copy()
                            frame0_lm = landmarks
                            reg_info0 = info
                            quality = registration_quality(
                                affine, info["thermal_mask"], info["rgb_mask"]
                            )
                            quality_frame_idx = i
                            if i > 0:
                                print(
                                    f"  [info] pierwsza udana rejestracja na klatce {i}",
                                    flush=True,
                                )
                    else:
                        n_affine_fail += 1
                        if quality is None:
                            print(f"  [warn] affine (klatka {i}): {info}")
                if last_lm is None and pending:
                    for pr, pt in pending:
                        _append(pr, pt, landmarks, False)
                    pending.clear()
                last_lm = landmarks
                _append(rgb, thermal, landmarks, True)
            else:
                if last_lm is None:
                    pending.append((rgb, thermal))
                else:
                    _append(rgb, thermal, last_lm, False)
            if (i + 1) % 200 == 0 or i + 1 == n_frames:
                print(f"  klatka {i + 1}/{n_frames}", flush=True)

        if quality is None:
            raise RuntimeError(
                f"{subject}/{scenario}: brak twarzy+affine "
                f"(OK/fail={n_affine_ok}/{n_affine_fail})"
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
            * (
                fallback_flags["left_cheek"].astype(float)
                + fallback_flags["right_cheek"].astype(float)
            ).mean()
        )

        def _mean_skin(key: str) -> float:
            vals = [v for v in skin_lists[key] if np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")

        skin_fore = _mean_skin("forehead")
        skin_l, skin_r = _mean_skin("left_cheek"), _mean_skin("right_cheek")
        cheek_vals = [v for v in (skin_l, skin_r) if np.isfinite(v)]
        skin_cheeks = float(np.mean(cheek_vals)) if cheek_vals else float("nan")
        return {
            "plain": plain,
            "gated": gated,
            "valid": valid,
            "quality": quality,
            "reg_info0": reg_info0,
            "affine": affine,
            "frame0_rgb": frame0_rgb,
            "frame0_th": frame0_th,
            "frame0_lm": frame0_lm,
            "quality_frame_idx": quality_frame_idx,
            "n_affine_ok": n_affine_ok,
            "n_affine_fail": n_affine_fail,
            "fb_fore": fb_fore,
            "fb_cheek": fb_cheek,
            "mask_mode": mask_mode,
            "skinfrac_forehead": skin_fore,
            "skinfrac_cheeks": skin_cheeks,
        }

    def _score_traces(ex: dict, *, chrom_bp: bool, save_figs: bool) -> list[dict]:
        plain, gated, valid = ex["plain"], ex["gated"], ex["valid"]
        quality = ex["quality"]
        method_map = _method_fns(chrom_window_bandpass=chrom_bp)
        variant = _variant_name(ex["mask_mode"], chrom_bp)
        trustworthy = bool(quality and quality["trustworthy"])
        gate_label = "trustworthy" if trustworthy else "NO-GO"

        # Peak-SNR (peak-to-median) — tylko informacyjnie; headline = HEADLINE_REGION.
        snr_peak: dict[str, float] = {}
        for region in REGIONS_REPORT:
            sig = methods.pos(plain[region], fs)
            clean = estimate.bandpass_filter(estimate.detrend_signal(sig), fs)
            snr_peak[region] = peak_snr_db(clean, fs)
        snr_hl = snr_peak.get(HEADLINE_REGION, float("nan"))
        if np.isfinite(snr_hl) and snr_hl >= PEAK_SNR_CONF_HIGH_DB:
            conf_info = "high"
        elif np.isfinite(snr_hl) and snr_hl >= PEAK_SNR_CONF_MED_DB:
            conf_info = "med"
        else:
            conf_info = "low"
        region_used = HEADLINE_REGION
        region_reason = f"headline={region_used} (stała reguła; peak-SNR tylko info)"
        print(
            f"  score variant={variant}  RMS={quality['contour_rms_px']:.1f} px  "
            f"gated {gate_label}  calib={reg_calib_source}  (4 równe okna n//4)",
            flush=True,
        )
        print(
            f"  region_used={region_used} confidence(info)={conf_info}  "
            f"skin_f={ex['skinfrac_forehead']:.2f} skin_c={ex['skinfrac_cheeks']:.2f}  "
            f"peakSNR_f={snr_peak['forehead']:.1f} peakSNR_c={snr_peak['cheeks']:.1f} dB",
            flush=True,
        )

        rows_v: list[dict] = []
        cleaned: dict[tuple[str, str, str], np.ndarray] = {}
        window_hrs: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
        pos_scene: dict[str, dict] = {}  # region → hr_scene plain/gated do headline

        for region in REGIONS_REPORT:
            for name, fn in method_map.items():
                sig_p = fn(plain[region], fs)
                sig_g = fn(gated[region], fs)
                clean_p = estimate.bandpass_filter(estimate.detrend_signal(sig_p), fs)
                clean_g = estimate.bandpass_filter(estimate.detrend_signal(sig_g), fs)
                try:
                    hr_scene_p = float(estimate.estimate_hr_welch(clean_p, fs))
                except (ValueError, np.linalg.LinAlgError):
                    hr_scene_p = float("nan")
                try:
                    hr_scene_g = float(estimate.estimate_hr_welch(clean_g, fs))
                except (ValueError, np.linalg.LinAlgError):
                    hr_scene_g = float("nan")
                hr_ref_scene = float(hr_ref)
                vp = validate.validate_against_hr_quarters(
                    sig_p,
                    fs,
                    polar.t_s,
                    polar.hr_bpm,
                    video_duration_s=duration_s,
                    valid=valid,
                )
                vg = validate.validate_against_hr_quarters(
                    sig_g,
                    fs,
                    polar.t_s,
                    polar.hr_bpm,
                    video_duration_s=duration_s,
                    valid=valid,
                )
                fb = ex["fb_fore"] if region == "forehead" else ex["fb_cheek"]
                if name == "POS":
                    pos_scene[region] = {
                        "hr_scene_plain": hr_scene_p,
                        "hr_scene_gated": hr_scene_g,
                        "abs_err_scene_plain": float(abs(hr_scene_p - hr_ref_scene))
                        if np.isfinite(hr_scene_p)
                        else float("nan"),
                        "abs_err_scene_gated": float(abs(hr_scene_g - hr_ref_scene))
                        if np.isfinite(hr_scene_g)
                        else float("nan"),
                    }
                row: dict = {
                    "variant": variant,
                    "roi_mask": ex["mask_mode"],
                    "chrom_window_bandpass": chrom_bp,
                    "subject": subject,
                    "scenario": scenario,
                    "region": region,
                    "method": name,
                    "hr_ref_scene": hr_ref_scene,
                    "hr_scene_plain": hr_scene_p,
                    "hr_scene_gated": hr_scene_g,
                    "abs_err_scene_plain": float(abs(hr_scene_p - hr_ref_scene))
                    if np.isfinite(hr_scene_p)
                    else float("nan"),
                    "abs_err_scene_gated": float(abs(hr_scene_g - hr_ref_scene))
                    if np.isfinite(hr_scene_g)
                    else float("nan"),
                    "hr_ref": hr_ref_scene,
                    "hr_plain": hr_scene_p,
                    "hr_gated": hr_scene_g,
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
                    "usable_t0_s": float(vp["usable_t0_s"]),
                    "usable_t1_s": float(vp["usable_t1_s"]),
                    "reg_rms_px": float(quality["contour_rms_px"]),
                    "reg_iou": float(quality["iou"]),
                    "gated_status": gate_label,
                    "sync_mode": polar.sync_mode,
                    "sync_offset_s": float(polar.sync_offset_s),
                    "ref_suspect": subject == "subject03",
                    "skinfrac_forehead": float(ex["skinfrac_forehead"]),
                    "skinfrac_cheeks": float(ex["skinfrac_cheeks"]),
                    "snr_peak_forehead": float(snr_peak["forehead"]),
                    "snr_peak_cheeks": float(snr_peak["cheeks"]),
                    "region_used": region_used,
                    "confidence": conf_info,
                    "region_select_reason": region_reason,
                    "reg_calib_source": reg_calib_source,
                }
                for i in range(4):
                    ta = float(vp["window_t0_s"][i])
                    tb = float(vp["window_t1_s"][i])
                    row[f"w{i+1}_t0_s"] = ta
                    row[f"w{i+1}_t1_s"] = tb
                    row[f"hr_w{i+1}_plain"] = float(vp["estimated_hr_bpm"][i])
                    row[f"hr_w{i+1}_gated"] = float(vg["estimated_hr_bpm"][i])
                    row[f"hr_w{i+1}_ref"] = float(vp["reference_hr_bpm"][i])
                    row[f"w{i+1}_used"] = bool(vp["window_used"][i])
                rows_v.append(row)
                cleaned[(region, name, "plain")] = clean_p
                cleaned[(region, name, "gated")] = clean_g
                t_centers = 0.5 * (vp["window_t0_s"] + vp["window_t1_s"])
                window_hrs[(region, name, "plain")] = (
                    t_centers,
                    np.asarray(vp["estimated_hr_bpm"], dtype=np.float64),
                )
                window_hrs[(region, name, "gated")] = (
                    t_centers,
                    np.asarray(vg["estimated_hr_bpm"], dtype=np.float64),
                )

        # Headline HR = POS z regionu stałego (HEADLINE_REGION = cheeks).
        used = region_used
        if used in pos_scene:
            hl = pos_scene[used]
        else:
            hl = {
                "hr_scene_plain": float("nan"),
                "hr_scene_gated": float("nan"),
                "abs_err_scene_plain": float("nan"),
                "abs_err_scene_gated": float("nan"),
            }
        for row in rows_v:
            row["hr_headline_plain"] = hl["hr_scene_plain"]
            row["hr_headline_gated"] = hl["hr_scene_gated"]
            row["abs_err_headline_plain"] = hl["abs_err_scene_plain"]
            row["abs_err_headline_gated"] = hl["abs_err_scene_gated"]

        if save_figs:
            _write_tables(out_dir, rows_v, quality, gate_label)
            if (
                ex["frame0_rgb"] is not None
                and ex["reg_info0"] is not None
                and ex["affine"] is not None
            ):
                _save_figures(
                    out_dir,
                    ex["frame0_rgb"],
                    ex["frame0_th"],
                    ex["frame0_lm"],
                    ex["reg_info0"],
                    ex["affine"],
                    quality,
                    cleaned,
                    window_hrs,
                    polar,
                    fs,
                    rows_v,
                )
            (out_dir / "reg_quality.json").write_text(
                json.dumps(
                    {
                        **quality,
                        "method": ex["reg_info0"]["method"] if ex["reg_info0"] else None,
                        "gated_status": gate_label,
                        "affine_ok": ex["n_affine_ok"],
                        "affine_fail": ex["n_affine_fail"],
                        "valid_pct": float(100.0 * valid.mean()),
                        "quality_frame_idx": ex["quality_frame_idx"],
                        "roi_mask": ex["mask_mode"],
                        "chrom_window_bandpass": chrom_bp,
                        "variant": variant,
                        "windowing": "quarters_4",
                        "region_used": region_used,
                        "confidence_info": conf_info,
                        "reg_calib_source": reg_calib_source,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return rows_v

    # Domyślnie: jedna ekstrakcja bbox + CHROM legacy. ab-sweep / hull / fixed — tylko za flagami.
    rows: list[dict] = []
    mask_modes = ("bbox", "hull") if ab_sweep else (roi_mask_mode,)
    dual = bool(ab_sweep or chrom_window_bandpass is None)
    chrom_modes: tuple[bool, ...] = (True, False) if dual else (bool(chrom_window_bandpass),)

    for mask_mode in mask_modes:
        ex = _extract_traces(mask_mode)
        for chrom_bp in chrom_modes:
            save_figs = (mask_mode == mask_modes[-1]) and (chrom_bp == chrom_modes[-1])
            rows.extend(_score_traces(ex, chrom_bp=chrom_bp, save_figs=save_figs))
        if ab_sweep:
            _write_ab_table(out_dir, rows)

    qpath = out_dir / "reg_quality.json"
    quality_dict = (
        json.loads(qpath.read_text(encoding="utf-8"))
        if qpath.is_file()
        else ex["quality"]
    )
    return {
        "rows": rows,
        "quality": quality_dict,
        "gated_status": rows[0]["gated_status"] if rows else "n/a",
    }


def _write_ab_table(out_dir: Path, rows: list[dict]) -> None:
    """Zbiorcza tabela A/B (4 warianty) → metrics_ab.csv + metrics_ab.md.

    Scala z istniejącym plikiem (np. osobny przebieg bbox, potem hull).
    """
    fields = [
        "variant",
        "roi_mask",
        "chrom_window_bandpass",
        "region",
        "method",
        "mae_plain",
        "mae_gated",
        "octave_frac_plain",
        "octave_frac_gated",
        "hr_ref",
        "hr_plain",
        "hr_gated",
        "delta_snr",
        "sync_offset_s",
        "gated_status",
        "reg_rms_px",
    ]
    csv_path = out_dir / "metrics_ab.csv"
    merged: dict[tuple, dict] = {}
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as f:
            for old in csv.DictReader(f):
                key = (old.get("variant"), old.get("region"), old.get("method"))
                # odtwórz typy numeryczne luźno — DictWriter i tak pisze stringi
                merged[key] = old
    for r in rows:
        key = (r.get("variant"), r.get("region"), r.get("method"))
        merged[key] = r
    out_rows = list(merged.values())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    rows = out_rows  # do markdown poniżej
    lines = [
        "# A/B ROI × CHROM",
        "",
        "| variant | region | method | MAE plain | MAE gated | % oktaw plain | "
        "HR plain | sync offset |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        mae_p = float(r["mae_plain"])
        mae_g = float(r["mae_gated"])
        oct_p = float(r["octave_frac_plain"])
        hr_p = float(r["hr_plain"])
        off = float(r["sync_offset_s"])
        lines.append(
            f"| {r['variant']} | {r['region']} | {r['method']} | "
            f"{mae_p:.2f} | {mae_g:.2f} | {100 * oct_p:.0f}% | {hr_p:.1f} | {off:.3f} |"
        )
    (out_dir / "metrics_ab.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"A/B → {csv_path}")


def _write_tables(out_dir: Path, rows: list[dict], quality: dict, gate_label: str) -> None:
    fields = [
        "region",
        "method",
        "hr_ref_scene",
        "hr_scene_plain",
        "hr_scene_gated",
        "abs_err_scene_plain",
        "abs_err_scene_gated",
        "hr_w1_plain",
        "hr_w1_gated",
        "hr_w1_ref",
        "hr_w2_plain",
        "hr_w2_gated",
        "hr_w2_ref",
        "hr_w3_plain",
        "hr_w3_gated",
        "hr_w3_ref",
        "hr_w4_plain",
        "hr_w4_gated",
        "hr_w4_ref",
        "mae_plain",
        "mae_gated",
        "rmse_plain",
        "rmse_gated",
        "octave_frac_plain",
        "octave_frac_gated",
        "delta_snr",
        "n_windows",
        "usable_t0_s",
        "usable_t1_s",
        "fallback_pct",
        "gated_status",
        "reg_rms_px",
        "reg_iou",
        "sync_mode",
        "sync_offset_s",
        "roi_mask",
        "ref_suspect",
        "skinfrac_forehead",
        "skinfrac_cheeks",
        "snr_peak_forehead",
        "snr_peak_cheeks",
        "region_used",
        "confidence",
        "hr_headline_plain",
        "abs_err_headline_plain",
        "reg_calib_source",
    ]
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    off = rows[0].get("sync_offset_s", float("nan")) if rows else float("nan")
    lines = [
        f"# Metryki — RMS={quality['contour_rms_px']:.1f} px, IoU={quality['iou']:.3f}, "
        f"gated **{gate_label}** (próg {REG_GO_RMS_PX:.0f} px), "
        f"sync_offset={off:.3f} s, okna: 4 kwartalne",
        "",
        "| region | metoda | HR_ref | HR_scena plain | |err| plain | "
        "w1 p/r | w2 p/r | w3 p/r | w4 p/r | MAE_p | RMSE_p | %okt_p | "
        "MAE_g | ΔSNR |",
        "|---|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def _wr(i: int) -> str:
            ep = r.get(f"hr_w{i}_plain", float("nan"))
            er = r.get(f"hr_w{i}_ref", float("nan"))
            if not (np.isfinite(ep) and np.isfinite(er)):
                return "—"
            return f"{ep:.0f}/{er:.0f}"

        lines.append(
            f"| {r['region']} | {r['method']} | {r['hr_ref_scene']:.1f} | "
            f"{r['hr_scene_plain']:.1f} | {r['abs_err_scene_plain']:.1f} | "
            f"{_wr(1)} | {_wr(2)} | {_wr(3)} | {_wr(4)} | "
            f"{r['mae_plain']:.2f} | {r['rmse_plain']:.2f} | "
            f"{100 * r['octave_frac_plain']:.0f}% | {r['mae_gated']:.2f} | "
            f"{r['delta_snr']:+.2f} |"
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


_SUMMARY_CSV_FIELDS = [
    "subject",
    "scenario",
    "region",
    "method",
    "hr_ref_scene",
    "hr_scene_plain",
    "abs_err_scene_plain",
    "hr_scene_gated",
    "abs_err_scene_gated",
    "hr_headline_plain",
    "hr_headline_gated",
    "abs_err_headline_plain",
    "abs_err_headline_gated",
    "hr_w1_plain",
    "hr_w2_plain",
    "hr_w3_plain",
    "hr_w4_plain",
    "hr_w1_gated",
    "hr_w2_gated",
    "hr_w3_gated",
    "hr_w4_gated",
    "hr_w1_ref",
    "hr_w2_ref",
    "hr_w3_ref",
    "hr_w4_ref",
    "mae_plain",
    "rmse_plain",
    "octave_frac_plain",
    "mae_gated",
    "rmse_gated",
    "octave_frac_gated",
    "delta_snr",
    "n_windows",
    "reg_rms_px",
    "reg_iou",
    "gated_status",
    "sync_mode",
    "sync_offset_s",
    "usable_t0_s",
    "usable_t1_s",
    "fallback_pct",
    "ref_suspect",
    "skinfrac_forehead",
    "skinfrac_cheeks",
    "snr_peak_forehead",
    "snr_peak_cheeks",
    "region_used",
    "confidence",
    "region_select_reason",
    "reg_calib_source",
]


def _fmt_cell(v) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        if not np.isfinite(v):
            return ""
        if isinstance(v, float) and abs(v - round(v)) < 1e-9:
            return f"{v:.4g}"
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    return "" if v is None else str(v)


def _write_all_subjects_csv(all_rows: list[dict], out_root: Path) -> Path:
    path = out_root / "all_subjects.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in _SUMMARY_CSV_FIELDS})
    print(f"CSV zbiorczy → {path}")
    return path


def _write_summary(all_rows: list[dict], out_root: Path) -> None:
    """Pełny zrzut do analizy + krótki werdykt; równolegle ``all_subjects.csv``."""
    _write_all_subjects_csv(all_rows, out_root)
    path = out_root / "summary.md"

    lines = [
        "# Podsumowanie demo_run — wszyscy badani (pipeline docelowy)",
        "",
        "Default: **bbox** · **POS** + **CHROM legacy** · plain/gated · "
        "**4 równe okna** · headline=**cheeks** (stała) · "
        "kalibracja rejestracji **per scena** · "
        "skinfrac/peak-SNR tylko informacyjnie.",
        "",
        "`ref_suspect=True` → **subject03** (Polar do ~190 BPM — wykluczaj z wniosków zbiorczych).",
        "",
        "Zbiorczy CSV: [`all_subjects.csv`](all_subjects.csv).",
        "",
        "## Pełna tabela (subject × scena × region × metoda)",
        "",
    ]

    # Markdown table header = wszystkie kolumny
    lines.append("| " + " | ".join(_SUMMARY_CSV_FIELDS) + " |")
    lines.append("|" + "|".join(["---"] * len(_SUMMARY_CSV_FIELDS)) + "|")
    for r in all_rows:
        cells = [_fmt_cell(r.get(k)) for k in _SUMMARY_CSV_FIELDS]
        lines.append("| " + " | ".join(cells) + " |")

    # Krótki werdykt na podstawie danych (bez naciągania)
    lines += ["", "## Krótki werdykt", ""]
    by_key: dict[tuple, list] = {}
    for r in all_rows:
        if r.get("ref_suspect"):
            continue
        key = (r["subject"], r["scenario"], r["method"], r["region"])
        by_key.setdefault(key, []).append(r)

    nogo = sorted(
        {
            (r["subject"], r["scenario"])
            for r in all_rows
            if r.get("gated_status") == "NO-GO"
        }
    )
    if nogo:
        lines.append(
            "- **NO-GO rejestracji** (RMS > próg): "
            + ", ".join(f"{s}/{c}" for s, c in nogo)
            + " — gated niewiarygodny."
        )
    else:
        lines.append("- Brak scen NO-GO w tym przebiegu.")

    # POS forehead |err| sceny — dobre / słabe
    pos_fore = [
        r
        for r in all_rows
        if r.get("method") == "POS"
        and r.get("region") == "forehead"
        and not r.get("ref_suspect")
    ]
    good, weak = [], []
    for r in pos_fore:
        err = r.get("abs_err_scene_plain", float("nan"))
        if not isinstance(err, (int, float)) or not np.isfinite(err):
            continue
        tag = f"{r['subject']}/{r['scenario']} (|err|={err:.1f})"
        if err <= 10.0 and r.get("gated_status") != "NO-GO":
            good.append(tag)
        elif err >= 25.0 or r.get("gated_status") == "NO-GO":
            weak.append(tag)
    if good:
        lines.append("- **POS czoło — wiarygodniejsze** (|err|_scena ≤ 10): " + "; ".join(good))
    if weak:
        lines.append("- **POS czoło — słabe** (|err| ≥ 25 lub NO-GO): " + "; ".join(weak))

    # plain vs gated: ile razy gated ma niższe MAE
    n_cmp = n_g_better = 0
    for r in all_rows:
        if r.get("ref_suspect") or r.get("gated_status") == "NO-GO":
            continue
        mp, mg = r.get("mae_plain"), r.get("mae_gated")
        if (
            isinstance(mp, (int, float))
            and isinstance(mg, (int, float))
            and np.isfinite(mp)
            and np.isfinite(mg)
        ):
            n_cmp += 1
            if mg < mp:
                n_g_better += 1
    if n_cmp:
        lines.append(
            f"- **Plain vs gated (MAE okienkowe, bez subject03/NO-GO):** "
            f"gated lepsze w {n_g_better}/{n_cmp} wierszach "
            f"({100 * n_g_better / n_cmp:.0f}%) — nie uznawać gated za domyślny zwycięzcę."
        )
    scene_meta = {}
    for r in all_rows:
        key = (r["subject"], r["scenario"])
        if key not in scene_meta:
            scene_meta[key] = r
    n_scenes = sum(1 for r in scene_meta.values() if not r.get("ref_suspect"))
    n_nogo = sum(
        1
        for r in scene_meta.values()
        if (not r.get("ref_suspect")) and r.get("gated_status") == "NO-GO"
    )
    n_fallback = sum(
        1
        for r in scene_meta.values()
        if str(r.get("reg_calib_source", "")).startswith("fallback")
    )
    lines.append(
        f"- Headline zawsze **cheeks**; czoło raportowane osobno (inset włosów "
        f"`FOREHEAD_TOP_INSET_FRAC`)."
    )
    lines.append(
        f"- Sceny **NO-GO** rejestracji (bez subject03): {n_nogo}/{n_scenes}; "
        f"kalibracja fallback: {n_fallback} "
        f"(`demo_results/<subject>/<scena>/reg_calib.json`)."
    )
    lines.append(
        "- `confidence` / `snr_peak_*` / `skinfrac_*` — kolumny informacyjne, "
        "nie bramkują danych."
    )
    lines.append(
        "- Sceny z ruchem (s2/s4) oraz approach (s5) zwykle gorsze niż rest/drone."
    )
    lines.append("")
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
    parser.add_argument(
        "--roi-mask",
        choices=("bbox", "hull"),
        default="bbox",
        help="Maska ROI (default: bbox = pipeline docelowy; hull = opcja eksperym.)",
    )
    parser.add_argument(
        "--chrom-mode",
        choices=("legacy", "fixed", "both"),
        default="legacy",
        help="CHROM (default: legacy = bandpass per-okno; fixed/both = opcja eksperym.)",
    )
    parser.add_argument(
        "--ab-sweep",
        action="store_true",
        help="Opcja eksperym.: A/B bbox/hull × chrom legacy/fixed (nie default)",
    )
    args = parser.parse_args()

    demo_data = args.demo_data.resolve()
    global CALIB_PATH
    CALIB_PATH = demo_data / "reg_calib.json"  # legacy path; nie używany per-subject

    DEMO_RESULTS.mkdir(parents=True, exist_ok=True)

    sessions = discover_demo_sessions(demo_data)
    if args.subject:
        sessions = [s for s in sessions if s[0] == args.subject]
    if args.scenario:
        sessions = [s for s in sessions if s[1] == args.scenario]
    if not sessions:
        raise SystemExit("Brak sesji do przetworzenia.")

    if args.chrom_mode == "both":
        chrom_bp: bool | None = None
    else:
        chrom_bp = args.chrom_mode == "legacy"
    all_rows: list[dict] = []
    for subject, scenario, session_dir in sessions:
        out_dir = DEMO_RESULTS / subject / scenario
        calib_log = out_dir / "calib_stderr.log"
        with _redirect_stderr_to(calib_log):
            _, calib_src = ensure_scene_calibration(
                demo_data,
                subject,
                scenario,
                session_dir,
                force=args.recalibrate,
            )
        result = run_session(
            subject,
            scenario,
            session_dir,
            out_dir,
            roi_mask_mode=args.roi_mask,
            chrom_window_bandpass=chrom_bp,
            ab_sweep=args.ab_sweep,
            reg_calib_source=calib_src,
        )
        if result.get("skipped"):
            print(f"\n→ {subject}/{scenario}: SKIP (sync sanity)")
            continue
        all_rows.extend(result["rows"])
        q = result["quality"] or {}
        sample = result["rows"][0] if result["rows"] else {}
        print(
            f"\n→ {subject}/{scenario}: RMS={q.get('contour_rms_px', float('nan')):.1f} px "
            f"({result['gated_status']}) region_used={sample.get('region_used')} "
            f"confidence={sample.get('confidence')}"
        )
        for r in result["rows"]:
            if r["method"] != "POS":
                continue
            err = r.get("abs_err_scene_plain", float("nan"))
            mark = "*" if r["region"] == r.get("region_used") else " "
            print(
                f"  {mark}{r['region']}/{r['method']}: |err|={err:.1f} "
                f"MAE={r['mae_plain']:.2f} "
                f"HR={r.get('hr_scene_plain', float('nan')):.0f}/"
                f"ref={r.get('hr_ref_scene', float('nan')):.0f}"
            )
        if sample:
            print(
                f"   headline(POS/{sample.get('region_used')}): "
                f"|err|={sample.get('abs_err_headline_plain', float('nan')):.1f} "
                f"HR={sample.get('hr_headline_plain', float('nan')):.0f}"
            )

    _write_summary(all_rows, DEMO_RESULTS)


if __name__ == "__main__":
    main()
