"""Potok end-to-end: data/<subject>/<scena>/ → <out>/<subject>/<scena>/ + agregat.

Sesje DJI: *_RGB.MP4/*_V.MP4 → RGB, *_T.MP4 → termika, dataHR_*.csv/*_HR.csv → Polar HR.
Per scena: synchronizacja creation_time↔Polar, kalibracja rejestracji per scena,
przebieg po klatkach (plain + gated), POS/CHROM × czoło/policzki, 4 równe okna,
tabele i figury. Agregat (``all_subjects.csv``, ``summary.md``) składany ze wszystkich
``metrics.csv`` w katalogu wyjściowym.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from src import estimate, methods, validate
from src.config import (
    AFFINE_EVERY_BY_SCENARIO,
    AFFINE_EVERY_DEFAULT,
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    POLAR_HR_SKIP_SAMPLES,
    REF_SUSPECT_SUBJECTS,
    REG_GO_RMS_PX,
)
from src.extract import (
    gated_means_per_frame_from_samples,
    mean_rgb_in_mask,
    sample_roi_temps_via_affine as _sample_roi_temps_via_affine,
)
from src.io_layer import (
    LoadedRecording,
    Recording,
    hr_video_time_overlap_ok,
    load_polar_hr_csv,
    probe_video,
)
from src.registration import (
    _to_gray,
    apply_affine,
    apply_nominal_calibration,
    calibrate_nominal_registration,
    estimate_affine_thermal_to_rgb,
    invert_affine,
    registration_quality,
    warp_thermal_to_rgb,
)
from src.roi import (
    make_cropping_detector,
    roi_mask_from_landmarks,
    select_roi_from_landmarks,
)

# OpenCV remap: dst.cols musi być < SHRT_MAX (~32k). Duże ROI (Zoom) → chunki.
_REMAP_CHUNK = 16_384

REGIONS_EXTRACT = ["forehead", "left_cheek", "right_cheek"]
REGIONS_REPORT = ["forehead", "cheeks"]
REGION_LABELS = {"forehead": "czoło", "cheeks": "policzki"}
METHODS = {"POS": methods.pos, "CHROM": methods.chrom}
WAVE_PREVIEW_SEC = 15.0
FIG_DPI = 160
HR_PLOT_CLIP_BPM = (40.0, 200.0)
HR_PLOT_PAD_BPM = 5.0

SCENE_CSV_FIELDS = [
    "subject",
    "scenario",
    "region",
    "method",
    "hr_ref_scene",
    "hr_scene_plain",
    "abs_err_scene_plain",
    "hr_scene_gated",
    "abs_err_scene_gated",
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
    "reg_rms_first_px",
    "reg_good_frac",
    "reg_n_attempts",
    "gated_status",
    "sync_mode",
    "sync_offset_s",
    "usable_t0_s",
    "usable_t1_s",
    "fallback_pct",
    "ref_suspect",
    "reg_calib_source",
]


def is_ref_suspect(subject: str) -> bool:
    """Badany z nietypową referencją HR (``REF_SUSPECT_SUBJECTS``)."""
    return subject in REF_SUSPECT_SUBJECTS


def _quiet_native_env() -> None:
    """Wycisza logi C++ MediaPipe/TF przed pierwszym użyciem detektora."""
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")


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
# Odkrywanie nagrań DJI
# ---------------------------------------------------------------------------


def _pick_one(paths: list[Path], label: str, session: Path) -> Path:
    if not paths:
        raise FileNotFoundError(f"Brak {label} w {session}")
    return sorted(paths)[0]


def discover_sessions(data_dir: Path) -> list[tuple[str, str, Path]]:
    """Lista (subject, scenario, session_dir) z <data_dir>/<subject>/<scenario>/."""
    out: list[tuple[str, str, Path]] = []
    if not data_dir.is_dir():
        return out
    for subject_dir in sorted(
        p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("subject")
    ):
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


def load_session(
    session_dir: Path, subject: str, scenario: str
) -> tuple[LoadedRecording, Path]:
    """Nagranie RGB+termika sesji i ścieżka Polar HR."""
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
# Kalibracja rejestracji per scena
# ---------------------------------------------------------------------------


def _try_calibrate_from_session(
    session_dir: Path,
    subject: str,
    scenario: str,
    max_frames: int = 90,
) -> dict | None:
    """Kalibracja z pierwszej klatki z twarzą w sesji; None gdy brak detekcji."""
    loaded, _ = load_session(session_dir, subject, scenario)
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


def _best_sibling_calib(
    out_root: Path, subject: str, exclude_scenario: str
) -> tuple[dict, str] | None:
    """Fallback: scena tego badanego z najniższym RMS w reg_quality.json."""
    subj_root = out_root / subject
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
    data_dir: Path,
    out_root: Path,
    subject: str,
    scenario: str,
    session_dir: Path,
    force: bool = False,
) -> tuple[dict, str]:
    """Kalibracja **per scena** → ``<out>/<subject>/<scena>/reg_calib.json``.

    Zwraca ``(calib, reg_calib_source)`` gdzie source ∈
    ``self`` | ``fallback:<scena>`` | ``fallback:subject`` | ``fallback:shared``.
    """
    out_calib = out_root / subject / scenario / "reg_calib.json"
    if out_calib.is_file() and not force:
        calib = json.loads(out_calib.read_text(encoding="utf-8"))
        apply_nominal_calibration(calib)
        src = str(calib.get("calib_source_tag", "self"))
        print(f"Kalibracja {subject}/{scenario} z {out_calib} ({src})")
        return calib, src

    calib = _try_calibrate_from_session(session_dir, subject, scenario)
    source_tag = "self"
    if calib is None:
        sib = _best_sibling_calib(out_root, subject, exclude_scenario=scenario)
        if sib is not None:
            calib, source_tag = sib
        else:
            subj_calib = out_root / subject / "reg_calib.json"
            shared = data_dir / "reg_calib.json"
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


# ---------------------------------------------------------------------------
# Potok jednego nagrania
# ---------------------------------------------------------------------------


def run_session(
    subject: str,
    scenario: str,
    session_dir: Path,
    out_dir: Path,
    *,
    reg_calib_source: str = "self",
) -> dict:
    """Jedna scena; natywny stderr (MediaPipe) → ``<out_dir>/run_stderr.log``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stderr_log = out_dir / "run_stderr.log"
    with _redirect_stderr_to(stderr_log):
        return _run_session_body(
            subject,
            scenario,
            session_dir,
            out_dir,
            stderr_log,
            reg_calib_source=reg_calib_source,
        )


def _accept_affine(
    candidate_rms: float,
    current_rms: float | None,
    current_trusted: bool,
    threshold: float = REG_GO_RMS_PX,
) -> bool:
    """Czy nowy affine ma zastąpić bieżący.

    RMS ≤ progu → przyjmij zawsze. Powyżej progu → przyjmij tylko, gdy jeszcze nie ma
    affine poniżej progu i kandydat jest lepszy od bieżącego (albo bieżącego brak).
    NaN RMS traktowany jak +∞.
    """
    cand = candidate_rms if np.isfinite(candidate_rms) else np.inf
    if cand <= threshold:
        return True
    if current_rms is None:
        return True
    if current_trusted:
        return False
    cur = current_rms if np.isfinite(current_rms) else np.inf
    return cand < cur


def _scene_registration_quality(
    attempts_rms: list[float],
    attempts_iou: list[float],
    first_quality: dict,
    threshold: float = REG_GO_RMS_PX,
) -> dict:
    """Jakość rejestracji całej sceny: mediany RMS/IoU udanych prób + udział dobrych."""
    rms = np.asarray(attempts_rms, dtype=np.float64)
    iou = np.asarray(attempts_iou, dtype=np.float64)
    rms_ok = rms[np.isfinite(rms)]
    iou_ok = iou[np.isfinite(iou)]
    rms_med = float(np.median(rms_ok)) if rms_ok.size else float("nan")
    return {
        "contour_rms_px": rms_med,
        "iou": float(np.median(iou_ok)) if iou_ok.size else float("nan"),
        "rms_first_px": float(first_quality["contour_rms_px"]),
        "iou_first": float(first_quality["iou"]),
        "good_frac": float(np.mean(np.isfinite(rms) & (rms <= threshold))) if rms.size else 0.0,
        "n_attempts": int(rms.size),
        "trustworthy": bool(np.isfinite(rms_med) and rms_med <= threshold),
        "go_threshold_px": float(threshold),
    }


def _extract_traces(loaded: LoadedRecording, subject: str, scenario: str) -> dict:
    """Jeden przebieg dekodowania: plain/gated RGB (ROI bbox) + jakość rejestracji sceny."""
    n_frames = loaded.rgb_meta.frame_count
    detector = make_cropping_detector()
    affine_every = AFFINE_EVERY_BY_SCENARIO.get(scenario, AFFINE_EVERY_DEFAULT)
    affine = None
    affine_rms: float | None = None
    affine_trusted = False
    # Hold-last landmarków: klatki bez detekcji mają valid=False (nie wliczane
    # do progu 50% okna), ale próbka RGB z zamrożonego ROI nadal wchodzi w sygnał.
    last_lm = None
    pending: list[tuple[np.ndarray, np.ndarray]] = []
    n_affine_ok = n_affine_fail = n_affine_rejected = 0
    attempts_rms: list[float] = []
    attempts_iou: list[float] = []
    plain_lists: dict[str, list] = {r: [] for r in REGIONS_EXTRACT}
    rgb_pix: dict[str, list] = {r: [] for r in REGIONS_EXTRACT}
    temps_lists: dict[str, list] = {r: [] for r in REGIONS_EXTRACT}
    valid_list: list[bool] = []
    frame0_rgb = frame0_th = frame0_lm = None
    reg_info0 = None
    first_quality = None
    quality_frame_idx: int | None = None

    def _append(rgb, thermal, landmarks, detected: bool) -> None:
        valid_list.append(detected)
        h, w = rgb.shape[:2]
        th_gray = _to_gray(thermal).astype(np.float32)
        for region in REGIONS_EXTRACT:
            roi_mask = roi_mask_from_landmarks(landmarks, region, (h, w))
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

    print(
        f"Przebieg synced RGB+termika (roi=bbox, affine co {affine_every} kl.)...",
        flush=True,
    )
    for i, (rgb, thermal, _) in enumerate(loaded.synced_pairs(reference="rgb")):
        landmarks = detector(rgb)
        if landmarks is not None:
            if affine is None or (i % affine_every == 0):
                new_aff, info = estimate_affine_thermal_to_rgb(rgb, thermal, landmarks)
                if new_aff is not None:
                    new_aff = new_aff.astype(np.float64)
                    n_affine_ok += 1
                    q = registration_quality(new_aff, info["thermal_mask"], info["rgb_mask"])
                    attempts_rms.append(float(q["contour_rms_px"]))
                    attempts_iou.append(float(q["iou"]))
                    if first_quality is None:
                        frame0_rgb = rgb.copy()
                        frame0_th = thermal.copy()
                        frame0_lm = landmarks
                        reg_info0 = info
                        first_quality = q
                        quality_frame_idx = i
                        if i > 0:
                            print(
                                f"  [info] pierwsza udana rejestracja na klatce {i}",
                                flush=True,
                            )
                    cand_rms = float(q["contour_rms_px"])
                    if _accept_affine(cand_rms, affine_rms, affine_trusted):
                        affine = new_aff
                        affine_rms = cand_rms
                        affine_trusted = bool(np.isfinite(cand_rms) and cand_rms <= REG_GO_RMS_PX)
                    else:
                        n_affine_rejected += 1
                else:
                    n_affine_fail += 1
                    if first_quality is None:
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

    if first_quality is None:
        raise RuntimeError(
            f"{subject}/{scenario}: brak twarzy+affine "
            f"(OK/fail={n_affine_ok}/{n_affine_fail})"
        )
    quality = _scene_registration_quality(attempts_rms, attempts_iou, first_quality)

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
    return {
        "plain": plain,
        "gated": gated,
        "valid": valid,
        "quality": quality,
        "first_quality": first_quality,
        "reg_info0": reg_info0,
        "frame0_rgb": frame0_rgb,
        "frame0_th": frame0_th,
        "frame0_lm": frame0_lm,
        "quality_frame_idx": quality_frame_idx,
        "affine_every": affine_every,
        "n_affine_ok": n_affine_ok,
        "n_affine_fail": n_affine_fail,
        "n_affine_rejected": n_affine_rejected,
        "fallback_pct": {"forehead": fb_fore, "cheeks": fb_cheek},
    }


def _scene_hr(clean: np.ndarray, fs: float) -> float:
    """HR całej sceny (Welch); NaN przy błędzie."""
    try:
        return float(estimate.estimate_hr_welch(clean, fs))
    except (ValueError, np.linalg.LinAlgError):
        return float("nan")


def _abs_err(hr: float, ref: float) -> float:
    return float(abs(hr - ref)) if np.isfinite(hr) else float("nan")


def _score_traces(
    ex: dict,
    *,
    subject: str,
    scenario: str,
    fs: float,
    duration_s: float,
    polar,
    hr_ref: float,
    reg_calib_source: str,
) -> tuple[list[dict], dict, dict]:
    """Metryki: region × metoda × plain/gated w 4 równych oknach.

    Zwraca ``(rows, cleaned, window_hrs)``; ``cleaned``/``window_hrs`` do figur.
    """
    plain, gated, valid = ex["plain"], ex["gated"], ex["valid"]
    quality = ex["quality"]
    gate_label = "trustworthy" if quality["trustworthy"] else "NO-GO"
    print(
        f"  RMS mediana={quality['contour_rms_px']:.1f} px "
        f"(1. próba {quality['rms_first_px']:.1f} px, dobre "
        f"{100 * quality['good_frac']:.0f}% z {quality['n_attempts']})  "
        f"gated {gate_label}  calib={reg_calib_source}  (4 równe okna n//4)",
        flush=True,
    )

    rows: list[dict] = []
    cleaned: dict[tuple[str, str, str], np.ndarray] = {}
    window_hrs: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for region in REGIONS_REPORT:
        for name, fn in METHODS.items():
            sig_p = fn(plain[region], fs)
            sig_g = fn(gated[region], fs)
            clean_p = estimate.bandpass_filter(estimate.detrend_signal(sig_p), fs)
            clean_g = estimate.bandpass_filter(estimate.detrend_signal(sig_g), fs)
            hr_scene_p = _scene_hr(clean_p, fs)
            hr_scene_g = _scene_hr(clean_g, fs)
            vp = validate.validate_against_hr_quarters(
                sig_p, fs, polar.t_s, polar.hr_bpm, video_duration_s=duration_s, valid=valid
            )
            vg = validate.validate_against_hr_quarters(
                sig_g, fs, polar.t_s, polar.hr_bpm, video_duration_s=duration_s, valid=valid
            )
            row: dict = {
                "subject": subject,
                "scenario": scenario,
                "region": region,
                "method": name,
                "hr_ref_scene": float(hr_ref),
                "hr_scene_plain": hr_scene_p,
                "hr_scene_gated": hr_scene_g,
                "abs_err_scene_plain": _abs_err(hr_scene_p, hr_ref),
                "abs_err_scene_gated": _abs_err(hr_scene_g, hr_ref),
                "snr_plain": float(vp["snr_mean"]),
                "snr_gated": float(vg["snr_mean"]),
                "delta_snr": float(vg["snr_mean"] - vp["snr_mean"]),
                "mae_plain": float(vp["mae_bpm"]),
                "mae_gated": float(vg["mae_bpm"]),
                "rmse_plain": float(vp["rmse_bpm"]),
                "rmse_gated": float(vg["rmse_bpm"]),
                "octave_frac_plain": float(vp["octave_error_frac"]),
                "octave_frac_gated": float(vg["octave_error_frac"]),
                "fallback_pct": float(ex["fallback_pct"][region]),
                "n_windows": int(vp["n_windows_used"]),
                "usable_t0_s": float(vp["usable_t0_s"]),
                "usable_t1_s": float(vp["usable_t1_s"]),
                "reg_rms_px": float(quality["contour_rms_px"]),
                "reg_iou": float(quality["iou"]),
                "reg_rms_first_px": float(quality["rms_first_px"]),
                "reg_good_frac": float(quality["good_frac"]),
                "reg_n_attempts": int(quality["n_attempts"]),
                "gated_status": gate_label,
                "sync_mode": polar.sync_mode,
                "sync_offset_s": float(polar.sync_offset_s),
                "ref_suspect": is_ref_suspect(subject),
                "reg_calib_source": reg_calib_source,
            }
            for i in range(4):
                row[f"hr_w{i+1}_plain"] = float(vp["estimated_hr_bpm"][i])
                row[f"hr_w{i+1}_gated"] = float(vg["estimated_hr_bpm"][i])
                row[f"hr_w{i+1}_ref"] = float(vp["reference_hr_bpm"][i])
            rows.append(row)
            cleaned[(region, name, "plain")] = clean_p
            cleaned[(region, name, "gated")] = clean_g
            t0 = np.asarray(vp["window_t0_s"], dtype=np.float64)
            t1 = np.asarray(vp["window_t1_s"], dtype=np.float64)
            window_hrs[(region, name, "plain")] = (
                t0,
                t1,
                np.asarray(vp["estimated_hr_bpm"], dtype=np.float64),
            )
            window_hrs[(region, name, "gated")] = (
                t0,
                t1,
                np.asarray(vg["estimated_hr_bpm"], dtype=np.float64),
            )
    return rows, cleaned, window_hrs


def _write_reg_quality(out_dir: Path, ex: dict, reg_calib_source: str) -> dict:
    """Jakość rejestracji sceny → ``reg_quality.json``."""
    quality = ex["quality"]
    gate_label = "trustworthy" if quality["trustworthy"] else "NO-GO"
    payload = {
        **quality,
        "first_frame": ex["first_quality"],
        "method": ex["reg_info0"]["method"] if ex["reg_info0"] else None,
        "gated_status": gate_label,
        "affine_every": ex["affine_every"],
        "affine_ok": ex["n_affine_ok"],
        "affine_fail": ex["n_affine_fail"],
        "affine_rejected": ex["n_affine_rejected"],
        "valid_pct": float(100.0 * ex["valid"].mean()),
        "quality_frame_idx": ex["quality_frame_idx"],
        "windowing": "quarters_4",
        "reg_calib_source": reg_calib_source,
    }
    (out_dir / "reg_quality.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _run_session_body(
    subject: str,
    scenario: str,
    session_dir: Path,
    out_dir: Path,
    stderr_log: Path,
    *,
    reg_calib_source: str = "self",
) -> dict:
    loaded, hr_path = load_session(session_dir, subject, scenario)
    fs = loaded.rgb_meta.fps
    n_frames = loaded.rgb_meta.frame_count
    duration_s = float(loaded.rgb_meta.duration_s)
    polar = load_polar_hr_csv(
        hr_path,
        skip_samples=POLAR_HR_SKIP_SAMPLES,
        video_path=loaded.recording.rgb_path,
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
        # Nieaktualne metryki z poprzedniego przebiegu nie mogą trafić do agregatu.
        (out_dir / "metrics.csv").unlink(missing_ok=True)
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

    ex = _extract_traces(loaded, subject, scenario)
    rows, cleaned, window_hrs = _score_traces(
        ex,
        subject=subject,
        scenario=scenario,
        fs=fs,
        duration_s=duration_s,
        polar=polar,
        hr_ref=hr_ref,
        reg_calib_source=reg_calib_source,
    )
    gate_label = rows[0]["gated_status"]
    _write_tables(out_dir, rows, ex["quality"], gate_label)
    if ex["frame0_rgb"] is not None and ex["reg_info0"] is not None:
        _save_figures(
            out_dir,
            ex["frame0_rgb"],
            ex["frame0_th"],
            ex["frame0_lm"],
            ex["reg_info0"],
            ex["first_quality"],
            ex["quality"],
            cleaned,
            window_hrs,
            polar,
            fs,
            duration_s,
            rows,
        )
    quality_dict = _write_reg_quality(out_dir, ex, reg_calib_source)
    return {"rows": rows, "quality": quality_dict, "gated_status": gate_label}


# ---------------------------------------------------------------------------
# Tabele per scena
# ---------------------------------------------------------------------------


def _write_tables(out_dir: Path, rows: list[dict], quality: dict, gate_label: str) -> None:
    """``metrics.csv`` (pełne kolumny, źródło agregatu) + ``metrics.md``."""
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCENE_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    off = rows[0].get("sync_offset_s", float("nan")) if rows else float("nan")
    lines = [
        f"# Metryki — RMS mediana={quality['contour_rms_px']:.1f} px, "
        f"IoU mediana={quality['iou']:.3f}, "
        f"gated **{gate_label}** (próg {REG_GO_RMS_PX:.0f} px), "
        f"sync_offset={off:.3f} s, okna: 4 równe",
        "",
        f"Rejestracja: RMS 1. próby={quality['rms_first_px']:.1f} px, "
        f"dobre próby (RMS ≤ {REG_GO_RMS_PX:.0f} px)={100 * quality['good_frac']:.0f}% "
        f"z {quality['n_attempts']}.",
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


# ---------------------------------------------------------------------------
# Figury per scena
# ---------------------------------------------------------------------------


def _face_crop_bounds(landmarks: np.ndarray) -> tuple[int, int, int, int]:
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    return (
        int(xs.min()) - 40,
        int(ys.min()) - 40,
        int(xs.max()) + 40,
        int(ys.max()) + 40,
    )


def _crop(img: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    return img[max(0, y0) : min(img.shape[0], y1), max(0, x0) : min(img.shape[1], x1)]


def _fig_rgb_roi(out_dir: Path, rgb: np.ndarray, landmarks: np.ndarray) -> None:
    vis = rgb.copy()
    for region, color in [
        ("forehead", (0, 255, 0)),
        ("left_cheek", (255, 200, 0)),
        ("right_cheek", (255, 200, 0)),
    ]:
        y0, x0, y1, x1 = (int(v) for v in select_roi_from_landmarks(landmarks, region))
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 3)
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.imshow(_crop(vis, _face_crop_bounds(landmarks)))
    ax.set_title("RGB — ROI (czoło / policzki)")
    ax.axis("off")
    fig.savefig(out_dir / "fig_rgb_roi.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _fig_thermal_seg(out_dir: Path, thermal: np.ndarray, info: dict) -> None:
    """IR + segment twarzy + maska perfuzji (względna jasność w segmencie)."""
    th_gray = _to_gray(thermal)
    th_vis = cv2.cvtColor(th_gray, cv2.COLOR_GRAY2RGB)
    seg_mask = info["thermal_mask"]
    overlay = th_vis.copy()
    overlay[seg_mask] = (0.45 * th_vis[seg_mask] + np.array([180, 40, 40])).astype(np.uint8)
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


def _fig_overlay_reg(
    out_dir: Path,
    rgb: np.ndarray,
    thermal: np.ndarray,
    landmarks: np.ndarray,
    info: dict,
    first_quality: dict,
    scene_quality: dict,
) -> None:
    """Termika warpowana na RGB + kontur maski — wszystko z tej samej (pierwszej) klatki.

    Affine bierzemy z ``info["affine"]`` (ta sama próba co obraz, maska i RMS klatki).
    """
    affine = np.asarray(info["affine"], dtype=np.float64)
    warped = warp_thermal_to_rgb(thermal, affine, rgb.shape[:2])
    rgb_h, rgb_w = rgb.shape[:2]
    ys_r, xs_r = np.mgrid[0:rgb_h, 0:rgb_w]
    pts = apply_affine(
        invert_affine(affine),
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
    status = "trustworthy" if scene_quality["trustworthy"] else "NO-GO"
    fig, ax = plt.subplots(figsize=(6, 7.5))
    ax.imshow(_crop(blend, _face_crop_bounds(landmarks)))
    ax.set_title(
        f"Nakładka termika→RGB (1. udana klatka)\n"
        f"RMS klatki={first_quality['contour_rms_px']:.1f} px  "
        f"IoU={first_quality['iou']:.2f}\n"
        f"scena: mediana RMS={scene_quality['contour_rms_px']:.1f} px, "
        f"dobre próby {100 * scene_quality['good_frac']:.0f}% "
        f"z {scene_quality['n_attempts']}  ({status})",
        fontsize=10,
    )
    ax.axis("off")
    fig.savefig(out_dir / "fig_overlay_reg.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_window_steps(ax, t0, t1, hrs, style: str, color: str, label: str) -> None:
    """HR per okno jako poziomy odcinek nad [t0, t1) + znacznik w środku okna."""
    first = True
    for a, b, h in zip(t0, t1, hrs, strict=True):
        if not np.isfinite(h):
            continue
        ax.plot([a, b], [h, h], style, color=color, lw=1.6, label=label if first else None)
        ax.plot([0.5 * (a + b)], [h], "o", color=color, ms=4)
        first = False


def _hr_ylim(polar_hr: np.ndarray, window_hrs: dict) -> tuple[float, float]:
    """Wspólny zakres Y paneli HR: min/max Polar i estymat okienkowych, przycięty do pasma."""
    vals = [np.asarray(polar_hr, dtype=float)]
    vals += [np.asarray(hrs, dtype=float) for _, _, hrs in window_hrs.values()]
    allv = np.concatenate([v.ravel() for v in vals])
    allv = allv[np.isfinite(allv)]
    lo_clip, hi_clip = HR_PLOT_CLIP_BPM
    if allv.size == 0:
        return lo_clip, hi_clip
    lo = max(lo_clip, float(allv.min()) - HR_PLOT_PAD_BPM)
    hi = min(hi_clip, float(allv.max()) + HR_PLOT_PAD_BPM)
    if hi <= lo:
        return lo_clip, hi_clip
    return lo, hi


def _fig_hr_windows(
    out_dir: Path,
    cleaned: dict,
    window_hrs: dict,
    polar,
    fs: float,
    duration_s: float,
) -> None:
    """Wiersze: HR czoło, HR policzki, fala czoło, fala policzki; kolumny: POS, CHROM.

    Panele HR mają oś czasu całego nagrania; panele fali własną oś (pierwsze 15 s).
    """
    method_names = list(METHODS)
    fig, axes = plt.subplots(4, len(method_names), figsize=(12, 13))
    in_vid = (polar.t_s >= 0.0) & (polar.t_s <= duration_s)
    hr_ylim = _hr_ylim(polar.hr_bpm[in_vid], window_hrs)
    for col, method in enumerate(method_names):
        for row_i, region in enumerate(REGIONS_REPORT):
            ax = axes[row_i, col]
            ax.plot(polar.t_s[in_vid], polar.hr_bpm[in_vid], "k-", lw=1.0, label="Polar HR")
            for kind, style, color in [
                ("plain", "-", "tab:blue"),
                ("gated", "--", "tab:orange"),
            ]:
                key = (region, method, kind)
                if key in window_hrs:
                    t0, t1, hrs = window_hrs[key]
                    _plot_window_steps(ax, t0, t1, hrs, style, color, kind)
            ax.set_xlim(0.0, duration_s)
            ax.set_ylim(*hr_ylim)
            ax.set_xlabel("czas [s]")
            ax.set_ylabel("HR [BPM]")
            ax.set_title(f"{method} — HR w 4 równych oknach ({REGION_LABELS[region]})")
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

        for row_i, region in enumerate(REGIONS_REPORT, start=len(REGIONS_REPORT)):
            ax = axes[row_i, col]
            sig_len = len(cleaned[(region, method, "plain")])
            n_show = min(sig_len, int(WAVE_PREVIEW_SEC * fs))
            t = np.arange(n_show) / fs
            for kind, color in [("plain", "tab:blue"), ("gated", "tab:orange")]:
                sig = cleaned[(region, method, kind)][:n_show]
                sig = (sig - sig.mean()) / (sig.std() + 1e-9)
                ax.plot(t, sig, color=color, lw=0.8, label=kind)
            ax.set_xlim(0.0, t[-1] if n_show else WAVE_PREVIEW_SEC)
            ax.set_xlabel("czas [s]")
            ax.set_ylabel("rPPG (znorm.)")
            ax.set_title(
                f"{method} — fragment fali, pierwsze {n_show / fs:.0f} s "
                f"({REGION_LABELS[region]})"
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_hr_windows.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _fig_spectrum(out_dir: Path, cleaned: dict, fs: float, rows: list[dict]) -> None:
    """Widmo Welcha (oś BPM): wiersze czoło/policzki, kolumny POS/CHROM; plain vs gated."""
    method_names = list(METHODS)
    fig, axes = plt.subplots(
        len(REGIONS_REPORT), len(method_names), figsize=(11, 8), squeeze=False
    )
    for row_i, region in enumerate(REGIONS_REPORT):
        for col, method in enumerate(method_names):
            ax = axes[row_i, col]
            for kind, color in [("plain", "tab:blue"), ("gated", "tab:orange")]:
                sig = cleaned[(region, method, kind)]
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
            for r in rows:
                if r["region"] == region and r["method"] == method:
                    ax.axvline(r["hr_ref_scene"], color="k", ls="--", alpha=0.5, label="ref")
                    break
            ax.set_xlabel("częstotliwość [BPM]")
            ax.set_ylabel("PSD")
            ax.set_title(f"Widmo Welcha — {REGION_LABELS[region]} / {method}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_spectrum.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _save_figures(
    out_dir: Path,
    rgb: np.ndarray,
    thermal: np.ndarray,
    landmarks: np.ndarray,
    info: dict,
    first_quality: dict,
    scene_quality: dict,
    cleaned: dict,
    window_hrs: dict,
    polar,
    fs: float,
    duration_s: float,
    rows: list[dict],
) -> None:
    _fig_rgb_roi(out_dir, rgb, landmarks)
    _fig_thermal_seg(out_dir, thermal, info)
    _fig_overlay_reg(out_dir, rgb, thermal, landmarks, info, first_quality, scene_quality)
    _fig_hr_windows(out_dir, cleaned, window_hrs, polar, fs, duration_s)
    _fig_spectrum(out_dir, cleaned, fs, rows)
    print(f"Figury → {out_dir}/fig_*.png")


# ---------------------------------------------------------------------------
# Agregat z dysku
# ---------------------------------------------------------------------------


def _parse_cell(value: str):
    """Komórka CSV → bool / float / str (puste → NaN)."""
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return value


def collect_scene_rows(out_root: Path) -> list[dict]:
    """Wszystkie wiersze z ``<out>/<subject>/<scena>/metrics.csv`` (posortowane)."""
    rows: list[dict] = []
    for path in sorted(out_root.glob("subject*/*/metrics.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                if not raw.get("subject"):
                    continue
                row = {k: _parse_cell(v) for k, v in raw.items()}
                row["subject"] = raw["subject"]
                row["scenario"] = raw["scenario"]
                rows.append(row)
    return rows


def _fmt_cell(v) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        if not np.isfinite(v):
            return ""
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    return "" if v is None else str(v)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and bool(np.isfinite(v))


def write_all_subjects_csv(all_rows: list[dict], out_root: Path) -> Path:
    path = out_root / "all_subjects.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCENE_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in SCENE_CSV_FIELDS})
    print(f"CSV zbiorczy → {path}")
    return path


def _verdict_lines(all_rows: list[dict]) -> list[str]:
    """Krótki werdykt liczony wyłącznie z wierszy tabeli."""
    suspects = ", ".join(REF_SUSPECT_SUBJECTS) or "—"
    lines = [
        "",
        "## Krótki werdykt",
        "",
        f"Statystyki poniżej liczone bez badanych z `ref_suspect=True` ({suspects}).",
        "",
    ]
    nogo = sorted(
        {
            (r["subject"], r["scenario"])
            for r in all_rows
            if r.get("gated_status") == "NO-GO" and not r.get("ref_suspect")
        }
    )
    if nogo:
        lines.append(
            f"- **NO-GO rejestracji** (mediana RMS sceny > {REG_GO_RMS_PX:.0f} px): "
            + ", ".join(f"{s}/{c}" for s, c in nogo)
            + " — gated niewiarygodny."
        )
    else:
        lines.append("- Brak scen NO-GO.")

    for region in REGIONS_REPORT:
        good, weak = [], []
        for r in all_rows:
            if r.get("method") != "POS" or r.get("region") != region or r.get("ref_suspect"):
                continue
            err = r.get("abs_err_scene_plain")
            if not _is_num(err):
                continue
            tag = f"{r['subject']}/{r['scenario']} (|err|={err:.1f})"
            if err <= 10.0:
                good.append(tag)
            elif err >= 25.0:
                weak.append(tag)
        label = REGION_LABELS[region]
        if good:
            lines.append(f"- **POS {label} — wiarygodniejsze** (|err|_scena ≤ 10): " + "; ".join(good))
        if weak:
            lines.append(f"- **POS {label} — słabe** (|err|_scena ≥ 25): " + "; ".join(weak))

    n_cmp = n_g_better = 0
    for r in all_rows:
        if r.get("ref_suspect") or r.get("gated_status") == "NO-GO":
            continue
        mp, mg = r.get("mae_plain"), r.get("mae_gated")
        if _is_num(mp) and _is_num(mg):
            n_cmp += 1
            if mg < mp:
                n_g_better += 1
    if n_cmp:
        lines.append(
            f"- **Plain vs gated (MAE okienkowe, bez scen NO-GO):** "
            f"gated lepsze w {n_g_better}/{n_cmp} wierszach "
            f"({100 * n_g_better / n_cmp:.0f}%) — nie uznawać gated za domyślny zwycięzcę."
        )

    scene_meta: dict[tuple[str, str], dict] = {}
    for r in all_rows:
        scene_meta.setdefault((r["subject"], r["scenario"]), r)
    n_scenes = sum(1 for r in scene_meta.values() if not r.get("ref_suspect"))
    n_nogo = sum(
        1
        for r in scene_meta.values()
        if (not r.get("ref_suspect")) and r.get("gated_status") == "NO-GO"
    )
    n_fallback = sum(
        1
        for r in scene_meta.values()
        if (not r.get("ref_suspect"))
        and str(r.get("reg_calib_source", "")).startswith("fallback")
    )
    lines.append(
        f"- Sceny **NO-GO** rejestracji: {n_nogo}/{n_scenes}; "
        f"kalibracja fallback: {n_fallback} (`<out>/<subject>/<scena>/reg_calib.json`)."
    )
    lines.append("- Sceny z ruchem (s2/s4) oraz approach (s5) zwykle gorsze niż rest/drone.")
    lines.append("")
    return lines


def write_summary(out_root: Path) -> list[dict]:
    """``all_subjects.csv`` + ``summary.md`` ze wszystkich ``metrics.csv`` w ``out_root``."""
    all_rows = collect_scene_rows(out_root)
    write_all_subjects_csv(all_rows, out_root)
    subjects = sorted({r["subject"] for r in all_rows})
    lines = [
        "# Podsumowanie potoku — wszyscy przetworzeni badani",
        "",
        "Default: **bbox** · **POS** + **CHROM** · plain/gated · czoło + policzki · "
        "**4 równe okna** · kalibracja rejestracji **per scena**.",
        "",
        f"Badani w agregacie ({len(subjects)}): {', '.join(subjects) if subjects else '—'}.",
        "",
        f"`ref_suspect=True` ({', '.join(REF_SUSPECT_SUBJECTS) or '—'}): flaga oznacza "
        "nietypowo wysokie HR referencji Polar. Wiersze tych badanych zostają w tabeli; "
        "werdykt liczy statystyki bez nich.",
        "",
        f"Rejestracja: `reg_rms_px`/`reg_iou` = mediana z prób odświeżenia affine w scenie, "
        f"`reg_rms_first_px` = pierwsza próba, `reg_good_frac` = udział prób z RMS ≤ "
        f"{REG_GO_RMS_PX:.0f} px; `gated_status=trustworthy`, gdy mediana ≤ progu.",
        "",
        "Zbiorczy CSV: [`all_subjects.csv`](all_subjects.csv).",
        "",
        "## Pełna tabela (subject × scena × region × metoda)",
        "",
        "| " + " | ".join(SCENE_CSV_FIELDS) + " |",
        "|" + "|".join(["---"] * len(SCENE_CSV_FIELDS)) + "|",
    ]
    for r in all_rows:
        lines.append("| " + " | ".join(_fmt_cell(r.get(k)) for k in SCENE_CSV_FIELDS) + " |")
    lines += _verdict_lines(all_rows)
    path = out_root / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Zbiorcza → {path}")
    return all_rows


# ---------------------------------------------------------------------------
# Orkiestracja
# ---------------------------------------------------------------------------


def _print_session_result(subject: str, scenario: str, result: dict) -> None:
    q = result["quality"] or {}
    print(
        f"\n→ {subject}/{scenario}: RMS mediana={q.get('contour_rms_px', float('nan')):.1f} px "
        f"(1. próba {q.get('rms_first_px', float('nan')):.1f} px, "
        f"dobre {100 * q.get('good_frac', float('nan')):.0f}%) ({result['gated_status']})"
    )
    for r in result["rows"]:
        print(
            f"  {r['region']}/{r['method']}: |err| plain={r['abs_err_scene_plain']:.1f} "
            f"gated={r['abs_err_scene_gated']:.1f}  "
            f"MAE plain={r['mae_plain']:.2f} gated={r['mae_gated']:.2f}  "
            f"HR={r['hr_scene_plain']:.0f}/ref={r['hr_ref_scene']:.0f}"
        )


def run_pipeline(
    data_dir: Path,
    out_root: Path,
    *,
    subject: str | None = None,
    recalibrate: bool = False,
) -> list[dict]:
    """Przetwarza sesje z ``data_dir`` (opcjonalnie jednego badanego) i odświeża agregat."""
    _quiet_native_env()
    matplotlib.use("Agg")
    data_dir = data_dir.resolve()
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sessions = discover_sessions(data_dir)
    if subject:
        sessions = [s for s in sessions if s[0] == subject]
    if not sessions:
        raise SystemExit(f"Brak sesji do przetworzenia w {data_dir}.")

    for subj, scenario, session_dir in sessions:
        out_dir = out_root / subj / scenario
        with _redirect_stderr_to(out_dir / "calib_stderr.log"):
            _, calib_src = ensure_scene_calibration(
                data_dir, out_root, subj, scenario, session_dir, force=recalibrate
            )
        result = run_session(subj, scenario, session_dir, out_dir, reg_calib_source=calib_src)
        if result.get("skipped"):
            print(f"\n→ {subj}/{scenario}: SKIP (sync sanity)")
            continue
        _print_session_result(subj, scenario, result)

    return write_summary(out_root)
