"""Potok end-to-end: data/<subject>/<scena>/ -> <out>/<subject>/<scena>/ + agregat.

Sesje DJI: *_RGB.MP4/*_V.MP4 (RGB), *_T.MP4 (termika), dataHR_*.csv/*_HR.csv (Polar HR).
Per scena: synchronizacja creation_time z Polarem, kalibracja rejestracji per scena,
przebieg po klatkach (plain + gated), POS/CHROM × czoło/policzki, 4 równe okna.
Tabele i agregat: ``src.report``; figury: ``src.figures``.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np

from src import estimate, validate
from src.config import (
    AFFINE_EVERY_BY_SCENARIO,
    AFFINE_EVERY_DEFAULT,
    POLAR_HR_SKIP_SAMPLES,
    REF_SUSPECT_SUBJECTS,
    REG_GO_RMS_PX,
    REGIONS_EXTRACT,
    REGIONS_REPORT,
)
from src.extract import (
    gated_means_per_frame_from_samples,
    mean_rgb_in_mask,
    sample_roi_temps_via_affine,
)
from src.figures import SignalKey, WindowHrs, save_registration_figures, save_scene_figures
from src.io_layer import (
    LoadedRecording,
    PolarHrSeries,
    Recording,
    hr_video_time_overlap_ok,
    load_polar_hr_csv,
    probe_video,
)
from src.methods import PIPELINE_METHODS
from src.registration import (
    SceneRegistrationQuality,
    apply_nominal_calibration,
    calibrate_nominal_registration,
    estimate_affine_thermal_to_rgb,
    registration_quality,
    scale_is_plausible,
    to_gray,
)
from src.report import write_scene_tables, write_summary
from src.roi import make_cropping_detector, roi_mask_from_landmarks


def is_ref_suspect(subject: str) -> bool:
    """Badany z nietypową referencją HR (``REF_SUSPECT_SUBJECTS``)."""
    return subject in REF_SUSPECT_SUBJECTS


def _quiet_native_env() -> None:
    """Wycisza logi C++ MediaPipe/TF przed pierwszym użyciem detektora."""
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")


@contextlib.contextmanager
def _redirect_stderr_to(log_path: Path) -> Iterator[Path]:
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
        with contextlib.suppress(OSError):
            os.dup2(saved_fd, stderr_fd)
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


def resolve_dji_streams(session_dir: Path) -> tuple[Path, Path, Path | None]:
    """Ścieżki (RGB, termika, HR): *_RGB.MP4/*_V.MP4, *_T.MP4, dataHR_*.csv/*_HR.csv.

    Brak pliku HR -> ``None`` (wymagany dopiero w trybie z metrykami).
    """
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
    hr = (
        _pick_one(hr_candidates, "HR (dataHR_*.csv/*_HR.csv)", session_dir)
        if hr_candidates
        else None
    )
    return rgb, thermal, hr


def load_session(
    session_dir: Path, subject: str, scenario: str
) -> tuple[LoadedRecording, Path | None]:
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
    """Kalibracja z pierwszej klatki z twarzą, dla której segment w termice ma skalę głowy.

    Klatki odrzucone przez ``calibrate_nominal_registration`` (``ValueError``) są pomijane
    w limicie ``max_frames``; brak udanej -> None (caller sięga po fallbacki).
    """
    loaded, _ = load_session(session_dir, subject, scenario)
    detector = make_cropping_detector()
    n_rejected = 0
    last_error = ""
    for i, (rgb, thermal, _) in enumerate(loaded.synced_pairs(reference="rgb")):
        if i >= max_frames:
            break
        lm = detector(rgb)
        if lm is None:
            continue
        try:
            calib = calibrate_nominal_registration(rgb, thermal, lm)
        except ValueError as exc:
            n_rejected += 1
            last_error = str(exc)
            continue
        if n_rejected:
            print(f"  kalibracja {subject}/{scenario}: odrzucono {n_rejected} kl. ({last_error})")
        calib["source"] = f"{subject}/{scenario}#frame{i}"
        return calib
    print(
        f"  kalibracja {subject}/{scenario}: brak udanej klatki w {max_frames} "
        f"(odrzucono {n_rejected} kl. z twarzą; ostatni błąd: {last_error or '—'})"
    )
    return None


def _best_sibling_calib(
    out_root: Path, subject: str, exclude_scenario: str
) -> tuple[dict, str] | None:
    """Fallback: scena tego badanego z najniższym RMS i skalą w ``REG_SCALE_RANGE``."""
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
            scale = float(json.loads(cpath.read_text(encoding="utf-8"))["REG_NOMINAL_SCALE"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if not scale_is_plausible(scale):
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
    """Kalibracja **per scena** -> ``<out>/<subject>/<scena>/reg_calib.json``.

    Zwraca ``(calib, reg_calib_source)``, gdzie source to jedno z:
    ``self`` | ``fallback:<scena>`` | ``fallback:subject`` | ``fallback:shared``.
    """
    out_calib = out_root / subject / scenario / "reg_calib.json"
    if out_calib.is_file() and not force:
        calib = json.loads(out_calib.read_text(encoding="utf-8"))
        apply_nominal_calibration(calib)
        src = str(calib.get("calib_source_tag", "self"))
        print(f"Kalibracja {subject}/{scenario} z {out_calib} ({src})")
        return calib, src

    found = _try_calibrate_from_session(session_dir, subject, scenario)
    source_tag = "self"
    if found is None:
        sib = _best_sibling_calib(out_root, subject, exclude_scenario=scenario)
        if sib is not None:
            found, source_tag = sib
        else:
            subj_calib = out_root / subject / "reg_calib.json"
            shared = data_dir / "reg_calib.json"
            if subj_calib.is_file():
                found = json.loads(subj_calib.read_text(encoding="utf-8"))
                source_tag = "fallback:subject"
            elif shared.is_file():
                found = json.loads(shared.read_text(encoding="utf-8"))
                source_tag = "fallback:shared"
            else:
                raise RuntimeError(
                    f"Kalibracja {subject}/{scenario}: brak twarzy i brak fallbacku"
                )

    calib = dict(cast(dict, found))
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
# Jakość rejestracji sceny
# ---------------------------------------------------------------------------


def _accept_affine(
    candidate_rms: float,
    current_rms: float | None,
    current_trusted: bool,
    threshold: float = REG_GO_RMS_PX,
) -> bool:
    """Czy nowy affine ma zastąpić bieżący.

    RMS ≤ progu -> przyjmij zawsze. Powyżej progu -> przyjmij tylko, gdy jeszcze nie ma
    affine poniżej progu i kandydat jest lepszy od bieżącego (albo bieżącego brak).
    NaN RMS traktowany jak +inf.
    """
    cand = candidate_rms if np.isfinite(candidate_rms) else np.inf
    if cand <= threshold:
        return True
    if current_rms is None:
        return True
    if current_trusted:
        return False
    cur = current_rms if np.isfinite(current_rms) else np.inf
    return bool(cand < cur)


def _scene_registration_quality(
    attempts_rms: list[float],
    attempts_iou: list[float],
    first_quality: dict,
    threshold: float = REG_GO_RMS_PX,
) -> SceneRegistrationQuality:
    """Jakość rejestracji całej sceny: mediany RMS/IoU udanych prób + udział dobrych.

    Próby z odrzuconym affine mają RMS = inf: wchodzą do mediany RMS (ciągną ją w górę),
    NaN nie wchodzi. Trustworthy tylko przy skończonej medianie ≤ progu.
    """
    rms = np.asarray(attempts_rms, dtype=np.float64)
    iou = np.asarray(attempts_iou, dtype=np.float64)
    rms_ok = rms[~np.isnan(rms)]
    iou_ok = iou[np.isfinite(iou)]
    rms_med = float(np.median(rms_ok)) if rms_ok.size else float("nan")
    return SceneRegistrationQuality(
        contour_rms_px=rms_med,
        iou=float(np.median(iou_ok)) if iou_ok.size else float("nan"),
        rms_first_px=float(first_quality["contour_rms_px"]),
        iou_first=float(first_quality["iou"]),
        good_frac=float(np.mean(np.isfinite(rms) & (rms <= threshold))) if rms.size else 0.0,
        n_attempts=int(rms.size),
        trustworthy=bool(np.isfinite(rms_med) and rms_med <= threshold),
        go_threshold_px=float(threshold),
    )


# ---------------------------------------------------------------------------
# Przebieg po klatkach: ślady plain/gated
# ---------------------------------------------------------------------------


def _per_region() -> dict[str, list]:
    return {r: [] for r in REGIONS_EXTRACT}


@dataclass
class _TraceState:
    """Stan jednego przebiegu po klatkach sceny."""

    affine_every: int
    affine: np.ndarray | None = None
    affine_rms: float | None = None
    affine_trusted: bool = False
    # Hold-last landmarków: klatki bez detekcji mają valid=False (nie wliczane
    # do progu 50% okna), ale próbka RGB z zamrożonego ROI nadal wchodzi w sygnał.
    last_lm: np.ndarray | None = None
    pending: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    n_affine_ok: int = 0
    n_affine_fail: int = 0
    n_affine_rejected: int = 0
    n_affine_implausible: int = 0
    attempts_rms: list[float] = field(default_factory=list)
    attempts_iou: list[float] = field(default_factory=list)
    plain_lists: dict[str, list] = field(default_factory=_per_region)
    rgb_pix: dict[str, list] = field(default_factory=_per_region)
    temps_lists: dict[str, list] = field(default_factory=_per_region)
    valid_list: list[bool] = field(default_factory=list)
    frame0_rgb: np.ndarray | None = None
    frame0_th: np.ndarray | None = None
    frame0_lm: np.ndarray | None = None
    reg_info0: dict | None = None
    first_quality: dict | None = None
    quality_frame_idx: int | None = None


@dataclass(frozen=True)
class SceneTraces:
    """Ślady RGB sceny (plain/gated per region raportu) + jakość rejestracji."""

    plain: dict[str, np.ndarray]
    gated: dict[str, np.ndarray]
    valid: np.ndarray
    quality: SceneRegistrationQuality
    first_quality: dict
    reg_info0: dict
    frame0_rgb: np.ndarray
    frame0_th: np.ndarray
    frame0_lm: np.ndarray
    quality_frame_idx: int
    affine_every: int
    n_affine_ok: int
    n_affine_fail: int
    n_affine_rejected: int
    n_affine_implausible: int
    fallback_pct: dict[str, float]


def _append_sample(
    st: _TraceState,
    rgb: np.ndarray,
    thermal: np.ndarray,
    landmarks: np.ndarray,
    detected: bool,
) -> None:
    """Dopisuje próbkę klatki: średnia RGB w ROI (plain) i próbki termiki przez affine."""
    st.valid_list.append(detected)
    h, w = rgb.shape[:2]
    th_gray = to_gray(thermal).astype(np.float32)
    for region in REGIONS_EXTRACT:
        roi_mask = roi_mask_from_landmarks(landmarks, region, (h, w))
        st.plain_lists[region].append(mean_rgb_in_mask(rgb, roi_mask))
        if st.affine is None:
            st.rgb_pix[region].append(None)
            st.temps_lists[region].append(None)
            continue
        ys, xs, tvals = sample_roi_temps_via_affine(th_gray, st.affine, roi_mask)
        if tvals.size == 0:
            st.rgb_pix[region].append(None)
            st.temps_lists[region].append(None)
        else:
            st.rgb_pix[region].append(rgb[ys, xs].astype(np.float64))
            st.temps_lists[region].append(tvals)


def _refresh_affine(
    st: _TraceState,
    i: int,
    rgb: np.ndarray,
    thermal: np.ndarray,
    landmarks: np.ndarray,
) -> None:
    """Próba nowego affine: jakość każdej próby, podmiana wg ``_accept_affine``.

    Próba z nieprawdopodobnym affine (``affine_is_plausible``: skala/anizotropia)
    liczy się jako nieudana (RMS = inf)
    i nigdy nie zastępuje bieżącego affine.
    """
    new_aff, info = estimate_affine_thermal_to_rgb(rgb, thermal, landmarks)
    if new_aff is None:
        st.n_affine_fail += 1
        if st.first_quality is None:
            print(f"  [warn] affine (klatka {i}): {info}")
        return
    info = cast(dict, info)
    new_aff = new_aff.astype(np.float64)
    st.n_affine_ok += 1
    q = registration_quality(new_aff, info["thermal_mask"], info["rgb_mask"])
    scale_ok = bool(info["scale_ok"])
    q["scale_ok"] = scale_ok
    q["seg_scale"] = float(info["seg_scale"])
    q["affine_scale"] = float(info["affine_scale"])
    q["affine_aniso"] = float(info["affine_aniso"])
    if scale_ok:
        st.attempts_rms.append(float(q["contour_rms_px"]))
        st.attempts_iou.append(float(q["iou"]))
    else:
        st.n_affine_implausible += 1
        st.attempts_rms.append(np.inf)
        st.attempts_iou.append(np.nan)
    if st.first_quality is None:
        st.frame0_rgb = rgb.copy()
        st.frame0_th = thermal.copy()
        st.frame0_lm = landmarks
        st.reg_info0 = info
        st.first_quality = q
        st.quality_frame_idx = i
        if i > 0:
            print(f"  [info] pierwsza udana rejestracja na klatce {i}", flush=True)
    if not scale_ok:
        return
    cand_rms = float(q["contour_rms_px"])
    if _accept_affine(cand_rms, st.affine_rms, st.affine_trusted):
        st.affine = new_aff
        st.affine_rms = cand_rms
        st.affine_trusted = bool(np.isfinite(cand_rms) and cand_rms <= REG_GO_RMS_PX)
    else:
        st.n_affine_rejected += 1


def _consume_frame(
    st: _TraceState,
    i: int,
    rgb: np.ndarray,
    thermal: np.ndarray,
    landmarks: np.ndarray | None,
) -> None:
    """Jedna klatka: ewentualne odświeżenie affine, potem próbka (hold-last przy braku twarzy).

    Klatki przed pierwszą detekcją czekają w ``pending`` i dostają landmarki pierwszej
    detekcji (valid=False).
    """
    if landmarks is None:
        if st.last_lm is None:
            st.pending.append((rgb, thermal))
        else:
            _append_sample(st, rgb, thermal, st.last_lm, False)
        return
    if st.first_quality is None or i % st.affine_every == 0:
        _refresh_affine(st, i, rgb, thermal, landmarks)
    if st.last_lm is None and st.pending:
        for pr, pt in st.pending:
            _append_sample(st, pr, pt, landmarks, False)
        st.pending.clear()
    st.last_lm = landmarks
    _append_sample(st, rgb, thermal, landmarks, True)


def _finalize_traces(st: _TraceState, subject: str, scenario: str) -> SceneTraces:
    """Bramkowanie per klatka, łączenie policzków i jakość sceny."""
    if (
        st.first_quality is None
        or st.reg_info0 is None
        or st.frame0_rgb is None
        or st.frame0_th is None
        or st.frame0_lm is None
        or st.quality_frame_idx is None
    ):
        raise RuntimeError(
            f"{subject}/{scenario}: brak twarzy+affine "
            f"(OK/fail={st.n_affine_ok}/{st.n_affine_fail})"
        )
    quality = _scene_registration_quality(st.attempts_rms, st.attempts_iou, st.first_quality)

    valid = np.array(st.valid_list, dtype=bool)
    plain_ex = {r: np.asarray(v, dtype=np.float64) for r, v in st.plain_lists.items()}
    gated_ex: dict[str, np.ndarray] = {}
    fallback_flags: dict[str, np.ndarray] = {}
    for region in REGIONS_EXTRACT:
        g, fb = gated_means_per_frame_from_samples(
            st.rgb_pix[region], st.temps_lists[region], plain_ex[region]
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
    return SceneTraces(
        plain=plain,
        gated=gated,
        valid=valid,
        quality=quality,
        first_quality=st.first_quality,
        reg_info0=st.reg_info0,
        frame0_rgb=st.frame0_rgb,
        frame0_th=st.frame0_th,
        frame0_lm=st.frame0_lm,
        quality_frame_idx=st.quality_frame_idx,
        affine_every=st.affine_every,
        n_affine_ok=st.n_affine_ok,
        n_affine_fail=st.n_affine_fail,
        n_affine_rejected=st.n_affine_rejected,
        n_affine_implausible=st.n_affine_implausible,
        fallback_pct={"forehead": fb_fore, "cheeks": fb_cheek},
    )


def _extract_traces(loaded: LoadedRecording, subject: str, scenario: str) -> SceneTraces:
    """Jeden przebieg dekodowania: plain/gated RGB (ROI bbox) + jakość rejestracji sceny."""
    n_frames = loaded.rgb_meta.frame_count
    detector = make_cropping_detector()
    st = _TraceState(affine_every=AFFINE_EVERY_BY_SCENARIO.get(scenario, AFFINE_EVERY_DEFAULT))
    print(
        f"Przebieg synced RGB+termika (roi=bbox, affine co {st.affine_every} kl.)...",
        flush=True,
    )
    for i, (rgb, thermal, _) in enumerate(loaded.synced_pairs(reference="rgb")):
        _consume_frame(st, i, rgb, thermal, detector(rgb))
        if (i + 1) % 200 == 0 or i + 1 == n_frames:
            print(f"  klatka {i + 1}/{n_frames}", flush=True)
    return _finalize_traces(st, subject, scenario)


# ---------------------------------------------------------------------------
# Metryki sceny
# ---------------------------------------------------------------------------


def _scene_hr(clean: np.ndarray, fs: float) -> float:
    """HR całej sceny (Welch); NaN przy błędzie."""
    try:
        return float(estimate.estimate_hr_welch(clean, fs))
    except (ValueError, np.linalg.LinAlgError):
        return float("nan")


def _abs_err(hr: float, ref: float) -> float:
    return float(abs(hr - ref)) if np.isfinite(hr) else float("nan")


def _score_traces(
    ex: SceneTraces,
    *,
    subject: str,
    scenario: str,
    fs: float,
    duration_s: float,
    polar: PolarHrSeries,
    hr_ref: float,
    reg_calib_source: str,
) -> tuple[list[dict], dict[SignalKey, np.ndarray], dict[SignalKey, WindowHrs]]:
    """Metryki: region × metoda × plain/gated w 4 równych oknach.

    Zwraca ``(rows, cleaned, window_hrs)``; ``cleaned``/``window_hrs`` do figur.
    """
    plain, gated, valid = ex.plain, ex.gated, ex.valid
    quality = ex.quality
    gate_label = quality.gate_label
    print(
        f"  RMS mediana={quality.contour_rms_px:.1f} px "
        f"(1. próba {quality.rms_first_px:.1f} px, dobre "
        f"{100 * quality.good_frac:.0f}% z {quality.n_attempts})  "
        f"gated {gate_label}  calib={reg_calib_source}  (4 równe okna n//4)",
        flush=True,
    )

    rows: list[dict] = []
    cleaned: dict[SignalKey, np.ndarray] = {}
    window_hrs: dict[SignalKey, WindowHrs] = {}

    for region in REGIONS_REPORT:
        for name, fn in PIPELINE_METHODS.items():
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
                "fallback_pct": float(ex.fallback_pct[region]),
                "n_windows": int(vp["n_windows_used"]),
                "usable_t0_s": float(vp["usable_t0_s"]),
                "usable_t1_s": float(vp["usable_t1_s"]),
                "reg_rms_px": float(quality.contour_rms_px),
                "reg_iou": float(quality.iou),
                "reg_rms_first_px": float(quality.rms_first_px),
                "reg_good_frac": float(quality.good_frac),
                "reg_n_attempts": int(quality.n_attempts),
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


def _write_reg_quality(out_dir: Path, ex: SceneTraces, reg_calib_source: str) -> dict:
    """Jakość rejestracji sceny -> ``reg_quality.json``."""
    payload = {
        **asdict(ex.quality),
        "first_frame": ex.first_quality,
        "method": ex.reg_info0["method"] if ex.reg_info0 else None,
        "gated_status": ex.quality.gate_label,
        "affine_every": ex.affine_every,
        "affine_ok": ex.n_affine_ok,
        "affine_fail": ex.n_affine_fail,
        "affine_rejected": ex.n_affine_rejected,
        "affine_implausible": ex.n_affine_implausible,
        "valid_pct": float(100.0 * ex.valid.mean()),
        "quality_frame_idx": ex.quality_frame_idx,
        "windowing": "quarters_4",
        "reg_calib_source": reg_calib_source,
    }
    (out_dir / "reg_quality.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Potok jednej sceny
# ---------------------------------------------------------------------------


def run_session(
    subject: str,
    scenario: str,
    session_dir: Path,
    out_dir: Path,
    *,
    reg_calib_source: str = "self",
    figures_only: bool = False,
) -> dict:
    """Jedna scena; natywny stderr (MediaPipe) -> ``<out_dir>/run_stderr.log``."""
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
            figures_only=figures_only,
        )


def _run_registration_only(
    loaded: LoadedRecording,
    subject: str,
    scenario: str,
    out_dir: Path,
    stderr_log: Path,
    reg_calib_source: str,
) -> dict:
    """Tryb ``figures_only``: rejestracja + ``reg_quality.json`` + figury rejestracji, bez HR."""
    print(f"\n{'=' * 60}")
    print(
        f"{subject}/{scenario}  {loaded.rgb_meta.frame_count} kl. @ "
        f"{loaded.rgb_meta.fps:.3f} fps  [figures-only: bez HR i metryk]"
    )
    print(f"  native stderr → {stderr_log}")
    ex = _extract_traces(loaded, subject, scenario)
    quality_dict = _write_reg_quality(out_dir, ex, reg_calib_source)
    save_registration_figures(
        out_dir,
        rgb=ex.frame0_rgb,
        thermal=ex.frame0_th,
        landmarks=ex.frame0_lm,
        reg_info=ex.reg_info0,
        first_quality=ex.first_quality,
        scene_quality=ex.quality,
    )
    print(f"Figury rejestracji → {out_dir}/fig_*.png")
    return {"rows": [], "quality": quality_dict, "gated_status": ex.quality.gate_label}


def _run_session_body(
    subject: str,
    scenario: str,
    session_dir: Path,
    out_dir: Path,
    stderr_log: Path,
    *,
    reg_calib_source: str = "self",
    figures_only: bool = False,
) -> dict:
    loaded, hr_path = load_session(session_dir, subject, scenario)
    if figures_only:
        return _run_registration_only(
            loaded, subject, scenario, out_dir, stderr_log, reg_calib_source
        )
    if hr_path is None:
        raise FileNotFoundError(f"Brak HR (dataHR_*.csv/*_HR.csv) w {session_dir}")
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
        # Pominięta scena nie może zostawić w agregacie metryk z innego przebiegu.
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
    write_scene_tables(out_dir, rows, ex.quality)
    save_scene_figures(
        out_dir,
        rgb=ex.frame0_rgb,
        thermal=ex.frame0_th,
        landmarks=ex.frame0_lm,
        reg_info=ex.reg_info0,
        first_quality=ex.first_quality,
        scene_quality=ex.quality,
        cleaned=cleaned,
        window_hrs=window_hrs,
        polar=polar,
        fs=fs,
        duration_s=duration_s,
        rows=rows,
    )
    quality_dict = _write_reg_quality(out_dir, ex, reg_calib_source)
    return {"rows": rows, "quality": quality_dict, "gated_status": ex.quality.gate_label}


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
    figures_only: bool = False,
) -> list[dict]:
    """Przetwarza sesje z ``data_dir`` (opcjonalnie jednego badanego) i odświeża agregat.

    ``figures_only``: tylko rejestracja i jej figury (bez Polar, metryk i agregatu).
    """
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
        result = run_session(
            subj,
            scenario,
            session_dir,
            out_dir,
            reg_calib_source=calib_src,
            figures_only=figures_only,
        )
        if result.get("skipped"):
            print(f"\n→ {subj}/{scenario}: SKIP (sync sanity)")
            continue
        _print_session_result(subj, scenario, result)

    if figures_only:
        return []
    return write_summary(out_root)
