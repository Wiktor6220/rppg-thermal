"""Figury per scena: ROI RGB, segment termiki, nakładka rejestracji, HR w oknach, widmo."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.signal import welch

from src.config import BAND_HIGH_HZ, BAND_LOW_HZ, REGION_LABELS, REGIONS_REPORT
from src.io_layer import PolarHrSeries
from src.methods import PIPELINE_METHODS
from src.registration import (
    SceneRegistrationQuality,
    apply_affine,
    invert_affine,
    to_gray,
    warp_thermal_to_rgb,
)
from src.roi import select_roi_from_landmarks

WAVE_PREVIEW_SEC = 15.0
FIG_DPI = 160
HR_PLOT_CLIP_BPM = (40.0, 200.0)
HR_PLOT_PAD_BPM = 5.0

# Klucz (region, metoda, "plain"|"gated")
SignalKey = tuple[str, str, str]
WindowHrs = tuple[np.ndarray, np.ndarray, np.ndarray]  # (t0, t1, hr) per okno


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
    th_gray = to_gray(thermal)
    th_vis = cv2.cvtColor(th_gray, cv2.COLOR_GRAY2RGB)
    seg_mask = info["thermal_mask"]
    overlay = th_vis.copy()
    overlay[seg_mask] = np.clip(
        0.45 * th_vis[seg_mask] + np.array([180, 40, 40]), 0, 255
    ).astype(np.uint8)
    vals = th_gray[seg_mask].astype(np.float64)
    if vals.size:
        thr = vals.mean() + 0.5 * vals.std()
        perf = seg_mask & (th_gray.astype(np.float64) >= thr)
        overlay[perf] = np.clip(
            0.35 * th_vis[perf] + np.array([40, 200, 80]), 0, 255
        ).astype(np.uint8)
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
    scene_quality: SceneRegistrationQuality,
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
    title = (
        f"Nakładka termika→RGB (1. udana klatka)\n"
        f"RMS klatki={first_quality['contour_rms_px']:.1f} px  "
        f"IoU={first_quality['iou']:.2f}\n"
        f"scena: mediana RMS={scene_quality.contour_rms_px:.1f} px, "
        f"dobre próby {100 * scene_quality.good_frac:.0f}% "
        f"z {scene_quality.n_attempts}  ({scene_quality.gate_label})"
    )
    if first_quality.get("scale_ok") is False:
        title += (
            f"\naffine poza zakresem: "
            f"skala={first_quality.get('affine_scale', float('nan')):.2f}, "
            f"anizotropia={first_quality.get('affine_aniso', float('nan')):.2f} "
            f"— próba odrzucona"
        )
    fig, ax = plt.subplots(figsize=(6, 7.5))
    ax.imshow(_crop(blend, _face_crop_bounds(landmarks)))
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.savefig(out_dir / "fig_overlay_reg.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_window_steps(
    ax: Axes,
    t0: np.ndarray,
    t1: np.ndarray,
    hrs: np.ndarray,
    style: str,
    color: str,
    label: str,
) -> None:
    """HR per okno jako poziomy odcinek nad [t0, t1) + znacznik w środku okna."""
    first = True
    for a, b, h in zip(t0, t1, hrs, strict=True):
        if not np.isfinite(h):
            continue
        ax.plot([a, b], [h, h], style, color=color, lw=1.6, label=label if first else None)
        ax.plot([0.5 * (a + b)], [h], "o", color=color, ms=4)
        first = False


def _hr_ylim(polar_hr: np.ndarray, window_hrs: dict[SignalKey, WindowHrs]) -> tuple[float, float]:
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
    cleaned: dict[SignalKey, np.ndarray],
    window_hrs: dict[SignalKey, WindowHrs],
    polar: PolarHrSeries,
    fs: float,
    duration_s: float,
) -> None:
    """Wiersze: HR czoło, HR policzki, fala czoło, fala policzki; kolumny: POS, CHROM.

    Panele HR mają oś czasu całego nagrania; panele fali własną oś (pierwsze 15 s).
    """
    method_names = list(PIPELINE_METHODS)
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


def _fig_spectrum(
    out_dir: Path, cleaned: dict[SignalKey, np.ndarray], fs: float, rows: list[dict]
) -> None:
    """Widmo Welcha (oś BPM): wiersze czoło/policzki, kolumny POS/CHROM; plain vs gated."""
    method_names = list(PIPELINE_METHODS)
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


def save_registration_figures(
    out_dir: Path,
    *,
    rgb: np.ndarray,
    thermal: np.ndarray,
    landmarks: np.ndarray,
    reg_info: dict,
    first_quality: dict,
    scene_quality: SceneRegistrationQuality,
) -> None:
    """Figury rejestracji: ``fig_rgb_roi``, ``fig_thermal_seg``, ``fig_overlay_reg``."""
    _fig_rgb_roi(out_dir, rgb, landmarks)
    _fig_thermal_seg(out_dir, thermal, reg_info)
    _fig_overlay_reg(out_dir, rgb, thermal, landmarks, reg_info, first_quality, scene_quality)


def save_scene_figures(
    out_dir: Path,
    *,
    rgb: np.ndarray,
    thermal: np.ndarray,
    landmarks: np.ndarray,
    reg_info: dict,
    first_quality: dict,
    scene_quality: SceneRegistrationQuality,
    cleaned: dict[SignalKey, np.ndarray],
    window_hrs: dict[SignalKey, WindowHrs],
    polar: PolarHrSeries,
    fs: float,
    duration_s: float,
    rows: list[dict],
) -> None:
    """Wszystkie ``fig_*.png`` sceny; obraz/maska/affine z pierwszej udanej rejestracji."""
    save_registration_figures(
        out_dir,
        rgb=rgb,
        thermal=thermal,
        landmarks=landmarks,
        reg_info=reg_info,
        first_quality=first_quality,
        scene_quality=scene_quality,
    )
    _fig_hr_windows(out_dir, cleaned, window_hrs, polar, fs, duration_s)
    _fig_spectrum(out_dir, cleaned, fs, rows)
    print(f"Figury → {out_dir}/fig_*.png")
