"""Ekstrakcja śladu RGB z ROI i maskowanie perfuzji z termiki."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import PERFUSION_MIN_ROI_FRAC, PERFUSION_TEMP_STD_FACTOR
from src.registration import apply_affine

_EPS = 1e-8


def _roi_to_mask(roi_position: np.ndarray, height: int, width: int) -> np.ndarray:
    """Sprowadza ROI (maska binarna (H, W) lub bbox [y0, x0, y1, x1]) do maski (H, W)."""
    roi_position = np.asarray(roi_position)
    if roi_position.ndim == 2 and roi_position.shape == (height, width):
        return roi_position.astype(bool)
    if roi_position.shape == (4,):
        y0, x0, y1, x1 = (int(v) for v in roi_position)
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask
    raise ValueError(
        f"ROI musi być maską (H, W)=({height}, {width}) lub bboxem (4,), "
        f"otrzymano kształt {roi_position.shape}"
    )


def mean_rgb_in_mask(rgb_frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Średnia R, G, B po pikselach maski; pusta maska → średnia po całej klatce."""
    rgb_frame = np.asarray(rgb_frame, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if mask.any():
        return rgb_frame[mask].mean(axis=0).astype(np.float64)
    return rgb_frame.reshape(-1, rgb_frame.shape[-1]).mean(axis=0).astype(np.float64)


def _check_lengths(n_frames: int, roi_positions: list, valid: np.ndarray) -> None:
    """Waliduje spójność długości sekwencji klatek, pozycji ROI i wektora valid."""
    if len(roi_positions) != n_frames:
        raise ValueError("Liczba pozycji ROI musi odpowiadać liczbie klatek.")
    if np.asarray(valid).shape[0] != n_frames:
        raise ValueError("Długość wektora valid[] musi odpowiadać liczbie klatek.")


def extract_rgb_trace(
    rgb_frames: np.ndarray, roi_positions: list[np.ndarray], valid: np.ndarray
) -> np.ndarray:
    """Średnie R, G, B w ROI dla każdej klatki.

    Args:
        rgb_frames: (N, H, W, 3).
        roi_positions: maska (H, W) lub bbox na klatkę.
        valid: bool[N] — detekcja ROI; nie usuwa klatek z ekstrakcji.

    Returns:
        (N, 3) float64.
    """
    rgb_frames = np.asarray(rgb_frames, dtype=np.float64)
    n_frames, height, width, _ = rgb_frames.shape
    _check_lengths(n_frames, roi_positions, valid)

    trace = np.empty((n_frames, 3), dtype=np.float64)
    for i in range(n_frames):
        mask = _roi_to_mask(roi_positions[i], height, width)
        trace[i] = mean_rgb_in_mask(rgb_frames[i], mask)
    return trace


def compute_perfusion_mask(thermal_frame: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Maska pikseli w ROI powyżej progu mean + k*std (termika, bez normalizacji per klatka).

    Args:
        thermal_frame: (H, W) wartości termiczne.
        roi_mask: (H, W) bool.

    Returns:
        (H, W) bool.
    """
    thermal_frame = np.asarray(thermal_frame, dtype=np.float64)
    roi_mask = np.asarray(roi_mask, dtype=bool)
    if thermal_frame.shape != roi_mask.shape:
        raise ValueError("Klatka termiczna i maska ROI muszą mieć ten sam kształt (H, W).")

    roi_values = thermal_frame[roi_mask]
    if roi_values.size == 0:
        return np.zeros_like(roi_mask, dtype=bool)

    threshold = roi_values.mean() + PERFUSION_TEMP_STD_FACTOR * roi_values.std()
    return roi_mask & (thermal_frame >= threshold)


def extract_rgb_trace_thermal_gated(
    rgb_frames: np.ndarray,
    thermal_frames: np.ndarray,
    roi_positions: list[np.ndarray],
    valid: np.ndarray,
) -> np.ndarray:
    """Średnie RGB w ROI ∩ masce perfuzji (termika już w układzie RGB).

    Args:
        rgb_frames: (N, H, W, 3).
        thermal_frames: (N, H, W) po warp termika→RGB.
        roi_positions: maska lub bbox na klatkę.
        valid: bool[N].

    Returns:
        (N, 3); przy pustej masce perfuzji — średnia po ROI.
    """
    rgb_frames = np.asarray(rgb_frames, dtype=np.float64)
    thermal_frames = np.asarray(thermal_frames, dtype=np.float64)
    n_frames, height, width, _ = rgb_frames.shape
    _check_lengths(n_frames, roi_positions, valid)
    if thermal_frames.shape[0] != n_frames:
        raise ValueError("Liczba klatek termicznych musi odpowiadać liczbie klatek RGB.")

    trace = np.empty((n_frames, 3), dtype=np.float64)
    for i in range(n_frames):
        roi_mask = _roi_to_mask(roi_positions[i], height, width)
        perfusion_mask = compute_perfusion_mask(thermal_frames[i], roi_mask)
        n_roi = int(roi_mask.sum())
        n_perf = int(perfusion_mask.sum())
        use_gated = n_roi > 0 and n_perf >= PERFUSION_MIN_ROI_FRAC * n_roi
        gated_mask = perfusion_mask if use_gated else roi_mask
        trace[i] = mean_rgb_in_mask(rgb_frames[i], gated_mask)
    return trace


def sample_roi_temps_via_affine(
    thermal_gray: np.ndarray,
    affine_th_to_rgb: np.ndarray,
    roi_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Próbkuje temperatury termiki w pikselach ROI RGB przez odwrotną affinę.

    Returns:
        (ys, xs, temps) — tylko piksele z trafieniem w kadr termiki; puste, gdy brak.
    """
    ys, xs = np.where(np.asarray(roi_mask, dtype=bool))
    if ys.size == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_i, empty_i, np.zeros(0, dtype=np.float64)

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
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_i, empty_i, np.zeros(0, dtype=np.float64)

    map_x = pts_th[inside, 0].astype(np.float32).reshape(1, -1)
    map_y = pts_th[inside, 1].astype(np.float32).reshape(1, -1)
    temps = (
        cv2.remap(
            np.asarray(thermal_gray, dtype=np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        .ravel()
        .astype(np.float64)
    )
    return ys[inside], xs[inside], temps


def gated_mean_rgb_affine(
    rgb: np.ndarray,
    thermal_gray: np.ndarray,
    affine_th_to_rgb: np.ndarray,
    roi_mask: np.ndarray,
    temp_std_factor: float = PERFUSION_TEMP_STD_FACTOR,
    min_roi_frac: float = PERFUSION_MIN_ROI_FRAC,
) -> tuple[np.ndarray, bool]:
    """Średnia RGB w ROI ∩ masce perfuzji (remap przez odwrotną affinę).

    Returns:
        (mean_rgb[3], used_fallback) — fallback gdy maska < ``min_roi_frac``.
    """
    plain = mean_rgb_in_mask(rgb, roi_mask)
    ys, xs, temps = sample_roi_temps_via_affine(thermal_gray, affine_th_to_rgb, roi_mask)
    if temps.size == 0:
        return plain, True

    thr = float(temps.mean() + temp_std_factor * temps.std())
    keep = temps >= thr
    n_keep = int(np.count_nonzero(keep))
    if n_keep < min_roi_frac * temps.size:
        return plain, True
    return rgb[ys[keep], xs[keep]].mean(axis=0).astype(np.float64), False


def gated_means_per_frame_from_samples(
    rgb_pixels: list[np.ndarray | None],
    temps: list[np.ndarray | None],
    plain_means: np.ndarray,
    temp_std_factor: float = PERFUSION_TEMP_STD_FACTOR,
    min_roi_frac: float = PERFUSION_MIN_ROI_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """Gated per klatka z wcześniej zebranych próbek (rgb w ROI, temps)."""
    n = len(plain_means)
    out = np.empty((n, 3), dtype=np.float64)
    fallback = np.zeros(n, dtype=bool)
    for i in range(n):
        rp = rgb_pixels[i]
        tp = temps[i]
        if rp is None or tp is None or tp.size == 0:
            out[i] = plain_means[i]
            fallback[i] = True
            continue
        thr = float(tp.mean() + temp_std_factor * tp.std())
        keep = tp >= thr
        if int(np.count_nonzero(keep)) < min_roi_frac * tp.size:
            out[i] = plain_means[i]
            fallback[i] = True
        else:
            out[i] = rp[keep].mean(axis=0)
            fallback[i] = False
    return out, fallback


def gated_means_per_window_from_samples(
    rgb_pixels: list[np.ndarray | None],
    temps: list[np.ndarray | None],
    plain_means: np.ndarray,
    fs: float,
    window_s: float = 10.0,
    temp_std_factor: float = PERFUSION_TEMP_STD_FACTOR,
    min_roi_frac: float = PERFUSION_MIN_ROI_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """Jedna decyzja gated/fallback i jeden próg na okno ``window_s``."""
    n = len(plain_means)
    out = plain_means.copy()
    fallback = np.ones(n, dtype=bool)
    win = max(1, int(round(window_s * fs)))
    start = 0
    while start < n:
        end = min(n, start + win)
        pool_t = [temps[i] for i in range(start, end) if temps[i] is not None and temps[i].size]
        if not pool_t:
            start = end
            continue
        all_t = np.concatenate(pool_t)
        thr = float(all_t.mean() + temp_std_factor * all_t.std())
        n_roi = int(all_t.size)
        n_keep = int(np.count_nonzero(all_t >= thr))
        use_gated = n_roi > 0 and n_keep >= min_roi_frac * n_roi
        for i in range(start, end):
            rp = rgb_pixels[i]
            tp = temps[i]
            if not use_gated or rp is None or tp is None or tp.size == 0:
                out[i] = plain_means[i]
                fallback[i] = True
                continue
            keep = tp >= thr
            if not np.any(keep):
                out[i] = plain_means[i]
                fallback[i] = True
            else:
                out[i] = rp[keep].mean(axis=0)
                fallback[i] = False
        start = end
    return out, fallback


def thermal_probe_points(thermal_mask: np.ndarray, n_contour: int = 16) -> np.ndarray | None:
    """Stałe punkty w układzie termiki: centroid + równomierna próbka konturu maski."""
    mask_u8 = np.asarray(thermal_mask, dtype=bool).astype(np.uint8)
    if not mask_u8.any():
        return None
    ys, xs = np.where(mask_u8)
    centroid = np.array([[xs.mean(), ys.mean()]], dtype=np.float64)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return centroid
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if cnt.shape[0] < 3:
        return centroid
    idx = np.linspace(0, cnt.shape[0] - 1, num=min(n_contour, cnt.shape[0]), dtype=int)
    return np.vstack([centroid, cnt[idx]])


def affine_point_dispersion_px(
    matrices: list[np.ndarray],
    probe_pts_th: np.ndarray,
) -> dict[str, float]:
    """Rozrzut pozycji stałych punktów termiki po zmapowaniu do RGB [px].

    Returns:
        median_abs_dev — mediana |p − med(p)| po punktach i estymatach,
        iqr_radial — IQR odległości od mediany pozycji (po wszystkich punktach×estymatach),
        max_median_dev — najgorszy punkt (mediana odchylenia).
    """
    if len(matrices) < 2 or probe_pts_th.size == 0:
        return {"median_abs_dev": 0.0, "iqr_radial": 0.0, "max_median_dev": 0.0}

    mapped = np.stack(
        [apply_affine(np.asarray(m, dtype=np.float64), probe_pts_th) for m in matrices],
        axis=0,
    )  # (N, K, 2)
    med = np.median(mapped, axis=0)  # (K, 2)
    delta = mapped - med[None, :, :]
    radial = np.linalg.norm(delta, axis=2)  # (N, K)
    per_point_mad = np.median(radial, axis=0)
    flat = radial.ravel()
    iqr = float(np.subtract(*np.percentile(flat, [75, 25])))
    return {
        "median_abs_dev": float(np.median(per_point_mad)),
        "iqr_radial": iqr,
        "max_median_dev": float(np.max(per_point_mad)),
    }


def consensus_median_affine(
    matrices: list[np.ndarray],
    probe_pts_th: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """Jedna affine z mediany pozycji punktów (nie mediana 6 parametrów).

    Returns:
        (affine_2x3, mean_residual_px) albo (None, nan).
    """
    if not matrices or probe_pts_th is None or len(probe_pts_th) < 3:
        return None, float("nan")

    mapped = np.stack(
        [apply_affine(np.asarray(m, dtype=np.float64), probe_pts_th) for m in matrices],
        axis=0,
    )
    med_xy = np.median(mapped, axis=0)
    resid = float(np.mean(np.linalg.norm(mapped - med_xy[None, :, :], axis=2)))
    src = probe_pts_th.astype(np.float32)
    dst = med_xy.astype(np.float32)
    matrix, _ = cv2.estimateAffine2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=1e6,
    )
    if matrix is None:
        return None, resid
    return matrix.astype(np.float64), resid


def median_affine(matrices: list[np.ndarray]) -> np.ndarray:
    """DEPRECATED: mediana parametrów — użyj ``consensus_median_affine``."""
    stack = np.stack([np.asarray(m, dtype=np.float64) for m in matrices], axis=0)
    return np.median(stack, axis=0)


def affine_translation_iqr_px(matrices: list[np.ndarray]) -> tuple[float, float]:
    """DEPRECATED: IQR tx/ty — mylące przy sprzężeniu ze skalą; zostawione dla testów."""
    if len(matrices) < 2:
        return 0.0, 0.0
    txs = np.array([float(m[0, 2]) for m in matrices], dtype=np.float64)
    tys = np.array([float(m[1, 2]) for m in matrices], dtype=np.float64)

    def _iqr(a: np.ndarray) -> float:
        return float(np.subtract(*np.percentile(a, [75, 25])))

    return _iqr(txs), _iqr(tys)
