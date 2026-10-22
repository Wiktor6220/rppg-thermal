"""Ekstrakcja śladu RGB z ROI i bramkowanie perfuzją z termiki."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import PERFUSION_MIN_ROI_FRAC, PERFUSION_TEMP_STD_FACTOR
from src.registration import apply_affine

# OpenCV remap: dst.cols musi być < SHRT_MAX (~32k), więc duże ROI próbkujemy porcjami.
_REMAP_CHUNK = 16_384


def mean_rgb_in_mask(rgb_frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Średnia RGB w masce; pusta maska -> średnia całej klatki."""
    rgb_frame = np.asarray(rgb_frame, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if mask.any():
        return rgb_frame[mask].mean(axis=0).astype(np.float64)
    return rgb_frame.reshape(-1, rgb_frame.shape[-1]).mean(axis=0).astype(np.float64)


def sample_roi_temps_via_affine(
    thermal_gray: np.ndarray,
    affine_th_to_rgb: np.ndarray,
    roi_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Temperatury termiki w pikselach ROI RGB (odwrotna affine + remap porcjami)."""
    ys, xs = np.where(np.asarray(roi_mask, dtype=bool))
    if ys.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float64)

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


def gated_means_per_frame_from_samples(
    rgb_pixels: list[np.ndarray | None],
    temps: list[np.ndarray | None],
    plain_means: np.ndarray,
    temp_std_factor: float = PERFUSION_TEMP_STD_FACTOR,
    min_roi_frac: float = PERFUSION_MIN_ROI_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """Gated z gotowych próbek per klatka -> (means, fallback_flags).

    Próg ``mean + k·std`` liczony na względnej jasności podglądu termiki w ROI
    (nie temperatura radiometryczna); za mało pikseli powyżej progu -> plain.
    """
    n = len(plain_means)
    out = np.empty((n, 3), dtype=np.float64)
    fallback = np.zeros(n, dtype=bool)
    for i in range(n):
        rp, tp = rgb_pixels[i], temps[i]
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
