"""Automatyczna korejestracja termika → RGB (kontury masek + refine linii oczu).

Bez człowieka w pętli i bez ML. Landmarki MediaPipe → maska twarzy na RGB;
na termice segmentacja jasnej plamy. Start: pełna affine z kątowo sparowanych
konturów. Refine: przesunięcie w Y, żeby najciemniejszy pas oczu na termice
trafił w linię oczu RGB (MediaPipe 33/263) — usuwa systematyczny bias sylwetki.

``extract`` zakłada termikę już zwarpowaną do rozmiaru RGB (``warp_thermal_to_rgb``).
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config import (
    REG_EYE_BAND_BOTTOM,
    REG_EYE_BAND_TOP,
    REG_EYE_LANDMARK_L,
    REG_EYE_LANDMARK_R,
    REG_MORPH_KERNEL,
    REG_NECK_WIDTH_FRAC,
    REG_NOMINAL_OFFSET,
    REG_NOMINAL_SCALE,
    REG_WINDOW_PAD,
)

# Liczba punktów konturu do estimateAffine2D (empirycznie stabilne 24–48).
_CONTOUR_SAMPLES: int = 36


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Klatka (H, W) lub (H, W, 3) → uint8 szarość."""
    if frame.ndim == 2:
        return frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def rgb_face_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Binarna maska twarzy RGB z wypukłej otoczki landmarków MediaPipe."""
    height, width = shape
    hull = cv2.convexHull(landmarks.astype(np.float32))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
    return mask.astype(bool)


def nominal_thermal_window(
    landmarks: np.ndarray, thermal_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Hojne okno w termice z nominalnego odwzorowania bboxa twarzy RGB."""
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    cx = 0.5 * (xs.min() + xs.max())
    cy = 0.5 * (ys.min() + ys.max())
    fw = max(1.0, float(xs.max() - xs.min()))
    fh = max(1.0, float(ys.max() - ys.min()))
    tcx = REG_NOMINAL_SCALE * cx + REG_NOMINAL_OFFSET[0]
    tcy = REG_NOMINAL_SCALE * cy + REG_NOMINAL_OFFSET[1]
    tw = REG_NOMINAL_SCALE * fw * REG_WINDOW_PAD
    th = REG_NOMINAL_SCALE * fh * REG_WINDOW_PAD
    height, width = thermal_shape
    x0 = max(0, int(tcx - tw / 2))
    x1 = min(width, int(tcx + tw / 2))
    y0 = max(0, int(tcy - th / 2))
    y1 = min(height, int(tcy + th / 2))
    return x0, y0, x1, y1


def segment_thermal_face(
    gray: np.ndarray, window: tuple[int, int, int, int]
) -> tuple[np.ndarray | None, dict | str]:
    """Segmentuje twarz w oknie: Otsu + morfologia + największa składowa + cięcie szyi."""
    x0, y0, x1, y1 = window
    win = gray[y0:y1, x0:x1]
    if win.size == 0:
        return None, "okno puste"

    _, binw = cv2.threshold(win, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((REG_MORPH_KERNEL, REG_MORPH_KERNEL), np.uint8)
    binw = cv2.morphologyEx(binw, cv2.MORPH_OPEN, kernel)
    binw = cv2.morphologyEx(binw, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binw, connectivity=8)
    if n_labels <= 1:
        return None, "brak spójnej składowej po progowaniu"
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (labels == biggest).astype(np.uint8)

    rows_w = comp.sum(axis=1)
    max_w = int(rows_w.max())
    widest = int(np.argmax(rows_w))
    for row in range(widest, comp.shape[0]):
        if rows_w[row] < REG_NECK_WIDTH_FRAC * max_w:
            comp[row:, :] = 0
            break

    mask = np.zeros_like(gray, dtype=bool)
    mask[y0:y1, x0:x1] = comp.astype(bool)
    if not mask.any():
        return None, "pusta maska po cięciu szyi"
    return mask, {"area": int(mask.sum()), "window": window}


def contour_sample_points(mask: np.ndarray, n_points: int = _CONTOUR_SAMPLES) -> np.ndarray | None:
    """Próbkuje kontur maski w równych odstępach kątowych wokół centroidu (N, 2)."""
    binary = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(contour) < n_points:
        return None
    center = contour.mean(axis=0)
    angles = np.arctan2(contour[:, 1] - center[1], contour[:, 0] - center[0])
    targets = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
    points = np.empty((n_points, 2), dtype=np.float64)
    for i, target in enumerate(targets):
        delta = np.abs((angles - target + np.pi) % (2 * np.pi) - np.pi)
        points[i] = contour[int(np.argmin(delta))]
    return points


def _mask_geometry(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Centroid, kąt głównej osi [rad] i skala (sqrt pola) — fallback gdy brak konturu."""
    moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    area = float(moments["m00"])
    if area < 1.0:
        return None
    cx = moments["m10"] / area
    cy = moments["m01"] / area
    mu20 = moments["mu20"] / area
    mu02 = moments["mu02"] / area
    mu11 = moments["mu11"] / area
    theta = 0.5 * float(np.arctan2(2.0 * mu11, mu20 - mu02))
    scale = float(np.sqrt(area))
    return cx, cy, theta, scale


def affine_from_mask_moments(src_mask: np.ndarray, dst_mask: np.ndarray) -> np.ndarray | None:
    """Podobieństwo z momentów (fallback). Kąt zawijany modulo π (oś główna)."""
    src = _mask_geometry(src_mask)
    dst = _mask_geometry(dst_mask)
    if src is None or dst is None:
        return None
    sx, sy, stheta, sscale = src
    dx, dy, dtheta, dscale = dst
    if sscale < 1e-6:
        return None
    scale = dscale / sscale
    angle = dtheta - stheta
    angle = (angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    a, b = scale * cos_a, -scale * sin_a
    c, d = scale * sin_a, scale * cos_a
    tx = dx - (a * sx + b * sy)
    ty = dy - (c * sx + d * sy)
    return np.array([[a, b, tx], [c, d, ty]], dtype=np.float64)


def affine_from_contours(src_mask: np.ndarray, dst_mask: np.ndarray) -> np.ndarray | None:
    """Pełna affine LS z kątowo sparowanych punktów konturu (src→dst)."""
    src = contour_sample_points(src_mask)
    dst = contour_sample_points(dst_mask)
    if src is None or dst is None:
        return None
    matrix, _ = cv2.estimateAffine2D(
        src.astype(np.float32),
        dst.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=1e6,
    )
    return None if matrix is None else matrix.astype(np.float64)


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Przekształca punkty (N, 2) macierzą afiniczną 2×3."""
    pts = np.asarray(points, dtype=np.float64)
    return pts @ matrix[:, :2].T + matrix[:, 2]


def compose_affine(second: np.ndarray, first: np.ndarray) -> np.ndarray:
    """Składa dwie macierze 2×3: wynik(p) = second(first(p))."""
    a = np.vstack([first.astype(np.float64), [0.0, 0.0, 1.0]])
    b = np.vstack([second.astype(np.float64), [0.0, 0.0, 1.0]])
    return (b @ a)[:2]


def thermal_eye_line(
    gray: np.ndarray,
    mask: np.ndarray,
    band_top: float = REG_EYE_BAND_TOP,
    band_bottom: float = REG_EYE_BAND_BOTTOM,
    min_cols: int = 8,
) -> tuple[float, float] | None:
    """Środek najciemniejszego wiersza w pasie oczu maski termicznej (cx, row).

    Pas to ułamek wysokości maski od jej góry — celowo nie od 0, żeby nie łapać
    ciemnej linii włosów / górnej krawędzi segmentacji.
    """
    ys = np.where(mask)[0]
    if ys.size == 0:
        return None
    top, bot = int(ys.min()), int(ys.max())
    height = max(1, bot - top)
    y0 = top + int(band_top * height)
    y1 = top + int(band_bottom * height)
    best_row, best_val = None, np.inf
    for row in range(y0, y1 + 1):
        cols = np.where(mask[row])[0]
        if cols.size < min_cols:
            continue
        val = float(gray[row, cols].mean())
        if val < best_val:
            best_val, best_row = val, row
    if best_row is None:
        return None
    cols = np.where(mask[best_row])[0]
    return float(0.5 * (cols.min() + cols.max())), float(best_row)


def refine_affine_eye_y(
    affine: np.ndarray,
    gray: np.ndarray,
    thermal_mask: np.ndarray,
    landmarks: np.ndarray,
) -> tuple[np.ndarray, dict | None]:
    """Koryguje ty affine, by linia oczu termiki trafiła w Y środków oczu RGB.

    Zwraca (affine_po_refine, info) albo (affine_bez_zmian, None) gdy brak detekcji.
    """
    eye_th = thermal_eye_line(gray, thermal_mask)
    if eye_th is None:
        return affine, None
    if landmarks.shape[0] <= max(REG_EYE_LANDMARK_L, REG_EYE_LANDMARK_R):
        return affine, None

    rgb_eye_y = 0.5 * (
        landmarks[REG_EYE_LANDMARK_L, 1] + landmarks[REG_EYE_LANDMARK_R, 1]
    )
    pred = apply_affine(affine, np.array([[eye_th[0], eye_th[1]]]))[0]
    dy = float(rgb_eye_y - pred[1])
    refined = affine.copy()
    refined[1, 2] += dy
    return refined, {"eye_thermal": eye_th, "dy": dy, "rgb_eye_y": float(rgb_eye_y)}


def estimate_affine_thermal_to_rgb(
    rgb_frame: np.ndarray,
    thermal_frame: np.ndarray,
    landmarks: np.ndarray,
) -> tuple[np.ndarray | None, dict | str]:
    """Estymuje pełną affine termika→RGB (kontury + refine Y linii oczu; bez HITL).

    Args:
        rgb_frame: klatka RGB (H, W, 3) — kształt kadru; maska z landmarków.
        thermal_frame: klatka termiczna (h, w) lub (h, w, 3) podglądu.
        landmarks: (K, 2) punkty MediaPipe w pikselach RGB.

    Returns:
        ``(affine_2x3, info_dict)`` albo ``(None, powód_str)``.
    """
    if landmarks is None or len(landmarks) < 3:
        return None, "brak landmarków RGB"

    rgb_h, rgb_w = rgb_frame.shape[:2]
    gray = _to_gray(thermal_frame)
    window = nominal_thermal_window(landmarks, gray.shape)
    thermal_mask, seg_info = segment_thermal_face(gray, window)
    if thermal_mask is None:
        return None, f"segmentacja: {seg_info}"

    face_mask = rgb_face_mask(landmarks, (rgb_h, rgb_w))
    if not face_mask.any():
        return None, "pusta maska RGB"

    method = "contour"
    affine = affine_from_contours(thermal_mask, face_mask)
    if affine is None:
        method = "moments_fallback"
        affine = affine_from_mask_moments(thermal_mask, face_mask)
    if affine is None:
        return None, "nie udało się policzyć affine z masek"

    init_affine = affine.copy()
    affine, eye_info = refine_affine_eye_y(affine, gray, thermal_mask, landmarks)
    if eye_info is not None:
        method = f"{method}+eye_y"

    info = {
        "affine": affine,
        "init_affine": init_affine,
        "method": method,
        "eye_refine": eye_info,
        "rgb_mask": face_mask,
        "thermal_mask": thermal_mask,
        "window": window,
        "seg": seg_info,
    }
    return affine, info


def warp_thermal_to_rgb(
    thermal_frame: np.ndarray,
    affine: np.ndarray,
    rgb_shape: tuple[int, int],
) -> np.ndarray:
    """Warpuje klatkę termiczną do rozmiaru RGB macierzą 2×3 (wynik: float64 szarość)."""
    gray = _to_gray(thermal_frame).astype(np.float64)
    height, width = rgb_shape
    return cv2.warpAffine(
        gray, affine.astype(np.float64), (width, height), flags=cv2.INTER_LINEAR
    )
