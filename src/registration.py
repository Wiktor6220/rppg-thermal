"""Korejestracja termika->RGB: segmentacja, affine z konturów, refine Y linii oczu."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.config import (
    REG_AFFINE_MAX_ANISO,
    REG_AFFINE_SCALE_RANGE,
    REG_CALIB_WINDOW_PAD,
    REG_EYE_BAND_BOTTOM,
    REG_EYE_BAND_TOP,
    REG_EYE_LANDMARK_L,
    REG_EYE_LANDMARK_R,
    REG_GO_RMS_PX,
    REG_MORPH_KERNEL,
    REG_NECK_MIN_FRAC,
    REG_NECK_WIDTH_FRAC,
    REG_NOMINAL_OFFSET,
    REG_NOMINAL_SCALE,
    REG_PRIOR_OFFSET,
    REG_PRIOR_SCALE,
    REG_SCALE_RANGE,
    REG_WINDOW_PAD,
)

# Ile największych składowych po progowaniu rozważać jako kandydatów na głowę.
_MAX_SEG_CANDIDATES: int = 5

# Liczba punktów konturu do estimateAffine2D (empirycznie stabilne 24–48).
_CONTOUR_SAMPLES: int = 36


def to_gray(frame: np.ndarray) -> np.ndarray:
    """Klatka (H, W) lub (H, W, 3) -> uint8 szarość."""
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


def landmark_bbox_size(landmarks: np.ndarray) -> tuple[float, float]:
    """(szerokość, wysokość) bboxa landmarków RGB, min. 1 px."""
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    return max(1.0, float(xs.max() - xs.min())), max(1.0, float(ys.max() - ys.min()))


def scale_is_plausible(scale: float) -> bool:
    """Czy skala termika/RGB mieści się w ``REG_SCALE_RANGE`` (segment = głowa)."""
    lo, hi = REG_SCALE_RANGE
    return bool(np.isfinite(scale) and lo <= scale <= hi)


def affine_scale(affine: np.ndarray) -> float:
    """Skala affine: sqrt(|det A[:2, :2]|)."""
    return float(np.sqrt(abs(np.linalg.det(np.asarray(affine, dtype=np.float64)[:2, :2]))))


def affine_anisotropy(affine: np.ndarray) -> float:
    """Anizotropia affine: σ1/σ2 z SVD A[:2, :2] (1 = podobieństwo); inf przy osobliwej."""
    sv = np.linalg.svd(np.asarray(affine, dtype=np.float64)[:2, :2], compute_uv=False)
    return float(sv[0] / sv[1]) if sv[1] > 0 else float("inf")


def affine_is_plausible(affine: np.ndarray) -> bool:
    """Skala w ``REG_AFFINE_SCALE_RANGE`` i anizotropia ≤ ``REG_AFFINE_MAX_ANISO``.

    Kamery RGB i termiczna są sztywno złączone, więc affine termika->RGB jest prawie
    podobieństwem o skali zbliżonej do stosunku rozdzielczości.
    """
    lo, hi = REG_AFFINE_SCALE_RANGE
    scale = affine_scale(affine)
    aniso = affine_anisotropy(affine)
    return bool(np.isfinite(scale) and lo <= scale <= hi and aniso <= REG_AFFINE_MAX_ANISO)


def thermal_window_from_mapping(
    landmarks: np.ndarray,
    thermal_shape: tuple[int, int],
    scale: float,
    offset: tuple[float, float],
    pad: float,
) -> tuple[int, int, int, int]:
    """Okno w termice z odwzorowania bboxa twarzy RGB: ``scale·p + offset``, rozmiar × ``pad``."""
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    cx = 0.5 * (xs.min() + xs.max())
    cy = 0.5 * (ys.min() + ys.max())
    fw, fh = landmark_bbox_size(landmarks)
    tcx = scale * cx + offset[0]
    tcy = scale * cy + offset[1]
    tw = scale * fw * pad
    th = scale * fh * pad
    height, width = thermal_shape
    x0 = max(0, int(tcx - tw / 2))
    x1 = min(width, int(tcx + tw / 2))
    y0 = max(0, int(tcy - th / 2))
    y1 = min(height, int(tcy + th / 2))
    return x0, y0, x1, y1


def nominal_thermal_window(
    landmarks: np.ndarray, thermal_shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Hojne okno w termice z bieżącego odwzorowania nominalnego (``REG_NOMINAL_*``)."""
    return thermal_window_from_mapping(
        landmarks, thermal_shape, REG_NOMINAL_SCALE, REG_NOMINAL_OFFSET, REG_WINDOW_PAD
    )


def _cut_neck_width_profile(
    comp: np.ndarray, min_cut_rows: int = 0
) -> tuple[np.ndarray, dict]:
    """Odcina barki w lokalnym minimum szerokości między głową a barkami.

    Profil wierszy: szerokość rośnie na twarzy, maleje na szyi, znów rośnie na barkach.
    Gdy barki nie są w kadrze — fallback: spadek poniżej ``REG_NECK_WIDTH_FRAC``
    względem maksimum głowy (nie globalnego najszerszego wiersza).
    Cięcie nie wyżej niż ``min_cut_rows`` wierszy od góry segmentu (przewężenie na
    wysokości oczu, np. ciemne szkła okularów, nie jest szyją).
    """
    rows_w = comp.sum(axis=1).astype(np.float64)
    nz = np.where(rows_w > 0)[0]
    info: dict = {"neck_method": "none", "cut_row": None, "head_row": None}
    if nz.size < 10:
        return comp, info

    y0, y1 = int(nz[0]), int(nz[-1])
    profile = rows_w[y0 : y1 + 1]
    k = max(5, len(profile) // 40)
    smooth = np.convolve(profile, np.ones(k) / k, mode="same")
    n = len(smooth)

    # Głowa != globalne max (to często barki). Przy barkach w kadrze bierz
    # maksimum w górnych ~40% sylwetki; inaczej globalne max = głowa.
    gmax = int(np.argmax(smooth))
    if gmax > 0.45 * n:
        head_rel = int(np.argmax(smooth[: max(3, int(0.40 * n))]))
    else:
        head_rel = gmax
    head_w = float(smooth[head_rel])
    if head_w < 1.0:
        return comp, info
    info["head_row"] = int(y0 + head_rel)

    neck_rel: int | None = None
    min_gap = max(3, n // 25)
    look_ahead = max(8, n // 8)
    search_end = n - 2
    if gmax > 0.45 * n:
        search_end = min(search_end, gmax)

    # (idx, score) — najgłębsze przewężenie z wzrostem szerokości poniżej
    best_pinch: tuple[int, float] | None = None
    for i in range(max(head_rel + min_gap, min_cut_rows), search_end):
        if not (smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1]):
            continue
        below = smooth[i + 1 : min(n, i + 1 + look_ahead)]
        if not below.size:
            continue
        below_max = float(below.max())
        ratio = float(smooth[i]) / head_w
        if below_max > smooth[i] * 1.12 and ratio < 0.92:
            score = (below_max / max(smooth[i], 1.0)) * (1.0 - ratio)
            if best_pinch is None or score > best_pinch[1]:
                best_pinch = (i, score)
        if ratio < REG_NECK_WIDTH_FRAC and below_max > smooth[i] * 1.05:
            neck_rel = i
            info["neck_method"] = "pinch_shoulders"
            break
        if ratio < 0.72:
            neck_rel = i
            info["neck_method"] = "pinch_deep"
            break

    if neck_rel is None and best_pinch is not None:
        neck_rel = best_pinch[0]
        info["neck_method"] = "pinch_best"

    if neck_rel is None:
        for i in range(max(head_rel + 1, min_cut_rows), search_end + 1):
            if smooth[i] < REG_NECK_WIDTH_FRAC * head_w:
                neck_rel = i
                info["neck_method"] = "head_frac_fallback"
                break

    out = comp.copy()
    if neck_rel is not None:
        cut = y0 + neck_rel
        out[cut:, :] = 0
        info["cut_row"] = int(cut)
    return out, info


def _component_bbox_size(comp: np.ndarray) -> tuple[float, float] | None:
    """(szerokość, wysokość) bboxa składowej (konwencja jak ``infer_nominal_from_masks``)."""
    ys, xs = np.nonzero(comp)
    if xs.size == 0:
        return None
    return max(1.0, float(xs.max() - xs.min())), max(1.0, float(ys.max() - ys.min()))


def _size_mismatch(wh: tuple[float, float], expected_wh: tuple[float, float]) -> float:
    """|ln(w/ew)| + |ln(h/eh)| — 0 przy idealnym rozmiarze."""
    return abs(float(np.log(wh[0] / expected_wh[0]))) + abs(float(np.log(wh[1] / expected_wh[1])))


def segment_thermal_face(
    gray: np.ndarray,
    window: tuple[int, int, int, int],
    expected_wh: tuple[float, float] | None = None,
) -> tuple[np.ndarray | None, dict | str]:
    """Segmentuje twarz w oknie: Otsu + morfologia + wybór składowej + cięcie szyi.

    Kandydaci: do ``_MAX_SEG_CANDIDATES`` największych składowych, każda po cięciu szyi.
    Bez ``expected_wh`` — największa. Z ``expected_wh`` (oczekiwany rozmiar głowy w px
    termiki) — kandydat o bboxie najbliższym oczekiwanemu (``_size_mismatch``), żeby
    jasny tułów nie wygrywał z głową tylko dlatego, że jest większy.
    """
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
    # Stabilne sortowanie: przy remisie pola wygrywa niższa etykieta (jak np.argmax).
    order = 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA], kind="stable")
    candidates = [int(lab) for lab in order[:_MAX_SEG_CANDIDATES]]

    min_cut_rows = 0 if expected_wh is None else int(REG_NECK_MIN_FRAC * expected_wh[1])
    picked_rank = 0
    comp, neck_info = _cut_neck_width_profile(
        (labels == candidates[0]).astype(np.uint8), min_cut_rows
    )
    if expected_wh is not None:
        best_cost = np.inf
        for rank, lab in enumerate(candidates):
            cut, info = _cut_neck_width_profile((labels == lab).astype(np.uint8), min_cut_rows)
            wh = _component_bbox_size(cut)
            if wh is None:
                continue
            cost = _size_mismatch(wh, expected_wh)
            if cost < best_cost:
                best_cost, picked_rank, comp, neck_info = cost, rank, cut, info

    mask = np.zeros_like(gray, dtype=bool)
    mask[y0:y1, x0:x1] = comp.astype(bool)
    if not mask.any():
        return None, "pusta maska po cięciu szyi"
    # cut_row jest lokalny w oknie — podaj też w współrzędnych pełnej klatki
    cut_local = neck_info.get("cut_row")
    head_local = neck_info.get("head_row")
    return mask, {
        "area": int(mask.sum()),
        "window": window,
        "neck_method": neck_info.get("neck_method"),
        "cut_row": None if cut_local is None else int(y0 + cut_local),
        "head_row": None if head_local is None else int(y0 + head_local),
        "n_candidates": len(candidates),
        "picked_rank": picked_rank,
    }


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
    """Pełna affine LS z kątowo sparowanych punktów konturu (src->dst)."""
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


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    """Odwrotność macierzy afinicznej 2×3."""
    return cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float32)).astype(np.float64)


def map_rgb_mask_to_thermal(
    rgb_mask: np.ndarray,
    affine_th_to_rgb: np.ndarray,
    thermal_shape: tuple[int, int],
) -> np.ndarray:
    """Mapuje maskę RGB -> termika: thermal(p) = rgb(affine_th_to_rgb(p))."""
    th_h, th_w = thermal_shape
    ys, xs = np.mgrid[0:th_h, 0:th_w]
    pts = apply_affine(
        affine_th_to_rgb,
        np.column_stack([xs.ravel().astype(np.float64), ys.ravel().astype(np.float64)]),
    )
    map_x = pts[:, 0].reshape(th_h, th_w).astype(np.float32)
    map_y = pts[:, 1].reshape(th_h, th_w).astype(np.float32)
    warped = cv2.remap(
        np.asarray(rgb_mask, dtype=np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped.astype(bool)


def constrain_thermal_mask_to_rgb_face(
    thermal_mask: np.ndarray,
    rgb_face: np.ndarray,
    affine_th_to_rgb: np.ndarray,
) -> np.ndarray:
    """Przecięcie maski termicznej z otoczką twarzy RGB zmapowaną w termikę."""
    hull_th = map_rgb_mask_to_thermal(
        rgb_face, affine_th_to_rgb, thermal_mask.shape[:2]
    )
    return np.asarray(thermal_mask, dtype=bool) & hull_th


def thermal_eye_line(
    gray: np.ndarray,
    mask: np.ndarray,
    band_top: float = REG_EYE_BAND_TOP,
    band_bottom: float = REG_EYE_BAND_BOTTOM,
    min_cols: int = 8,
) -> tuple[float, float] | None:
    """(cx, row) najciemniejszego wiersza w pasie oczu maski (REG_EYE_BAND_*)."""
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
    """Koryguje ty affine wg linii oczu termika->RGB."""
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


def _fit_affine_masks(
    thermal_mask: np.ndarray, face_mask: np.ndarray
) -> tuple[np.ndarray | None, str]:
    """Affine termika->RGB z konturów, z fallbackiem na momenty."""
    affine = affine_from_contours(thermal_mask, face_mask)
    if affine is not None:
        return affine, "contour"
    affine = affine_from_mask_moments(thermal_mask, face_mask)
    if affine is not None:
        return affine, "moments_fallback"
    return None, "fail"


def estimate_affine_thermal_to_rgb(
    rgb_frame: np.ndarray,
    thermal_frame: np.ndarray,
    landmarks: np.ndarray,
) -> tuple[np.ndarray | None, dict | str]:
    """Estymuje affine termika->RGB: kontury masek + otoczka RGB + refine Y oczu."""
    if landmarks is None or len(landmarks) < 3:
        return None, "brak landmarków RGB"

    rgb_h, rgb_w = rgb_frame.shape[:2]
    gray = to_gray(thermal_frame)
    window = nominal_thermal_window(landmarks, gray.shape)
    fw, fh = landmark_bbox_size(landmarks)
    expected_wh = (REG_NOMINAL_SCALE * fw, REG_NOMINAL_SCALE * fh)
    thermal_mask, seg_info = segment_thermal_face(gray, window, expected_wh)
    if thermal_mask is None:
        return None, f"segmentacja: {seg_info}"
    # Skala segmentu względem twarzy RGB — tylko diagnostyka (nie wpływa na scale_ok).
    seg_scale = infer_nominal_from_masks(landmarks, thermal_mask)[0]

    face_mask = rgb_face_mask(landmarks, (rgb_h, rgb_w))
    if not face_mask.any():
        return None, "pusta maska RGB"

    affine, method = _fit_affine_masks(thermal_mask, face_mask)
    if affine is None:
        return None, "nie udało się policzyć affine z masek"
    init_affine = affine.copy()
    seg_mask = thermal_mask  # oryginał po neck-pinch — baza do kolejnych hull

    # Iteracyjnie: otoczka RGB w termice -> ponowne dopasowanie (zwykle 2–3× do <18 px).
    hull_frac = 1.0
    n_hull = 0
    for _ in range(3):
        refined = constrain_thermal_mask_to_rgb_face(seg_mask, face_mask, affine)
        hull_frac = float(refined.sum()) / max(1, int(seg_mask.sum()))
        min_keep = max(500, int(0.15 * int(seg_mask.sum())))
        if int(refined.sum()) < min_keep:
            method = f"{method}+hull_skip"
            break
        aff2, method2 = _fit_affine_masks(refined, face_mask)
        if aff2 is None:
            thermal_mask = refined
            method = f"{method}+hull"
            break
        thermal_mask = refined
        affine = aff2
        n_hull += 1
        method = f"{method}+hull+{method2}"

    pre_eye = affine.copy()
    affine, eye_info = refine_affine_eye_y(affine, gray, thermal_mask, landmarks)
    if eye_info is not None:
        # Odrzuć korektę Y oczu, gdy psuje dopasowanie konturów (fałszywy pas oczu).
        src_pts = contour_sample_points(thermal_mask)
        dst_pts = contour_sample_points(face_mask)
        if src_pts is not None and dst_pts is not None:
            def _crms(mat: np.ndarray) -> float:
                r = np.linalg.norm(apply_affine(mat, src_pts) - dst_pts, axis=1)
                return float(np.sqrt(np.mean(r**2)))

            if _crms(affine) <= _crms(pre_eye) + 1.0:
                method = f"{method}+eye_y"
            else:
                eye_info = {**eye_info, "rejected": True, "dy_rejected": eye_info["dy"]}
                affine = pre_eye
        else:
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
        "hull_frac": hull_frac,
        "n_hull_iters": n_hull,
        "seg_scale": float(seg_scale),
        "affine_scale": affine_scale(affine),
        "affine_aniso": affine_anisotropy(affine),
        # Końcowy affine (po hull i eye_y): segment spoza głowy (tułów, samo czoło)
        # daje skalę/anizotropię niemożliwą dla sztywnej pary kamer.
        "scale_ok": affine_is_plausible(affine),
    }
    return affine, info


def warp_thermal_to_rgb(
    thermal_frame: np.ndarray,
    affine: np.ndarray,
    rgb_shape: tuple[int, int],
) -> np.ndarray:
    """Warpuje klatkę termiczną do rozmiaru RGB macierzą 2×3 (wynik: float64 szarość)."""
    gray = to_gray(thermal_frame).astype(np.float64)
    height, width = rgb_shape
    return cv2.warpAffine(
        gray, affine.astype(np.float64), (width, height), flags=cv2.INTER_LINEAR
    )


def infer_nominal_from_masks(
    landmarks: np.ndarray, thermal_mask: np.ndarray
) -> tuple[float, tuple[float, float], dict]:
    """Estymuje REG_NOMINAL_SCALE/OFFSET z bboxa RGB vs maski termicznej (twarz)."""
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    rcx = 0.5 * (float(xs.min()) + float(xs.max()))
    rcy = 0.5 * (float(ys.min()) + float(ys.max()))
    rw = max(1.0, float(xs.max() - xs.min()))
    rh = max(1.0, float(ys.max() - ys.min()))
    ty, tx = np.where(thermal_mask)
    if tx.size == 0:
        raise ValueError("pusta maska termiczna — brak inferencji nominalnej")
    tcx, tcy = float(tx.mean()), float(ty.mean())
    tw = max(1.0, float(tx.max() - tx.min()))
    th = max(1.0, float(ty.max() - ty.min()))
    scale = 0.5 * (tw / rw + th / rh)
    offset = (tcx - scale * rcx, tcy - scale * rcy)
    meta = {
        "rgb_size": (rw, rh),
        "th_size": (tw, th),
        "rgb_centroid": (rcx, rcy),
        "th_centroid": (tcx, tcy),
        "area": int(thermal_mask.sum()),
    }
    return float(scale), (float(offset[0]), float(offset[1])), meta


def calibrate_nominal_registration(
    rgb_frame: np.ndarray,
    thermal_frame: np.ndarray,
    landmarks: np.ndarray,
    window_pad: float = 1.6,
) -> dict:
    """Auto-kalibracja REG_NOMINAL_* z jednej pary klatek.

    Segmentacja w oknie z priora (``REG_PRIOR_*``, ``REG_CALIB_WINDOW_PAD``) z oczekiwanym
    rozmiarem głowy ``REG_PRIOR_SCALE`` × bbox twarzy RGB. ``ValueError``, gdy segmentacja
    się nie uda albo szerokość lub wysokość segmentu względem twarzy RGB wypada poza
    ``REG_SCALE_RANGE`` (caller może wziąć inną klatkę).
    """
    gray = to_gray(thermal_frame)
    window = thermal_window_from_mapping(
        landmarks, gray.shape[:2], REG_PRIOR_SCALE, REG_PRIOR_OFFSET, REG_CALIB_WINDOW_PAD
    )
    fw, fh = landmark_bbox_size(landmarks)
    expected_wh = (REG_PRIOR_SCALE * fw, REG_PRIOR_SCALE * fh)
    mask, seg = segment_thermal_face(gray, window, expected_wh)
    if mask is None:
        raise ValueError(f"kalibracja: segmentacja w oknie priora nieudana ({seg})")
    scale, offset, meta = infer_nominal_from_masks(landmarks, mask)
    # Osobno szerokość i wysokość: segment ucięty na oczach (samo czoło) ma dobrą
    # szerokość, ale za małą wysokość, a średnia obu mogłaby zmieścić się w zakresie.
    scale_w = meta["th_size"][0] / meta["rgb_size"][0]
    scale_h = meta["th_size"][1] / meta["rgb_size"][1]
    if not (scale_is_plausible(scale_w) and scale_is_plausible(scale_h)):
        raise ValueError(
            f"nieprawdopodobna skala segmentu: szer. {scale_w:.2f}, wys. {scale_h:.2f} "
            f"(zakres {REG_SCALE_RANGE[0]}–{REG_SCALE_RANGE[1]}) — segment w termice to nie głowa"
        )
    return {
        "REG_NOMINAL_SCALE": scale,
        "REG_NOMINAL_OFFSET": [offset[0], offset[1]],
        "REG_WINDOW_PAD": float(window_pad),
        "meta": {**meta, "seg": seg if isinstance(seg, dict) else {"msg": str(seg)}},
    }


def apply_nominal_calibration(calib: dict) -> None:
    """Ustawia stałe nominalne w module (runtime; bez zapisu config.py)."""
    global REG_NOMINAL_SCALE, REG_NOMINAL_OFFSET, REG_WINDOW_PAD
    REG_NOMINAL_SCALE = float(calib["REG_NOMINAL_SCALE"])
    off = calib["REG_NOMINAL_OFFSET"]
    REG_NOMINAL_OFFSET = (float(off[0]), float(off[1]))
    if "REG_WINDOW_PAD" in calib:
        REG_WINDOW_PAD = float(calib["REG_WINDOW_PAD"])


def registration_quality(
    affine: np.ndarray, thermal_mask: np.ndarray, rgb_mask: np.ndarray
) -> dict:
    """Contour RMS, błąd centroidu i IoU maski termicznej po warp na RGB."""
    src = contour_sample_points(thermal_mask)
    dst = contour_sample_points(rgb_mask)
    if src is None or dst is None:
        return {
            "contour_rms_px": float("nan"),
            "centroid_err_px": float("nan"),
            "iou": float("nan"),
            "trustworthy": False,
        }
    resid = np.linalg.norm(apply_affine(affine, src) - dst, axis=1)
    crms = float(np.sqrt(np.mean(resid**2)))
    ys, xs = np.where(thermal_mask)
    ry, rx = np.where(rgb_mask)
    cent = float(
        np.linalg.norm(
            apply_affine(affine, np.array([[xs.mean(), ys.mean()]]))[0]
            - np.array([rx.mean(), ry.mean()])
        )
    )
    # IoU: maska termiczna zmapowana na RGB przez remap (poprawna geometria)
    rgb_h, rgb_w = rgb_mask.shape[:2]
    ys_r, xs_r = np.mgrid[0:rgb_h, 0:rgb_w]
    inv = invert_affine(affine)
    pts = apply_affine(
        inv,
        np.column_stack([xs_r.ravel().astype(np.float64), ys_r.ravel().astype(np.float64)]),
    )
    map_x = pts[:, 0].reshape(rgb_h, rgb_w).astype(np.float32)
    map_y = pts[:, 1].reshape(rgb_h, rgb_w).astype(np.float32)
    warped = cv2.remap(
        thermal_mask.astype(np.uint8),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    inter = np.logical_and(warped, rgb_mask).sum()
    union = np.logical_or(warped, rgb_mask).sum()
    iou = float(inter / max(1, union))
    return {
        "contour_rms_px": crms,
        "centroid_err_px": cent,
        "iou": iou,
        "trustworthy": bool(crms <= REG_GO_RMS_PX),
        "go_threshold_px": float(REG_GO_RMS_PX),
    }


@dataclass(frozen=True)
class SceneRegistrationQuality:
    """Jakość rejestracji całej sceny z prób odświeżenia affine.

    Kolejność pól = kolejność kluczy w ``reg_quality.json``.
    """

    contour_rms_px: float  # mediana RMS konturu z udanych prób
    iou: float  # mediana IoU z udanych prób
    rms_first_px: float
    iou_first: float
    good_frac: float  # udział prób z RMS <= progu
    n_attempts: int
    trustworthy: bool  # mediana RMS <= progu
    go_threshold_px: float

    @property
    def gate_label(self) -> str:
        """Status gated: ``trustworthy`` albo ``NO-GO``."""
        return "trustworthy" if self.trustworthy else "NO-GO"
