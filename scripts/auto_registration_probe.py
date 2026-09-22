"""scripts/auto_registration_probe.py — automat korejestracji RGB↔termika (go/no-go).

DIAGNOSTYKA subject01/s1 (nie rusza src/, nie generalizuje na inne nagrania). Klasa
transformacji zablokowana = pełna affine (estimateAffine2D, LS). Strona RGB automatyczna
(make_cropping_detector, 468 landmarków). Strona termiczna automatyczna: okno z NOMINALNEGO
odwzorowania RGB→termika (odcięcie torsu) + segmentacja (Otsu+morfologia+największa składowa
+ cięcie szyi) → cechy twarzy. Walidacja względem RĘCZNEGO ground truth (MANUAL_THERMAL).

Uruchomienie: uv run python scripts/auto_registration_probe.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.config import PERFUSION_TEMP_STD_FACTOR, RESULTS_DIR  # noqa: E402
from src.io_layer import load_recording  # noqa: E402
from src.roi import make_cropping_detector, select_roi_from_landmarks  # noqa: E402

SUBJECT, SCENARIO = "subject01", "s1_rest_rest"

# --- Nominalne stałe odwzorowania RGB->termika (tylko do okna, nie do precyzji) ---
NOMINAL_SCALE = 0.52  # odwrotność ~1.93 z kalibracji
NOMINAL_OFFSET = (-340.0, -60.0)  # [px termiki]; twarz na termice siedzi wyżej
WINDOW_PAD = 2.0  # hojne powiększenie okna wokół nominalnej twarzy
MORPH_KERNEL = 7
NECK_WIDTH_FRAC = 0.62  # próg przewężenia szyi (ułamek maks. szerokości twarzy)

# --- RĘCZNY GROUND TRUTH (subject01/s1, t=0) — z probe_registration.py ---
# (indeks landmarku RGB, etykieta, punkt termiczny [px], rejon perfuzji?)
GT_POINTS = [
    (127, "skroń L", (667.1, 491.6), True),
    (356, "skroń P", (789.4, 491.6), True),
    (9, "czoło", (721.6, 453.4), True),
    (205, "policzek L", (681.1, 540.7), True),
    (425, "policzek P", (767.6, 544.6), True),
    (61, "usta L", (701.4, 554.0), False),
    (291, "usta P", (735.7, 552.4), False),
]

# Indeksy RGB dla AUTO-korespondencji (usta/oczy jako środki).
IDX_TEMPLE_L, IDX_TEMPLE_R, IDX_FOREHEAD = 127, 356, 10
IDX_EYE_L, IDX_EYE_R = 33, 263  # zewnętrzne kąciki oczu
IDX_MOUTH_L, IDX_MOUTH_R = 61, 291


def rgb_auto_targets(landmarks: np.ndarray) -> dict[str, np.ndarray]:
    """Punkty docelowe RGB dla auto-korespondencji (z landmarków)."""
    return {
        "temple_L": landmarks[IDX_TEMPLE_L],
        "temple_R": landmarks[IDX_TEMPLE_R],
        "forehead": landmarks[IDX_FOREHEAD],
        "eye_center": (landmarks[IDX_EYE_L] + landmarks[IDX_EYE_R]) / 2.0,
        "mouth": (landmarks[IDX_MOUTH_L] + landmarks[IDX_MOUTH_R]) / 2.0,
    }


def nominal_window(landmarks: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Hojne okno w termice z nominalnego odwzorowania bboxa twarzy RGB (odcina tors)."""
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    fw, fh = xs.max() - xs.min(), ys.max() - ys.min()
    tcx = NOMINAL_SCALE * cx + NOMINAL_OFFSET[0]
    tcy = NOMINAL_SCALE * cy + NOMINAL_OFFSET[1]
    tw, th = NOMINAL_SCALE * fw * WINDOW_PAD, NOMINAL_SCALE * fh * WINDOW_PAD
    height, width = shape
    x0 = max(0, int(tcx - tw / 2))
    x1 = min(width, int(tcx + tw / 2))
    y0 = max(0, int(tcy - th / 2))
    y1 = min(height, int(tcy + th / 2))
    return x0, y0, x1, y1


def segment_thermal_face(gray: np.ndarray, window: tuple[int, int, int, int]):
    """Segmentuje twarz w oknie: Otsu+morfologia+największa składowa+cięcie szyi.

    Zwraca (mask_full_bool, info) albo (None, powód).
    """
    x0, y0, x1, y1 = window
    win = gray[y0:y1, x0:x1]
    if win.size == 0:
        return None, "okno puste"
    _, binw = cv2.threshold(win, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    binw = cv2.morphologyEx(binw, cv2.MORPH_OPEN, kernel)
    binw = cv2.morphologyEx(binw, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binw, connectivity=8)
    if n <= 1:
        return None, "brak spójnej składowej po progowaniu"
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (labels == biggest).astype(np.uint8)

    # Cięcie szyi: poniżej najszerszego wiersza, pierwszy wyraźnie węższy wiersz.
    rows_w = comp.sum(axis=1)
    max_w = int(rows_w.max())
    widest = int(np.argmax(rows_w))
    for r in range(widest, comp.shape[0]):
        if rows_w[r] < NECK_WIDTH_FRAC * max_w:
            comp[r:, :] = 0
            break

    mask = np.zeros_like(gray, dtype=bool)
    mask[y0:y1, x0:x1] = comp.astype(bool)
    if not mask.any():
        return None, "pusta maska po cięciu szyi"
    return mask, {"area": int(mask.sum()), "window": window}


def thermal_features(gray: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Cechy twarzy z maski termicznej: skronie/czoło/środek oczu/usta (px termiki)."""
    ys, xs = np.where(mask)
    top, bot = int(ys.min()), int(ys.max())
    height = bot - top
    mid = (top + bot) // 2
    feats: dict[str, np.ndarray] = {}

    # Pas oczu: najciemniejszy wiersz w górnej połowie maski (min. 8 px w wierszu).
    best_row, best_val = None, np.inf
    for r in range(top, mid + 1):
        cols = np.where(mask[r])[0]
        if cols.size < 8:
            continue
        val = float(gray[r, cols].mean())
        if val < best_val:
            best_val, best_row = val, r
    if best_row is not None:
        cols = np.where(mask[best_row])[0]
        feats["temple_L"] = np.array([cols.min(), best_row], float)
        feats["temple_R"] = np.array([cols.max(), best_row], float)
        feats["eye_center"] = np.array([(cols.min() + cols.max()) / 2.0, best_row], float)

    # Czoło: górny skraj bboxa, środek x z pasa górnych wierszy.
    band = max(1, int(0.10 * height))
    top_cols = np.where(mask[top:top + band].any(axis=0))[0]
    if top_cols.size:
        feats["forehead"] = np.array([(top_cols.min() + top_cols.max()) / 2.0, top], float)

    # Usta: najciemniejszy wiersz w DOLNEJ tercji maski (poniżej nozdrzy).
    lower_start = top + int(0.66 * height)
    best_row, best_val = None, np.inf
    for r in range(lower_start, bot + 1):
        cols = np.where(mask[r])[0]
        if cols.size < 6:
            continue
        val = float(gray[r, cols].mean())
        if val < best_val:
            best_val, best_row = val, r
    if best_row is not None:
        cols = np.where(mask[best_row])[0]
        feats["mouth"] = np.array([(cols.min() + cols.max()) / 2.0, best_row], float)
    return feats


def fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Affine LS (wysoki próg RANSAC → wszystkie punkty jako inliery)."""
    matrix, _ = cv2.estimateAffine2D(
        src.astype(np.float32), dst.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=1e6,
    )
    return matrix


def apply_affine(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ matrix[:, :2].T + matrix[:, 2]


def auto_affine(rgb: np.ndarray, thermal: np.ndarray, detector):
    """Pełny auto-pipeline dla jednej klatki. Zwraca (affine, landmarks, feats, mask) lub None."""
    landmarks = detector(rgb)
    if landmarks is None:
        return None, "brak detekcji twarzy RGB"
    gray = cv2.cvtColor(thermal, cv2.COLOR_RGB2GRAY)
    window = nominal_window(landmarks, gray.shape)
    mask, info = segment_thermal_face(gray, window)
    if mask is None:
        return None, f"segmentacja: {info}"
    feats = thermal_features(gray, mask)
    targets = rgb_auto_targets(landmarks)
    common = [k for k in targets if k in feats]
    if len(common) < 4:
        return None, f"za mało cech ({len(common)}): {common}"
    src = np.array([feats[k] for k in common], float)
    dst = np.array([targets[k] for k in common], float)
    affine = fit_affine(src, dst)
    return {"affine": affine, "landmarks": landmarks, "feats": feats,
            "targets": targets, "common": common, "mask": mask, "gray": gray}, None


def _test_grid() -> np.ndarray:
    """Siatka punktów testowych w przestrzeni termiki, pokrywająca twarz (z GT bbox)."""
    tp = np.array([p[2] for p in GT_POINTS])
    xs = np.linspace(tp[:, 0].min() - 10, tp[:, 0].max() + 10, 5)
    ys = np.linspace(tp[:, 1].min() - 10, tp[:, 1].max() + 10, 5)
    gx, gy = np.meshgrid(xs, ys)
    return np.c_[gx.ravel(), gy.ravel()]


def count_mask_pixels(warped_gray: np.ndarray, bbox: np.ndarray) -> int:
    """Liczba pikseli maski względnej (mean + k*std) w bboxie ROI (RGB)."""
    y0, x0, y1, x1 = (int(v) for v in bbox)
    region = warped_gray[y0:y1, x0:x1].astype(np.float64)
    if region.size == 0:
        return 0
    thr = region.mean() + PERFUSION_TEMP_STD_FACTOR * region.std()
    return int((region >= thr).sum())


def main() -> None:
    loaded = load_recording(SUBJECT, SCENARIO)
    n_frames = loaded.rgb_meta.frame_count
    sample_idx = list(np.linspace(0, n_frames - 1, 5, dtype=int))
    print(f"{SUBJECT}/{SCENARIO}: {n_frames} klatek; próbki stabilności: {sample_idx}")

    detector = make_cropping_detector()
    frames: dict[int, tuple] = {}
    for i, (rgb, thermal, _) in enumerate(loaded.synced_pairs(reference="rgb")):
        if i in sample_idx:
            frames[i] = (rgb.copy(), thermal.copy())
        if i >= sample_idx[-1]:
            break

    out_dir = RESULTS_DIR / "registration_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Klatka t=0: pełna diagnostyka ===
    rgb0, thermal0 = frames[sample_idx[0]]
    res, err = auto_affine(rgb0, thermal0, detector)
    if res is None:
        print(f"\n[NO-GO] Auto-detekcja zawiodła na t=0: {err}")
        return

    print("\n=== 1–2. Auto-detekcja termiki i korespondencje (t=0) ===")
    print(f"  wykryte cechy termiczne: {sorted(res['feats'])}")
    all_feats = {"temple_L", "temple_R", "forehead", "eye_center", "mouth"}
    print(f"  użyte w dopasowaniu: {res['common']}")
    missing = sorted(all_feats - set(res["common"]))
    print(f"  brakujące/odrzucone: {missing if missing else 'brak'}")

    affine_auto = res["affine"]
    landmarks0 = res["landmarks"]

    # Ręczna affine (GT): MANUAL_THERMAL -> landmarki RGB w tych indeksach.
    gt_thermal = np.array([p[2] for p in GT_POINTS], float)
    gt_rgb = np.array([landmarks0[p[0]] for p in GT_POINTS], float)
    affine_manual = fit_affine(gt_thermal, gt_rgb)

    # === 4a. Residuum auto-affine w ręcznych punktach GT ===
    pred = apply_affine(affine_auto, gt_thermal)
    resid = np.linalg.norm(pred - gt_rgb, axis=1)
    print("\n=== 4a. Residuum AUTO-affine w ręcznych punktach GT [px RGB] ===")
    for (_, label, _, _), r in zip(GT_POINTS, resid, strict=True):
        print(f"  {label:<11} {r:7.1f}")
    print(f"  {'RMS':<11} {np.sqrt(np.mean(resid**2)):7.1f}   max {resid.max():7.1f}")

    # === 4c. Rozbicie na rejony ===
    perf_fh_temple = [i for i, p in enumerate(GT_POINTS) if p[3] and "policzek" not in p[1]]
    cheeks = [i for i, p in enumerate(GT_POINTS) if "policzek" in p[1]]
    rms_ft = float(np.sqrt(np.mean(resid[perf_fh_temple] ** 2)))
    rms_ch = float(np.sqrt(np.mean(resid[cheeks] ** 2)))
    print("\n=== 4c. Residuum wg rejonów [px RGB] ===")
    print(f"  czoło+skronie: RMS {rms_ft:.1f}")
    print(f"  policzki:      RMS {rms_ch:.1f}")

    # === 4b. Rozjazd macierzy auto vs ręczna na siatce testowej ===
    grid = _test_grid()
    diff = np.linalg.norm(
        apply_affine(affine_auto, grid) - apply_affine(affine_manual, grid), axis=1
    )
    print("\n=== 4b. Rozjazd AUTO vs RĘCZNA na siatce testowej twarzy [px RGB] ===")
    print(f"  RMS {np.sqrt(np.mean(diff**2)):.1f}   max {diff.max():.1f}")

    # === 5. Stabilność na 5 klatkach ===
    print("\n=== 5. Stabilność auto-affine na 5 klatkach (rozjazd mapowania siatki) ===")
    mapped_grids: dict[int, np.ndarray] = {}
    per_frame = {}
    for idx in sample_idx:
        rgb_i, thermal_i = frames[idx]
        res_i, err_i = auto_affine(rgb_i, thermal_i, detector)
        if res_i is None:
            print(f"  klatka {idx}: [FAIL] {err_i}")
            continue
        mapped_grids[idx] = apply_affine(res_i["affine"], grid)
        per_frame[idx] = res_i
    if mapped_grids:
        mean_map = np.mean(list(mapped_grids.values()), axis=0)
        for idx, mg in mapped_grids.items():
            d = np.linalg.norm(mg - mean_map, axis=1)
            print(f"  klatka {idx:>4}: RMS odchył od średniej {np.sqrt(np.mean(d**2)):6.1f} px, "
                  f"max {d.max():6.1f} px  (cechy: {len(per_frame[idx]['common'])})")

    # === 6. Licznik pikseli maski per rejon per klatka ===
    print("\n=== 6. Piksele maski perfuzji (mean+k*std, k="
          f"{PERFUSION_TEMP_STD_FACTOR}) per rejon per klatka ===")
    regions = ["forehead", "left_cheek", "right_cheek"]
    print(f"  {'klatka':>6}  " + "".join(f"{r:>13}" for r in regions))
    for idx in sample_idx:
        if idx not in per_frame:
            continue
        res_i = per_frame[idx]
        rh, rw = frames[idx][0].shape[:2]
        warped = cv2.warpAffine(res_i["gray"], res_i["affine"], (rw, rh))
        counts = []
        for region in regions:
            bbox = select_roi_from_landmarks(res_i["landmarks"], region)
            counts.append(count_mask_pixels(warped, bbox))
        print(f"  {idx:>6}  " + "".join(f"{c:>13}" for c in counts))

    # === 7. Nakładka auto-affine (t=0) ===
    warped0 = cv2.warpAffine(res["gray"], affine_auto, (rgb0.shape[1], rgb0.shape[0]))
    thermal_red = np.zeros_like(rgb0)
    thermal_red[..., 0] = warped0
    blended = cv2.addWeighted(rgb0, 1.0, thermal_red, 0.5, 0)
    for k in res["common"]:
        x, y = res["targets"][k]
        cv2.circle(blended, (int(x), int(y)), 12, (0, 255, 0), 2)  # cel RGB auto
    for _, _, tpt, _ in GT_POINTS:
        px = apply_affine(affine_manual, np.array([tpt]))[0]
        cv2.circle(blended, (int(px[0]), int(px[1])), 8, (0, 255, 255), 2)  # GT ręczny (mapowany)
    full = out_dir / "overlay_auto_affine.png"
    cv2.imwrite(str(full), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    ph = int(round(blended.shape[0] * 1600 / blended.shape[1]))
    cv2.imwrite(str(out_dir / "overlay_auto_affine_preview.png"),
                cv2.cvtColor(cv2.resize(blended, (1600, ph)), cv2.COLOR_RGB2BGR))
    print(f"\nNakładka: overlay_auto_affine.png (+ _preview) w {out_dir}")


if __name__ == "__main__":
    main()
