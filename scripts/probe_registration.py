"""scripts/probe_registration.py — DIAGNOSTYKA korejestracji RGB↔termika (rozdz. 4.4).

Tylko podgląd — NIE liczy modułu, nie rusza extract.py ani src/. Bierze JEDNĄ parę klatek
RGB+termika z tej samej chwili (parowanie po CZASIE; fps 29.97 vs 30.0) z subject01/s1 i:

1. Punkty po stronie RGB wyznacza AUTOMATYCZNIE — landmarki twarzy (make_cropping_detector,
   ta sama ścieżka detekcji co w potoku), indeksy 127/356 (skronie), 168 (nasada nosa),
   4 (czubek nosa), 61/291 (kąciki ust) → 6 punktów w pikselach oryginału.
2. Odpowiadające punkty termiczne są stałą THERMAL_POINTS (do doprecyzowania na gridzie).
3. Dopasowuje i porównuje TRZY transformacje termika->RGB: similarity
   (estimateAffinePartial2D), pełna affine (estimateAffine2D) i homografia (findHomography)
   — residua per punkt, RMS i max [px RGB] w jednej tabeli.
4. Zapisuje nakładki (RGB + zwarpowana termika w czerwonym, alpha 0.5) dla wszystkich trzech.

Uruchomienie: uv run python scripts/probe_registration.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402
from src.io_layer import load_recording  # noqa: E402
from src.roi import make_cropping_detector  # noqa: E402

SUBJECT, SCENARIO = "subject01", "s1_rest_rest"

# Landmarki twarzy (Face Mesh 468) po stronie RGB — kolejność == THERMAL_POINTS.
RGB_LANDMARK_INDICES = [127, 356, 168, 4, 61, 291]
POINT_LABELS = [
    "skroń L (127)", "skroń P (356)", "nasada nosa (168)",
    "czubek nosa (4)", "kącik ust L (61)", "kącik ust P (291)",
]
# Punkty termiczne w pikselach termiki (1280x1024), ta sama kolejność co wyżej.
THERMAL_POINTS = np.array(
    [[615, 370], [742, 370], [675, 378], [675, 440], [652, 468], [700, 468]],
    dtype=np.float64,
)


def _save_grid(frame_rgb: np.ndarray, path: Path, major: int, minor: int, title: str) -> None:
    """Zapisuje klatkę z osiami w PIKSELACH i siatką (major/minor) do odczytu współrzędnych."""
    height, width = frame_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(width / 190.0, height / 190.0))
    ax.imshow(frame_rgb, extent=(0, width, height, 0), interpolation="nearest")
    ax.set_xticks(np.arange(0, width + 1, major))
    ax.set_yticks(np.arange(0, height + 1, major))
    ax.set_xticks(np.arange(0, width + 1, minor), minor=True)
    ax.set_yticks(np.arange(0, height + 1, minor), minor=True)
    ax.grid(which="major", color="yellow", alpha=0.7, linewidth=0.8)
    ax.grid(which="minor", color="yellow", alpha=0.28, linewidth=0.4)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_title(title, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _apply_affine(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Przekształca punkty (N,2) macierzą afiniczną 2x3."""
    return pts @ matrix[:, :2].T + matrix[:, 2]


def _apply_homography(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Przekształca punkty (N,2) homografią 3x3 (z normalizacją perspektywy)."""
    homog = np.c_[pts, np.ones(len(pts))] @ matrix.T
    return homog[:, :2] / homog[:, 2:3]


def _save_overlay(
    rgb: np.ndarray, warped_thermal: np.ndarray, rgb_points: np.ndarray, full_path: Path
) -> Path:
    """RGB + półprzezroczysta termika w kanale czerwonym (alpha 0.5) + zielone punkty RGB.

    Zapisuje pełną rozdzielczość oraz lżejszy podgląd (*_preview.png); zwraca ścieżkę podglądu.
    """
    thermal_gray = cv2.cvtColor(warped_thermal, cv2.COLOR_RGB2GRAY)
    thermal_red = np.zeros_like(rgb)
    thermal_red[..., 0] = thermal_gray
    blended = cv2.addWeighted(rgb, 1.0, thermal_red, 0.5, 0)
    for x, y in rgb_points:
        cv2.circle(blended, (int(x), int(y)), 12, (0, 255, 0), 2)

    cv2.imwrite(str(full_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    preview_path = full_path.with_name(full_path.stem + "_preview.png")
    ph = int(round(blended.shape[0] * 1600 / blended.shape[1]))
    preview = cv2.resize(blended, (1600, ph), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    return preview_path


def _detect_rgb_points(rgb_frame: np.ndarray) -> np.ndarray:
    """Wykrywa landmarki twarzy na klatce RGB i zwraca 6 punktów (px oryginału)."""
    detector = make_cropping_detector()  # ta sama ścieżka co w potoku (twarz jest mała)
    landmarks = detector(rgb_frame)
    if landmarks is None:
        raise RuntimeError("Nie wykryto twarzy na klatce RGB — nie da się pobrać landmarków.")
    return landmarks[RGB_LANDMARK_INDICES]


def _fit_and_report(rgb_frame: np.ndarray, thermal_frame: np.ndarray, rgb_pts: np.ndarray,
                    out_dir: Path) -> None:
    """Dopasowuje 3 transformacje termika->RGB, warpuje, zapisuje nakładki i wypisuje residua."""
    height, width = rgb_frame.shape[:2]
    src = THERMAL_POINTS.astype(np.float32)
    dst = rgb_pts.astype(np.float32)
    # Wysoki próg RANSAC → wszystkie punkty jako inliery (dopasowanie LS na 6 punktach).
    big = 1e6
    similarity, _ = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=big
    )
    affine, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=big)
    homography, _ = cv2.findHomography(src, dst, method=0)

    variants = {
        "similarity": (similarity, _apply_affine, cv2.warpAffine),
        "affine": (affine, _apply_affine, cv2.warpAffine),
        "homography": (homography, _apply_homography, cv2.warpPerspective),
    }

    residuals: dict[str, np.ndarray] = {}
    for name, (matrix, apply_fn, warp_fn) in variants.items():
        residuals[name] = np.linalg.norm(apply_fn(matrix, THERMAL_POINTS) - rgb_pts, axis=1)
        warped = warp_fn(thermal_frame, matrix, (width, height))
        _save_overlay(rgb_frame, warped, rgb_pts, out_dir / f"overlay_{name}.png")

    names = list(variants)
    print("\n=== Residua reprojekcji termika->RGB [piksele RGB] ===")
    print(f"{'punkt':<20}" + "".join(f"{n:>13}" for n in names))
    for i, label in enumerate(POINT_LABELS):
        print(f"{label:<20}" + "".join(f"{residuals[n][i]:>13.1f}" for n in names))
    print(f"{'RMS':<20}" + "".join(f"{np.sqrt(np.mean(residuals[n] ** 2)):>13.1f}" for n in names))
    print(f"{'max':<20}" + "".join(f"{residuals[n].max():>13.1f}" for n in names))
    print("\nNakładki: " + ", ".join(f"overlay_{n}.png (+_preview)" for n in names))


def main() -> None:
    loaded = load_recording(SUBJECT, SCENARIO)
    rgb_frame, thermal_frame, t_seconds = next(loaded.synced_pairs(reference="rgb"))
    print(f"Nagranie: {SUBJECT}/{SCENARIO}  para w czasie t={t_seconds:.3f}s")
    print(f"  RGB {rgb_frame.shape}  termika {thermal_frame.shape}")

    out_dir = RESULTS_DIR / "registration_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "rgb_full.png"), cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "thermal_full.png"), cv2.cvtColor(thermal_frame, cv2.COLOR_RGB2BGR))
    rh, rw = rgb_frame.shape[:2]
    th, tw = thermal_frame.shape[:2]
    _save_grid(rgb_frame, out_dir / "rgb_grid.png", 256, 64, f"RGB {rw}x{rh} (siatka 256/64 px)")
    _save_grid(thermal_frame, out_dir / "thermal_grid.png", 128, 32,
               f"Termika {tw}x{th} (siatka 128/32 px)")

    rgb_pts = _detect_rgb_points(rgb_frame)
    print("\nPunkty RGB z landmarków (px oryginału):")
    for label, (x, y) in zip(POINT_LABELS, rgb_pts, strict=True):
        print(f"  {label:<20} ({x:.1f}, {y:.1f})")

    _fit_and_report(rgb_frame, thermal_frame, rgb_pts, out_dir)
    print(f"\nZapisano do: {out_dir}")


if __name__ == "__main__":
    main()
