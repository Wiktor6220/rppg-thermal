"""scripts/probe_registration.py — DIAGNOSTYKA korejestracji RGB↔termika (rozdz. 4.4).

Tylko podgląd — NIE liczy żadnej transformacji, nie rusza extract.py. Bierze JEDNĄ parę
klatek RGB+termika z tej samej chwili (parowanie po CZASIE z io_layer; fps różny:
29.97 vs 30.0) z subject01/s1_rest_rest i zapisuje do results/registration_probe/:
  - pełne klatki (dokładne piksele): rgb_full.png (3840x2160), thermal_full.png (1280x1024),
  - wersje z SIATKĄ współrzędnych pikselowych (osie + grid) do ręcznego odczytu punktów
    charakterystycznych wspólnych dla obu obrazów: rgb_grid.png, thermal_grid.png.

Następny krok (dopiero po podaniu 6–8 par odpowiadających sobie punktów): policzenie
transformacji i nałożenie obrazów.

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

SUBJECT, SCENARIO = "subject01", "s1_rest_rest"

# Przybliżone pary punktów (piksele) — WSZYSTKIE na osobie (tło ma inną głębię/paralaksę).
# Format: ((x_rgb, y_rgb), (x_term, y_term)).
POINT_PAIRS = [
    ((2000, 880), (770, 400)),
    ((1660, 1400), (600, 640)),
    ((2420, 1400), (910, 640)),
    ((2010, 1240), (775, 560)),
    ((1760, 2010), (640, 950)),
]


def _save_grid(frame_rgb: np.ndarray, path: Path, major: int, minor: int, title: str) -> None:
    """Zapisuje klatkę z osiami w PIKSELACH i siatką (major/minor) do odczytu współrzędnych."""
    height, width = frame_rgb.shape[:2]
    fig_w = width / 190.0
    fig_h = height / 190.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
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


def _save_overlay(rgb: np.ndarray, warped_thermal: np.ndarray, full_path: Path) -> Path:
    """RGB + półprzezroczysta przekształcona termika w kanale czerwonym (alpha 0.5).

    Zapisuje pełną rozdzielczość oraz lżejszy podgląd (*_preview.png); zwraca ścieżkę podglądu.
    """
    thermal_gray = cv2.cvtColor(warped_thermal, cv2.COLOR_RGB2GRAY)
    thermal_red = np.zeros_like(rgb)
    thermal_red[..., 0] = thermal_gray  # kanał R (obraz w układzie RGB)
    blended = cv2.addWeighted(rgb, 1.0, thermal_red, 0.5, 0)

    cv2.imwrite(str(full_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    preview_path = full_path.with_name(full_path.stem + "_preview.png")
    ph = int(round(blended.shape[0] * 1600 / blended.shape[1]))
    preview = cv2.resize(blended, (1600, ph), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    return preview_path


def _run_registration(rgb_frame: np.ndarray, thermal_frame: np.ndarray, out_dir: Path) -> None:
    """Liczy transformacje termika->RGB (similarity + homografia), warping, nakładki, residua."""
    rgb_pts = np.array([pair[0] for pair in POINT_PAIRS], dtype=np.float64)
    thermal_pts = np.array([pair[1] for pair in POINT_PAIRS], dtype=np.float64)
    height, width = rgb_frame.shape[:2]

    src = thermal_pts.astype(np.float32)
    dst = rgb_pts.astype(np.float32)
    affine, inliers_aff = cv2.estimateAffinePartial2D(src, dst)
    homography, inliers_h = cv2.findHomography(src, dst, method=0)

    warp_affine = cv2.warpAffine(thermal_frame, affine, (width, height))
    warp_homography = cv2.warpPerspective(thermal_frame, homography, (width, height))

    prev_aff = _save_overlay(rgb_frame, warp_affine, out_dir / "overlay_affine.png")
    prev_h = _save_overlay(rgb_frame, warp_homography, out_dir / "overlay_homography.png")

    res_affine = np.linalg.norm(_apply_affine(affine, thermal_pts) - rgb_pts, axis=1)
    res_homography = np.linalg.norm(_apply_homography(homography, thermal_pts) - rgb_pts, axis=1)

    n_aff = int(inliers_aff.sum()) if inliers_aff is not None else len(POINT_PAIRS)
    n_h = int(inliers_h.sum()) if inliers_h is not None else len(POINT_PAIRS)
    n_pairs = len(POINT_PAIRS)
    print("\n=== Residua reprojekcji termika->RGB [piksele RGB] ===")
    print(f"(similarity inliers: {n_aff}/{n_pairs}, homografia inliers: {n_h}/{n_pairs})")
    print(f"{'punkt (x_rgb,y_rgb)':<22}{'similarity':>12}{'homografia':>12}")
    for i, (rgb_pt, _) in enumerate(POINT_PAIRS):
        label = f"({rgb_pt[0]},{rgb_pt[1]})"
        print(f"{label:<22}{res_affine[i]:>12.1f}{res_homography[i]:>12.1f}")
    rms_aff = float(np.sqrt(np.mean(res_affine**2)))
    rms_h = float(np.sqrt(np.mean(res_homography**2)))
    print(f"{'RMS':<22}{rms_aff:>12.1f}{rms_h:>12.1f}")
    print(f"{'max':<22}{res_affine.max():>12.1f}{res_homography.max():>12.1f}")

    print("\nNakładki (RGB + termika w czerwonym, alpha 0.5):")
    for name in ("overlay_affine.png", "overlay_homography.png"):
        print(f"  - {name}  (+ {name.replace('.png', '_preview.png')})")
    print(f"Podglądy do szybkiego obejrzenia: {prev_aff.name}, {prev_h.name}")


def main() -> None:
    loaded = load_recording(SUBJECT, SCENARIO)
    rgb_frame, thermal_frame, t_seconds = next(loaded.synced_pairs(reference="rgb"))

    print(f"Nagranie: {SUBJECT}/{SCENARIO}")
    print(f"  para zsynchronizowana w czasie: t = {t_seconds:.3f} s (odniesienie: RGB)")
    print(f"  RGB    : {loaded.rgb_meta.fps:.3f} fps, klatka {rgb_frame.shape}")
    print(f"  Termika: {loaded.thermal_meta.fps:.3f} fps, klatka {thermal_frame.shape}")

    out_dir = RESULTS_DIR / "registration_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pełne klatki — dokładne piksele (RGB->BGR do zapisu przez OpenCV).
    cv2.imwrite(str(out_dir / "rgb_full.png"), cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "thermal_full.png"), cv2.cvtColor(thermal_frame, cv2.COLOR_RGB2BGR))

    # Wersje z siatką współrzędnych.
    rh, rw = rgb_frame.shape[:2]
    rgb_title = f"RGB {rw}x{rh} — t={t_seconds:.3f}s (siatka 256/64 px)"
    _save_grid(rgb_frame, out_dir / "rgb_grid.png", major=256, minor=64, title=rgb_title)
    _save_grid(
        thermal_frame, out_dir / "thermal_grid.png", major=128, minor=32,
        title=f"Termika {thermal_frame.shape[1]}x{thermal_frame.shape[0]} — t={t_seconds:.3f}s "
        f"(siatka 128/32 px)",
    )

    print(f"\nZapisano klatki do: {out_dir}")
    for name in ("rgb_full.png", "thermal_full.png", "rgb_grid.png", "thermal_grid.png"):
        print(f"  - {name}")

    # Transformacja termika->RGB na podanych parach punktów.
    _run_registration(rgb_frame, thermal_frame, out_dir)


if __name__ == "__main__":
    main()
