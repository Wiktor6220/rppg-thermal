"""scripts/probe_registration.py — DIAGNOSTYKA korejestracji RGB↔termika (rozdz. 4.4).

Tylko podgląd — NIE liczy modułu, nie rusza extract.py ani src/. Bierze JEDNĄ parę klatek
RGB+termika z tej samej chwili (parowanie po CZASIE; fps 29.97 vs 30.0) z subject01/s1 i:

1. Punkty po stronie RGB wyznacza AUTOMATYCZNIE — landmarki twarzy (make_cropping_detector,
   ta sama ścieżka detekcji co w potoku).
2. Odpowiadające punkty termiczne są stałą (THERMAL). Punkty bez współrzędnych termicznych
   (None) są POMIJANE w dopasowaniu — czekają na wartości odczytane z gridu.
3. Dopasowuje TYLKO affine (estimateAffine2D, LS) termika->RGB; residua per punkt, RMS, max,
   oraz osobno RMS na rejonach PERFUZJI (skronie/policzki/czoło) — to metryka decydująca
   o masce ROI.
4. Leave-one-out: dla każdego punktu affine na pozostałych → błąd predykcji na wyłączonym.
5. Zapisuje nakładkę affine (RGB + termika w czerwonym, alpha 0.5).

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

# Definicje punktów kontrolnych: (indeks landmarku RGB, etykieta, punkt termiczny lub None,
# czy należy do rejonu PERFUZJI — skronie/policzki/czoło). Czubek nosa (4) usunięty (poza
# płaszczyzną twarzy). Punkty z termika=None czekają na odczyt z gridu.
POINTS = [
    (127, "skroń L (127)", (615.0, 370.0), True),
    (356, "skroń P (356)", (742.0, 370.0), True),
    (168, "nasada nosa (168)", (675.0, 378.0), False),
    (61, "kącik ust L (61)", (652.0, 468.0), False),
    (291, "kącik ust P (291)", (700.0, 468.0), False),
    (205, "policzek L (205)", None, True),  # TODO: termika z gridu
    (425, "policzek P (425)", None, True),  # TODO: termika z gridu
    (9, "czoło środek (9)", None, True),  # TODO: termika z gridu
    (152, "broda (152)", None, False),  # TODO: termika z gridu
]


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


def _fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Dopasowanie affine LS (wysoki próg RANSAC → wszystkie punkty jako inliery)."""
    matrix, _ = cv2.estimateAffine2D(
        src.astype(np.float32), dst.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=1e6,
    )
    return matrix


def _save_overlay(rgb: np.ndarray, warped_thermal: np.ndarray, marks, full_path: Path) -> Path:
    """RGB + termika w kanale czerwonym (alpha 0.5) + kolorowe punkty; zwraca ścieżkę podglądu."""
    thermal_gray = cv2.cvtColor(warped_thermal, cv2.COLOR_RGB2GRAY)
    thermal_red = np.zeros_like(rgb)
    thermal_red[..., 0] = thermal_gray
    blended = cv2.addWeighted(rgb, 1.0, thermal_red, 0.5, 0)
    for x, y, color in marks:
        cv2.circle(blended, (int(x), int(y)), 12, color, 2)

    cv2.imwrite(str(full_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    preview_path = full_path.with_name(full_path.stem + "_preview.png")
    ph = int(round(blended.shape[0] * 1600 / blended.shape[1]))
    preview = cv2.resize(blended, (1600, ph), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    return preview_path


def main() -> None:
    loaded = load_recording(SUBJECT, SCENARIO)
    rgb_frame, thermal_frame, t_seconds = next(loaded.synced_pairs(reference="rgb"))
    print(f"Nagranie: {SUBJECT}/{SCENARIO}  para w czasie t={t_seconds:.3f}s")
    print(f"  RGB {rgb_frame.shape}  termika {thermal_frame.shape}")

    out_dir = RESULTS_DIR / "registration_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    rh, rw = rgb_frame.shape[:2]
    th, tw = thermal_frame.shape[:2]
    _save_grid(rgb_frame, out_dir / "rgb_grid.png", 256, 64, f"RGB {rw}x{rh} (siatka 256/64 px)")
    _save_grid(thermal_frame, out_dir / "thermal_grid.png", 128, 32,
               f"Termika {tw}x{th} (siatka 128/32 px)")

    detector = make_cropping_detector()
    landmarks = detector(rgb_frame)
    if landmarks is None:
        raise RuntimeError("Nie wykryto twarzy na klatce RGB.")

    # RGB współrzędne dla wszystkich zdefiniowanych punktów.
    rgb_xy = {idx: landmarks[idx] for idx, _, _, _ in POINTS}
    print("\nPunkty RGB z landmarków (px oryginału):")
    for idx, label, thermal, _ in POINTS:
        status = "" if thermal is not None else "  [brak termiki — czekam na wartość z gridu]"
        x, y = rgb_xy[idx]
        print(f"  {label:<18} ({x:7.1f}, {y:7.1f}){status}")

    used = [p for p in POINTS if p[2] is not None]
    pending = [p for p in POINTS if p[2] is None]
    src = np.array([t for _, _, t, _ in used], dtype=np.float64)
    dst = np.array([rgb_xy[idx] for idx, _, _, _ in used], dtype=np.float64)

    # Affine na wszystkich użytych punktach.
    affine = _fit_affine(src, dst)
    residuals = np.linalg.norm(_apply_affine(affine, src) - dst, axis=1)

    print("\n=== Affine (estimateAffine2D, LS) — residua reprojekcji [px RGB] ===")
    print(f"(użyte punkty: {len(used)}; oczekujące na termikę: {len(pending)})")
    for i, (_, label, _, _) in enumerate(used):
        print(f"  {label:<18} {residuals[i]:8.1f}")
    print(f"  {'RMS (wszystkie)':<18} {np.sqrt(np.mean(residuals**2)):8.1f}")
    print(f"  {'max':<18} {residuals.max():8.1f}")

    perf_mask = np.array([perf for _, _, _, perf in used])
    if perf_mask.any():
        perf_rms = float(np.sqrt(np.mean(residuals[perf_mask] ** 2)))
        perf_labels = [label for (_, label, _, perf) in used if perf]
        print(f"  {'RMS (perfuzja)':<18} {perf_rms:8.1f}   [{', '.join(perf_labels)}]")

    # Leave-one-out: affine na pozostałych, błąd na wyłączonym.
    print("\n=== Leave-one-out (affine na pozostałych) — błąd predykcji [px RGB] ===")
    loo_errors = []
    for i, (_, label, thermal, _) in enumerate(used):
        others = [j for j in range(len(used)) if j != i]
        matrix = _fit_affine(src[others], dst[others])
        pred = _apply_affine(matrix, np.array([thermal], dtype=np.float64))[0]
        err = float(np.linalg.norm(pred - dst[i]))
        loo_errors.append(err)
        print(f"  {label:<18} {err:8.1f}")
    print(f"  {'RMS LOO':<18} {np.sqrt(np.mean(np.square(loo_errors))):8.1f}")

    # Nakładka affine: użyte punkty na zielono, oczekujące na cyjan (RGB pozycje).
    warped = cv2.warpAffine(thermal_frame, affine, (rw, rh))
    marks = [(*rgb_xy[idx], (0, 255, 0)) for idx, _, _, _ in used]
    marks += [(*rgb_xy[idx], (0, 255, 255)) for idx, _, _, _ in pending]
    preview = _save_overlay(rgb_frame, warped, marks, out_dir / "overlay_affine.png")

    print(f"\nNakładka: overlay_affine.png (+ {preview.name})")
    if pending:
        print("\nOczekują na termikę (odczytaj z thermal_grid.png); pozycje RGB wypisane wyżej:")
        for _, label, _, _ in pending:
            print(f"  - {label}")
    print(f"\nZapisano do: {out_dir}")


if __name__ == "__main__":
    main()
