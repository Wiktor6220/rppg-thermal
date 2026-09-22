"""scripts/pick_thermal_points.py — ręczny wybór punktów termicznych klikaniem (narzędzie).

Wyświetla pierwszą klatkę termiczną nagrania w pełnej rozdzielczości (matplotlib) i zbiera
N kliknięć przez `ginput`, w kolejności etykiet z LABELS. Na stdout wypisuje tablicę
THERMAL_POINTS gotową do wklejenia do scripts/probe_registration.py.

Narzędzie interaktywne (wymaga GUI) — uruchamiane ręcznie, NIE zapisuje do src/.
Uruchomienie: uv run python scripts/pick_thermal_points.py [--subject S] [--scenario SC]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402 - GUI backend (bez Agg, potrzebny ginput)

from src.io_layer import load_recording  # noqa: E402

# Kolejność MUSI odpowiadać punktom kontrolnym w probe_registration.py.
LABELS = [
    "skroń L (127)", "skroń P (356)", "czoło środek (9)",
    "policzek L (205)", "policzek P (425)", "kącik ust L (61)", "kącik ust P (291)",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Klikanie punktów termicznych (ginput).")
    parser.add_argument("--subject", default="subject01")
    parser.add_argument("--scenario", default="s1_rest_rest")
    args = parser.parse_args()

    loaded = load_recording(args.subject, args.scenario)
    frame = next(loaded.thermal_frames(to_rgb=True))
    height, width = frame.shape[:2]
    print(f"Termika {args.subject}/{args.scenario}: {width}x{height}")
    print(f"Kliknij {len(LABELS)} punktów w kolejności: {', '.join(LABELS)}")

    fig, ax = plt.subplots(figsize=(width / 120.0, height / 120.0))
    ax.imshow(frame, interpolation="nearest")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    picked = []
    for i, label in enumerate(LABELS):
        ax.set_title(f"Kliknij: {label}   ({i + 1}/{len(LABELS)})")
        fig.canvas.draw()
        clicks = plt.ginput(1, timeout=0)
        if not clicks:
            print("Przerwano — brak kliknięcia.")
            plt.close(fig)
            return
        x, y = clicks[0]
        picked.append((x, y))
        ax.plot(x, y, "g+", markersize=14, markeredgewidth=2)
        ax.annotate(str(i + 1), (x, y), color="lime", fontsize=9)
        fig.canvas.draw()
    plt.close(fig)

    print("\n# Wklej do POINTS w scripts/probe_registration.py (kolumna termiczna):")
    print("THERMAL_POINTS = [")
    for (x, y), label in zip(picked, LABELS, strict=True):
        print(f"    ({x:.1f}, {y:.1f}),  # {label}")
    print("]")


if __name__ == "__main__":
    main()
