"""scripts/demo_pipeline_rgb.py — pierwszy pełny przelot potoku RGB-only na realnym nagraniu.

Bez termiki, bez warpingu, bez Polara. Dla subject01/s1_rest_rest:
  1. wczytuje klatki RGB (io_layer) i śledzi ROI (roi.track_roi_across_frames + detektor
     wycinkowy, ten sam co dał 100% pokrycia),
  2. dla regionów forehead/left_cheek/right_cheek liczy przez extract.py średnie RGB w
     czasie (cały bbox ROI, bez maski),
  3. przepuszcza przez CHROM i POS (methods.py), potem estimate.py: detrend, bandpass
     0.7–4 Hz, HR z Welcha — osobno dla każdego regionu i metody,
  4. zapisuje wykresy + tabelę HR do results/pipeline_rgb/,
  5. wypisuje tabelę HR i rozrzut między regionami/metodami.

fs = rzeczywiste fps nagrania (nie zakładamy 30). Uruchomienie:
    uv run python scripts/demo_pipeline_rgb.py
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

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import welch  # noqa: E402 - tylko do WIZUALIZACJI widma

from src import estimate, extract, methods  # noqa: E402
from src.config import BAND_HIGH_HZ, BAND_LOW_HZ, RESULTS_DIR  # noqa: E402
from src.io_layer import load_recording  # noqa: E402
from src.roi import (  # noqa: E402
    make_cropping_detector,
    select_roi_from_landmarks,
    track_roi_across_frames,
)

REGIONS = ["forehead", "left_cheek", "right_cheek"]
METHODS = {"CHROM": methods.chrom, "POS": methods.pos}
SUBJECT, SCENARIO = "subject01", "s1_rest_rest"


def _multi_roi_builder(landmarks: np.ndarray, region: str) -> np.ndarray:
    """Bboxy wszystkich regionów naraz jako tablica (len(REGIONS), 4)."""
    return np.stack([select_roi_from_landmarks(landmarks, r) for r in REGIONS])


def _region_mean_rgb(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Średnia RGB w bboxie regionu — policzona przez extract.py (bbox = całość wycinka)."""
    y0, x0, y1, x1 = (int(v) for v in bbox)
    crop = frame[y0:y1, x0:x1]
    full_bbox = np.array([0, 0, crop.shape[0], crop.shape[1]])
    return extract.extract_rgb_trace(crop[None], [full_bbox], np.array([True]))[0]


def _extract_region_traces(loaded, roi_positions) -> dict[str, np.ndarray]:
    """Przebieg (drugi, dekodujący) po klatkach — średnie RGB per region, przez extract.py."""
    traces: dict[str, list] = {r: [] for r in REGIONS}
    for i, frame in enumerate(loaded.rgb_frames(to_rgb=True)):
        boxes = roi_positions[i]
        for k, region in enumerate(REGIONS):
            traces[region].append(_region_mean_rgb(frame, boxes[k]))
    return {r: np.asarray(v, dtype=np.float64) for r, v in traces.items()}


def _plot_raw_rgb(t, trace, out_dir) -> None:
    fig, ax = plt.subplots(figsize=(11, 3))
    for idx, (chan, color) in enumerate([("R", "tab:red"), ("G", "tab:green"), ("B", "tab:blue")]):
        ax.plot(t, trace[:, idx], color=color, lw=0.8, label=chan)
    ax.set_xlim(0, 10)
    ax.set_xlabel("czas [s]")
    ax.set_ylabel("średnia wartość kanału w ROI [j.u.]")
    ax.set_title("Surowe średnie RGB w ROI (forehead) — subject01/s1")
    ax.legend(loc="upper right")
    fig.savefig(out_dir / "raw_rgb_forehead.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_method_signals(t, signals, out_dir) -> None:
    fig, ax = plt.subplots(figsize=(11, 3))
    for name, sig in signals.items():
        z = (sig - sig.mean()) / (sig.std() + 1e-9)
        ax.plot(t, z, lw=0.9, label=name)
    ax.set_xlim(2, 10)
    ax.set_xlabel("czas [s]")
    ax.set_ylabel("sygnał rPPG (standaryzowany)")
    ax.set_title("Sygnał rPPG po CHROM i POS (forehead) — subject01/s1")
    ax.legend(loc="upper right")
    fig.savefig(out_dir / "chrom_pos_forehead.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_spectrum(cleaned_signals, hr_forehead, fs, out_dir) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for name, sig in cleaned_signals.items():
        freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), int(10 * fs)))
        line, = ax.plot(freqs * 60.0, psd, label=name)
        ax.axvline(hr_forehead[name], color=line.get_color(), ls=":", lw=1)
    ax.set_xlim(BAND_LOW_HZ * 60, BAND_HIGH_HZ * 60)
    ax.set_xlabel("częstość [BPM]")
    ax.set_ylabel("gęstość mocy [j.u.]")
    ax.set_title("Widmo mocy (forehead) z zaznaczonym pikiem HR — subject01/s1")
    ax.legend(loc="upper right")
    fig.savefig(out_dir / "spectrum_forehead.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _write_table(hr, out_dir) -> None:
    lines = ["# Estymowane HR [BPM] — subject01/s1_rest_rest (RGB-only)", "",
             "| region | " + " | ".join(METHODS) + " |",
             "|---|" + "---|" * len(METHODS)]
    for region in REGIONS:
        cells = " | ".join(f"{hr[(region, m)]:.2f}" for m in METHODS)
        lines.append(f"| {region} | {cells} |")
    lines.append("")
    (out_dir / "hr_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    loaded = load_recording(SUBJECT, SCENARIO)
    fs = loaded.rgb_meta.fps
    print(f"Nagranie: {SUBJECT}/{SCENARIO}  {loaded.rgb_meta.frame_count} klatek @ {fs:.3f} fps")

    # 1. Detekcja + śledzenie ROI (3 regiony w jednym przebiegu).
    print("Detekcja + śledzenie ROI...", flush=True)
    roi_positions, valid = track_roi_across_frames(
        loaded.rgb_frames(to_rgb=True), detector=make_cropping_detector(),
        roi_builder=_multi_roi_builder, region=REGIONS[0],
    )
    print(f"  pokrycie detekcji: {100.0 * valid.mean():.1f}%  ({int(valid.sum())}/{valid.size})")

    # 2. Ekstrakcja średnich RGB per region (przez extract.py), drugi przebieg dekodujący.
    print("Ekstrakcja średnich RGB w ROI...", flush=True)
    traces = _extract_region_traces(loaded, roi_positions)
    n = traces[REGIONS[0]].shape[0]
    t = np.arange(n) / fs

    # 3. CHROM/POS + estymacja HR per region i metoda.
    hr: dict[tuple[str, str], float] = {}
    signals_forehead: dict[str, np.ndarray] = {}
    cleaned_forehead: dict[str, np.ndarray] = {}
    for region in REGIONS:
        for name, method_fn in METHODS.items():
            sig = method_fn(traces[region], fs)
            cleaned = estimate.bandpass_filter(estimate.detrend_signal(sig), fs)
            hr[(region, name)] = estimate.estimate_hr_welch(cleaned, fs)
            if region == "forehead":
                signals_forehead[name] = sig
                cleaned_forehead[name] = cleaned

    # 4. Wykresy + tabela.
    out_dir = RESULTS_DIR / "pipeline_rgb"
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_raw_rgb(t, traces["forehead"], out_dir)
    _plot_method_signals(t, signals_forehead, out_dir)
    _plot_spectrum(cleaned_forehead, {m: hr[("forehead", m)] for m in METHODS}, fs, out_dir)
    _write_table(hr, out_dir)

    # 5. Tabela na stdout + rozrzut.
    print(f"\n=== Estymowane HR [BPM] (fs={fs:.3f} Hz) ===")
    print(f"{'region':<14}" + "".join(f"{m:>10}" for m in METHODS))
    for region in REGIONS:
        print(f"{region:<14}" + "".join(f"{hr[(region, m)]:>10.2f}" for m in METHODS))

    values = np.array([hr[(r, m)] for r in REGIONS for m in METHODS])
    print("\n=== Rozrzut ===")
    print(f"  wszystkie estymaty: min={values.min():.2f}, max={values.max():.2f}, "
          f"rozstęp={values.max() - values.min():.2f}, std={values.std():.2f} BPM")
    for m in METHODS:
        vals = np.array([hr[(r, m)] for r in REGIONS])
        print(f"  {m:<6} między regionami: rozstęp={vals.max() - vals.min():.2f} BPM")
    for region in REGIONS:
        vals = np.array([hr[(region, m)] for m in METHODS])
        print(f"  {region:<12} CHROM vs POS: różnica={abs(vals[0] - vals[1]):.2f} BPM")
    print(f"\nWykresy i tabela: {out_dir}")


if __name__ == "__main__":
    main()
