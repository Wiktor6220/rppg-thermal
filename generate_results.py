"""Uruchamia funkcje z src/ i tests/synthetic.py na danych syntetycznych i ZAPISUJE
wykresy oraz podsumowanie do folderu results/.

Skrypt niczego nie liczy „sam z siebie" — wyłącznie importuje i wywołuje istniejące
funkcje (metody rPPG, estymacja HR, SNR, maska perfuzji, walidacja) oraz generatory
sygnału/klatek z `tests/synthetic.py`. Wizualizacja (matplotlib) i drobne przeliczenia
błędu są jedynie prezentacją wyników tych funkcji.
"""

import warnings

import matplotlib

matplotlib.use("Agg")  # zapis do plików, bez okna/kernela

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch  # tylko do WIZUALIZACJI widma mocy

from src.config import BAND_HIGH_HZ, BAND_LOW_HZ, FS, RESULTS_DIR, VALIDATION_WINDOW_SEC
from src.estimate import bandpass_filter, detrend_signal, estimate_hr_welch, snr_rppg
from src.extract import (
    compute_perfusion_mask,
    extract_rgb_trace,
    extract_rgb_trace_thermal_gated,
)
from src.methods import chrom, green, ica_method, pos
from src.validate import validate_signal
from tests.synthetic import (
    dominant_hr_bpm,
    generate_hard_synthetic_rgb,
    generate_synthetic_frames,
    generate_synthetic_rgb,
    generate_two_stage_pulse,
)

warnings.filterwarnings("ignore")  # m.in. FastICA ConvergenceWarning na krótkich oknach

TRUE_HR_BPM = 72.0
DPI = 120


def _save(fig, name: str) -> str:
    """Zapisuje figurę do results/ i zwraca nazwę pliku."""
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_raw_signal() -> str:
    """1. Surowe przebiegi RGB (trzy kanały)."""
    rgb_trace, t = generate_synthetic_rgb(fs=FS, duration_s=30.0, hr_bpm=TRUE_HR_BPM, seed=0)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    for idx, (chan, color) in enumerate([("R", "tab:red"), ("G", "tab:green"), ("B", "tab:blue")]):
        ax.plot(t, rgb_trace[:, idx], color=color, lw=0.8, label=chan)
    ax.set_xlim(0, 10)
    ax.set_xlabel("czas [s]")
    ax.set_ylabel("wartość kanału [j.u.]")
    ax.set_title("Surowe przebiegi kanałów RGB (syntetyk, HR=72 BPM, pierwsze 10 s)")
    ax.legend(loc="upper right")
    return _save(fig, "raw_signal.png")


def fig_methods_comparison() -> tuple[str, dict]:
    """2. Sygnał po GREEN/CHROM/POS na wspólnej osi czasu; zwraca też estymaty HR."""
    rgb_trace, t = generate_synthetic_rgb(fs=FS, duration_s=30.0, hr_bpm=TRUE_HR_BPM, seed=0)
    method_fns = {"GREEN": green, "CHROM": chrom, "POS": pos, "ICA": ica_method}

    hr_table = {}
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for name, fn in method_fns.items():
        sig = fn(rgb_trace, FS)
        hr_peak = dominant_hr_bpm(sig, FS)
        hr_welch = estimate_hr_welch(bandpass_filter(detrend_signal(sig), FS), FS)
        hr_table[name] = {
            "hr_peak_bpm": hr_peak,
            "hr_welch_bpm": hr_welch,
            "err_bpm": abs(hr_peak - TRUE_HR_BPM),
        }
        if name != "ICA":  # ICA rysujemy osobno? nie — pokazujemy 3 główne dla czytelności
            z = (sig - sig.mean()) / (sig.std() + 1e-9)
            ax.plot(t, z, lw=0.9, label=name)
    ax.set_xlim(2, 10)
    ax.set_xlabel("czas [s]")
    ax.set_ylabel("sygnał rPPG (standaryzowany do wykresu)")
    ax.set_title("Sygnał rPPG po GREEN / CHROM / POS")
    ax.legend(loc="upper right")
    return _save(fig, "methods_comparison.png"), hr_table


def fig_spectrum() -> tuple[str, float]:
    """3. Widmo mocy sygnału po POS z zaznaczonym pikiem HR; zwraca odczytane HR."""
    rgb_trace, _ = generate_synthetic_rgb(fs=FS, duration_s=30.0, hr_bpm=TRUE_HR_BPM, seed=0)
    sig = pos(rgb_trace, FS)
    hr_pos = dominant_hr_bpm(sig, FS)

    freqs, psd = welch(sig, fs=FS, nperseg=min(len(sig), int(10 * FS)))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(freqs * 60.0, psd, color="tab:purple", label="widmo mocy (POS)")
    ax.axvline(hr_pos, color="tab:red", ls=":", lw=1.5, label=f"pik HR = {hr_pos:.1f} BPM")
    ax.axvline(
        TRUE_HR_BPM, color="k", ls="--", lw=1.2, label=f"prawdziwe HR = {TRUE_HR_BPM:.0f} BPM"
    )
    ax.set_xlim(BAND_LOW_HZ * 60, BAND_HIGH_HZ * 60)
    ax.set_xlabel("częstość [BPM]")
    ax.set_ylabel("gęstość mocy [j.u.]")
    ax.set_title("Widmo mocy sygnału rPPG (POS) z odczytanym pikiem HR")
    ax.legend(loc="upper right")
    return _save(fig, "spectrum.png"), hr_pos


def fig_interference_robustness() -> tuple[str, dict]:
    """4. Błąd HR (GREEN/CHROM/POS/ICA) vs SIŁA wspólnego artefaktu jasności.

    Oś X to `interference_amplitude` (siła wspólnej, jednakowej dla wszystkich kanałów
    składowej jasności w paśmie tętna) przy USTALONYM umiarkowanym szumie. To ta
    składowa — a nie sam szum — generuje błąd GREEN, dlatego badamy ją bezpośrednio.
    Oczekiwanie: gdy siła artefaktu przekroczy amplitudę pulsu, GREEN „przykleja się"
    do częstości artefaktu (błąd skacze do |96-72|=24 BPM), a CHROM/POS pozostają
    niskie, bo znoszą wspólną składową jasności w projekcji chrominancji.
    """
    fixed_noise_std = 0.5  # umiarkowany, stały szum pomiarowy
    pulse_amplitude = 0.02  # siła pulsu — próg, powyżej którego artefakt dominuje w GREEN
    interference_levels = np.linspace(0.0, 0.05, 11)
    seeds = range(6)
    method_fns = {"GREEN": green, "CHROM": chrom, "POS": pos, "ICA": ica_method}

    mean_err = {m: [] for m in method_fns}
    std_err = {m: [] for m in method_fns}
    for ia in interference_levels:
        per = {m: [] for m in method_fns}
        for sd in seeds:
            rgb_h, _ = generate_hard_synthetic_rgb(
                fs=FS,
                duration_s=30.0,
                hr_bpm=TRUE_HR_BPM,
                noise_std=fixed_noise_std,
                pulse_amplitude=pulse_amplitude,
                interference_amplitude=float(ia),
                seed=sd,
            )
            for m, fn in method_fns.items():
                per[m].append(abs(dominant_hr_bpm(fn(rgb_h, FS), FS) - TRUE_HR_BPM))
        for m in method_fns:
            mean_err[m].append(float(np.mean(per[m])))
            std_err[m].append(float(np.std(per[m])))

    fig, ax = plt.subplots(figsize=(10, 4))
    for m in method_fns:
        me = np.array(mean_err[m])
        se = np.array(std_err[m])
        ax.plot(interference_levels, me, marker="o", label=m)
        ax.fill_between(interference_levels, me - se, me + se, alpha=0.15)
    ax.axvline(
        pulse_amplitude, color="gray", ls="--", lw=1.2,
        label=f"próg = amplituda pulsu ({pulse_amplitude:.2f})",
    )
    ax.set_xlim(interference_levels[0], interference_levels[-1])
    ax.set_xlabel("siła wspólnego artefaktu jasności (interference_amplitude)")
    ax.set_ylabel("średni |błąd HR| [BPM]  (6 ziaren)")
    ax.set_title(
        f"Błąd HR vs siła wspólnego artefaktu jasności (noise_std={fixed_noise_std} stałe)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    summary = {
        "levels": interference_levels,
        "mean_err": mean_err,
        "fixed_noise_std": fixed_noise_std,
        "pulse_amplitude": pulse_amplitude,
    }
    return _save(fig, "interference_robustness.png"), summary


def fig_thermal_mask() -> tuple[str, dict]:
    """5. Klatka termiczna, maska ROI i maska perfuzji obok siebie."""
    frames = generate_synthetic_frames(fs=FS, duration_s=20.0, hr_bpm=TRUE_HR_BPM, seed=0)
    thermal0 = frames["thermal_frames"][0]
    roi_mask = frames["roi_mask"]
    perfusion = compute_perfusion_mask(thermal0, roi_mask)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im = axes[0].imshow(thermal0, cmap="inferno")
    axes[0].set_title("Klatka termiczna [°C]")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="°C")
    axes[1].imshow(roi_mask, cmap="gray")
    axes[1].set_title("Maska ROI")
    axes[2].imshow(perfusion, cmap="gray")
    axes[2].set_title("Maska perfuzji")
    for a in axes:
        a.set_xlabel("x [px]")
        a.set_ylabel("y [px]")
    fig.suptitle("Termika → ROI → maska perfuzji")
    info = {"roi_px": int(roi_mask.sum()), "perfusion_px": int(perfusion.sum())}
    return _save(fig, "thermal_mask.png"), info


def fig_snr_gain() -> tuple[str, dict]:
    """6. rPPG z pełnego ROI vs bramka termiczna + zysk SNR w dB."""
    frames = generate_synthetic_frames(fs=FS, duration_s=20.0, hr_bpm=TRUE_HR_BPM, seed=0)
    plain_trace = extract_rgb_trace(frames["rgb_frames"], frames["roi_positions"], frames["valid"])
    gated_trace = extract_rgb_trace_thermal_gated(
        frames["rgb_frames"], frames["thermal_frames"], frames["roi_positions"], frames["valid"]
    )
    sig_plain = green(plain_trace, FS)
    sig_gated = green(gated_trace, FS)
    snr_plain = snr_rppg(sig_plain, FS, TRUE_HR_BPM)
    snr_gated = snr_rppg(sig_gated, FS, TRUE_HR_BPM)
    t = frames["t"]

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(
        t, (sig_plain - sig_plain.mean()) / (sig_plain.std() + 1e-9),
        lw=0.9, label=f"pełne ROI (SNR={snr_plain:.1f} dB)",
    )
    ax.plot(
        t, (sig_gated - sig_gated.mean()) / (sig_gated.std() + 1e-9),
        lw=0.9, label=f"bramka termiczna (SNR={snr_gated:.1f} dB)",
    )
    ax.set_xlim(2, 10)
    ax.set_xlabel("czas [s]")
    ax.set_ylabel("sygnał rPPG (standaryzowany)")
    ax.set_title(f"rPPG: pełne ROI vs bramka termiczna (zysk SNR = {snr_gated - snr_plain:.1f} dB)")
    ax.legend(loc="upper right")
    info = {
        "snr_plain_db": float(snr_plain),
        "snr_gated_db": float(snr_gated),
        "gain_db": float(snr_gated - snr_plain),
    }
    return _save(fig, "snr_gain.png"), info


def fig_validation_windows() -> tuple[str, dict]:
    """7. Estymaty HR per okno vs referencja, z zaznaczonym oknem pominiętym (utrata ROI)."""
    reference, estimated = generate_two_stage_pulse(
        fs=FS, duration_s=40.0, hr_bpm_stage1=65.0, hr_bpm_stage2=85.0, noise_std=0.2, seed=1
    )
    n = estimated.shape[0]
    valid = np.ones(n, dtype=bool)
    valid[300:600] = False  # 10 s utraty ROI (t = 10..20 s przy FS=30)
    result = validate_signal(estimated, reference, FS, valid=valid)

    ws = result["window_start_s"]
    est = result["estimated_hr_bpm"]
    ref = result["reference_hr_bpm"]
    used = result["window_used"]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ws, ref, marker="s", label="referencja HR (per okno)")
    ax.plot(ws, est, marker="o", label="estymacja HR (per okno)")
    shaded = False
    for i in range(len(ws)):
        if not used[i]:
            ax.axvspan(
                ws[i], ws[i] + VALIDATION_WINDOW_SEC, color="tab:red", alpha=0.12,
                label=("okno pominięte (utrata ROI)" if not shaded else None),
            )
            shaded = True
    ax.axvline(20.0, color="k", ls="--", lw=1, label="skok HR 65→85 (t=20 s)")
    ax.set_xlabel("początek okna [s]")
    ax.set_ylabel("HR [BPM]")
    ax.set_title("Walidacja per okno; okno z utratą ROI pominięte (NaN → przerwa)")
    ax.legend(loc="best")
    info = {
        "n_windows_total": int(result["n_windows_total"]),
        "n_windows_used": int(result["n_windows_used"]),
        "mae_bpm": float(result["mae_bpm"]),
        "rmse_bpm": float(result["rmse_bpm"]),
    }
    return _save(fig, "validation_windows.png"), info


def _fmt(x) -> str:
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def write_summary(
    hr_table: dict,
    hr_pos: float,
    interf: dict,
    thermal: dict,
    snr: dict,
    val: dict,
    files: list[str],
) -> str:
    """8. Tabela zbiorcza results_summary.md z liczb policzonych powyżej."""
    lines = [
        "# Podsumowanie wyników (dane syntetyczne)",
        "",
        f"Prawdziwe HR (sekcje 1–3, 5–6): **{TRUE_HR_BPM:.0f} BPM**.",
        "Wszystkie liczby policzone na żywo funkcjami z `src/` i `tests/synthetic.py`.",
        "",
        "## 1. Estymacja HR metodami rPPG (czysty syntetyk, seed=0)",
        "",
        "| Metoda | HR pik w paśmie [BPM] | HR Welch (pipeline) [BPM] | |błąd| [BPM] |",
        "|---|---|---|---|",
    ]
    for name, row in hr_table.items():
        lines.append(
            f"| {name} | {_fmt(row['hr_peak_bpm'])} | {_fmt(row['hr_welch_bpm'])} | "
            f"{_fmt(row['err_bpm'])} |"
        )
    lines += [
        "",
        f"Odczyt piku HR z widma POS (spectrum.png): **{hr_pos:.2f} BPM**.",
        "",
        "## 2. Odporność na siłę wspólnego artefaktu jasności",
        "",
        f"Ustalony szum: noise_std = {interf['fixed_noise_std']:.2f}; "
        f"amplituda pulsu (próg) = {interf['pulse_amplitude']:.2f}. "
        "Oś X = `interference_amplitude` (siła wspólnej składowej jasności w paśmie).",
        "",
        "Średni |błąd HR| [BPM] (6 ziaren) w funkcji siły artefaktu:",
        "",
        "| Metoda | " + " | ".join(f"{lvl:.3f}" for lvl in interf["levels"]) + " |",
        "|---|" + "---|" * len(interf["levels"]),
    ]
    for m, errs in interf["mean_err"].items():
        lines.append(f"| {m} | " + " | ".join(_fmt(e) for e in errs) + " |")
    lines += [
        "",
        "## 3. Bramkowanie termiczne (maska perfuzji)",
        "",
        f"- Piksele ROI: **{thermal['roi_px']}**, piksele perfuzji: **{thermal['perfusion_px']}**",
        f"- SNR pełne ROI: **{snr['snr_plain_db']:.2f} dB**",
        f"- SNR z bramką termiczną: **{snr['snr_gated_db']:.2f} dB**",
        f"- **Zysk SNR: {snr['gain_db']:.2f} dB**",
        "",
        "## 4. Walidacja per okno (skok HR 65→85, 10 s utraty ROI)",
        "",
        "| Metryka | Wartość |",
        "|---|---|",
        f"| Okna łącznie | {val['n_windows_total']} |",
        f"| Okna użyte w metrykach | {val['n_windows_used']} |",
        f"| MAE [BPM] | {_fmt(val['mae_bpm'])} |",
        f"| RMSE [BPM] | {_fmt(val['rmse_bpm'])} |",
        "",
        "## Wygenerowane pliki",
        "",
        *[f"- `{f}`" for f in files],
        "",
    ]
    text = "\n".join(lines)
    (RESULTS_DIR / "results_summary.md").write_text(text + "\n", encoding="utf-8")
    return "results_summary.md"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = []

    files.append(fig_raw_signal())
    methods_png, hr_table = fig_methods_comparison()
    files.append(methods_png)
    spectrum_png, hr_pos = fig_spectrum()
    files.append(spectrum_png)
    interf_png, interf = fig_interference_robustness()
    files.append(interf_png)
    thermal_png, thermal = fig_thermal_mask()
    files.append(thermal_png)
    snr_png, snr = fig_snr_gain()
    files.append(snr_png)
    val_png, val = fig_validation_windows()
    files.append(val_png)

    summary_md = write_summary(hr_table, hr_pos, interf, thermal, snr, val, files)
    files.append(summary_md)

    print(f"Zapisano {len(files)} plików do: {RESULTS_DIR}")
    for f in files:
        print(f"  - {f}")
    print()
    print("Kluczowe liczby:")
    for name, row in hr_table.items():
        print(f"  HR {name:<6}: pik {row['hr_peak_bpm']:6.2f} BPM  (błąd {row['err_bpm']:.2f})")
    print(f"  Pik HR (POS, spectrum): {hr_pos:.2f} BPM")
    print(f"  SNR pełne ROI / bramka: {snr['snr_plain_db']:.2f} / {snr['snr_gated_db']:.2f} dB "
          f"(zysk {snr['gain_db']:.2f} dB)")
    print(f"  Walidacja: MAE {val['mae_bpm']:.2f} BPM, RMSE {val['rmse_bpm']:.2f} BPM, "
          f"okna {val['n_windows_used']}/{val['n_windows_total']}")


if __name__ == "__main__":
    main()
