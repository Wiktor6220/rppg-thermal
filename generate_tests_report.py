"""Uruchamia testy syntetyczne z src/methods.py, src/estimate.py i src/validate.py
i zbiera wyniki do jednego raportu (results/tests_report.md).

Wykorzystuje wyłącznie istniejące funkcje generujące sygnał syntetyczny
(`_generate_synthetic_rgb`, `_generate_synthetic_pulse`, `_generate_two_stage_pulse`)
oraz istniejące metody/estymatory z tych modułów — nie tworzy nowych danych
ani nowej logiki. Parametry (fs, długość, zadana częstość, seed) są takie same,
jak w blokach `if __name__ == "__main__":` poszczególnych modułów, więc wynik
jest powtarzalny.

Awaria pojedynczego testu (wyjątek albo przekroczona tolerancja) nie przerywa
raportu — jest w nim odnotowana jako FAIL/ERROR.
"""

from datetime import datetime

import numpy as np

from src import config
from src.estimate import (
    _generate_synthetic_pulse,
    bandpass_filter,
    detrend_signal,
    estimate_hr_peaks,
    estimate_hr_welch,
)
from src.methods import _dominant_hr_bpm, _generate_synthetic_rgb, chrom, green, ica_method, pos
from src.validate import _generate_two_stage_pulse, validate_signal

TOLERANCE_BPM = 5.0  # tolerancja błędu HR — spójna z testami __main__ w każdym module
REPORT_PATH = config.RESULTS_DIR / "tests_report.md"


def run_methods_test() -> dict:
    """Powtarza test z src/methods.py: GREEN/CHROM/POS/ICA na syntetycznym RGB."""
    params = {"fs_hz": 30.0, "duration_s": 30.0, "hr_bpm": 72.0, "noise_std": 0.3, "seed": 0}

    rgb_signal, _ = _generate_synthetic_rgb(
        fs=params["fs_hz"], duration_s=params["duration_s"], hr_bpm=params["hr_bpm"], seed=params["seed"]
    )

    methods_under_test = {"GREEN": green, "CHROM": chrom, "POS": pos, "ICA": ica_method}
    rows = []
    for name, method_fn in methods_under_test.items():
        try:
            pulse_signal = method_fn(rgb_signal, params["fs_hz"])
            estimated_bpm = _dominant_hr_bpm(pulse_signal, params["fs_hz"])
            error_bpm = abs(estimated_bpm - params["hr_bpm"])
            status = "OK" if error_bpm <= TOLERANCE_BPM else "FAIL (poza tolerancją)"
            rows.append((name, params["hr_bpm"], estimated_bpm, error_bpm, status))
        except Exception as exc:  # noqa: BLE001 - raport ma pokazać awarię, nie przerwać się
            rows.append((name, params["hr_bpm"], None, None, f"ERROR: {exc}"))

    return {"params": params, "rows": rows}


def run_estimate_test() -> dict:
    """Powtarza test z src/estimate.py: HR z widma Welcha vs. z detekcji pików."""
    params = {"fs_hz": 30.0, "duration_s": 60.0, "hr_bpm": 72.0, "noise_std": 0.15, "seed": 0}

    rows = []
    try:
        raw_signal = _generate_synthetic_pulse(
            fs=params["fs_hz"], duration_s=params["duration_s"], hr_bpm=params["hr_bpm"], seed=params["seed"]
        )
        filtered = bandpass_filter(detrend_signal(raw_signal), params["fs_hz"])

        estimators = {"Welch (widmo mocy)": estimate_hr_welch, "find_peaks (dziedzina czasu)": estimate_hr_peaks}
        for name, estimator_fn in estimators.items():
            try:
                estimated_bpm = estimator_fn(filtered, params["fs_hz"])
                error_bpm = abs(estimated_bpm - params["hr_bpm"])
                status = "OK" if error_bpm <= TOLERANCE_BPM else "FAIL (poza tolerancją)"
                rows.append((name, params["hr_bpm"], estimated_bpm, error_bpm, status))
            except Exception as exc:  # noqa: BLE001
                rows.append((name, params["hr_bpm"], None, None, f"ERROR: {exc}"))
    except Exception as exc:  # noqa: BLE001 - awaria przygotowania sygnału (detrend/bandpass)
        rows.append(("przygotowanie sygnału (detrend + bandpass)", params["hr_bpm"], None, None, f"ERROR: {exc}"))

    return {"params": params, "rows": rows}


def run_validate_test() -> dict:
    """Powtarza test z src/validate.py: MAE/RMSE per okno, z pominięciem okna bez ROI."""
    params = {
        "fs_hz": 30.0,
        "duration_s": 40.0,
        "hr_bpm": "65.0 -> 85.0 (skok w połowie nagrania)",
        "noise_std": 0.2,
        "seed": 1,
    }

    try:
        reference_clean, estimated_noisy = _generate_two_stage_pulse(
            fs=30.0, duration_s=40.0, hr_bpm_stage1=65.0, hr_bpm_stage2=85.0, noise_std=0.2, seed=1
        )
        n_samples = estimated_noisy.shape[0]
        valid_vector = np.ones(n_samples, dtype=bool)
        valid_vector[300:600] = False  # symulacja 10 s utraty ROI, tak jak w validate.py

        result = validate_signal(estimated_noisy, reference_clean, params["fs_hz"], valid=valid_vector)
        status = "OK" if not np.isnan(result["mae_bpm"]) and not np.isnan(result["rmse_bpm"]) else "FAIL"
        outcome = {
            "mae_bpm": result["mae_bpm"],
            "rmse_bpm": result["rmse_bpm"],
            "n_windows_total": result["n_windows_total"],
            "n_windows_used": result["n_windows_used"],
            "status": status,
        }
    except Exception as exc:  # noqa: BLE001
        outcome = {
            "mae_bpm": None,
            "rmse_bpm": None,
            "n_windows_total": None,
            "n_windows_used": None,
            "status": f"ERROR: {exc}",
        }

    return {"params": params, "outcome": outcome}


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _params_table_rows(*named_params: tuple[str, dict]) -> list[str]:
    lines = [
        "| Moduł | fs [Hz] | długość sygnału [s] | zadana częstość [BPM] | poziom szumu (std) | ziarno losowe |",
        "|---|---|---|---|---|---|",
    ]
    for module_name, params in named_params:
        lines.append(
            f"| {module_name} | {_fmt(params['fs_hz'])} | {_fmt(params['duration_s'])} | "
            f"{params['hr_bpm']} | {_fmt(params['noise_std'])} | {_fmt(params['seed'])} |"
        )
    return lines


def _hr_table(rows: list[tuple]) -> list[str]:
    lines = [
        "| Metoda | Częstość referencyjna [BPM] | Częstość estymowana [BPM] | Błąd bezwzględny [BPM] | Status |",
        "|---|---|---|---|---|",
    ]
    for name, ref_bpm, est_bpm, err_bpm, status in rows:
        lines.append(f"| {name} | {_fmt(ref_bpm)} | {_fmt(est_bpm)} | {_fmt(err_bpm)} | {status} |")
    return lines


def build_report(timestamp: str, methods_result: dict, estimate_result: dict, validate_result: dict) -> str:
    lines = [
        "# Raport testów syntetycznych",
        "",
        f"Data uruchomienia: {timestamp}",
        f"Tolerancja błędu HR: ±{TOLERANCE_BPM} BPM (jak w testach `__main__` poszczególnych modułów).",
        "",
        "## Parametry testów",
        "",
        *_params_table_rows(
            ("methods.py", methods_result["params"]),
            ("estimate.py", estimate_result["params"]),
            ("validate.py", validate_result["params"]),
        ),
        "",
        "## 1. methods.py — metody rPPG (GREEN, CHROM, POS, ICA)",
        "",
        *_hr_table(methods_result["rows"]),
        "",
        "## 2. estimate.py — estymacja HR: Welch vs. find_peaks",
        "",
        *_hr_table(estimate_result["rows"]),
        "",
        "## 3. validate.py — MAE/RMSE na sygnale syntetycznym (per okno)",
        "",
        "| Metryka | Wartość |",
        "|---|---|",
        f"| MAE [BPM] | {_fmt(validate_result['outcome']['mae_bpm'])} |",
        f"| RMSE [BPM] | {_fmt(validate_result['outcome']['rmse_bpm'])} |",
        f"| Liczba okien łącznie | {_fmt(validate_result['outcome']['n_windows_total'])} |",
        f"| Liczba okien użytych w metrykach | {_fmt(validate_result['outcome']['n_windows_used'])} |",
        f"| Status | {validate_result['outcome']['status']} |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    methods_result = run_methods_test()
    estimate_result = run_estimate_test()
    validate_result = run_validate_test()

    report_text = build_report(timestamp, methods_result, estimate_result, validate_result)

    print(report_text)

    REPORT_PATH.write_text(report_text + "\n", encoding="utf-8")
    print(f"Raport zapisany do: {REPORT_PATH}")


if __name__ == "__main__":
    main()
