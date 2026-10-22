"""Tabele wyników: ``metrics.csv``/``metrics.md`` per scena oraz agregat z dysku.

Agregat (``all_subjects.csv``, ``summary.md``) składany ze wszystkich
``<out>/<subject>/<scena>/metrics.csv``, więc przebieg jednego badanego nie kasuje reszty.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypeGuard

import numpy as np

from src.config import REF_SUSPECT_SUBJECTS, REG_GO_RMS_PX, REGION_LABELS, REGIONS_REPORT
from src.registration import SceneRegistrationQuality

SCENE_CSV_FIELDS = [
    "subject",
    "scenario",
    "region",
    "method",
    "hr_ref_scene",
    "hr_scene_plain",
    "abs_err_scene_plain",
    "hr_scene_gated",
    "abs_err_scene_gated",
    "hr_w1_plain",
    "hr_w2_plain",
    "hr_w3_plain",
    "hr_w4_plain",
    "hr_w1_gated",
    "hr_w2_gated",
    "hr_w3_gated",
    "hr_w4_gated",
    "hr_w1_ref",
    "hr_w2_ref",
    "hr_w3_ref",
    "hr_w4_ref",
    "mae_plain",
    "rmse_plain",
    "octave_frac_plain",
    "mae_gated",
    "rmse_gated",
    "octave_frac_gated",
    "delta_snr",
    "n_windows",
    "reg_rms_px",
    "reg_iou",
    "reg_rms_first_px",
    "reg_good_frac",
    "reg_n_attempts",
    "gated_status",
    "sync_mode",
    "sync_offset_s",
    "usable_t0_s",
    "usable_t1_s",
    "fallback_pct",
    "ref_suspect",
    "reg_calib_source",
]


# ---------------------------------------------------------------------------
# Tabele per scena
# ---------------------------------------------------------------------------


def _window_cell(row: dict, i: int) -> str:
    """Komórka ``estymata/referencja`` okna ``i`` (plain) albo ``—``."""
    ep = row.get(f"hr_w{i}_plain", float("nan"))
    er = row.get(f"hr_w{i}_ref", float("nan"))
    if not (np.isfinite(ep) and np.isfinite(er)):
        return "—"
    return f"{ep:.0f}/{er:.0f}"


def write_scene_tables(
    out_dir: Path, rows: list[dict], quality: SceneRegistrationQuality
) -> None:
    """``metrics.csv`` (pełne kolumny, źródło agregatu) + ``metrics.md``."""
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCENE_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    off = rows[0].get("sync_offset_s", float("nan")) if rows else float("nan")
    lines = [
        f"# Metryki — RMS mediana={quality.contour_rms_px:.1f} px, "
        f"IoU mediana={quality.iou:.3f}, "
        f"gated **{quality.gate_label}** (próg {REG_GO_RMS_PX:.0f} px), "
        f"sync_offset={off:.3f} s, okna: 4 równe",
        "",
        f"Rejestracja: RMS 1. próby={quality.rms_first_px:.1f} px, "
        f"dobre próby (RMS ≤ {REG_GO_RMS_PX:.0f} px)={100 * quality.good_frac:.0f}% "
        f"z {quality.n_attempts}.",
        "",
        "| region | metoda | HR_ref | HR_scena plain | |err| plain | "
        "w1 p/r | w2 p/r | w3 p/r | w4 p/r | MAE_p | RMSE_p | %okt_p | "
        "MAE_g | ΔSNR |",
        "|---|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['region']} | {r['method']} | {r['hr_ref_scene']:.1f} | "
            f"{r['hr_scene_plain']:.1f} | {r['abs_err_scene_plain']:.1f} | "
            f"{_window_cell(r, 1)} | {_window_cell(r, 2)} | "
            f"{_window_cell(r, 3)} | {_window_cell(r, 4)} | "
            f"{r['mae_plain']:.2f} | {r['rmse_plain']:.2f} | "
            f"{100 * r['octave_frac_plain']:.0f}% | {r['mae_gated']:.2f} | "
            f"{r['delta_snr']:+.2f} |"
        )
    (out_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Tabela → {csv_path}")


# ---------------------------------------------------------------------------
# Agregat z dysku
# ---------------------------------------------------------------------------


def _parse_cell(value: str) -> bool | float | str:
    """Komórka CSV -> bool / float / str (puste -> NaN)."""
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return value


def collect_scene_rows(out_root: Path) -> list[dict]:
    """Wszystkie wiersze z ``<out>/<subject>/<scena>/metrics.csv`` (posortowane)."""
    rows: list[dict] = []
    for path in sorted(out_root.glob("subject*/*/metrics.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                if not raw.get("subject"):
                    continue
                row = {k: _parse_cell(v) for k, v in raw.items()}
                row["subject"] = raw["subject"]
                row["scenario"] = raw["scenario"]
                rows.append(row)
    return rows


def _fmt_cell(v: object) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        if not np.isfinite(v):
            return ""
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    return "" if v is None else str(v)


def _is_num(v: object) -> TypeGuard[float]:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and bool(np.isfinite(v))


def write_all_subjects_csv(all_rows: list[dict], out_root: Path) -> Path:
    path = out_root / "all_subjects.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCENE_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in SCENE_CSV_FIELDS})
    print(f"CSV zbiorczy → {path}")
    return path


def _verdict_lines(all_rows: list[dict]) -> list[str]:
    """Krótki werdykt liczony wyłącznie z wierszy tabeli."""
    suspects = ", ".join(REF_SUSPECT_SUBJECTS) or "—"
    lines = [
        "",
        "## Krótki werdykt",
        "",
        f"Statystyki poniżej liczone bez badanych z `ref_suspect=True` ({suspects}).",
        "",
    ]
    nogo = sorted(
        {
            (r["subject"], r["scenario"])
            for r in all_rows
            if r.get("gated_status") == "NO-GO" and not r.get("ref_suspect")
        }
    )
    if nogo:
        lines.append(
            f"- **NO-GO rejestracji** (mediana RMS sceny > {REG_GO_RMS_PX:.0f} px): "
            + ", ".join(f"{s}/{c}" for s, c in nogo)
            + " — gated niewiarygodny."
        )
    else:
        lines.append("- Brak scen NO-GO.")

    for region in REGIONS_REPORT:
        good, weak = [], []
        for r in all_rows:
            if r.get("method") != "POS" or r.get("region") != region or r.get("ref_suspect"):
                continue
            err = r.get("abs_err_scene_plain")
            if not _is_num(err):
                continue
            tag = f"{r['subject']}/{r['scenario']} (|err|={err:.1f})"
            if err <= 10.0:
                good.append(tag)
            elif err >= 25.0:
                weak.append(tag)
        label = REGION_LABELS[region]
        if good:
            lines.append(
                f"- **POS {label} — wiarygodniejsze** (|err|_scena ≤ 10): " + "; ".join(good)
            )
        if weak:
            lines.append(f"- **POS {label} — słabe** (|err|_scena ≥ 25): " + "; ".join(weak))

    n_cmp = n_g_better = 0
    for r in all_rows:
        if r.get("ref_suspect") or r.get("gated_status") == "NO-GO":
            continue
        mp, mg = r.get("mae_plain"), r.get("mae_gated")
        if _is_num(mp) and _is_num(mg):
            n_cmp += 1
            if mg < mp:
                n_g_better += 1
    if n_cmp:
        lines.append(
            f"- **Plain vs gated (MAE okienkowe, bez scen NO-GO):** "
            f"gated lepsze w {n_g_better}/{n_cmp} wierszach "
            f"({100 * n_g_better / n_cmp:.0f}%)."
        )

    scene_meta: dict[tuple[str, str], dict] = {}
    for r in all_rows:
        scene_meta.setdefault((r["subject"], r["scenario"]), r)
    n_scenes = sum(1 for r in scene_meta.values() if not r.get("ref_suspect"))
    n_nogo = sum(
        1
        for r in scene_meta.values()
        if (not r.get("ref_suspect")) and r.get("gated_status") == "NO-GO"
    )
    n_fallback = sum(
        1
        for r in scene_meta.values()
        if (not r.get("ref_suspect"))
        and str(r.get("reg_calib_source", "")).startswith("fallback")
    )
    lines.append(
        f"- Sceny **NO-GO** rejestracji: {n_nogo}/{n_scenes}; "
        f"kalibracja fallback: {n_fallback} (`<out>/<subject>/<scena>/reg_calib.json`)."
    )
    lines.append("- Sceny z ruchem (s2/s4) oraz approach (s5) zwykle gorsze niż rest/drone.")
    lines.append("")
    return lines


def write_summary(out_root: Path) -> list[dict]:
    """``all_subjects.csv`` + ``summary.md`` ze wszystkich ``metrics.csv`` w ``out_root``."""
    all_rows = collect_scene_rows(out_root)
    write_all_subjects_csv(all_rows, out_root)
    subjects = sorted({r["subject"] for r in all_rows})
    lines = [
        "# Podsumowanie potoku — wszyscy przetworzeni badani",
        "",
        "Default: **bbox** · **POS** + **CHROM** · plain/gated · czoło + policzki · "
        "**4 równe okna** · kalibracja rejestracji **per scena**.",
        "",
        f"Badani w agregacie ({len(subjects)}): {', '.join(subjects) if subjects else '—'}.",
        "",
        f"`ref_suspect=True` ({', '.join(REF_SUSPECT_SUBJECTS) or '—'}): flaga oznacza "
        "nietypowo wysokie HR referencji Polar. Wiersze tych badanych zostają w tabeli; "
        "werdykt liczy statystyki bez nich.",
        "",
        f"Rejestracja: `reg_rms_px`/`reg_iou` = mediana z prób odświeżenia affine w scenie, "
        f"`reg_rms_first_px` = pierwsza próba, `reg_good_frac` = udział prób z RMS ≤ "
        f"{REG_GO_RMS_PX:.0f} px; `gated_status=trustworthy`, gdy mediana ≤ progu.",
        "",
        "Zbiorczy CSV: [`all_subjects.csv`](all_subjects.csv).",
        "",
        "## Pełna tabela (subject × scena × region × metoda)",
        "",
        "| " + " | ".join(SCENE_CSV_FIELDS) + " |",
        "|" + "|".join(["---"] * len(SCENE_CSV_FIELDS)) + "|",
    ]
    for r in all_rows:
        lines.append("| " + " | ".join(_fmt_cell(r.get(k)) for k in SCENE_CSV_FIELDS) + " |")
    lines += _verdict_lines(all_rows)
    path = out_root / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Zbiorcza → {path}")
    return all_rows
