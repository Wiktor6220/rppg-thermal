"""Stabilność affine refresh: rozrzut stałych punktów termiki w układzie RGB [px].

Uruchomienie:
    uv run python scripts/probe_affine_stability.py
    uv run python scripts/probe_affine_stability.py --subject subject02 --scenario s5_approach
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    AFFINE_EVERY_BY_SCENARIO,
    AFFINE_EVERY_DEFAULT,
    EVAL_SUBJECT,
    RESULTS_DIR,
)
from src.extract import affine_point_dispersion_px, thermal_probe_points  # noqa: E402
from src.io_layer import list_eval_recordings, load_recording  # noqa: E402
from src.registration import estimate_affine_thermal_to_rgb  # noqa: E402
from src.roi import make_cropping_detector  # noqa: E402

OUT = RESULTS_DIR / "affine_stability"


def _affine_every(scenario: str) -> int:
    return AFFINE_EVERY_BY_SCENARIO.get(scenario, AFFINE_EVERY_DEFAULT)


def probe_one(subject: str, scenario: str) -> dict:
    loaded = load_recording(subject, scenario)
    detector = make_cropping_detector()
    every = _affine_every(scenario)
    samples = []
    probe_pts = None
    n_fail = 0
    for i, (rgb, thermal, _) in enumerate(loaded.synced_pairs(reference="rgb")):
        if i % every != 0 and samples:
            continue
        landmarks = detector(rgb)
        if landmarks is None:
            continue
        aff, info = estimate_affine_thermal_to_rgb(rgb, thermal, landmarks)
        if aff is None:
            n_fail += 1
            continue
        samples.append(aff.astype(float))
        if probe_pts is None and isinstance(info, dict):
            probe_pts = thermal_probe_points(info["thermal_mask"])

    if probe_pts is None:
        probe_pts = __import__("numpy").array(
            [[640.0, 400.0], [500.0, 550.0], [780.0, 550.0], [640.0, 700.0]]
        )
    disp = affine_point_dispersion_px(samples, probe_pts) if len(samples) >= 2 else {
        "median_abs_dev": float("nan"),
        "iqr_radial": float("nan"),
        "max_median_dev": float("nan"),
    }
    row = {
        "subject": subject,
        "scenario": scenario,
        "affine_every": every,
        "n_ok": len(samples),
        "n_fail": n_fail,
        **disp,
    }
    print(
        f"{subject}/{scenario}: n={len(samples)}  "
        f"MAD={disp['median_abs_dev']:.1f}  "
        f"IQR={disp['iqr_radial']:.1f}  "
        f"maxMAD={disp['max_median_dev']:.1f} px"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=EVAL_SUBJECT)
    parser.add_argument("--scenario", default=None)
    parser.add_argument(
        "--all-eval",
        action="store_true",
        help=f"wszystkie scenariusze {EVAL_SUBJECT} (domyślnie przy braku --scenario)",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.scenario:
        rows = [probe_one(args.subject, args.scenario)]
    else:
        rows = []
        for rec in list_eval_recordings():
            if args.subject and rec.subject != args.subject:
                continue
            rows.append(probe_one(rec.subject, rec.scenario))

    csv_path = OUT / "affine_point_dispersion.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = [
        "# Stabilność affine (refresh) — rozrzut punktów w RGB [px]",
        "",
        "Stałe punkty z pierwszej udanej maski termicznej mapowane przez każdą estymatę.",
        "MAD / IQR radial ≫ 10–20 px ⇒ rejestracja niestabilna.",
        "",
        "| subject | scenario | n | MAD [px] | IQR radial [px] | maxMAD [px] |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['subject']} | {r['scenario']} | {r['n_ok']} | "
            f"{r['median_abs_dev']:.1f} | {r['iqr_radial']:.1f} | {r['max_median_dev']:.1f} |"
        )
    md.append("")
    (OUT / "affine_point_dispersion.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Zapisano: {csv_path}")


if __name__ == "__main__":
    main()
