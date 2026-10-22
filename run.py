"""CLI potoku rPPG RGB+termika — cała logika w ``src.pipeline``.

    uv run run.py                      # wszyscy badani z data/
    uv run run.py --subject subject04  # jeden badany (agregat obejmuje cały --out)
    uv run run.py --recalibrate        # wymuś ponowną kalibrację rejestracji per scena
"""

import argparse
from pathlib import Path

from src.config import DATA_DIR, PIPELINE_RESULTS_DIR
from src.pipeline import run_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Potok rPPG RGB+termika (DJI) → metryki i figury.")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="Katalog z danymi (domyślnie data/)")
    parser.add_argument("--subject", default=None, help="np. subject04 (domyślnie wszyscy)")
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Przelicz reg_calib.json każdej sceny",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PIPELINE_RESULTS_DIR,
        help="Katalog wyjściowy (domyślnie results/pipeline/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(args.data, args.out, subject=args.subject, recalibrate=args.recalibrate)
