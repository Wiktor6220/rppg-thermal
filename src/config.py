"""Jedyne źródło stałych używanych w projekcie.

Nie powielać tych wartości w innych plikach — importować stąd.
"""

from pathlib import Path

# --- Parametry sygnału ---

FS: int = 30  # częstotliwość próbkowania klatek (Hz), wspólna dla RGB i termiki (M4T)

# Pasmo zainteresowania odpowiadające fizjologicznemu zakresowi HR (42–240 bpm)
BAND_LOW_HZ: float = 0.7
BAND_HIGH_HZ: float = 4.0

# --- Ścieżki ---

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
RESULTS_DIR: Path = ROOT_DIR / "results"

# --- Indeksy landmarków MediaPipe Face Mesh (do zdefiniowania ROI) ---

# TODO: uzupełnić konkretnymi indeksami po wyborze ROI (czoło, policzki) w roi.py
FACE_MESH_LANDMARK_INDICES: dict[str, list[int]] = {}
