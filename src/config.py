"""Stałe konfiguracyjne projektu."""

from pathlib import Path

# --- Sygnał (fs zawsze z metadanych wideo) ---
BAND_LOW_HZ: float = 0.7  # ~42 BPM
BAND_HIGH_HZ: float = 4.0  # ~240 BPM

# --- Estymacja ---
DETREND_LAMBDA: float = 300.0  # smoothness priors (Tarvainen 2002)
BUTTERWORTH_ORDER: int = 3
WELCH_SEGMENT_SEC: float = 10.0

# --- Eval ---
# Badani z nietypowo wysokim HR referencji: zostają w tabelach,
# werdykt w summary.md liczy statystyki bez nich.
REF_SUSPECT_SUBJECTS: tuple[str, ...] = ()
MIN_VALID_RATIO: float = 0.5  # min. udział valid[] w oknie
# 4 równe okna (n//4 klatek; ostatnie bierze resztę)
N_QUARTER_WINDOWS: int = 4

# --- Maska perfuzji ---
PERFUSION_TEMP_STD_FACTOR: float = 0.5  # próg: mean + k*std w ROI
PERFUSION_MIN_ROI_FRAC: float = 0.10  # poniżej -> fallback do pełnego ROI

# --- Polar HR.csv ---
POLAR_HR_COLUMN: int = 3  # 0-based; kolumna BPM
POLAR_HR_SKIP_SAMPLES: int = 0

# --- Affine refresh [klatki]; klucz = nazwa sceny, brak klucza -> DEFAULT ---
AFFINE_EVERY_BY_SCENARIO: dict[str, int] = {
    "s1_rest_rest": 30,
    "s2_person_move": 30,
    "s3_drone_move": 10,
    "s4_both_move": 10,
    "s5_approach": 5,
}
AFFINE_EVERY_DEFAULT: int = 30

# --- Korejestracja ---
# Prior rozmiaru/położenia głowy w termice (nigdy nie nadpisywany w runtime).
# Rozmiar głowy w termice / bbox twarzy RGB; mediana poprawnych kalibracji.
REG_PRIOR_SCALE: float = 0.67
REG_PRIOR_OFFSET: tuple[float, float] = (-690.0, -220.0)  # px termiki
REG_SCALE_RANGE: tuple[float, float] = (0.5, 1.0)  # poza zakresem = segment nie jest głową
REG_CALIB_WINDOW_PAD: float = 2.5
# Cięcie szyi nie wyżej niż ten ułamek oczekiwanej wysokości głowy od góry segmentu.
REG_NECK_MIN_FRAC: float = 0.75
# Skala affine termika->RGB (sqrt|det|); poprawne 1.45–2.5.
REG_AFFINE_SCALE_RANGE: tuple[float, float] = (1.4, 2.7)
# sigma1/sigma2 macierzy 2×2 affine; kamery sztywne -> prawie podobieństwo.
REG_AFFINE_MAX_ANISO: float = 1.15
# Bieżące odwzorowanie nominalne; per scena nadpisywane z reg_calib.json.
REG_NOMINAL_SCALE: float = REG_PRIOR_SCALE
REG_NOMINAL_OFFSET: tuple[float, float] = REG_PRIOR_OFFSET  # px termiki
REG_WINDOW_PAD: float = 2.0
REG_MORPH_KERNEL: int = 7
REG_NECK_WIDTH_FRAC: float = 0.62
REG_EYE_BAND_TOP: float = 0.35
REG_EYE_BAND_BOTTOM: float = 0.55
REG_EYE_LANDMARK_L: int = 33
REG_EYE_LANDMARK_R: int = 263
REG_GO_RMS_PX: float = 18.0

# --- Ścieżki ---
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
RESULTS_DIR: Path = ROOT_DIR / "results"
PIPELINE_RESULTS_DIR: Path = RESULTS_DIR / "pipeline"  # domyślne --out w run.py
MODELS_DIR: Path = ROOT_DIR / "models"
FACE_LANDMARKER_MODEL_PATH: Path = MODELS_DIR / "face_landmarker.task"

# Regiony: ekstrakcja per ROI, raport (policzki = średnia lewego i prawego)
REGIONS_EXTRACT: tuple[str, ...] = ("forehead", "left_cheek", "right_cheek")
REGIONS_REPORT: tuple[str, ...] = ("forehead", "cheeks")
REGION_LABELS: dict[str, str] = {"forehead": "czoło", "cheeks": "policzki"}

# MediaPipe Face Mesh — indeksy ROI (468 punktów)
FACE_MESH_LANDMARK_INDICES: dict[str, list[int]] = {
    "forehead": [10, 67, 69, 66, 107, 108, 109, 151, 337, 338, 297, 299, 296, 336],
    "left_cheek": [50, 101, 118, 117, 116, 123, 147, 187, 205, 36, 142],
    "right_cheek": [280, 330, 347, 346, 345, 352, 376, 411, 425, 266, 371],
}
