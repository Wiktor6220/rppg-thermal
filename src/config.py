"""Stałe konfiguracyjne projektu."""

from pathlib import Path

# --- Parametry sygnału ---

FS: int = 30  # częstotliwość próbkowania klatek (Hz), wspólna dla RGB i termiki (M4T)

# Pasmo HR ~42–240 bpm
BAND_LOW_HZ: float = 0.7
BAND_HIGH_HZ: float = 4.0

# --- Detrending / filtracja / estymacja HR (estimate.py) ---

# Smoothness priors (Tarvainen et al., 2002); większa lambda → silniejsze tłumienie trendu
DETREND_LAMBDA: float = 300.0

BUTTERWORTH_ORDER: int = 3

# Segment Welcha [s], spójny z oknem walidacji
WELCH_SEGMENT_SEC: float = 10.0

# --- Walidacja (validate.py) ---

VALIDATION_WINDOW_SEC: float = 10.0
VALIDATION_STEP_SEC: float = 5.0

# Poniżej tego udziału valid[] w oknie — okno pomijane w metrykach
MIN_VALID_RATIO: float = 0.5

# --- Maska perfuzji z termiki (extract.py) ---

# Próg: mean(ROI) + k * std(ROI) na wartościach termicznych w ROI
PERFUSION_TEMP_STD_FACTOR: float = 0.5
# Zbyt mała maska perfuzji → fallback do pełnego ROI
PERFUSION_MIN_ROI_FRAC: float = 0.10

# --- Referencja Polar H10 (plik *_HR.csv) ---
# Kolumna 4 (1-based; nagłówek „2026”) = HR [BPM]. Pierwsze N próbek = kalibracja.
POLAR_HR_COLUMN: int = 3  # 0-based index
POLAR_HR_SKIP_SAMPLES: int = 5

# --- Odświeżanie affine termika→RGB (klatki) wg scenariusza ---
AFFINE_EVERY_BY_SCENARIO: dict[str, int] = {
    "s1_rest_rest": 30,
    "s2_person_move": 30,
    "s3_drone_move": 10,
    "s4_both_move": 10,
    "s5_approach": 5,
}
AFFINE_EVERY_DEFAULT: int = 30

# --- Korejestracja termika → RGB (registration.py) ---
REG_NOMINAL_SCALE: float = 0.52
REG_NOMINAL_OFFSET: tuple[float, float] = (-340.0, -60.0)  # [px termiki]
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
MODELS_DIR: Path = ROOT_DIR / "models"

# MediaPipe FaceLandmarker (.task) — plik w models/ (poza gitem)
FACE_LANDMARKER_MODEL_PATH: Path = MODELS_DIR / "face_landmarker.task"

# --- Indeksy landmarków MediaPipe Face Mesh (468 punktów, bez iris) ---

FACE_MESH_LANDMARK_INDICES: dict[str, list[int]] = {
    "forehead": [10, 67, 69, 66, 107, 108, 109, 151, 337, 338, 297, 299, 296, 336],
    "left_cheek": [50, 101, 118, 117, 116, 123, 147, 187, 205, 36, 142],
    "right_cheek": [280, 330, 347, 346, 345, 352, 376, 411, 425, 266, 371],
}
