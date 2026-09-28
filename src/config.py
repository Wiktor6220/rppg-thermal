"""Stałe konfiguracyjne projektu."""

from pathlib import Path

# --- Sygnał ---
FS: int = 30  # Hz (nominalnie; runtime bierze fps z metadanych)
BAND_LOW_HZ: float = 0.7  # ~42 BPM
BAND_HIGH_HZ: float = 4.0  # ~240 BPM

# --- Estymacja ---
DETREND_LAMBDA: float = 300.0  # smoothness priors (Tarvainen 2002)
BUTTERWORTH_ORDER: int = 3
WELCH_SEGMENT_SEC: float = 10.0

# --- Eval ---
EVAL_SUBJECT: str = "subject02"
REF_SOURCE: str = "hr_csv"
VALIDATION_WINDOW_SEC: float = 10.0
VALIDATION_STEP_SEC: float = 5.0
MIN_VALID_RATIO: float = 0.5  # min. udział valid[] w oknie

# --- Maska perfuzji ---
PERFUSION_TEMP_STD_FACTOR: float = 0.5  # próg: mean + k*std w ROI
PERFUSION_MIN_ROI_FRAC: float = 0.10  # poniżej → fallback do pełnego ROI

# --- Polar HR.csv ---
POLAR_HR_COLUMN: int = 3  # 0-based; kolumna BPM
POLAR_HR_SKIP_SAMPLES: int = 0

# --- Polar ECG (tylko skrypty diagnostyczne) ---
ECG_FS_HZ: float = 130.0
ECG_SKIP_SEC: float = 10.0
ECG_FRAME_MARKERS: tuple[tuple[int, int, int, int], ...] = (
    (67, 82, 8, 0),
    (68, 82, 8, 0),
)
ECG_HR_MIN_BPM: float = 45.0
ECG_HR_MAX_BPM: float = 140.0

# --- Affine refresh [klatki] ---
AFFINE_EVERY_BY_SCENARIO: dict[str, int] = {
    "s1_rest_rest": 30,
    "s2_person_move": 30,
    "s3_drone_move": 10,
    "s4_both_move": 10,
    "s5_approach": 5,
}
AFFINE_EVERY_DEFAULT: int = 30
AFFINE_FIXED_MEDIAN_N: int = 30

# --- Korejestracja ---
REG_NOMINAL_SCALE: float = 0.52
REG_NOMINAL_OFFSET: tuple[float, float] = (-340.0, -60.0)  # px termiki
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
FACE_LANDMARKER_MODEL_PATH: Path = MODELS_DIR / "face_landmarker.task"

# MediaPipe Face Mesh — indeksy ROI (468 punktów)
FACE_MESH_LANDMARK_INDICES: dict[str, list[int]] = {
    "forehead": [10, 67, 69, 66, 107, 108, 109, 151, 337, 338, 297, 299, 296, 336],
    "left_cheek": [50, 101, 118, 117, 116, 123, 147, 187, 205, 36, 142],
    "right_cheek": [280, 330, 347, 346, 345, 352, 376, 411, 425, 266, 371],
}
