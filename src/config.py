"""Jedyne źródło stałych używanych w projekcie.

Nie powielać tych wartości w innych plikach — importować stąd.
"""

from pathlib import Path

# --- Parametry sygnału ---

FS: int = 30  # częstotliwość próbkowania klatek (Hz), wspólna dla RGB i termiki (M4T)

# Pasmo zainteresowania odpowiadające fizjologicznemu zakresowi HR (42–240 bpm)
BAND_LOW_HZ: float = 0.7
BAND_HIGH_HZ: float = 4.0

# --- Detrending / filtracja / estymacja HR (estimate.py) ---

# Parametr regularyzacji dla detrendingu metodą smoothness priors
# (Tarvainen et al., 2002) — im większy, tym silniejsze tłumienie wolnych
# składowych (silniejszy efekt górnoprzepustowy). Dobrany empirycznie dla FS=30.
DETREND_LAMBDA: float = 300.0

BUTTERWORTH_ORDER: int = 3  # rząd filtru pasmowoprzepustowego Butterwortha

# Długość segmentu dla estymatora widma mocy (Welch), w sekundach —
# spójna z długością okna walidacyjnego (10 s, patrz CONTEXT.md)
WELCH_SEGMENT_SEC: float = 10.0

# --- Walidacja (validate.py) ---

VALIDATION_WINDOW_SEC: float = 10.0  # długość okna walidacyjnego (CONTEXT.md: "np. 10 s")
VALIDATION_STEP_SEC: float = 5.0  # krok przesuwanego okna (50% zakładki)

# Minimalny odsetek ważnych klatek w oknie (wg valid[]), poniżej którego okno
# jest pomijane w metrykach — odrzucamy OKNA z przewagą nieważnych klatek,
# nigdy pojedyncze klatki (CLAUDE.md).
MIN_VALID_RATIO: float = 0.5

# --- Maska perfuzji z termiki (extract.py) ---

# Piksel uznajemy za „wysokiej perfuzji", gdy jego temperatura (wartość
# radiometryczna, bezwzględna) przekracza średnią ROI o `PERFUSION_TEMP_STD_FACTOR`
# odchyleń standardowych liczonych w obrębie ROI. Próg względny wobec rozkładu ROI,
# a nie normalizacja per klatka — pracujemy na temperaturze bezwzględnej (CLAUDE.md).
PERFUSION_TEMP_STD_FACTOR: float = 0.5

# --- Ścieżki ---

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
RESULTS_DIR: Path = ROOT_DIR / "results"
MODELS_DIR: Path = ROOT_DIR / "models"

# Model MediaPipe Tasks FaceLandmarker (.task) — pobierany osobno (poza gitem), bo
# ten build mediapipe nie dostarcza offline `solutions.face_mesh` ani modelu w paczce.
# Pobranie: storage.googleapis.com/mediapipe-models/face_landmarker/.../face_landmarker.task
FACE_LANDMARKER_MODEL_PATH: Path = MODELS_DIR / "face_landmarker.task"

# --- Indeksy landmarków MediaPipe Face Mesh (siatka 468 punktów, bez iris) ---

# Kuratorowane klastry punktów wyznaczające ROI wysokiej perfuzji (czoło, policzki).
# Indeksy w zakresie 0..467 (refine_landmarks=False). ROI liczymy jako bounding box
# obejmujący dany klaster. Zestawy są przybliżone i można je doprecyzować po podglądzie.
FACE_MESH_LANDMARK_INDICES: dict[str, list[int]] = {
    # Czoło: od linii brwi (66/296, 107/336) w górę do czubka (10), boki (67/297).
    "forehead": [10, 67, 69, 66, 107, 108, 109, 151, 337, 338, 297, 299, 296, 336],
    # Policzek lewy (strona obrazu): od okolic nosa (50) po środek policzka (205).
    "left_cheek": [50, 101, 118, 117, 116, 123, 147, 187, 205, 36, 142],
    # Policzek prawy (strona obrazu), lustrzane odpowiedniki punktów lewego.
    "right_cheek": [280, 330, 347, 346, 345, 352, 376, 411, 425, 266, 371],
}
