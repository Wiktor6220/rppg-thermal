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
# Gdy maska perfuzji pokrywa mniej niż ten ułamek ROI → fallback do pełnego ROI
# (nie psuje sygnału przy zbyt ostrej / przesuniętej masce).
PERFUSION_MIN_ROI_FRAC: float = 0.10

# --- Referencja Polar H10 (plik *_HR.csv) ---
# Kolumna 4 (1-based; nagłówek „2026”) = HR [BPM]. Pierwsze N próbek = kalibracja.
POLAR_HR_COLUMN: int = 3  # 0-based index
POLAR_HR_SKIP_SAMPLES: int = 5

# --- Odświeżanie affine termika→RGB (klatki) wg scenariusza ---
# s1/s2: spokojniej; s3/s4: ruch; s5: zmienny dystans / paralaksa.
AFFINE_EVERY_BY_SCENARIO: dict[str, int] = {
    "s1_rest_rest": 30,
    "s2_person_move": 30,
    "s3_drone_move": 10,
    "s4_both_move": 10,
    "s5_approach": 5,
}
AFFINE_EVERY_DEFAULT: int = 30

# --- Korejestracja termika → RGB (registration.py) ---
# Nominalne odwzorowanie RGB→termika służy TYLKO do wycięcia okna segmentacji
# (nie do precyzji). Affine: kontury masek + refine Y linii oczu.
# Wartości empiryczne z subject01/s1 (probe).
REG_NOMINAL_SCALE: float = 0.52
REG_NOMINAL_OFFSET: tuple[float, float] = (-340.0, -60.0)  # [px termiki]
REG_WINDOW_PAD: float = 2.0  # powiększenie okna wokół nominalnej twarzy
REG_MORPH_KERNEL: int = 7
REG_NECK_WIDTH_FRAC: float = 0.62  # cięcie szyi: ułamek maks. szerokości twarzy
# Pas wyszukiwania linii oczu na masce termicznej (ułamek wysokości maski od góry).
# Nie od 0 — góra maski (włosy/czoło) jest ciemniejsza i fałszywie wygrywa.
REG_EYE_BAND_TOP: float = 0.35
REG_EYE_BAND_BOTTOM: float = 0.55
# Zewnętrzne kąciki oczu MediaPipe (Face Mesh) — cel refine Y.
REG_EYE_LANDMARK_L: int = 33
REG_EYE_LANDMARK_R: int = 263
# Próg go/no-go (RMS vs ręczny GT). Ręczna ~10.5; kontur+Y-oczy ~13 px na s1.
REG_GO_RMS_PX: float = 18.0

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
