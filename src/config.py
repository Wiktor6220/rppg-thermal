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

# --- Indeksy landmarków MediaPipe Face Mesh (do zdefiniowania ROI) ---

# TODO: uzupełnić konkretnymi indeksami po wyborze ROI (czoło, policzki) w roi.py
FACE_MESH_LANDMARK_INDICES: dict[str, list[int]] = {}
