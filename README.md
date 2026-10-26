# rppg-thermal

Estymacja HR metodą rPPG z nagrań RGB + termika (DJI), z bramkowaniem ROI maską perfuzji z termiki.
Referencja: Polar H10 (`*_HR.csv` / `dataHR_*.csv`).

## Uruchomienie

Wymagania: Python 3.11 przez [uv](https://docs.astral.sh/uv/), model `models/face_landmarker.task`.

```bash
uv run run.py                         # wszyscy badani z data/
uv run run.py --subject subject04     # jeden badany
uv run run.py --recalibrate           # wymuś ponowną kalibrację rejestracji każdej sceny
uv run run.py --figures-only          # tylko reg_quality.json + figury rejestracji (bez HR/Polar, metryk i agregatu)
uv run run.py --data data --out results/pipeline   # wartości domyślne
uv run pytest                         # testy
```

Wejście: `data/<subject>/<scena>/` z plikami `*_RGB.MP4` lub `*_V.MP4` (RGB), `*_T.MP4` (termika)
i `dataHR_*.csv` lub `*_HR.csv` (Polar). Opcjonalnie `data/reg_calib.json` jako wspólny fallback kalibracji.

## Co liczy potok

Per scena: synchronizacja wideo↔Polar po `creation_time` (strefa Europe/Warsaw), kalibracja rejestracji
termika→RGB z pierwszej dobrze wykrytej klatki sceny, ROI bbox z landmarków MediaPipe (czoło, policzki),
sygnał **plain** (średnie RGB w ROI) i **gated** (RGB w ROI ∩ cieplejsze piksele termiki), metody **POS**
i **CHROM**, walidacja w **4 równych oknach** (`n//4` klatek, ostatnie bierze resztę).
Wynik: oba regiony × POS/CHROM × plain/gated — bez jednoliczbowego „headline”.

**Rejestracja.** Kalibracja sceny segmentuje termikę w oknie z priora (`REG_PRIOR_SCALE`,
`REG_PRIOR_OFFSET`, `REG_CALIB_WINDOW_PAD`) i spośród do 5 największych jasnych składowych wybiera tę
o rozmiarze najbliższym oczekiwanej głowie. Cięcie szyi (profil szerokości) nie może wypaść wyżej niż
`REG_NECK_MIN_FRAC` (0,75) × oczekiwana wysokość głowy od góry segmentu — inaczej u osób w okularach
(szkła w termice są czarne) potok ciąłby twarz na wysokości oczu. Klatka jest odrzucana (bierzemy
następną z twarzą), gdy szerokość LUB wysokość segmentu termicznego względem bboxa twarzy RGB wypada
poza `REG_SCALE_RANGE` (0,5–1,0) — wtedy segment to nie głowa (np. tułów albo samo czoło). Brak udanej
klatki → fallback z innej sceny (też tylko ze skalą w zakresie).

**Jakość rejestracji.** Affine jest odświeżany co `AFFINE_EVERY_BY_SCENARIO[scena]` klatek. Próba jest odrzucana,
gdy końcowy affine termika→RGB jest nieprawdopodobny: skala √|det A| poza `REG_AFFINE_SCALE_RANGE`
(1,4–2,7) albo anizotropia σ1/σ2 macierzy 2×2 > `REG_AFFINE_MAX_ANISO` (1,15; kamery są sztywno
złączone, więc affine to prawie podobieństwo). Odrzucona próba nie zastępuje bieżącego affine i liczy się do mediany RMS jako `inf` (`affine_implausible` w
`reg_quality.json`). `gated_status=trustworthy`, gdy mediana RMS sceny jest skończona i ≤ 18 px.

## Katalog wyjściowy (`--out`, domyślnie `results/pipeline/`)

```text
results/pipeline/
├── all_subjects.csv          # wszystkie wiersze ze wszystkich metrics.csv w --out
├── summary.md                # pełna tabela + krótki werdykt (z tych samych wierszy)
└── <subject>/<scena>/
    ├── metrics.csv           # region × metoda: HR sceny, HR w 4 oknach, MAE/RMSE plain/gated, RMS rejestracji
    ├── metrics.md            # to samo w czytelnej tabeli
    ├── reg_calib.json        # kalibracja rejestracji tej sceny (+ calib_source_tag: self / fallback:…)
    ├── reg_quality.json      # RMS konturu, IoU, status trustworthy/NO-GO (próg 18 px)
    ├── fig_rgb_roi.png       # ROI czoła i policzków na klatce RGB
    ├── fig_thermal_seg.png   # segment twarzy w termice + maska perfuzji
    ├── fig_overlay_reg.png   # termika nałożona na RGB po rejestracji
    ├── fig_hr_windows.png    # HR w 4 równych oknach (czoło/policzki × POS/CHROM) + fragment fali
    ├── fig_spectrum.png      # widmo Welcha (czoło/policzki × POS/CHROM)
    ├── run_stderr.log        # natywne logi MediaPipe z przebiegu
    └── calib_stderr.log      # natywne logi z kalibracji
```

`summary.md` i `all_subjects.csv` są składane z dysku, więc przebieg z `--subject` odświeża tylko tego
badanego, a agregat dalej zawiera pozostałych.

## Struktura kodu

| Moduł | Zawartość |
|---|---|
| `src/config.py` | wszystkie stałe (pasmo, progi, refresh affine per scena, regiony, ścieżki) |
| `src/io_layer.py` | metadane wideo, pary klatek RGB/termika po czasie, Polar HR.csv + sync `creation_time` |
| `src/roi.py` | MediaPipe FaceLandmarker na wycinku, ROI bbox z landmarków |
| `src/registration.py` | affine termika→RGB z konturów masek, jakość rejestracji, kalibracja nominalna |
| `src/extract.py` | średnie RGB w ROI, próbki termiki przez affine, bramkowanie perfuzją |
| `src/methods.py` | GREEN, CHROM, POS, ICA (sygnał→sygnał) |
| `src/estimate.py` | detrend, bandpass, HR z Welcha, SNR |
| `src/validate.py` | 4 równe okna, MAE/RMSE, błędy oktawowe |
| `src/pipeline.py` | sesje w `data/`, kalibracja per scena, przebieg po klatkach, metryki, `run_pipeline` |
| `src/figures.py` | wszystkie `fig_*.png` sceny |
| `src/report.py` | `metrics.csv`/`metrics.md`, agregat `all_subjects.csv` + `summary.md` |

`run.py` to cienkie CLI nad `src.pipeline.run_pipeline`; sam nic nie liczy.
`scripts/check_data.py` — raport spójności plików w `data/`.
Testy (`tests/`) działają wyłącznie na danych syntetycznych, bez `data/`.
