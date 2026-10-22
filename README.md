# rppg-thermal

Estymacja HR metodą rPPG z nagrań RGB + termika (DJI), z bramkowaniem ROI maską perfuzji z termiki.
Referencja: Polar H10 (`*_HR.csv` / `dataHR_*.csv`).

## Uruchomienie

Wymagania: Python 3.11 przez [uv](https://docs.astral.sh/uv/), model `models/face_landmarker.task`.

```bash
uv run run.py                         # wszyscy badani z data/
uv run run.py --subject subject04     # jeden badany
uv run run.py --recalibrate           # wymuś ponowną kalibrację rejestracji każdej sceny
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
    ├── fig_spectrum.png      # widmo Welcha (czoło)
    ├── run_stderr.log        # natywne logi MediaPipe z przebiegu
    └── calib_stderr.log      # natywne logi z kalibracji
```

`summary.md` i `all_subjects.csv` są składane z dysku, więc przebieg z `--subject` odświeża tylko tego
badanego, a agregat dalej zawiera pozostałych.

## Struktura kodu

`src/config.py → io_layer → roi → registration → extract → methods → estimate → validate → pipeline`.
`run.py` to cienkie CLI nad `src.pipeline.run_pipeline`; sam nic nie liczy.
