# CLAUDE.md — konwencje techniczne projektu

Instrukcje dla Claude Code przy pracy na tym repozytorium.
Kontekst merytoryczny (cel pracy, architektura, decyzje) — patrz `CONTEXT.md`.

## Środowisko

- **Python 3.11**, zarządzane przez **uv**. Nigdy nie używaj `pip` bezpośrednio.
- Instalacja zależności: `uv add <pakiet>`. Uruchamianie: `uv run python ...`, `uv run <cmd>`.
- Zależności deklarowane w `pyproject.toml` — nie w kodzie, nie w osobnym pliku „wszystkie importy".
- Platforma: MacBook M1 (arm64). mediapipe działa natywnie.

## Struktura kodu

- Cała logika w plikach `.py` w `src/`. Notebooki (`notebooks/`) tylko do eksploracji i wykresów — **importują** z `src/`, nigdy nie definiują logiki.
- **Moduły w `src/` nigdy nie uruchamiają się przy imporcie.** Definiują funkcje. Kod wykonywalny tylko pod `if __name__ == "__main__":` (do szybkiego testu modułu) albo w `run.py`/notebooku.
- Każdy plik importuje **tylko to, czego sam używa**. Bez wspólnego pliku ze wszystkimi importami.
- **`config.py` to jedyne źródło stałych** (fs=30, pasmo 0.7–4 Hz, ścieżki, indeksy landmarków). Nie powielać stałych w innych plikach.
- **`methods.py` = czyste funkcje sygnał→sygnał.** Bez wczytywania plików, bez rysowania. Dzięki temu testowalne na sygnale syntetycznym.
- `run.py` niczego nie liczy sam — wywołuje funkcje z modułów w ustalonej kolejności.

## Zasady merytoryczne (krytyczne — źródło błędów w poprzedniej wersji)

- **Nigdy nie normalizuj pojedynczej klatki** (min-max per frame kasuje sygnał tętna). Normalizacja tylko po osi **czasu** (np. `x / x.mean()` po całym oknie). Termika — na temperaturze **bezwzględnej** (radiometrycznej).
- **Nigdy nie usuwaj klatki** przy braku ROI. Przytrzymaj ostatnią pozycję lub interpoluj, oznacz w wektorze `valid[]`. Odrzucaj OKNA, nie klatki. Oś czasu musi mieć stały krok (wymóg FFT/Welcha).
- **ICA/PCA tylko na wielu kanałach** (3× RGB lub wiele ROI). Nigdy na sygnale 1D — to operacja pusta.
- Walidacja **w oknach** (np. 10 s, przesuwane), MAE/RMSE per okno względem referencji — nie jedna liczba na całość.

## Styl

- Identyfikatory (nazwy funkcji, zmiennych) po **angielsku**. Komentarze i docstringi mogą być po **polsku** (praca jest po polsku).
- Funkcje krótkie, jedna odpowiedzialność. Typowanie argumentów tam, gdzie pomaga czytelności.
- Zanim napiszesz metodę referencyjną (CHROM/POS/…), sprawdź ją na sygnale syntetycznym o znanej częstości.

## Git

- `data/` i `results/` są poza gitem (`.gitignore`). Nie commituj danych ani dużych plików wynikowych.
- Notebooki commituj oszczędnie (czyść output przed commitem, jeśli to możliwe).

## Czego NIE robić bez pytania

- Nie dodawaj deep learningu ani ciężkich zależności (torch itd.) — projekt jest klasyczny. DL to osobna, późniejsza decyzja.
- Nie instaluj `dji-thermal-sdk` na tym etapie (natywne libki, kłopotliwe na macOS arm; dopiero przy własnych danych M4T).
- Nie zmieniaj przyjętej architektury modułów bez uzgodnienia.
