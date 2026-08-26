"""Wczytywanie danych wejściowych: klatki wideo (RGB/termika) i referencyjny sygnał PPG.

Obsługiwane zbiory: UBFC-rPPG (RGB + PPG), iBVP (RGB + termika + PPG).
Moduł tylko odczytuje dane z dysku — nie przetwarza sygnału.
"""

from pathlib import Path

import numpy as np


def load_ubfc_subject(subject_dir: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Wczytuje jedną sesję ze zbioru UBFC-rPPG.

    Args:
        subject_dir: ścieżka do katalogu osoby badanej (zawiera wideo i plik referencji PPG).

    Returns:
        Krotka (frames, ppg_reference, ppg_fs):
            frames: tablica klatek RGB o kształcie (N, H, W, 3), dtype uint8.
            ppg_reference: 1D tablica referencyjnego sygnału PPG.
            ppg_fs: częstotliwość próbkowania sygnału referencyjnego (Hz).
    """
    raise NotImplementedError


def load_ibvp_subject(
    subject_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Wczytuje jedną sesję ze zbioru iBVP (RGB + termika + referencja PPG).

    Args:
        subject_dir: ścieżka do katalogu sesji (np. `p01_a/`), zawierającego
            podkatalogi `_rgb/`, `_t/` oraz plik `_bvp.csv`.

    Returns:
        Krotka (rgb_frames, thermal_frames, ppg_reference, ppg_fs):
            rgb_frames: tablica klatek RGB o kształcie (N, H, W, 3), dtype uint8.
            thermal_frames: tablica klatek termicznych o kształcie (N, H, W),
                wartości radiometryczne (temperatura bezwzględna), bez normalizacji.
            ppg_reference: 1D tablica referencyjnego sygnału PPG (PhysioKit, z ucha).
            ppg_fs: częstotliwość próbkowania sygnału referencyjnego (Hz).
    """
    raise NotImplementedError


def load_ppg_reference(csv_path: Path) -> tuple[np.ndarray, float]:
    """Wczytuje referencyjny sygnał PPG z pliku CSV.

    Args:
        csv_path: ścieżka do pliku z referencyjnym przebiegiem PPG.

    Returns:
        Krotka (ppg_signal, fs): 1D sygnał referencyjny i jego częstotliwość próbkowania (Hz).
    """
    raise NotImplementedError
