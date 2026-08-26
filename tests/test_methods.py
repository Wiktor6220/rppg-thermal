"""Testy metod rPPG (GREEN, CHROM, POS, ICA) na sygnale syntetycznym o znanej częstości."""

import numpy as np
import pytest

from src.estimate import snr_rppg
from src.methods import chrom, green, ica_method, pos
from tests.synthetic import (
    dominant_hr_bpm,
    generate_hard_synthetic_rgb,
    generate_synthetic_rgb,
)

FS_TEST = 30.0
TRUE_HR_BPM = 72.0
TOLERANCE_BPM = 5.0

METHODS = {"GREEN": green, "CHROM": chrom, "POS": pos, "ICA": ica_method}


@pytest.mark.parametrize("method_fn", METHODS.values(), ids=METHODS.keys())
def test_method_recovers_known_hr(method_fn):
    """Każda metoda odzyskuje zadane HR z syntetycznego RGB w granicach tolerancji."""
    rgb_signal, _ = generate_synthetic_rgb(fs=FS_TEST, duration_s=30.0, hr_bpm=TRUE_HR_BPM)

    pulse_signal = method_fn(rgb_signal, FS_TEST)
    estimated_bpm = dominant_hr_bpm(pulse_signal, FS_TEST)

    assert abs(estimated_bpm - TRUE_HR_BPM) <= TOLERANCE_BPM


@pytest.mark.parametrize("method_fn", METHODS.values(), ids=METHODS.keys())
def test_method_returns_1d_signal_of_matching_length(method_fn):
    """Każda metoda zwraca sygnał 1D o długości wejścia."""
    rgb_signal, _ = generate_synthetic_rgb(fs=FS_TEST, duration_s=30.0, hr_bpm=TRUE_HR_BPM)

    pulse_signal = method_fn(rgb_signal, FS_TEST)

    assert pulse_signal.ndim == 1
    assert pulse_signal.shape[0] == rgb_signal.shape[0]


@pytest.mark.parametrize("method_fn", METHODS.values(), ids=METHODS.keys())
def test_method_rejects_wrong_shape(method_fn):
    """Wejście o złym kształcie (nie (N, 3)) jest odrzucane."""
    with pytest.raises(ValueError):
        method_fn(np.zeros((100, 2)), FS_TEST)


# --- Trudny sygnał: różnicowanie GREEN vs CHROM/POS przez SNR i odporność na artefakt ---

HARD_SEEDS = [0, 1, 2, 3, 4]


@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_chrom_pos_outperform_green_on_common_mode_artifact(seed):
    """Na trudnym sygnale ze wspólnym artefaktem w paśmie CHROM/POS mają wyższe SNR niż GREEN.

    Na czystym sinusie wszystkie metody dają identyczny wynik (błąd 0). Różnicę widać
    dopiero przy wspólnym multiplikatywnym artefakcie jasności w paśmie tętna, który
    CHROM/POS znoszą w projekcji chrominancji, a GREEN przepuszcza.
    """
    rgb, _ = generate_hard_synthetic_rgb(fs=FS_TEST, duration_s=30.0, hr_bpm=TRUE_HR_BPM, seed=seed)

    snr_green = snr_rppg(green(rgb, FS_TEST), FS_TEST, TRUE_HR_BPM)
    snr_chrom = snr_rppg(chrom(rgb, FS_TEST), FS_TEST, TRUE_HR_BPM)
    snr_pos = snr_rppg(pos(rgb, FS_TEST), FS_TEST, TRUE_HR_BPM)

    # Margines 3 dB — wyraźnie ponad rozrzut między ziarnami (patrz probe w historii).
    assert snr_chrom > snr_green + 3.0
    assert snr_pos > snr_green + 3.0


@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_chrom_pos_recover_hr_where_green_fails(seed):
    """CHROM/POS odzyskują prawdziwe HR na trudnym sygnale; GREEN daje się zwieść artefaktowi."""
    rgb, _ = generate_hard_synthetic_rgb(fs=FS_TEST, duration_s=30.0, hr_bpm=TRUE_HR_BPM, seed=seed)

    hr_chrom = dominant_hr_bpm(chrom(rgb, FS_TEST), FS_TEST)
    hr_pos = dominant_hr_bpm(pos(rgb, FS_TEST), FS_TEST)
    hr_green = dominant_hr_bpm(green(rgb, FS_TEST), FS_TEST)

    assert abs(hr_chrom - TRUE_HR_BPM) <= TOLERANCE_BPM
    assert abs(hr_pos - TRUE_HR_BPM) <= TOLERANCE_BPM
    # GREEN zostaje przyciągnięty do artefaktu (~96 bpm) — potwierdza różnicę metod.
    assert abs(hr_green - TRUE_HR_BPM) > TOLERANCE_BPM
