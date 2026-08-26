"""Testy potoku estymacji HR: detrend, filtracja pasmowa, Welch i find_peaks."""

import numpy as np
import pytest

from src.estimate import (
    bandpass_filter,
    detrend_signal,
    estimate_hr_peaks,
    estimate_hr_welch,
    snr_rppg,
)
from tests.synthetic import generate_synthetic_pulse

FS_TEST = 30.0
TRUE_HR_BPM = 72.0
TOLERANCE_BPM = 5.0


@pytest.fixture
def cleaned_pulse():
    """Syntetyczny puls 1D po detrendzie i filtracji pasmowej."""
    raw = generate_synthetic_pulse(fs=FS_TEST, duration_s=60.0, hr_bpm=TRUE_HR_BPM)
    return bandpass_filter(detrend_signal(raw), FS_TEST)


@pytest.mark.parametrize("estimator", [estimate_hr_welch, estimate_hr_peaks])
def test_estimator_recovers_known_hr(cleaned_pulse, estimator):
    """Welch i find_peaks odzyskują zadane HR w granicach tolerancji."""
    hr_bpm = estimator(cleaned_pulse, FS_TEST)
    assert abs(hr_bpm - TRUE_HR_BPM) <= TOLERANCE_BPM


def test_welch_and_peaks_agree(cleaned_pulse):
    """Obie metody estymacji zgadzają się ze sobą w granicach tolerancji."""
    hr_welch = estimate_hr_welch(cleaned_pulse, FS_TEST)
    hr_peaks = estimate_hr_peaks(cleaned_pulse, FS_TEST)
    assert abs(hr_welch - hr_peaks) <= TOLERANCE_BPM


def test_detrend_removes_slow_trend():
    """Detrend usuwa wolny trend — średnia sygnału po detrendzie bliska zeru."""
    raw = generate_synthetic_pulse(fs=FS_TEST, duration_s=60.0, hr_bpm=TRUE_HR_BPM)
    detrended = detrend_signal(raw)
    assert abs(detrended.mean()) < abs(raw.mean())
    assert detrended.shape == raw.shape


def test_estimate_hr_peaks_raises_on_flat_signal():
    """Sygnał płaski: find_peaks nie znajduje pików -> ValueError."""
    flat = np.zeros(int(FS_TEST * 30))
    with pytest.raises(ValueError):
        estimate_hr_peaks(flat, FS_TEST)


def test_snr_high_for_pure_sine_at_ref_hr():
    """Czysty sinus o częstości referencyjnej ma wysokie SNR (moc skupiona w prążku HR)."""
    n = int(FS_TEST * 30)
    t = np.arange(n) / FS_TEST
    sine = np.sin(2 * np.pi * (TRUE_HR_BPM / 60.0) * t)
    assert snr_rppg(sine, FS_TEST, TRUE_HR_BPM) > 20.0


def test_snr_low_for_broadband_noise():
    """Szum szerokopasmowy ma niskie SNR (brak skupienia mocy w prążku HR)."""
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(int(FS_TEST * 30))
    assert snr_rppg(noise, FS_TEST, TRUE_HR_BPM) < 3.0


def test_snr_drops_when_signal_at_wrong_reference():
    """SNR liczone względem błędnej częstości jest niższe niż względem prawdziwej."""
    n = int(FS_TEST * 30)
    t = np.arange(n) / FS_TEST
    sine = np.sin(2 * np.pi * (TRUE_HR_BPM / 60.0) * t)
    assert snr_rppg(sine, FS_TEST, TRUE_HR_BPM) > snr_rppg(sine, FS_TEST, TRUE_HR_BPM + 30.0)
