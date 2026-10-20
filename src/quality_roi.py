"""Jakość ROI bez referencji: skin-fraction + peak-SNR (tylko informacyjnie)."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import periodogram

from src.config import (
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    PEAK_SNR_BIN_WIDTH_HZ,
    PEAK_SNR_CONF_HIGH_DB,
    PEAK_SNR_CONF_MED_DB,
    PEAK_SNR_N_HARMONICS,
    REGION_MIN_VALID_FRAC,
    SKIN_FRAC_MIN,
)

_EPS = 1e-12


def skin_mask_ycrcb(rgb_frame: np.ndarray) -> np.ndarray:
    """Maska skóry w YCrCb; wejście **RGB** (nie BGR)."""
    rgb = np.asarray(rgb_frame)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Oczekiwano RGB HxWx3, otrzymano {rgb.shape}")
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    # Klasyczny zakres skóry (Chai/Ngan): Cr∈[133,173], Cb∈[77,127]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)


def skin_fraction_in_mask(rgb_frame: np.ndarray, roi_mask: np.ndarray) -> float:
    """Udział pikseli skóry w ROI; pusta maska → NaN."""
    mask = np.asarray(roi_mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    skin = skin_mask_ycrcb(rgb_frame)
    return float(np.mean(skin[mask]))


def peak_snr_db(
    signal: np.ndarray,
    fs: float,
    n_harmonics: int = PEAK_SNR_N_HARMONICS,
    bin_width_hz: float = PEAK_SNR_BIN_WIDTH_HZ,
) -> float:
    """Peak-to-median prominence [dB] w paśmie HR — **bez** referencji Polar.

    ``10·log10(max(PSD_band) / mediana(PSD_band))``. Parametry ``n_harmonics`` /
    ``bin_width_hz`` zachowane dla kompatybilności API (nieużywane).
    Tylko diagnostyka — nie służy do wyboru regionu.
    """
    del n_harmonics, bin_width_hz
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < 8 or fs <= 0:
        return float("nan")
    freqs, psd = periodogram(signal, fs=fs)
    band = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    if not np.any(band):
        return float("nan")
    band_psd = psd[band]
    med = float(np.median(band_psd))
    peak = float(np.max(band_psd))
    return float(10.0 * np.log10((peak + _EPS) / (med + _EPS)))


def select_report_region(
    skin_forehead: float,
    skin_cheeks: float,
    snr_forehead: float,
    snr_cheeks: float,
    valid_frac: float,
    *,
    tau_skin: float = SKIN_FRAC_MIN,
    min_valid_frac: float = REGION_MIN_VALID_FRAC,
    snr_high_db: float = PEAK_SNR_CONF_HIGH_DB,
    snr_med_db: float = PEAK_SNR_CONF_MED_DB,
) -> dict:
    """Legacy: historyczny selektor regionu (skin_frac + peak-SNR).

    **Nieużywany** w pipeline — headline = ``HEADLINE_REGION`` (cheeks).
    Zostawione dla kompatybilności testów / diagnostyki.
    """
    enough_det = bool(np.isfinite(valid_frac) and valid_frac >= min_valid_frac)
    viable_f = bool(
        enough_det and np.isfinite(skin_forehead) and skin_forehead >= tau_skin
    )
    viable_c = bool(
        enough_det and np.isfinite(skin_cheeks) and skin_cheeks >= tau_skin
    )

    if viable_f and viable_c:
        # Oba ważne → wyższy peak-SNR
        snr_f = snr_forehead if np.isfinite(snr_forehead) else -np.inf
        snr_c = snr_cheeks if np.isfinite(snr_cheeks) else -np.inf
        if snr_c > snr_f:
            used, snr_sel = "cheeks", snr_cheeks
            reason = (
                f"oba ważne (skin_f={skin_forehead:.2f}, skin_c={skin_cheeks:.2f}); "
                f"wybrano policzki (peak-SNR {snr_c:.1f}>{snr_f:.1f} dB)"
            )
        else:
            used, snr_sel = "forehead", snr_forehead
            reason = (
                f"oba ważne (skin_f={skin_forehead:.2f}, skin_c={skin_cheeks:.2f}); "
                f"wybrano czoło (peak-SNR {snr_f:.1f}>={snr_c:.1f} dB)"
            )
    elif viable_f:
        used, snr_sel = "forehead", snr_forehead
        reason = (
            f"policzki odrzucone (skin={skin_cheeks:.2f}<{tau_skin}); użyto czoła"
        )
    elif viable_c:
        used, snr_sel = "cheeks", snr_cheeks
        reason = (
            f"czoło odrzucone (skin={skin_forehead:.2f}<{tau_skin}); użyto policzków"
        )
    else:
        used, snr_sel = "none", float("nan")
        reason = (
            f"żaden region ważny (skin_f={skin_forehead:.2f}, skin_c={skin_cheeks:.2f}, "
            f"valid_frac={valid_frac:.2f})"
        )

    if used == "none" or not np.isfinite(snr_sel):
        confidence = "low"
    elif snr_sel >= snr_high_db:
        confidence = "high"
    elif snr_sel >= snr_med_db:
        confidence = "med"
    else:
        confidence = "low"

    return {
        "region_used": used,
        "confidence": confidence,
        "viable_forehead": viable_f,
        "viable_cheeks": viable_c,
        "snr_selected_db": float(snr_sel) if np.isfinite(snr_sel) else float("nan"),
        "reason": reason,
    }
