"""Skleja potok: io_layer -> roi -> extract -> methods -> estimate -> validate.

Nie liczy niczego samodzielnie — wyłącznie wywołuje funkcje z `src/` w ustalonej
kolejności. Logika modułów nie jest jeszcze zaimplementowana (patrz src/).
"""

from src import config, estimate, extract, io_layer, methods, roi, validate


def run(subject_dir):
    rgb_frames, thermal_frames, ppg_reference, ppg_fs = io_layer.load_ibvp_subject(subject_dir)

    roi_positions, valid = roi.track_roi_across_frames(rgb_frames)

    rgb_trace = extract.extract_rgb_trace_thermal_gated(
        rgb_frames, thermal_frames, roi_positions, valid
    )

    rppg_signal = methods.pos(rgb_trace, config.FS)

    detrended_signal = estimate.detrend_signal(rppg_signal)
    filtered_signal = estimate.bandpass_filter(detrended_signal, config.FS)
    hr_bpm = estimate.estimate_hr_welch(filtered_signal, config.FS)

    return hr_bpm


if __name__ == "__main__":
    run(config.DATA_DIR)
