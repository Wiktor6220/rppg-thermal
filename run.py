"""Szkic potoku: io_layer → roi → extract → methods → estimate → validate."""

from src import config, estimate, extract, io_layer, methods, roi


def run(subject_dir):
    rgb_frames, thermal_frames, ppg_reference, ppg_fs = io_layer.load_ibvp_subject(subject_dir)

    roi_positions, valid = roi.track_roi_across_frames(rgb_frames)
    rgb_trace = extract.extract_rgb_trace(rgb_frames, roi_positions, valid)

    rppg_signal = methods.pos(rgb_trace, config.FS)
    cleaned = estimate.bandpass_filter(estimate.detrend_signal(rppg_signal), config.FS)
    return estimate.estimate_hr_welch(cleaned, config.FS)


if __name__ == "__main__":
    run(config.DATA_DIR)
