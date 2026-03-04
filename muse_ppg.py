from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from muse_config import HR_BAND_HZ, HR_MIN_BPM, HR_MAX_BPM, HRV_MIN_BEATS
from muse_filters import butter_bandpass_sos
from scipy.signal import sosfiltfilt

def _detect_beats(ts: np.ndarray, x: np.ndarray, fs: float):
    ts = np.asarray(ts, dtype=float)
    x = np.asarray(x, dtype=float)
    if ts.size < int(fs * 3.0):
        return None, np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=float)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    sos = butter_bandpass_sos(HR_BAND_HZ[0], HR_BAND_HZ[1], fs, order=3)
    xf = sosfiltfilt(sos, x)

    xf = xf - np.median(xf)
    mad = np.median(np.abs(xf)) + 1e-9
    xf = xf / mad
    xf = np.clip(xf, -10.0, 10.0)

    min_dist = int(fs * (60.0 / HR_MAX_BPM))
    min_dist = max(1, min_dist)

    peaks, _props = find_peaks(xf, distance=min_dist, prominence=0.8)
    if peaks.size < 2:
        return xf, peaks, np.array([], dtype=float), np.array([], dtype=float)

    peak_times = ts[peaks]
    ibi = np.diff(peak_times)

    ibi_min = 60.0 / HR_MAX_BPM
    ibi_max = 60.0 / HR_MIN_BPM
    ok = (ibi >= ibi_min) & (ibi <= ibi_max)

    peak_times = peak_times[np.concatenate(([True], ok))]
    ibi = np.diff(peak_times)

    return xf, peaks, peak_times, ibi

def estimate_bpm_from_ibi(ibi_s: np.ndarray) -> float | None:
    if ibi_s.size < 2:
        return None
    bpm = 60.0 / float(np.median(ibi_s))
    if not np.isfinite(bpm):
        return None
    return float(np.clip(bpm, HR_MIN_BPM, HR_MAX_BPM))

def rmssd_ms_from_ibi(ibi_s: np.ndarray) -> float | None:
    if ibi_s.size < (HRV_MIN_BEATS - 1):
        return None
    d = np.diff(ibi_s)
    if d.size < 2:
        return None
    rmssd = np.sqrt(np.mean(d * d)) * 1000.0
    if not np.isfinite(rmssd):
        return None
    return float(rmssd)

def ppg_quality_score(ts: np.ndarray, raw: np.ndarray, fs: float):
    ts = np.asarray(ts, dtype=float)
    raw = np.asarray(raw, dtype=float)
    if ts.size < int(fs * 3.0):
        return 0.0, "Bad", {"reason": "too_short"}

    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    dropout = float(np.mean(raw == 0.0))

    xf, _peaks, peak_times, ibi = _detect_beats(ts, raw, fs)
    if xf is None:
        return 0.0, "Bad", {"reason": "no_xf"}

    n_peaks = int(peak_times.size)
    cv = float(np.std(ibi) / (np.mean(ibi) + 1e-9)) if ibi.size >= 2 else 9.9
    band_power = float(np.std(xf))

    s_dropout = float(np.clip(1.0 - dropout / 0.25, 0.0, 1.0))
    s_peaks = float(np.clip((n_peaks - 3) / 6.0, 0.0, 1.0))
    s_cv = float(np.clip(1.0 - (cv - 0.08) / (0.25 - 0.08), 0.0, 1.0))
    s_pow = float(np.clip((band_power - 0.6) / (1.8 - 0.6), 0.0, 1.0))

    score = 0.35 * s_dropout + 0.25 * s_peaks + 0.25 * s_cv + 0.15 * s_pow
    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.70:
        label = "Good"
    elif score >= 0.45:
        label = "OK"
    else:
        label = "Bad"

    return score, label, {
        "dropout": dropout,
        "n_peaks": n_peaks,
        "ibi_cv": cv,
        "band_std": band_power,
    }