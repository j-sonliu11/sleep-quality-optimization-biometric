from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, welch, filtfilt

def butter_bandpass_sos(low, high, fs, order=4):
    low_n = low / (fs / 2.0)
    high_n = high / (fs / 2.0)
    low_n = max(1e-6, min(0.999, low_n))
    high_n = max(low_n + 1e-6, min(0.999, high_n))
    return butter(order, [low_n, high_n], btype="bandpass", output="sos")

def apply_filters(X, fs, bp_lo, bp_hi, do_notch, notch_f0):
    if X.size == 0:
        return X
    X = np.nan_to_num(np.asarray(X, dtype=float))
    sos = butter_bandpass_sos(bp_lo, bp_hi, fs, order=4)
    Y = sosfiltfilt(sos, X, axis=0)
    if do_notch:
        b_notch, a_notch = iirnotch(w0=notch_f0 / (fs / 2.0), Q=30.0)
        Y = filtfilt(b_notch, a_notch, Y, axis=0)
    return np.nan_to_num(Y)

def bandpower_welch(x, fs, fmin, fmax, nperseg):
    x = np.nan_to_num(np.asarray(x, dtype=float))
    if x.size < 64:
        return 0.0
    nperseg = int(max(64, min(nperseg, x.size)))
    f, pxx = welch(x, fs=fs, nperseg=nperseg)
    idx = (f >= fmin) & (f <= fmax)
    if not np.any(idx):
        return 0.0
    bp = float(np.trapezoid(pxx[idx], f[idx]))
    return bp if np.isfinite(bp) else 0.0

def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-9
    z = (x - med) / mad
    z = np.clip(z, -8.0, 8.0)
    return z
