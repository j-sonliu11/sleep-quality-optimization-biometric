# muse.py
# Run:
#   python -m streamlit run muse.py
#
# One-time install:
#   python -m pip install streamlit pylsl numpy scipy plotly muselsl

from __future__ import annotations

import os
import re
import sys
import json
import time
import socket
import threading
import subprocess
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, welch, filtfilt, find_peaks
import streamlit as st
import streamlit.components.v1 as components
from pylsl import StreamInlet, resolve_byprop

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import traceback
import streamlit as st

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.units import inch

import io
import textwrap

import traceback  # <-- ensure this import exists near top of file

from reportlab.pdfbase.pdfmetrics import stringWidth
from datetime import timedelta

# Show full exceptions in the app (not just terminal)
try:
    st.set_option("client.showErrorDetails", True)
except Exception:
    pass

def _show_fatal(e: Exception):
    st.error("❌ App error (see details below)")
    st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
# -----------------------------
# Config
# -----------------------------
EEG_FS = 256.0
EEG_CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]
MAX_BUF_SECONDS = 120.0

# Muse PPG via muselsl: typically 3 channels (IR, RED, AMB)
PPG_FS = 64.0
PPG_CHANNEL_NAMES = ["PPG_IR", "PPG_RED", "PPG_AMB"]

# HR estimation defaults
HR_BAND_HZ = (0.7, 3.0)     # ~42–180 bpm
HR_MIN_BPM = 40.0
HR_MAX_BPM = 200.0
HR_INST_WINDOW_SEC = 10.0
HR_SMOOTH_WINDOW_SEC = 30.0
HRV_MIN_BEATS = 8          # require enough beats for RMSSD

# “one-click” connect behavior: wait/retry
CONNECT_RETRY_TOTAL_S = 5.0   # <-- increased (Windows LSL publish can be slow)
CONNECT_RETRY_STEP_S = 0.35

# Collector: allow a short wait for first samples
COLLECTOR_WAIT_FOR_SAMPLES_S = 1.0

BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta":  (13, 30),
    "Gamma": (30, 45),
}
BAND_COLORS = {
    "Delta": "#4C78A8",
    "Theta": "#72B7B2",
    "Alpha": "#54A24B",
    "Beta":  "#F58518",
    "Gamma": "#E45756",
}

DARK_BG = "#0E1117"
DARK_PLOT = "#111827"
DARK_GRID = "rgba(255,255,255,0.10)"
DARK_TEXT = "rgba(255,255,255,0.92)"

EPOCH_SEC = 30.0

# -----------------------------
# Muse -> DreamT feature adaptation (domain bridging)
# -----------------------------
MUSE_TO_DREAMT_ADAPT = True

# Optional amplitude matching toward DreamT-like EEG scale (per Muse channel, µV std targets)
EEG_MATCH_DREAMT_STD = True
MUSE_TARGET_STD_UV = {
    "TP9": 38.0,   # temporal/occipital-ish proxy
    "AF7": 45.0,   # frontal proxy
    "AF8": 45.0,   # frontal proxy
    "TP10": 38.0,  # temporal/occipital-ish proxy
}
EEG_STD_GAIN_MIN = 0.35
EEG_STD_GAIN_MAX = 4.00

# Feature-space channel aliases to bridge Muse channels -> DreamT-style feature names
# (crude proxies, but much better than zero-filled alignment)
ENABLE_MUSE_DREAMT_FEATURE_ALIAS = True

# If amplitudes look too large (common with dry electrodes / motion), compress/clamp
EEG_CLIP_UV = 120.0            # hard clip after preprocessing
EEG_SOFTSCALE_UV = 35.0        # tanh compression scale (µV), keeps shape but reduces outliers
EEG_USE_CAR = True             # common-average reference to reduce reference mismatch
EEG_USE_ROBUST_STD = False     # keep False by default to preserve absolute amplitude info
EEG_REMOVE_DC_PER_EPOCH = True # center each channel before features

# -----------------------------
# Page + CSS
# -----------------------------
st.set_page_config(page_title="Muse 2 Live", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] > div { padding-top: 0.35rem; padding-bottom: 0.8rem; }
div.stButton > button { border-radius: 14px; padding: 0.75rem 1rem; }
.element-container { margin-bottom: 0.4rem; }

/* Small spinner badge */
.run-badge { display:inline-flex; align-items:center; gap:10px; }
.spinner {
  width:14px; height:14px;
  border:2px solid rgba(255,255,255,0.25);
  border-top-color: rgba(255,255,255,0.85);
  border-radius:50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.small-muted { color: rgba(255,255,255,0.7); font-size: 0.95rem; }

/* Hide Streamlit's built-in status widget */
div[data-testid="stStatusWidget"] { display: none !important; }

/* Keep columns top-aligned */
div[data-testid="stHorizontalBlock"] {
  align-items: flex-start !important;
}

/* Let sticky work (ancestors must not clip) */
div[data-testid="stHorizontalBlock"],
div[data-testid="column"],
div[data-testid="stVerticalBlock"],
div[data-testid="stMainBlockContainer"],
div[data-testid="stBlock"],
div[data-testid="stElementContainer"],
div.block-container {
  overflow: visible !important;
}

/* The anchor itself should not show */
#dc-row-anchor {
  display: none;
}

/*
  Target the FIRST stHorizontalBlock immediately after the anchor's element container.
  That is your charts + data collection row.
*/
div[data-testid="stElementContainer"]:has(#dc-row-anchor)
  + div[data-testid="stHorizontalBlock"] {
  align-items: flex-start !important;
}

/* Make the RIGHT column in that specific row sticky */
div[data-testid="stElementContainer"]:has(#dc-row-anchor)
  + div[data-testid="stHorizontalBlock"]
  > div[data-testid="column"]:nth-of-type(2) {
  position: sticky !important;
  top: 0.85rem !important;
  align-self: flex-start !important;
  z-index: 40 !important;
}

/* Style the panel content inside the sticky right column */
div[data-testid="stElementContainer"]:has(#dc-row-anchor)
  + div[data-testid="stHorizontalBlock"]
  > div[data-testid="column"]:nth-of-type(2)
  > div[data-testid="stVerticalBlock"] {
  background: rgba(17, 24, 39, 0.72);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 0.85rem 0.9rem 0.9rem 0.9rem;
  box-shadow: 0 8px 20px rgba(0,0,0,0.22);

  max-height: calc(100vh - 1.7rem);
  overflow: auto !important;
}

/* Mobile: disable sticky */
@media (max-width: 899px) {
  div[data-testid="stElementContainer"]:has(#dc-row-anchor)
    + div[data-testid="stHorizontalBlock"]
    > div[data-testid="column"]:nth-of-type(2) {
    position: static !important;
    top: auto !important;
    z-index: auto !important;
  }

  div[data-testid="stElementContainer"]:has(#dc-row-anchor)
    + div[data-testid="stHorizontalBlock"]
    > div[data-testid="column"]:nth-of-type(2)
    > div[data-testid="stVerticalBlock"] {
    max-height: none !important;
    overflow: visible !important;
  }
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧠 Muse 2 Live Dashboard")

# -----------------------------
# Thread-safe CONFIG for server thread
# -----------------------------
@dataclass
class ConfigStore:
    lock: threading.Lock
    cfg: dict

    def set(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                self.cfg[k] = float(v) if isinstance(v, (int, float)) else v

    def get(self):
        with self.lock:
            return dict(self.cfg)

def get_config_store() -> ConfigStore:
    if "config_store" not in st.session_state:
        st.session_state["config_store"] = ConfigStore(
            lock=threading.Lock(),
            cfg={
                "rolling_sec": 10.0,
                "bp_low": 1.0,
                "bp_high": 45.0,
                "use_notch": 1.0,
                "notch_freq": 60.0,
                "psd_len": 4.0,
                "update_ms": 200.0,
                "offset_uv": 320.0,
            },
        )
    return st.session_state["config_store"]

cfg_store = get_config_store()

# -----------------------------
# Sidebar: clean sliders (no yellow warnings)
# -----------------------------
def synced_slider_number(
    label: str,
    min_v: float,
    max_v: float,
    default: float,
    step: float,
    key: str,
    fmt: str = "%.2f",
):
    s_key = f"{key}__s"
    n_key = f"{key}__n"

    if key not in st.session_state:
        st.session_state[key] = float(default)
    if s_key not in st.session_state:
        st.session_state[s_key] = float(st.session_state[key])
    if n_key not in st.session_state:
        st.session_state[n_key] = float(st.session_state[key])

    def _from_slider():
        v = float(st.session_state[s_key])
        v = max(min_v, min(max_v, v))
        st.session_state[key] = v
        st.session_state[n_key] = v

    def _from_number():
        v = float(st.session_state[n_key])
        v = max(min_v, min(max_v, v))
        st.session_state[key] = v
        st.session_state[s_key] = v

    c1, c2 = st.columns([2, 1], gap="small")
    with c1:
        st.slider(
            label,
            min_value=float(min_v),
            max_value=float(max_v),
            step=float(step),
            key=s_key,
            on_change=_from_slider,
        )
    with c2:
        st.number_input(
            " ",
            min_value=float(min_v),
            max_value=float(max_v),
            step=float(step),
            format=fmt,
            key=n_key,
            on_change=_from_number,
        )
    return float(st.session_state[key])

with st.sidebar:
    st.header("Settings")

    rolling_sec = synced_slider_number("Rolling window (s)", 2.0, 30.0, 10.0, 1.0, "rolling_sec", fmt="%.2f")
    bp_low = synced_slider_number("Band-pass low (Hz)", 0.1, 30.0, 1.0, 0.1, "bp_low", fmt="%.2f")
    bp_high = synced_slider_number("Band-pass high (Hz)", 10.0, 60.0, 45.0, 0.5, "bp_high", fmt="%.2f")

    use_notch = st.checkbox("Notch filter", value=True)

    notch_label = st.selectbox(
        "Line freq (Hz)",
        ["60.0 Hz (NA)", "50.0 Hz (EU, AS, AF)"],
        index=0,
    )
    notch_freq = 60.0 if notch_label.startswith("60") else 50.0

    psd_len = synced_slider_number("PSD window (s)", 1.0, 8.0, 4.0, 1.0, "psd_len", fmt="%.2f")
    update_ms = synced_slider_number("Graph update (ms)", 50.0, 1000.0, 200.0, 50.0, "update_ms", fmt="%.0f")

    offset_uv = synced_slider_number("Trace vertical offset (µV)", 50.0, 800.0, 320.0, 10.0, "offset_uv", fmt="%.0f")

    show_debug = st.checkbox("Show debug panel", value=False)

cfg_store.set(
    rolling_sec=rolling_sec,
    bp_low=bp_low,
    bp_high=bp_high,
    use_notch=1.0 if use_notch else 0.0,
    notch_freq=float(notch_freq),
    psd_len=psd_len,
    update_ms=update_ms,
    offset_uv=offset_uv,
)

# -----------------------------
# Filtering + helpers
# -----------------------------
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

def ensure_eeg_uv(X: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Try to infer whether EEG is in volts or microvolts and convert to µV.
    Returns (X_uv, detected_units).
    Heuristic:
      - If typical magnitudes are ~1e-6 to 1e-3, assume volts.
      - Otherwise assume already µV.
    """
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.size == 0:
        return X, "unknown"

    p95 = float(np.percentile(np.abs(X), 95))
    if 1e-8 < p95 < 1e-2:
        return X * 1e6, "V_to_uV"
    return X, "uV_assumed"

def common_average_reference(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        return X
    avg = np.mean(X, axis=1, keepdims=True)
    return X - avg

def winsorize_clip(X: np.ndarray, lo: float = -EEG_CLIP_UV, hi: float = EEG_CLIP_UV) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.clip(X, lo, hi)

def soft_compress_tanh(X: np.ndarray, scale_uv: float = EEG_SOFTSCALE_UV) -> np.ndarray:
    """
    Smoothly compress large amplitudes while preserving small-signal structure.
    """
    X = np.asarray(X, dtype=float)
    s = max(1e-6, float(scale_uv))
    return s * np.tanh(X / s)

def robust_channel_standardize(X: np.ndarray) -> np.ndarray:
    """
    Per-channel robust standardization (optional; can hurt if model uses absolute amplitude).
    """
    X = np.asarray(X, dtype=float)
    Y = np.zeros_like(X, dtype=float)
    for i in range(X.shape[1]):
        xi = np.nan_to_num(X[:, i], nan=0.0, posinf=0.0, neginf=0.0)
        med = np.median(xi)
        mad = np.median(np.abs(xi - med)) + 1e-9
        zi = (xi - med) / (1.4826 * mad)
        Y[:, i] = np.clip(zi, -8.0, 8.0)
    return Y

def match_channel_std_to_targets(
    X: np.ndarray,
    ch_names: list[str],
    target_std_uv: dict[str, float],
    gain_min: float = EEG_STD_GAIN_MIN,
    gain_max: float = EEG_STD_GAIN_MAX,
) -> tuple[np.ndarray, dict]:
    """
    Per-channel linear gain so epoch std roughly matches DreamT-like target amplitudes.
    Returns scaled X and diagnostics.
    """
    X = np.asarray(X, dtype=float)
    Y = np.array(X, dtype=float, copy=True)
    gains = {}
    pre_std = {}
    post_std = {}

    if Y.ndim != 2 or Y.shape[1] != len(ch_names):
        return Y, {"gains": gains, "pre_std": pre_std, "post_std": post_std}

    for i, ch in enumerate(ch_names):
        xi = np.nan_to_num(Y[:, i], nan=0.0, posinf=0.0, neginf=0.0)
        s = float(np.std(xi))
        pre_std[ch] = s
        tgt = float(target_std_uv.get(ch, s if s > 0 else 1.0))
        if s <= 1e-9:
            g = 1.0
        else:
            g = tgt / s
        g = float(np.clip(g, gain_min, gain_max))
        Y[:, i] = xi * g
        gains[ch] = g
        post_std[ch] = float(np.std(Y[:, i]))

    return Y, {"gains": gains, "pre_std": pre_std, "post_std": post_std}


def _safe_mean(vals: list[float]) -> float:
    vv = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not vv:
        return 0.0
    return float(np.mean(vv))


def build_muse_to_dreamt_feature_aliases(feat: dict) -> dict:
    """
    Create DreamT-style feature aliases from Muse feature names so alignment can populate
    expected model columns instead of zero-filling everything.

    Handles names like:
      TP9, TP9__std, TP9__bp_Alpha, TP9__frac_Delta, ...
    Produces aliases like:
      C4-M1__std, F4-M1__bp_Alpha, etc. (proxy mappings)
    """
    # Base Muse channels available in this app
    muse_chs = ["TP9", "AF7", "AF8", "TP10"]

    # DreamT-ish target channels often seen in your training data
    target_chs = ["C4-M1", "F4-M1", "O2-M1", "Fp1-O2", "CZ-T4", "T3-CZ"]

    # Parse Muse channel feature family suffixes
    # e.g. "TP9" => suffix="", "TP9__std" => suffix="__std"
    suffix_map = {ch: {} for ch in muse_chs}
    for k, v in feat.items():
        for ch in muse_chs:
            if k == ch:
                suffix_map[ch][""] = v
            elif k.startswith(ch + "__"):
                suffix = k[len(ch):]  # includes leading "__"
                suffix_map[ch][suffix] = v

    def get(ch: str, suffix: str):
        return suffix_map.get(ch, {}).get(suffix, None)

    # Build all suffixes observed in Muse features
    all_suffixes = set()
    for ch in muse_chs:
        all_suffixes.update(suffix_map[ch].keys())

    aliases = {}

    # Proxy mapping strategy (feature-space, scalar-level)
    # You can refine these later, but this immediately prevents all-zero vectors.
    for suffix in all_suffixes:
        tp9 = get("TP9", suffix)
        af7 = get("AF7", suffix)
        af8 = get("AF8", suffix)
        tp10 = get("TP10", suffix)

        # Frontal proxy (combine AF7/AF8)
        frontal = _safe_mean([af7, af8])

        # Posterior/temporal proxy (combine TP9/TP10)
        posterior = _safe_mean([tp9, tp10])

        # Right-ish temporal/parietal proxy
        right_temp = _safe_mean([tp10, af8])

        # Left-ish temporal/frontal proxy
        left_temp = _safe_mean([tp9, af7])

        # Single-source fallbacks
        if suffix not in aliases:
            pass

        # DreamT-style aliases
        aliases[f"F4-M1{suffix}"] = frontal
        aliases[f"C4-M1{suffix}"] = right_temp
        aliases[f"O2-M1{suffix}"] = posterior
        aliases[f"Fp1-O2{suffix}"] = _safe_mean([af7, posterior])  # rough mixed frontal/posterior proxy
        aliases[f"CZ-T4{suffix}"] = tp10 if (tp10 is not None and np.isfinite(tp10)) else right_temp
        aliases[f"T3-CZ{suffix}"] = tp9 if (tp9 is not None and np.isfinite(tp9)) else left_temp

    # Also include some exact passthrough aliases for generic/global features if helpful
    # (No-op for most models but harmless)
    if "EEG_GLOBAL_STD_MEAN" in feat:
        aliases["GLOBAL_STD_MEAN"] = feat["EEG_GLOBAL_STD_MEAN"]
    if "EEG_GLOBAL_RMS_MEAN" in feat:
        aliases["GLOBAL_RMS_MEAN"] = feat["EEG_GLOBAL_RMS_MEAN"]
    if "EEG_GLOBAL_PTP_MEAN" in feat:
        aliases["GLOBAL_PTP_MEAN"] = feat["EEG_GLOBAL_PTP_MEAN"]

    return aliases

def adapt_muse_eeg_toward_dreamt(X_epoch_raw: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """
    Domain-bridging preprocessing for Muse dry-electrode EEG to behave more like DreamT features.
    Returns:
      X_adapted (µV-domain, filtered, artifact-suppressed),
      diag dict
    """
    diag = {
        "units_detected": None,
        "pre_p95_abs": None,
        "post_p95_abs": None,
        "used_car": bool(EEG_USE_CAR),
        "used_robust_std": bool(EEG_USE_ROBUST_STD),
        "clip_uv": float(EEG_CLIP_UV),
        "softscale_uv": float(EEG_SOFTSCALE_UV),
        "used_std_match": bool(EEG_MATCH_DREAMT_STD),
        "std_match_gains": {},
        "std_pre": {},
        "std_post": {},
    }

    X = np.asarray(X_epoch_raw, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.size == 0:
        return X, diag

    diag["pre_p95_abs"] = float(np.percentile(np.abs(X), 95))

    # 1) Force consistent units (µV)
    X, unit_mode = ensure_eeg_uv(X)
    diag["units_detected"] = unit_mode

    # 2) Filter (same app pipeline, but now operating in µV)
    do_notch = bool(int(cfg["use_notch"]))
    X = apply_filters(X, EEG_FS, float(cfg["bp_low"]), float(cfg["bp_high"]), do_notch, float(cfg["notch_freq"]))

    # 3) Remove per-epoch DC offsets (per channel)
    if EEG_REMOVE_DC_PER_EPOCH and X.ndim == 2:
        X = X - np.mean(X, axis=0, keepdims=True)

    # 4) Re-reference (CAR helps reduce montage/reference mismatch)
    if EEG_USE_CAR and X.ndim == 2 and X.shape[1] >= 2:
        X = common_average_reference(X)

    # 5) Artifact suppression (outliers dominate Muse dry sensors)
    X = winsorize_clip(X, -EEG_CLIP_UV, EEG_CLIP_UV)
    X = soft_compress_tanh(X, EEG_SOFTSCALE_UV)

    # 5.5) Optional amplitude matching to DreamT-like channel std (linear gain)
    if EEG_MATCH_DREAMT_STD and X.ndim == 2 and X.shape[1] == len(EEG_CHANNEL_NAMES):
        X, std_diag = match_channel_std_to_targets(
            X,
            ch_names=EEG_CHANNEL_NAMES,
            target_std_uv=MUSE_TARGET_STD_UV,
            gain_min=EEG_STD_GAIN_MIN,
            gain_max=EEG_STD_GAIN_MAX,
        )
        diag["std_match_gains"] = std_diag.get("gains", {})
        diag["std_pre"] = std_diag.get("pre_std", {})
        diag["std_post"] = std_diag.get("post_std", {})

    # 6) Optional robust standardization (OFF by default)
    if EEG_USE_ROBUST_STD:
        X = robust_channel_standardize(X)

    diag["post_p95_abs"] = float(np.percentile(np.abs(X), 95))
    return X, diag



# -----------------------------
# PPG: HR / HRV / Quality
# -----------------------------
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

# -----------------------------
# Generic LSL Receiver (EEG + PPG)
# -----------------------------
class LSLReceiver:
    def __init__(self, stream_type: str, fs: float, n_chan: int, max_seconds: float = MAX_BUF_SECONDS):
        self.stream_type = str(stream_type)
        self.fs = float(fs)
        self.n_chan = int(n_chan)
        self.maxlen = int(max_seconds * fs)

        self.ts = deque(maxlen=self.maxlen)
        self.buf = deque(maxlen=self.maxlen)
        self.lock = threading.RLock()

        self.running = False
        self.paused = False
        self.thread = None
        self.inlet = None
        self.last_error = None
        self.stream_meta = None

        self.sample_count = 0
        self.last_ts = None

        # NEW: prevent concurrent reconnect attempts
        self._reconnecting = False

    def start(self, timeout_s: float = 2.0, prefer_name_contains: str = "muse") -> bool:
        self.paused = False
        deadline = time.time() + max(0.0, float(timeout_s))

        # Muse PPG can show as AUX on some setups
        type_candidates = [self.stream_type]
        if self.stream_type.upper() == "PPG":
            type_candidates = ["PPG", "AUX"]

        while True:
            try:
                streams = []
                for tval in type_candidates:
                    streams = resolve_byprop("type", tval, timeout=0.35)
                    if streams:
                        break

                if streams:
                    best, best_score = None, -1
                    for s in streams:
                        try:
                            ch = s.channel_count()
                            fs = s.nominal_srate()
                            name = s.name()
                            typ = s.type()
                        except Exception:
                            continue

                        score = 0
                        n = str(name).lower()
                        if prefer_name_contains and prefer_name_contains in n:
                            score += 4
                        if ch == self.n_chan:
                            score += 3
                        if fs > 0 and abs(fs - self.fs) < max(6.0, 0.25 * self.fs):
                            score += 2
                        if str(typ).lower() in [x.lower() for x in type_candidates]:
                            score += 1

                        if score > best_score:
                            best_score, best = score, s

                    if best is not None:
                        new_inlet = StreamInlet(best, max_buflen=60)

                        try:
                            source_id = best.source_id()
                        except Exception:
                            source_id = None

                        with self.lock:
                            self.inlet = new_inlet
                            self.stream_meta = {
                                "name": best.name(),
                                "type": best.type(),
                                "source_id": source_id,
                                "channels": best.channel_count(),
                                "fs": best.nominal_srate(),
                            }
                            self.running = True
                            self.paused = False
                            self.last_error = None

                            # NEW: reset state on fresh start
                            self.ts.clear()
                            self.buf.clear()
                            self.sample_count = 0
                            self.last_ts = None

                        self.thread = threading.Thread(target=self._run, daemon=True)
                        self.thread.start()
                        return True

            except Exception as e:
                self.last_error = repr(e)

            if time.time() >= deadline:
                return False

    def _run(self):
        last_ok = time.time()
        backoff = 0.6  # NEW: start smaller; we’ll ramp up

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            try:
                with self.lock:
                    inlet = self.inlet

                if inlet is None:
                    self.last_error = "Inlet is None -> reconnect"
                    self._reconnect_inlet(backoff=backoff)
                    backoff = min(8.0, backoff * 1.5)
                    last_ok = time.time()
                    continue

                sample, ts = inlet.pull_sample(timeout=0.5)

            except Exception as e:
                self.last_error = f"pull_sample error: {e!r} -> reconnect"
                self._reconnect_inlet(backoff=backoff)
                backoff = min(8.0, backoff * 1.5)
                last_ok = time.time()
                continue

            if sample is None or ts is None:
                if time.time() - last_ok > 10.0:
                    self.last_error = "No samples for 10s -> reconnect"
                    self._reconnect_inlet(backoff=backoff)
                    backoff = min(8.0, backoff * 1.5)
                    last_ok = time.time()
                continue

            # Got data
            backoff = 0.6
            last_ok = time.time()

            try:
                s = np.asarray(sample, dtype=float).reshape(-1)

                if s.size < self.n_chan:
                    s = np.pad(s, (0, self.n_chan - s.size), constant_values=np.nan)
                elif s.size > self.n_chan:
                    s = s[: self.n_chan]

                with self.lock:
                    self.ts.append(float(ts))
                    self.buf.append(s.tolist())
                    self.sample_count += 1
                    self.last_ts = float(ts)
            except Exception as e:
                self.last_error = f"Buffer append error: {e!r}"
                continue

    def _reconnect_inlet(self, backoff: float = 1.0):
        # NEW: prevent overlapping reconnect attempts
        if self._reconnecting:
            return
        self._reconnecting = True

        try:
            time.sleep(float(backoff))

            target_type = getattr(self, "stream_type", None)
            target_name = None
            target_source_id = None

            meta = getattr(self, "stream_meta", None)
            if meta:
                target_name = meta.get("name")
                target_source_id = meta.get("source_id")

            streams = []

            # 1) Prefer source_id
            if target_source_id:
                try:
                    streams.extend(resolve_byprop("source_id", target_source_id, timeout=1.0))
                except Exception:
                    pass

            # 2) Then name
            if not streams and target_name:
                try:
                    streams.extend(resolve_byprop("name", target_name, timeout=1.0))
                except Exception:
                    pass

            # 3) Fallback to type — IMPORTANT: mirror your start() behavior for PPG/AUX
            if not streams and target_type:
                type_candidates = [str(target_type)]
                if str(target_type).upper() == "PPG":
                    type_candidates = ["PPG", "AUX"]

                for tval in type_candidates:
                    try:
                        streams = resolve_byprop("type", tval, timeout=1.0)
                        if streams:
                            break
                    except Exception:
                        continue

            if not streams:
                self.last_error = (
                    f"Reconnect failed: no LSL streams found "
                    f"(source_id={target_source_id!r}, name={target_name!r}, type={target_type!r})"
                )
                return

            # Score best stream
            best, best_score = None, -1
            for s in streams:
                try:
                    ch = s.channel_count()
                    fs = s.nominal_srate()
                    name = s.name()
                    typ = s.type()
                    sid = getattr(s, "source_id", lambda: None)()
                except Exception:
                    continue

                score = 0
                if target_source_id and sid == target_source_id:
                    score += 10
                if target_name and str(name) == str(target_name):
                    score += 6
                if ch == self.n_chan:
                    score += 3
                if fs > 0 and abs(fs - self.fs) < max(6.0, 0.25 * self.fs):
                    score += 2
                if target_type and str(typ).lower() == str(target_type).lower():
                    score += 1

                if score > best_score:
                    best_score, best = score, s

            if best is None:
                self.last_error = "Reconnect failed: streams found but none scored as usable."
                return

            new_inlet = StreamInlet(best, max_buflen=60)

            # Quick verify (short timeouts so we don't “freeze” receiver thread)
            ok = False
            for _ in range(2):
                samp, tss = new_inlet.pull_sample(timeout=0.25)
                if samp is not None and tss is not None:
                    ok = True
                    break

            with self.lock:
                # NEW: swap inlet + reset buffers so downstream doesn't hang on old timestamps
                self.inlet = new_inlet
                self.ts.clear()
                self.buf.clear()
                self.sample_count = 0
                self.last_ts = None

                try:
                    self.stream_meta = {
                        "name": best.name(),
                        "type": best.type(),
                        "source_id": getattr(best, "source_id", lambda: None)(),
                        "channels": best.channel_count(),
                        "fs": best.nominal_srate(),
                    }
                except Exception:
                    pass

            self.last_error = "Reconnected." if ok else "Reconnected (waiting for samples...)"

        except Exception as e:
            self.last_error = f"Reconnect exception: {e!r}"
        finally:
            self._reconnecting = False

    def get_window(self, seconds: float):
        with self.lock:
            if not self.ts:
                return np.array([]), np.array([[]], dtype=float)
            ts = np.array(self.ts, dtype=float)
            X = np.array(self.buf, dtype=float)
        t_end = ts[-1]
        mask = ts >= (t_end - seconds)
        return ts[mask], X[mask]

    def get_since(self, last_ts: float | None):
        with self.lock:
            if not self.ts:
                return np.array([]), np.array([[]], dtype=float)
            ts = np.array(self.ts, dtype=float)
            X = np.array(self.buf, dtype=float)

        if last_ts is None or ts.size == 0:
            return np.array([]), np.array([[]], dtype=float)

        # timestamps can reset after reconnect
        if ts[-1] < float(last_ts):
            return ts, X

        mask = ts > float(last_ts)
        return ts[mask], X[mask]

    def pause(self):
        self.paused = True

    def resume(self):
        if self.running:
            self.paused = False

    def stop_and_clear(self):
        self.running = False
        self.paused = False
        try:
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2)
        except Exception:
            pass

        with self.lock:
            self.ts.clear()
            self.buf.clear()
            self.inlet = None
            self.stream_meta = None
            self.sample_count = 0
            self.last_ts = None

# -----------------------------
# muselsl helpers
# -----------------------------
def start_muselsl_stream():
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW
    # enable PPG stream
    return subprocess.Popen(
        [sys.executable, "-m", "muselsl", "stream", "--ppg"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

def stop_proc(p):
    if p is None:
        return
    try:
        p.terminate()
    except Exception:
        pass

def scan_for_muse(timeout_s: int = 10):
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "muselsl", "list"],
            capture_output=True,
            text=True,
            timeout=max(5, int(timeout_s) + 2),
        )
        raw = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        macs = re.findall(r"([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})", raw)
        macs = list(dict.fromkeys([m.upper() for m in macs]))
        return (len(macs) > 0), macs, raw
    except Exception as e:
        return False, [], repr(e)

def connect_with_retries(eeg: LSLReceiver, ppg: LSLReceiver) -> tuple[bool, bool]:
    """
    Fixes the “two-click” issue by retrying connection while muselsl spins up.
    """
    t0 = time.time()
    ok_eeg = False
    ok_ppg = False

    if eeg.running:
        ok_eeg = True
    if ppg.running:
        ok_ppg = True

    while time.time() - t0 < CONNECT_RETRY_TOTAL_S:
        if not ok_eeg:
            ok_eeg = eeg.start(timeout_s=CONNECT_RETRY_STEP_S)
        if not ok_ppg:
            ok_ppg = ppg.start(timeout_s=CONNECT_RETRY_STEP_S)

        if ok_eeg and ok_ppg:
            break

        time.sleep(0.15)

    return ok_eeg, ok_ppg

# -----------------------------
# Local JSON server (iframe polling)
# -----------------------------
def find_free_port(preferred=8765):
    def can_bind(p):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", p))
            s.close()
            return True
        except Exception:
            return False

    if can_bind(preferred):
        return preferred
    for p in range(8766, 8850):
        if can_bind(p):
            return p
    raise RuntimeError("No free port found")

def _align_nearest(ts_src: np.ndarray, X_src: np.ndarray, ts_target: np.ndarray) -> np.ndarray:
    if ts_src.size == 0 or X_src.size == 0 or ts_target.size == 0:
        return np.full((len(ts_target), X_src.shape[1] if X_src.ndim == 2 else 1), np.nan, dtype=float)

    ts_src = np.asarray(ts_src, dtype=float)
    X_src = np.asarray(X_src, dtype=float)
    ts_target = np.asarray(ts_target, dtype=float)

    idx = np.searchsorted(ts_src, ts_target, side="left")
    idx0 = np.clip(idx - 1, 0, len(ts_src) - 1)
    idx1 = np.clip(idx, 0, len(ts_src) - 1)

    d0 = np.abs(ts_target - ts_src[idx0])
    d1 = np.abs(ts_target - ts_src[idx1])
    pick = np.where(d1 < d0, idx1, idx0)

    return X_src[pick, :]

def build_payload(eeg: LSLReceiver, ppg: LSLReceiver, cfg_store: ConfigStore):
    cfg = cfg_store.get()
    rolling = float(cfg["rolling_sec"])

    # ---- Pull windows
    ts_e, X_e = eeg.get_window(rolling)
    ts_p_roll, X_p_roll = ppg.get_window(rolling)

    # ---- Receiver health (NEW)
    eeg_meta = {
        "running": bool(eeg.running),
        "paused": bool(eeg.paused),
        "sample_count": int(getattr(eeg, "sample_count", 0)),
        "last_ts": (None if getattr(eeg, "last_ts", None) is None else float(eeg.last_ts)),
        "last_error": (None if getattr(eeg, "last_error", None) is None else str(eeg.last_error)),
        "reconnecting": bool(getattr(eeg, "_reconnecting", False)),
    }
    ppg_meta = {
        "running": bool(ppg.running),
        "paused": bool(ppg.paused),
        "sample_count": int(getattr(ppg, "sample_count", 0)),
        "last_ts": (None if getattr(ppg, "last_ts", None) is None else float(ppg.last_ts)),
        "last_error": (None if getattr(ppg, "last_error", None) is None else str(ppg.last_error)),
        "reconnecting": bool(getattr(ppg, "_reconnecting", False)),
    }

    # ---- Decide if we have enough data to render
    eeg_ok = (X_e.size != 0) and (ts_e.size >= 5) and (X_e.shape[0] >= 5)
    ppg_ok = (X_p_roll.size != 0) and (ts_p_roll.size >= 5) and (X_p_roll.shape[0] >= 5)

    waiting_for_samples = not eeg_ok and not ppg_ok

    payload = {
        "ok": True,
        "waiting_for_samples": bool(waiting_for_samples),
        "update_ms": float(cfg.get("update_ms", 200.0)),

        # keep these for your JS
        "paused": bool(eeg.paused),
        "running": bool(eeg.running),
        "ppg_running": bool(ppg.running),

        # NEW: expose health so you can see reconnect progress in JS/Streamlit
        "eeg_status": eeg_meta,
        "ppg_status": ppg_meta,
    }

    # ---- EEG payload
    if eeg_ok:
        do_notch = bool(int(cfg["use_notch"]))
        Xf = apply_filters(
            X_e,
            EEG_FS,
            float(cfg["bp_low"]),
            float(cfg["bp_high"]),
            do_notch,
            float(cfg["notch_freq"]),
        )
        t = (ts_e - ts_e[-1]).astype(float)

        nchan = Xf.shape[1]
        off = float(cfg["offset_uv"])
        base = []
        y_list = []
        for i in range(nchan):
            b = (nchan - 1 - i) * off
            base.append(b)
            y_list.append((Xf[:, i] + b).tolist())

        nper_bp = int(max(256, float(cfg["psd_len"]) * EEG_FS))
        bp = np.zeros((len(EEG_CHANNEL_NAMES), len(BANDS)), dtype=float)
        for ch in range(len(EEG_CHANNEL_NAMES)):
            for bi, (_lbl, (f1, f2)) in enumerate(BANDS.items()):
                bp[ch, bi] = bandpower_welch(Xf[:, ch], EEG_FS, f1, f2, nper_bp)

        denom = np.sum(bp, axis=1, keepdims=True)
        denom[denom <= 0] = 1e-12
        bp_frac = (bp / denom).tolist()

        nper = int(max(256, float(cfg["psd_len"]) * EEG_FS))
        nper = min(nper, max(64, len(Xf)))
        f, pxx = welch(Xf, fs=EEG_FS, nperseg=nper, axis=0)
        pxx_mean = np.mean(np.nan_to_num(pxx), axis=1)
        psd_db = (10 * np.log10(pxx_mean + 1e-12)).tolist()

        payload.update(
            {
                "t": t.tolist(),
                "y": y_list,
                "base": base,
                "channels": EEG_CHANNEL_NAMES,
                "bands": list(BANDS.keys()),
                "band_edges": {k: list(v) for k, v in BANDS.items()},
                "band_colors": BAND_COLORS,
                "bp_frac": bp_frac,
                "f": f.tolist(),
                "psd_db": psd_db,
            }
        )
    else:
        payload.update(
            {
                "t": [],
                "y": [],
                "base": [],
                "channels": EEG_CHANNEL_NAMES,
                "bands": list(BANDS.keys()),
                "band_edges": {k: list(v) for k, v in BANDS.items()},
                "band_colors": BAND_COLORS,
                "bp_frac": [],
                "f": [],
                "psd_db": [],
            }
        )

    # ---- PPG payload
    if ppg_ok:
        tppg = (ts_p_roll - ts_p_roll[-1]).astype(float)
        ppg_y_raw = [X_p_roll[:, i].tolist() for i in range(min(X_p_roll.shape[1], len(PPG_CHANNEL_NAMES)))]

        ts_inst, X_inst = ppg.get_window(HR_INST_WINDOW_SEC)
        ts_smooth, X_smooth = ppg.get_window(HR_SMOOTH_WINDOW_SEC)

        hr_inst = None
        hr_smooth = None
        rmssd = None
        q_label = "Bad"
        q_score = 0.0
        q_details = {}

        band_t = []
        band_y = []
        peak_t = []
        ibi_t = []
        ibi_ms = []

        if X_smooth.size and X_smooth.shape[1] >= 2 and ts_smooth.size >= 10:
            s_ir, l_ir, d_ir = ppg_quality_score(ts_smooth, X_smooth[:, 0], PPG_FS)
            s_rd, l_rd, d_rd = ppg_quality_score(ts_smooth, X_smooth[:, 1], PPG_FS)

            use_idx = 0 if s_ir >= s_rd else 1
            q_score, q_label, q_details = (s_ir, l_ir, d_ir) if use_idx == 0 else (s_rd, l_rd, d_rd)

            if X_inst.size and X_inst.shape[1] > use_idx:
                _xf_i, _peaks_i, _pt_i, ibi_i = _detect_beats(ts_inst, X_inst[:, use_idx], PPG_FS)
                hr_inst = estimate_bpm_from_ibi(ibi_i)

            xf_s, _peaks_s, pt_s, ibi_s = _detect_beats(ts_smooth, X_smooth[:, use_idx], PPG_FS)
            hr_smooth = estimate_bpm_from_ibi(ibi_s)

            if xf_s is not None and ts_smooth.size:
                t_end = ts_smooth[-1]
                m = ts_smooth >= (t_end - rolling)
                ts_disp = ts_smooth[m]
                xf_disp = xf_s[m]

                band_t = (ts_disp - ts_disp[-1]).astype(float).tolist()
                band_y = robust_z(xf_disp).tolist()

                if pt_s.size:
                    pt_disp = pt_s[pt_s >= (t_end - rolling)]
                    peak_t = (pt_disp - t_end).astype(float).tolist()

                if ibi_s.size:
                    ibi_t_abs = pt_s[1:]
                    ibi_t = (ibi_t_abs - t_end).astype(float).tolist()
                    ibi_ms = (ibi_s * 1000.0).astype(float).tolist()

            if q_label == "Good":
                rmssd = rmssd_ms_from_ibi(ibi_s)

        payload.update(
            {
                "ppg_t": tppg.tolist(),
                "ppg_y": ppg_y_raw,
                "ppg_channels": PPG_CHANNEL_NAMES[: min(X_p_roll.shape[1], len(PPG_CHANNEL_NAMES))],
                "hr_bpm_inst": (None if hr_inst is None else float(hr_inst)),
                "hr_bpm_smooth": (None if hr_smooth is None else float(hr_smooth)),
                "ppg_quality": q_label,
                "ppg_quality_score": float(q_score),
                "ppg_quality_details": q_details,
                "ppg_band_t": band_t,
                "ppg_band_y": band_y,
                "ppg_peak_t": peak_t,
                "ibi_t": ibi_t,
                "ibi_ms": ibi_ms,
                "rmssd_ms": (None if rmssd is None else float(rmssd)),
            }
        )
    else:
        payload.update(
            {
                "ppg_t": [],
                "ppg_y": [],
                "ppg_channels": [],
                "hr_bpm_inst": None,
                "hr_bpm_smooth": None,
                "ppg_quality": "Bad",
                "ppg_quality_score": 0.0,
                "ppg_quality_details": {},
                "ppg_band_t": [],
                "ppg_band_y": [],
                "ppg_peak_t": [],
                "ibi_t": [],
                "ibi_ms": [],
                "rmssd_ms": None,
            }
        )

    return payload

class DataHandler(BaseHTTPRequestHandler):
    eeg_ref = None
    ppg_ref = None
    cfg_store_ref: ConfigStore | None = None

    def _send(self, code, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/data"):
            try:
                if self.eeg_ref is None or self.ppg_ref is None or self.cfg_store_ref is None:
                    payload = {"ok": False, "reason": "server_not_ready"}
                else:
                    payload = build_payload(self.eeg_ref, self.ppg_ref, self.cfg_store_ref)
                self._send(200, json.dumps(payload).encode("utf-8"))
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "err": repr(e)}).encode("utf-8"))
        else:
            self._send(404, json.dumps({"ok": False}).encode("utf-8"))

    def log_message(self, format, *args):
        return

def ensure_data_server(eeg: LSLReceiver, ppg: LSLReceiver, cfg_store: ConfigStore):
    if "data_server_port" in st.session_state and "data_server_obj" in st.session_state:
        handler_cls = st.session_state.get("data_server_handler_cls", None)
        if handler_cls is not None:
            handler_cls.eeg_ref = eeg
            handler_cls.ppg_ref = ppg
            handler_cls.cfg_store_ref = cfg_store
        return st.session_state["data_server_port"]

    port = find_free_port(8765)

    DataHandler.eeg_ref = eeg
    DataHandler.ppg_ref = ppg
    DataHandler.cfg_store_ref = cfg_store
    st.session_state["data_server_handler_cls"] = DataHandler

    httpd = HTTPServer(("127.0.0.1", port), DataHandler)

    def _serve():
        try:
            httpd.serve_forever()
        except Exception:
            pass

    t = threading.Thread(target=_serve, daemon=True)
    t.start()

    st.session_state["data_server_port"] = port
    st.session_state["data_server_thread"] = t
    st.session_state["data_server_obj"] = httpd
    return port

# -----------------------------
# Data Collection (30s epoch logger)
# -----------------------------
def ensure_sleep_data_dir() -> Path:
    # Root folder for all live prediction sessions
    out_dir = Path(__file__).resolve().parent / "live prediction runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def epoch_features(ts_epoch: np.ndarray, X_epoch: np.ndarray, cfg: dict):
    """
    Extract per-epoch EEG features.
    Includes Muse->DreamT domain adaptation so live Muse features are closer
    to the DreamT feature distribution used by the XGBoost cascade.
    """
    # --- Domain adaptation / preprocessing ---
    if MUSE_TO_DREAMT_ADAPT:
        Xf, adapt_diag = adapt_muse_eeg_toward_dreamt(X_epoch, cfg)
    else:
        do_notch = bool(int(cfg["use_notch"]))
        X_uv, unit_mode = ensure_eeg_uv(X_epoch)
        Xf = apply_filters(X_uv, EEG_FS, float(cfg["bp_low"]), float(cfg["bp_high"]), do_notch, float(cfg["notch_freq"]))
        adapt_diag = {
            "units_detected": unit_mode,
            "pre_p95_abs": float(np.percentile(np.abs(np.nan_to_num(X_uv)), 95)) if np.size(X_uv) else None,
            "post_p95_abs": float(np.percentile(np.abs(np.nan_to_num(Xf)), 95)) if np.size(Xf) else None,
            "used_car": False,
            "used_robust_std": False,
            "clip_uv": None,
            "softscale_uv": None,
            "used_std_match": False,
            "std_match_gains": {},
            "std_pre": {},
            "std_post": {},
        }

    row = {}

    # Basic channel moments
    for i, ch in enumerate(EEG_CHANNEL_NAMES):
        xi = np.nan_to_num(Xf[:, i], nan=0.0, posinf=0.0, neginf=0.0)
        row[ch] = float(np.mean(xi))
        row[f"{ch}__std"] = float(np.std(xi))
        row[f"{ch}__rms"] = float(np.sqrt(np.mean(xi * xi)))
        row[f"{ch}__ptp"] = float(np.max(xi) - np.min(xi))
        row[f"{ch}__absmean"] = float(np.mean(np.abs(xi)))

    # Bandpowers + fractions
    nper = int(max(256, float(cfg["psd_len"]) * EEG_FS))
    for i, ch in enumerate(EEG_CHANNEL_NAMES):
        bps = []
        for band, (f1, f2) in BANDS.items():
            bp = bandpower_welch(Xf[:, i], EEG_FS, f1, f2, nper)
            row[f"{ch}__bp_{band}"] = float(bp)
            bps.append(bp)
        denom = float(np.sum(bps))
        if denom <= 0:
            denom = 1e-12
        for band in BANDS.keys():
            row[f"{ch}__frac_{band}"] = float(row[f"{ch}__bp_{band}"] / denom)

    # Cross-channel summary features
    try:
        row["EEG_GLOBAL_STD_MEAN"] = float(np.mean([row[f"{ch}__std"] for ch in EEG_CHANNEL_NAMES]))
        row["EEG_GLOBAL_RMS_MEAN"] = float(np.mean([row[f"{ch}__rms"] for ch in EEG_CHANNEL_NAMES]))
        row["EEG_GLOBAL_PTP_MEAN"] = float(np.mean([row[f"{ch}__ptp"] for ch in EEG_CHANNEL_NAMES]))
    except Exception:
        row["EEG_GLOBAL_STD_MEAN"] = 0.0
        row["EEG_GLOBAL_RMS_MEAN"] = 0.0
        row["EEG_GLOBAL_PTP_MEAN"] = 0.0

    # Diagnostics persisted in CSV
    row["EEG_UNITS_MODE"] = str(adapt_diag.get("units_detected", "unknown"))
    row["EEG_P95_ABS_PRE"] = "" if adapt_diag.get("pre_p95_abs") is None else float(adapt_diag["pre_p95_abs"])
    row["EEG_P95_ABS_POST"] = "" if adapt_diag.get("post_p95_abs") is None else float(adapt_diag["post_p95_abs"])
    row["EEG_USED_CAR"] = int(bool(adapt_diag.get("used_car", False)))
    row["EEG_USED_ROBUST_STD"] = int(bool(adapt_diag.get("used_robust_std", False)))
    row["EEG_USED_STD_MATCH"] = int(bool(adapt_diag.get("used_std_match", False)))

    # Persist per-channel std-match diagnostics (helps compare against DreamT target scale)
    std_gains = adapt_diag.get("std_match_gains", {}) or {}
    std_pre = adapt_diag.get("std_pre", {}) or {}
    std_post = adapt_diag.get("std_post", {}) or {}
    for ch in EEG_CHANNEL_NAMES:
        row[f"{ch}__stdmatch_gain"] = "" if ch not in std_gains else float(std_gains[ch])
        row[f"{ch}__std_pre_adapt"] = "" if ch not in std_pre else float(std_pre[ch])
        row[f"{ch}__std_post_adapt"] = "" if ch not in std_post else float(std_post[ch])

    row["Sleep_Stage"] = ""
    return row

class FeatureCalibrator:
    """
    Calibrates live features into the training feature distribution.

    Expects a JSON file with:
      {
        "impute": {"mode": "mean"},  # or "zero"
        "stats": {
          "feature_name": {"mean": 0.12, "std": 0.03, "clip_z": 5.0},
          ...
        }
      }

    - Standardizes to z-scores using training mean/std
    - Clips extreme z-scores
    - Imputes missing features using training mean (or 0)
    """
    def __init__(self, stats_json_path: Path | None):
        self.enabled = False
        self.impute_mode = "mean"
        self.stats = {}

        if stats_json_path is None:
            return

        try:
            obj = json.load(open(stats_json_path, "r", encoding="utf-8"))
            self.impute_mode = obj.get("impute", {}).get("mode", "mean")
            self.stats = obj.get("stats", {})
            self.enabled = bool(self.stats)
        except Exception:
            self.enabled = False
            self.stats = {}

    def transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled or X.empty:
            return X

        X2 = X.copy()

        # Impute missing expected features
        if self.impute_mode == "mean":
            for f, s in self.stats.items():
                if f in X2.columns:
                    mu = float(s.get("mean", 0.0))
                    X2[f] = X2[f].astype(float)
                    X2[f] = X2[f].where(np.isfinite(X2[f]), np.nan)
                    X2[f] = X2[f].fillna(mu)
        else:
            # zero-impute
            X2 = X2.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Z-score normalize + clip
        for f, s in self.stats.items():
            if f not in X2.columns:
                continue

            mu = float(s.get("mean", 0.0))
            sd = float(s.get("std", 1.0))
            if sd <= 1e-12:
                sd = 1.0

            clip_z = float(s.get("clip_z", 5.0))

            v = X2[f].astype(float).values
            z = (v - mu) / sd
            z = np.clip(z, -clip_z, clip_z)
            X2[f] = z

        return X2

# -----------------------------
# Live XGBoost Sleep Stage Inference (DreamT cascade)
# -----------------------------
class LiveSleepInference:
    """
    Thread-safe live inference wrapper for your DreamT-trained XGBoost cascade.
    Reuses the same logic as predict_muse.py (feature alignment + cascade prediction).
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.lock = threading.RLock()

        self.ready = False
        self.last_err = None

        self.xgb = None
        self.boosterA = None
        self.boosterB = None
        self.boosterC = None
        self.expected_cols: list[str] | None = None

        self.calibrator = FeatureCalibrator(self.run_dir / "train_feature_stats.json")

        # history of epoch predictions
        self.history: list[dict] = []

        self._load_models()

    def _safe_import_xgboost(self):
        import xgboost as xgb
        return xgb

    def _try_load_joblib_bundle(self, bundle_path: Path):
        try:
            return joblib.load(bundle_path)
        except Exception as e:
            self.last_err = f"Could not load joblib bundle: {bundle_path} ({e})"
            return None

    def _load_booster_from_json(self, json_path: Path):
        booster = self.xgb.Booster()
        booster.load_model(str(json_path))
        return booster

    def _align_columns(self, X: pd.DataFrame, expected_cols: list[str] | None):
        if not expected_cols:
            return X
        # Reindex creates all expected columns at once; missing get fill_value
        return X.reindex(columns=expected_cols, fill_value=np.nan).copy()

    def _coerce_numeric(self, X: pd.DataFrame):
        Xn = X.apply(pd.to_numeric, errors="coerce")
        Xn = Xn.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return Xn

    def _augment_with_feature_aliases(self, feat: dict) -> tuple[dict, dict]:
        """
        Add DreamT-style alias feature names derived from Muse feature names.
        Returns (augmented_features, alias_diag).
        """
        feat_aug = dict(feat)
        alias_diag = {
            "alias_enabled": int(bool(ENABLE_MUSE_DREAMT_FEATURE_ALIAS)),
            "alias_added_total": 0,
            "alias_added_matching_expected": 0,
            "alias_matching_expected_preview": [],
        }

        if not ENABLE_MUSE_DREAMT_FEATURE_ALIAS:
            return feat_aug, alias_diag

        try:
            aliases = build_muse_to_dreamt_feature_aliases(feat)
            added = 0
            matched = []

            for k, v in aliases.items():
                # only add if source key absent, or if existing is empty/non-finite
                if k not in feat_aug:
                    feat_aug[k] = v
                    added += 1
                    if self.expected_cols and (k in self.expected_cols):
                        matched.append(k)

            alias_diag["alias_added_total"] = int(added)
            alias_diag["alias_added_matching_expected"] = int(len(matched))
            alias_diag["alias_matching_expected_preview"] = matched[:15]
            return feat_aug, alias_diag

        except Exception as e:
            alias_diag["alias_error"] = repr(e)
            return feat_aug, alias_diag

    def _normalize_binary_proba(self, p):
        """
        Normalize predictions for a 2-class model into shape (n_rows, 2).

        Handles:
          - Booster + multi:softprob returning flat array of length (n_rows * 2)
          - Proper 2D output (n_rows, 2)
          - Logistic-style outputs (n_rows,) or (n_rows, 1)
        """
        arr = np.asarray(p, dtype=float)

        # Case A: already shaped (n, 2)
        if arr.ndim == 2 and arr.shape[1] == 2:
            s = np.sum(arr, axis=1, keepdims=True)
            s[s <= 0] = 1.0
            return arr / s

        # Case B: flat softprob output (n * 2,)
        if arr.ndim == 1 and arr.size % 2 == 0:
            arr = arr.reshape(-1, 2)
            s = np.sum(arr, axis=1, keepdims=True)
            s[s <= 0] = 1.0
            return arr / s

        # Case C: logistic output (n,)
        if arr.ndim == 1:
            pos = np.clip(arr, 0.0, 1.0)
            neg = 1.0 - pos
            return np.column_stack([neg, pos])

        # Case D: logistic output (n, 1)
        if arr.ndim == 2 and arr.shape[1] == 1:
            pos = np.clip(arr[:, 0], 0.0, 1.0)
            neg = 1.0 - pos
            return np.column_stack([neg, pos])

        raise ValueError(f"Unexpected binary prediction shape: {arr.shape}")


    def _normalize_multiclass_proba(self, p, n_classes: int):
        """
        Normalize predictions for an n-class model into shape (n_rows, n_classes).

        Handles:
          - Booster + multi:softprob returning flat array of length (n_rows * n_classes)
          - Proper 2D output (n_rows, n_classes)
          - Single-row 1D output (n_classes,)
        """
        arr = np.asarray(p, dtype=float)

        # Case A: already shaped (n, n_classes)
        if arr.ndim == 2 and arr.shape[1] == n_classes:
            s = np.sum(arr, axis=1, keepdims=True)
            s[s <= 0] = 1.0
            return arr / s

        # Case B: flat softprob output (n * n_classes,)
        if arr.ndim == 1 and arr.size % n_classes == 0:
            arr = arr.reshape(-1, n_classes)
            s = np.sum(arr, axis=1, keepdims=True)
            s[s <= 0] = 1.0
            return arr / s

        # Case C: single-row 1D (n_classes,)
        if arr.ndim == 1 and arr.size == n_classes:
            arr = arr.reshape(1, n_classes)
            s = np.sum(arr, axis=1, keepdims=True)
            s[s <= 0] = 1.0
            return arr / s

        raise ValueError(
            f"Unexpected multiclass prediction shape: {arr.shape}, expected (?, {n_classes})"
        )

    def _predict_cascade_from_boosters(self, X: pd.DataFrame):
        dmat = self.xgb.DMatrix(X, feature_names=list(X.columns))

        # -------------------------
        # Stage A: classes ["S", "W"]
        # -------------------------
        rawA = self.boosterA.predict(dmat)
        probaA = self._normalize_binary_proba(rawA)

        if not hasattr(self, "_dbg_prints"):
            self._dbg_prints = 0
        if self._dbg_prints < 5:
            print(
                "StageA raw shape:", np.asarray(rawA).shape,
                "probaA shape:", np.asarray(probaA).shape,
                "probaA[0]:", probaA[0],
                "sum:", float(np.sum(probaA[0])),
            )
            self._dbg_prints += 1

        pA_S = probaA[:, 0]
        pA_W = probaA[:, 1]

        isW = pA_W >= 0.80
        pred = np.array(["S"] * len(X), dtype=object)
        pred[isW] = "W"

        # -------------------------
        # Stage B: classes ["NREM", "R"]
        # Only for rows predicted as Sleep (not W)
        # -------------------------
        probaB = np.full((len(X), 2), np.nan, dtype=float)
        probaC = np.full((len(X), 3), np.nan, dtype=float)

        idxS = np.where(~isW)[0]
        if len(idxS) > 0:
            dS = self.xgb.DMatrix(X.iloc[idxS], feature_names=list(X.columns))
            rawB = self.boosterB.predict(dS)
            pb = self._normalize_binary_proba(rawB)
            probaB[idxS] = pb

            pB_NREM = pb[:, 0]
            pB_R = pb[:, 1]

            isR = pB_R >= 0.5
            pred[idxS[isR]] = "R"
            pred[idxS[~isR]] = "NREM"

            # -------------------------
            # Stage C: classes ["N1", "N2", "N3"]
            # Only for rows predicted as NREM
            # -------------------------
            idxNREM = idxS[~isR]
            if len(idxNREM) > 0:
                dN = self.xgb.DMatrix(X.iloc[idxNREM], feature_names=list(X.columns))
                rawC = self.boosterC.predict(dN)
                pc = self._normalize_multiclass_proba(rawC, n_classes=3)
                probaC[idxNREM] = pc

                c_idx = np.argmax(pc, axis=1)
                pred[idxNREM[c_idx == 0]] = "N1"
                pred[idxNREM[c_idx == 1]] = "N2"
                pred[idxNREM[c_idx == 2]] = "N3"

        return pred, probaA, probaB, probaC

    def _load_models(self):
        try:
            self.xgb = self._safe_import_xgboost()

            bundle_path = self.run_dir / "model_bundle.joblib"
            bundle = self._try_load_joblib_bundle(bundle_path)

            self.expected_cols = None
            if isinstance(bundle, dict):
                for k in ["feature_cols", "feature_columns", "expected_cols", "columns"]:
                    if k in bundle and isinstance(bundle[k], (list, tuple)):
                        self.expected_cols = list(bundle[k])
                        break

            # Try bundle models first
            boosterA = boosterB = boosterC = None
            if isinstance(bundle, dict):
                boosterA = bundle.get("stageA") or bundle.get("model_stageA") or bundle.get("modelA") or bundle.get("xgb_stageA")
                boosterB = bundle.get("stageB") or bundle.get("model_stageB") or bundle.get("modelB") or bundle.get("xgb_stageB")
                boosterC = bundle.get("stageC") or bundle.get("model_stageC") or bundle.get("modelC") or bundle.get("xgb_stageC")

                def to_booster(m):
                    if m is None:
                        return None
                    if hasattr(m, "get_booster"):
                        return m.get_booster()
                    if isinstance(m, self.xgb.Booster):
                        return m
                    return None

                boosterA, boosterB, boosterC = map(to_booster, [boosterA, boosterB, boosterC])

            # JSON fallback
            if boosterA is None or boosterB is None or boosterC is None:
                jsonA = self.run_dir / "xgb_stageA_W_vs_S.json"
                jsonB = self.run_dir / "xgb_stageB_R_vs_NREM.json"
                jsonC = self.run_dir / "xgb_stageC_N1_vs_N2_vs_N3.json"

                if not (jsonA.exists() and jsonB.exists() and jsonC.exists()):
                    raise RuntimeError(
                        f"Missing model files in {self.run_dir}. Need model_bundle.joblib or stage JSONs."
                    )

                boosterA = self._load_booster_from_json(jsonA)
                boosterB = self._load_booster_from_json(jsonB)
                boosterC = self._load_booster_from_json(jsonC)

            self.boosterA = boosterA
            self.boosterB = boosterB
            self.boosterC = boosterC
            self.ready = True
            self.last_err = None

        except Exception as e:
            self.ready = False
            self.last_err = repr(e)

    def predict_epoch_row(self, row: dict) -> dict | None:
        """
        row: one epoch feature row (same dict before writing CSV)
        returns a result dict and appends to internal history
        """
        if not self.ready:
            return None

        try:
            # model features only (drop labels/time-ish columns + non-numeric debug columns)
            drop = {
                "Sleep_Stage", "TIMESTAMP", "ISO_TIME", "pred_stage",
                # string debug columns that should not go to model
                "EEG_UNITS_MODE",
                "PPG_QUALITY",
            }
            feat_base = {k: v for k, v in row.items() if k not in drop}

            # ---- Add DreamT-style aliases from Muse feature names BEFORE alignment ----
            feat, alias_diag = self._augment_with_feature_aliases(feat_base)

            X_raw = pd.DataFrame([feat])

            # Diagnostics before align
            present_before = list(X_raw.columns)
            n_present_before = len(present_before)

            X = self._coerce_numeric(X_raw)

            missing_before_align = []
            extra_before_align = []
            if self.expected_cols:
                missing_before_align = [c for c in self.expected_cols if c not in X.columns]
                extra_before_align = [c for c in X.columns if c not in self.expected_cols]

            X = self._align_columns(X, self.expected_cols)
            X = self._coerce_numeric(X)

            # NEW: Calibrate into DreamT training distribution (z-score + impute)
            X = self.calibrator.transform_df(X)

            # Make sure numeric / no NaNs remain
            X = self._coerce_numeric(X)

            # Post-align diagnostics
            vals = X.iloc[0].astype(float).values
            nonzero_mask = np.abs(vals) > 0.0
            nonzero_cols = [c for c, nz in zip(X.columns.tolist(), nonzero_mask.tolist()) if nz]
            nonzero_vals = [float(v) for v, nz in zip(vals.tolist(), nonzero_mask.tolist()) if nz]

            n_nonzero = len(nonzero_cols)
            n_expected = len(self.expected_cols) if self.expected_cols else len(X.columns)
            nonzero_frac = (n_nonzero / n_expected) if n_expected > 0 else None

            # top nonzero by magnitude preview
            top_pairs = sorted(
                zip(nonzero_cols, nonzero_vals),
                key=lambda kv: abs(float(kv[1])),
                reverse=True
            )[:12]

            pred, probaA, probaB, probaC = self._predict_cascade_from_boosters(X)
            label = str(pred[0])

            out = {
                "epoch_index": None,  # filled by caller if desired
                "timestamp_s": row.get("TIMESTAMP", None),
                "iso_time": row.get("ISO_TIME", None),
                "pred_stage": label,

                "pA_S": float(probaA[0, 0]) if np.isfinite(probaA[0, 0]) else None,
                "pA_W": float(probaA[0, 1]) if np.isfinite(probaA[0, 1]) else None,
                "pB_NREM": (None if np.isnan(probaB[0, 0]) else float(probaB[0, 0])),
                "pB_R": (None if np.isnan(probaB[0, 1]) else float(probaB[0, 1])),
                "pC_N1": (None if np.isnan(probaC[0, 0]) else float(probaC[0, 0])),
                "pC_N2": (None if np.isnan(probaC[0, 1]) else float(probaC[0, 1])),
                "pC_N3": (None if np.isnan(probaC[0, 2]) else float(probaC[0, 2])),

                # Diagnostics for UI / CSV
                "diag_n_expected": int(n_expected),
                "diag_n_present_before_align": int(n_present_before),
                "diag_n_missing_before_align": int(len(missing_before_align)),
                "diag_n_extra_before_align": int(len(extra_before_align)),
                "diag_n_nonzero": int(n_nonzero),
                "diag_nonzero_frac": (None if nonzero_frac is None else float(nonzero_frac)),
                "diag_nonzero_cols_preview": nonzero_cols[:12],
                "diag_missing_cols_preview": missing_before_align[:12],
                "diag_extra_cols_preview": extra_before_align[:12],

                # New alias diagnostics
                "diag_alias_enabled": int(alias_diag.get("alias_enabled", 0)),
                "diag_alias_added_total": int(alias_diag.get("alias_added_total", 0)),
                "diag_alias_added_matching_expected": int(alias_diag.get("alias_added_matching_expected", 0)),
                "diag_alias_matching_expected_preview": alias_diag.get("alias_matching_expected_preview", []),

                # New magnitude preview (helps prove vector is not all zeros)
                "diag_nonzero_vals_preview": [
                    {"feature": k, "value": round(float(v), 6)} for k, v in top_pairs
                ],
            }

            # final display confidence (path-based)
            if label == "W":
                out["confidence"] = out["pA_W"]
            elif label == "R":
                out["confidence"] = out["pB_R"] if out["pB_R"] is not None else None
            elif label in {"N1", "N2", "N3"}:
                key = f"pC_{label}"
                out["confidence"] = out.get(key, None)
            else:
                out["confidence"] = None

            MAX_PRED_HISTORY = 2880  # 24 hours at 30s epochs

            with self.lock:
                self.history.append(out)
                if len(self.history) > MAX_PRED_HISTORY:
                    self.history = self.history[-MAX_PRED_HISTORY:]
            
            return out

        except Exception as e:
            with self.lock:
                self.last_err = repr(e)
            return None

    def status(self):
        with self.lock:
            return {
                "ready": bool(self.ready),
                "last_err": self.last_err,
                "n_preds": len(self.history),
                "latest": (self.history[-1] if self.history else None),
            }

    def get_history(self, n: int | None = 20):
        with self.lock:
            if n is None:
                return list(self.history)
            return list(self.history[-int(max(1, n)):])

class EpochCollector:
    def __init__(
        self,
        eeg: LSLReceiver,
        ppg: LSLReceiver,
        cfg_store: ConfigStore,
        infer_engine: LiveSleepInference | None = None,
    ):
        self.eeg = eeg
        self.ppg = ppg
        self.cfg_store = cfg_store
        self.thread = None
        self.running = False
        self.lock = threading.Lock()

        self.start_ts = None
        self.last_seen_ts_eeg = None
        self.last_seen_ts_ppg = None
        self.epoch_start_ts = None

        self.epoch_ts = []
        self.epoch_X = []

        self.ppg_ts = []
        self.ppg_X = []

        self.csv_path: Path | None = None
        self.session_dir: Path | None = None
        self.rows_written = 0
        self.last_write_iso = None
        self.last_err = None

        # --- NEW debug / anti-silent-stop ---
        self.log_path: Path | None = None
        self.last_stop_reason: str | None = None
        self.last_stop_iso: str | None = None
        self.last_heartbeat_iso: str | None = None

        self.infer_engine = infer_engine

    # -----------------------------
    # Logging helpers
    # -----------------------------
    def _log(self, msg: str):
        # Avoid locks + avoid any chance of UI deadlock/stall.
        lp = getattr(self, "log_path", None)
        if lp is None:
            return
        try:
            ts = datetime.now().isoformat(timespec="seconds")
            p = Path(lp)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8", newline="") as f:
                f.write(f"[{ts}] {msg}\n")
        except Exception:
            pass

    def start(self) -> bool:
        with self.lock:
            if self.running:
                return True
            if not self.eeg.running:
                self.last_err = "EEG receiver not running (connect to LSL first)."
                return False

        # Wait briefly for samples
        t0 = time.time()
        while time.time() - t0 < COLLECTOR_WAIT_FOR_SAMPLES_S:
            ts_w, X_w = self.eeg.get_window(1.0)
            if ts_w.size and X_w.size:
                break
            time.sleep(0.15)

        csv_name = None  # capture for logging outside the lock

        with self.lock:
            ts_w, X_w = self.eeg.get_window(1.0)
            if ts_w.size == 0:
                self.last_err = "No EEG samples in buffer yet (wait a moment after connecting)."
                return False

            out_root = ensure_sleep_data_dir()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = out_root / f"run_{stamp}"
            self.session_dir.mkdir(parents=True, exist_ok=True)

            self.csv_path = self.session_dir / f"muse_epochs_{stamp}.csv"
            csv_name = self.csv_path.name

            # per-run debug log
            self.log_path = self.session_dir / "collector_log.txt"
            self.last_stop_reason = None
            self.last_stop_iso = None
            self.last_err = None
            self.last_heartbeat_iso = None

            self.running = True
            self.rows_written = 0
            self.last_write_iso = None

            self.start_ts = float(ts_w[-1])
            self.last_seen_ts_eeg = float(ts_w[-1])
            self.epoch_start_ts = float(ts_w[-1])
            self.epoch_ts = []
            self.epoch_X = []

            ts_p, _ = self.ppg.get_window(1.0)
            self.last_seen_ts_ppg = float(ts_p[-1]) if ts_p.size else None
            self.ppg_ts = []
            self.ppg_X = []

            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

        # LOG OUTSIDE LOCK (no UI hang)
        try:
            if csv_name:
                self._log(f"START | csv={csv_name}")
        except Exception:
            pass

        return True

    def stop(self, reason: str = "STOP called"):
        with self.lock:
            self.running = False
            self.last_stop_reason = reason
            self.last_stop_iso = datetime.now().isoformat(timespec="seconds")
        try:
            self._log(f"STOP | reason={reason}")
        except Exception:
            pass

    def join(self, timeout: float = 10.0):
        t = None
        with self.lock:
            t = self.thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)

    def finalize_and_save_hypnogram(self) -> Path | None:
        """
        Stops collector, waits for thread to exit, then saves hypnogram_pred.png
        into the same session_dir as the epochs CSV.
        """
        # stop + join
        self.stop(reason="Finalize requested (save hypnogram)")
        self.join(timeout=15.0)

        with self.lock:
            csv_path = self.csv_path
            session_dir = self.session_dir

        if not csv_path or not session_dir:
            self._log("HYP | skipped (no csv_path or session_dir)")
            return None

        csv_path = Path(csv_path)
        session_dir = Path(session_dir)

        # Ensure CSV exists
        if not csv_path.exists():
            self.last_err = f"Hypnogram save skipped: epochs CSV not found: {csv_path}"
            self._log(f"HYP | skipped (csv missing): {csv_path}")
            return None

        # Ensure CSV has at least 1 data row (not just header)
        try:
            # Fast-ish check: read a tiny amount
            df_head = pd.read_csv(csv_path, nrows=1)
            if df_head.empty:
                self.last_err = "Hypnogram save skipped: epochs CSV has no rows yet."
                self._log("HYP | skipped (csv has 0 rows)")
                return None
        except Exception as e:
            self.last_err = f"Hypnogram save skipped: could not read epochs CSV ({e!r})"
            self._log(f"HYP | skipped (csv read error): {e!r}")
            return None

        try:
            out_png = session_dir / "hypnogram_pred.png"

            p = save_hypnogram_from_epochs_csv(csv_path, out_path=out_png)

            # Robust: handle None returns
            if p is None:
                self.last_err = "Hypnogram save error: save_hypnogram_from_epochs_csv returned None"
                self._log("HYP | ERROR save returned None")
                return None

            p = Path(p)
            self._log(f"HYP | saved={p.name}")
            return p

        except Exception as e:
            self.last_err = f"Hypnogram save error: {e!r}"
            self._log(f"HYP | ERROR {e!r}")
            self._log(traceback.format_exc())
            return None

    def status(self):
        with self.lock:
            return {
                "collecting": self.running,
                "csv_path": str(self.csv_path) if self.csv_path else None,
                "session_dir": str(self.session_dir) if self.session_dir else None,
                "rows_written": int(self.rows_written),
                "last_write_iso": self.last_write_iso,
                "last_err": self.last_err,
                # NEW
                "last_stop_reason": self.last_stop_reason,
                "last_stop_iso": self.last_stop_iso,
                "last_heartbeat_iso": self.last_heartbeat_iso,
                "log_path": str(self.log_path) if self.log_path else None,
            }

    def _append_row(self, row: dict):
        assert self.csv_path is not None
        write_header = not self.csv_path.exists()
        import csv

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    def _run(self):
        try:
            # NEW: detect long stalls + ts resets
            last_eeg_progress_wall = time.time()
            last_ppg_progress_wall = time.time()

            while True:
                with self.lock:
                    if not self.running:
                        self._log("THREAD | exit (running=False)")
                        return
                    self.last_heartbeat_iso = datetime.now().isoformat(timespec="seconds")

                # ---- Pull new EEG samples
                ts_new, X_new = self.eeg.get_since(self.last_seen_ts_eeg)

                if ts_new.size == 0:
                    time.sleep(0.10)
                else:
                    last_eeg_progress_wall = time.time()

                    # NEW: if timestamps jumped backwards or we lost continuity, reset epoch assembly
                    if self.last_seen_ts_eeg is not None:
                        try:
                            if float(ts_new[0]) < float(self.last_seen_ts_eeg) - 1e-6:
                                self._log("EEG | timestamp reset detected -> reset epoch state")
                                self.epoch_ts = []
                                self.epoch_X = []
                                self.epoch_start_ts = None
                        except Exception:
                            pass

                    self.last_seen_ts_eeg = float(ts_new[-1])

                    for t, x in zip(ts_new.tolist(), X_new.tolist()):
                        self.epoch_ts.append(float(t))
                        self.epoch_X.append(x)

                # ---- Pull new PPG samples (optional)
                if self.ppg.running and self.last_seen_ts_ppg is not None:
                    tsp_new, Xp_new = self.ppg.get_since(self.last_seen_ts_ppg)
                    if tsp_new.size:
                        last_ppg_progress_wall = time.time()

                        # NEW: timestamp reset for PPG
                        try:
                            if float(tsp_new[0]) < float(self.last_seen_ts_ppg) - 1e-6:
                                self._log("PPG | timestamp reset detected -> clear PPG epoch buffer")
                                self.ppg_ts = []
                                self.ppg_X = []
                        except Exception:
                            pass

                        self.last_seen_ts_ppg = float(tsp_new[-1])
                        for t, x in zip(tsp_new.tolist(), Xp_new.tolist()):
                            self.ppg_ts.append(float(t))
                            self.ppg_X.append(x)

                # ---- Watchdog: if EEG stalls for a long time, keep looping but log it
                if time.time() - last_eeg_progress_wall > 20.0:
                    self._log("EEG | stalled >20s (waiting for samples / reconnect)")
                    last_eeg_progress_wall = time.time()  # avoid log spam

                # ---- Establish epoch_start_ts when we have data
                if self.epoch_start_ts is None and self.epoch_ts:
                    self.epoch_start_ts = float(self.epoch_ts[0])

                if not self.epoch_ts or self.epoch_start_ts is None:
                    continue

                # ---- Only attempt an epoch if we have reached end of window
                if (float(self.epoch_ts[-1]) - float(self.epoch_start_ts)) < EPOCH_SEC:
                    continue

                # Build candidate epoch window
                epoch_start = float(self.epoch_start_ts)
                epoch_end = epoch_start + EPOCH_SEC

                ts_arr = np.asarray(self.epoch_ts, dtype=float)
                X_arr = np.asarray(self.epoch_X, dtype=float)

                m = (ts_arr >= epoch_start) & (ts_arr < epoch_end)
                ts_epoch = ts_arr[m]
                X_epoch = X_arr[m]

                # NEW: DO NOT advance epoch_start_ts yet — only if epoch is accepted
                # If too few samples, resync instead of burning time.
                min_samples = int(EEG_FS * EPOCH_SEC * 0.70)
                if ts_epoch.size < min_samples:
                    self._log(f"EPOCH | skipped (too few EEG samples: {ts_epoch.size}/{min_samples})")

                    # Resync strategy:
                    # Move epoch_start_ts up near the newest data so we can recover quickly.
                    # Keep a small lookback so we can accumulate a clean 30s.
                    newest = float(ts_arr[-1])
                    self.epoch_start_ts = max(newest - (0.25 * EPOCH_SEC), newest)  # safe fallback
                    # Better: just restart epoch at newest (prevents repeated partial windows)
                    self.epoch_start_ts = newest

                    # Also trim old buffers aggressively so we don’t keep scanning junk
                    keep = ts_arr >= newest
                    self.epoch_ts = ts_arr[keep].tolist()
                    self.epoch_X = X_arr[keep].tolist()
                    continue

                # ---- ACCEPT epoch: now advance and trim
                self.epoch_start_ts = epoch_end
                keep = ts_arr >= epoch_end
                self.epoch_ts = ts_arr[keep].tolist()
                self.epoch_X = X_arr[keep].tolist()

                # ---- Features + row
                cfg = self.cfg_store.get()
                row = epoch_features(ts_epoch, X_epoch, cfg)

                # ---- PPG-derived metrics for this epoch
                hr_inst = None
                hr_smooth = None
                rmssd = None
                q_label = "Bad"
                q_score = 0.0

                if self.ppg_ts and self.ppg_X:
                    tp = np.asarray(self.ppg_ts, dtype=float)
                    Xp = np.asarray(self.ppg_X, dtype=float)

                    m_epoch = (tp >= epoch_start) & (tp < epoch_end)
                    m_smooth = tp >= (epoch_end - HR_SMOOTH_WINDOW_SEC)

                    if np.any(m_smooth) and Xp.shape[1] >= 2:
                        ts_sm = tp[m_smooth]
                        X_sm = Xp[m_smooth]

                        s_ir, l_ir, _ = ppg_quality_score(ts_sm, X_sm[:, 0], PPG_FS)
                        s_rd, l_rd, _ = ppg_quality_score(ts_sm, X_sm[:, 1], PPG_FS)
                        use_idx = 0 if s_ir >= s_rd else 1
                        q_score, q_label = (s_ir, l_ir) if use_idx == 0 else (s_rd, l_rd)

                        _xf_s, _peaks_s, _pt_s, ibi_s = _detect_beats(ts_sm, X_sm[:, use_idx], PPG_FS)
                        hr_smooth = estimate_bpm_from_ibi(ibi_s)
                        if q_label == "Good":
                            rmssd = rmssd_ms_from_ibi(ibi_s)

                    if np.any(m_epoch) and Xp.shape[1] >= 1:
                        ts_ep = tp[m_epoch]
                        X_ep = Xp[m_epoch]
                        _xf_i, _peaks_i, _pt_i, ibi_i = _detect_beats(ts_ep, X_ep[:, 0], PPG_FS)
                        hr_inst = estimate_bpm_from_ibi(ibi_i)

                    keep_p = tp >= (epoch_end - 2 * HR_SMOOTH_WINDOW_SEC)
                    self.ppg_ts = tp[keep_p].tolist()
                    self.ppg_X = Xp[keep_p].tolist()

                row["HR_BPM_INST"] = "" if hr_inst is None else float(hr_inst)
                row["HR_BPM_SMOOTH"] = "" if hr_smooth is None else float(hr_smooth)
                row["RMSSD_MS"] = "" if rmssd is None else float(rmssd)
                row["PPG_QUALITY"] = q_label
                row["PPG_QUALITY_SCORE"] = float(q_score)

                if self.start_ts is None:
                    self.start_ts = epoch_start
                row["TIMESTAMP"] = float(epoch_start - float(self.start_ts))
                row["ISO_TIME"] = datetime.now().isoformat(timespec="seconds")

                # ---- Live inference (never allowed to kill collector)
                if self.infer_engine is not None and self.infer_engine.ready:
                    try:
                        pred_out = self.infer_engine.predict_epoch_row(row)
                        if pred_out is not None:
                            pred_out["epoch_index"] = int(self.rows_written + 1)

                            row["PRED_STAGE"] = pred_out.get("pred_stage", "")
                            row["PRED_CONFIDENCE"] = "" if pred_out.get("confidence") is None else float(pred_out["confidence"])

                            row["P_A_S"] = "" if pred_out.get("pA_S") is None else float(pred_out["pA_S"])
                            row["P_A_W"] = "" if pred_out.get("pA_W") is None else float(pred_out["pA_W"])
                            row["P_B_NREM"] = "" if pred_out.get("pB_NREM") is None else float(pred_out["pB_NREM"])
                            row["P_B_R"] = "" if pred_out.get("pB_R") is None else float(pred_out["pB_R"])
                            row["P_C_N1"] = "" if pred_out.get("pC_N1") is None else float(pred_out["pC_N1"])
                            row["P_C_N2"] = "" if pred_out.get("pC_N2") is None else float(pred_out["pC_N2"])
                            row["P_C_N3"] = "" if pred_out.get("pC_N3") is None else float(pred_out["pC_N3"])

                            row["DIAG_N_EXPECTED"] = pred_out.get("diag_n_expected", "")
                            row["DIAG_N_PRESENT_BEFORE_ALIGN"] = pred_out.get("diag_n_present_before_align", "")
                            row["DIAG_N_MISSING_BEFORE_ALIGN"] = pred_out.get("diag_n_missing_before_align", "")
                            row["DIAG_N_EXTRA_BEFORE_ALIGN"] = pred_out.get("diag_n_extra_before_align", "")
                            row["DIAG_N_NONZERO"] = pred_out.get("diag_n_nonzero", "")
                            row["DIAG_NONZERO_FRAC"] = "" if pred_out.get("diag_nonzero_frac") is None else float(pred_out["diag_nonzero_frac"])
                    except Exception as e:
                        self.last_err = f"Live inference error (collector kept running): {e!r}"
                        self._log(f"INFER | ERROR {e!r}")
                        self._log(traceback.format_exc())

                # ---- CSV append (never silent)
                try:
                    self._append_row(row)
                    with self.lock:
                        self.rows_written += 1
                        self.last_write_iso = datetime.now().isoformat(timespec="seconds")
                    self._log(f"EPOCH | wrote row {self.rows_written}")
                except Exception as e:
                    self.last_err = f"CSV write error: {e!r}"
                    self._log(f"CSV | ERROR {e!r}")
                    self._log(traceback.format_exc())
                    continue

        except Exception as e:
            with self.lock:
                self.last_err = f"Collector crash: {e!r}"
                self.running = False
                self.last_stop_reason = "CRASH"
                self.last_stop_iso = datetime.now().isoformat(timespec="seconds")
            self._log(f"CRASH | {e!r}")
            self._log(traceback.format_exc())
            return

# -----------------------------
# Snapshot save (time-series window to CSV)
# -----------------------------
def save_snapshot_csv(eeg: LSLReceiver, ppg: LSLReceiver, seconds: float) -> Path | None:
    ts_e, X_e = eeg.get_window(seconds)
    if X_e.size == 0:
        return None

    ts_p, X_p = ppg.get_window(max(seconds, HR_SMOOTH_WINDOW_SEC))

    out_dir = ensure_sleep_data_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = out_dir / f"muse_snapshot_{stamp}.csv"

    t0 = float(ts_e[0])
    rel_t = ts_e - t0

    if X_p.size != 0:
        Xp_aligned = _align_nearest(ts_p, X_p, ts_e)
    else:
        Xp_aligned = np.full((len(ts_e), len(PPG_CHANNEL_NAMES)), np.nan, dtype=float)

    hr_inst = None
    hr_smooth = None
    rmssd = None
    q_label = "Bad"
    q_score = 0.0

    if ts_p.size and X_p.size and X_p.shape[1] >= 2:
        s_ir, l_ir, _ = ppg_quality_score(ts_p, X_p[:, 0], PPG_FS)
        s_rd, l_rd, _ = ppg_quality_score(ts_p, X_p[:, 1], PPG_FS)
        use_idx = 0 if s_ir >= s_rd else 1
        q_score, q_label = (s_ir, l_ir) if use_idx == 0 else (s_rd, l_rd)

        _xf_s, _peaks_s, _pt_s, ibi_s = _detect_beats(ts_p, X_p[:, use_idx], PPG_FS)
        hr_smooth = estimate_bpm_from_ibi(ibi_s)
        if q_label == "Good":
            rmssd = rmssd_ms_from_ibi(ibi_s)

        t_end = ts_p[-1]
        m10 = ts_p >= (t_end - HR_INST_WINDOW_SEC)
        if np.any(m10):
            _xf_i, _peaks_i, _pt_i, ibi_i = _detect_beats(ts_p[m10], X_p[m10, use_idx], PPG_FS)
            hr_inst = estimate_bpm_from_ibi(ibi_i)

    import csv
    fieldnames = (
        ["TIMESTAMP"]
        + EEG_CHANNEL_NAMES
        + PPG_CHANNEL_NAMES
        + ["HR_BPM_INST", "HR_BPM_SMOOTH", "RMSSD_MS", "PPG_QUALITY", "PPG_QUALITY_SCORE"]
        + ["Sleep_Stage"]
    )

    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(X_e.shape[0]):
            row = {"TIMESTAMP": float(rel_t[i]), "Sleep_Stage": ""}
            for ci, ch in enumerate(EEG_CHANNEL_NAMES):
                row[ch] = float(X_e[i, ci])
            for pi, pch in enumerate(PPG_CHANNEL_NAMES):
                v = Xp_aligned[i, pi]
                row[pch] = float(v) if np.isfinite(v) else ""
            row["HR_BPM_INST"] = "" if hr_inst is None else float(hr_inst)
            row["HR_BPM_SMOOTH"] = "" if hr_smooth is None else float(hr_smooth)
            row["RMSSD_MS"] = "" if rmssd is None else float(rmssd)
            row["PPG_QUALITY"] = q_label
            row["PPG_QUALITY_SCORE"] = float(q_score)
            w.writerow(row)

    return p

def save_prediction_summary_image(infer_engine, out_dir: Path | None = None, max_epochs: int = 120) -> Path | None:
    """
    Save a PNG summary image of recent predicted sleep stages:
      - Top: histogram of stage counts
      - Bottom: epoch strip (temporal order)
    """
    try:
        hist = infer_engine.get_history(max_epochs)
        if not hist:
            return None

        if out_dir is None:
            out_dir = ensure_sleep_data_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"muse_predictions_summary_{stamp}.png"

        stage_order = ["W", "N1", "N2", "N3", "R"]
        stage_colors = {
            "W": "#f59e0b",   # amber
            "N1": "#60a5fa",  # blue
            "N2": "#22c55e",  # green
            "N3": "#8b5cf6",  # purple
            "R": "#ef4444",   # red
        }

        # Count histogram
        counts = {k: 0 for k in stage_order}
        for h in hist:
            stg = str(h.get("pred_stage", ""))
            if stg in counts:
                counts[stg] += 1

        # Figure (dark theme to match dashboard)
        fig = plt.figure(figsize=(10.5, 4.8), dpi=180, facecolor="#0E1117")
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.2, 1.4], hspace=0.45)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # ---------------- Histogram ----------------
        x = list(range(len(stage_order)))
        y = [counts[k] for k in stage_order]
        colors = [stage_colors[k] for k in stage_order]

        ax1.set_facecolor("#111827")
        bars = ax1.bar(x, y, color=colors, edgecolor=(1, 1, 1, 0.15), linewidth=1.0)

        ax1.set_xticks(x)
        ax1.set_xticklabels(stage_order, color="white", fontsize=10)
        ax1.tick_params(axis="y", colors="white", labelsize=9)
        ax1.grid(axis="y", alpha=0.15, color="white", linewidth=0.8)
        ax1.set_axisbelow(True)

        ymax = max(y) if y else 0
        ax1.set_ylim(0, max(1, ymax + 1))

        ax1.set_title(
            f"Predicted Sleep Stage Distribution (last {len(hist)} epochs)",
            color="white",
            fontsize=12,
            pad=10,
        )

        for rect, val in zip(bars, y):
            ax1.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + 0.05,
                str(val),
                ha="center",
                va="bottom",
                color="white",
                fontsize=9,
            )

        for spine in ax1.spines.values():
            spine.set_color((1, 1, 1, 0.15))

        # ---------------- Epoch strip ----------------
        ax2.set_facecolor("#111827")
        ax2.set_title("Epoch Strip (oldest → newest)", color="white", fontsize=11, pad=8)

        n = len(hist)
        for i, h in enumerate(hist):
            stg = str(h.get("pred_stage", ""))
            c = stage_colors.get(stg, "#9ca3af")
            ax2.add_patch(
                Rectangle(
                    (i, 0),
                    0.9,
                    1.0,
                    facecolor=c,
                    edgecolor=(1, 1, 1, 0.18),
                    linewidth=0.6,
                )
            )

        ax2.set_xlim(0, max(1, n))
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])

        if n <= 12:
            tick_idx = list(range(n))
        else:
            step = max(1, n // 8)
            tick_idx = list(range(0, n, step))
            if (n - 1) not in tick_idx:
                tick_idx.append(n - 1)

        tick_labels = []
        for i in tick_idx:
            idx_val = hist[i].get("epoch_index", i + 1)
            tick_labels.append(str(idx_val))

        ax2.set_xticks(tick_idx)
        ax2.set_xticklabels(tick_labels, color="white", fontsize=8)
        ax2.tick_params(axis="x", colors="white")

        for spine in ax2.spines.values():
            spine.set_color((1, 1, 1, 0.15))

        legend_handles = [
            Rectangle((0, 0), 1, 1, facecolor=stage_colors[k], edgecolor=(1, 1, 1, 0.15))
            for k in stage_order
        ]
        leg = ax2.legend(
            legend_handles,
            stage_order,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.32),
            ncol=5,
            frameon=False,
            fontsize=8,
        )
        for t in leg.get_texts():
            t.set_color("white")

        fig.suptitle("Muse Live Prediction Summary", color="white", fontsize=13, y=0.99)

        fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_path

    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None

def save_hypnogram_from_epochs_csv(
    epochs_csv_path: Path,
    out_path: Path | None = None,
    title: str = "Hypnogram (Predicted Stages)",
    epoch_seconds: int = 30,
) -> Path:
    """
    Save a dark-themed, colored-block hypnogram PNG from an epochs CSV.

    Guaranteed behavior:
      - Returns Path on success
      - Raises Exception on failure (never returns None)
    """
    epochs_csv_path = Path(epochs_csv_path)
    if not epochs_csv_path.exists():
        raise FileNotFoundError(f"Epochs CSV not found: {epochs_csv_path}")

    df = pd.read_csv(epochs_csv_path)
    if df.empty:
        raise ValueError("Epochs CSV has no rows yet (nothing to plot).")

    # Accept multiple possible stage columns (robust to partial runs / older CSVs)
    stage_col = None
    for cand in ["PRED_STAGE", "pred_stage", "Sleep_Stage", "SLEEP_STAGE"]:
        if cand in df.columns:
            stage_col = cand
            break
    if stage_col is None:
        raise ValueError(
            "CSV missing a stage column. Expected one of: PRED_STAGE, pred_stage, Sleep_Stage, SLEEP_STAGE."
        )

    stages = (
        df[stage_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"REM": "R"})
        .tolist()
    )
    if not stages:
        raise ValueError("Stage column is present but empty.")

    # Output path
    if out_path is None:
        out_path = epochs_csv_path.parent / "hypnogram_pred.png"
    out_path = Path(out_path)

    # ---- Dark themed colored-block hypnogram ----
    stage_order = ["W", "R", "N1", "N2", "N3"]
    y_pos = {s: i for i, s in enumerate(stage_order)}
    stage_colors = {
        "W":  "#f59e0b",
        "N1": "#60a5fa",
        "N2": "#22c55e",
        "N3": "#8b5cf6",
        "R":  "#ef4444",
    }
    unknown_color = "#9ca3af"

    n = len(stages)
    epoch_min = float(epoch_seconds) / 60.0
    total_min = n * epoch_min

    fig = plt.figure(figsize=(11.0, 3.2), dpi=200, facecolor="#0E1117")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#111827")

    block_h = 0.85
    for i, s in enumerate(stages):
        # Conservative fallback: unknown -> W (keeps y mapping valid)
        s2 = s if s in y_pos else "W"
        yp = y_pos.get(s2, y_pos["W"])
        c = stage_colors.get(s2, unknown_color)
        x = i * epoch_min
        ax.add_patch(
            Rectangle(
                (x, yp - block_h / 2),
                epoch_min,
                block_h,
                facecolor=c,
                edgecolor=(1, 1, 1, 0.10),
                linewidth=0.8,
            )
        )

    ax.set_xlim(0, total_min)
    ax.set_ylim(-0.75, len(stage_order) - 0.25)
    ax.invert_yaxis()

    ax.set_yticks([y_pos[s] for s in stage_order])
    ax.set_yticklabels(stage_order, fontsize=11, color="white")
    ax.set_xlabel("Time (minutes)", color="white", fontsize=11)
    ax.set_ylabel("Stage", color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=14, pad=10)

    ax.grid(True, axis="x", alpha=0.12, color="white", linewidth=0.9)
    ax.grid(True, axis="y", alpha=0.06, color="white", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.15))
    ax.tick_params(colors="white")

    ax.text(
        0.99, 0.03,
        f"{n} epochs • {total_min:.1f} min",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=10,
        color=(1, 1, 1, 0.75),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    return out_path

def _safe_parse_iso(ts_str: str):
    try:
        return datetime.fromisoformat(str(ts_str))
    except Exception:
        return None

def compute_sleep_metrics_from_epochs(df: pd.DataFrame, epoch_seconds: int = 30) -> dict:
    """
    Compute practical sleep metrics from epoch-level predictions.
    Works on your muse_epochs_*.csv that contains PRED_STAGE and (optionally) HR fields.
    """
    if df is None or df.empty:
        return {"ok": False, "reason": "empty_df"}

    if "PRED_STAGE" not in df.columns:
        return {"ok": False, "reason": "missing_PRED_STAGE"}

    stages = df["PRED_STAGE"].astype(str).str.strip().str.upper().replace({"REM": "R"}).tolist()
    n = len(stages)
    epoch_min = epoch_seconds / 60.0
    duration_min = n * epoch_min

    counts = {k: 0 for k in ["W", "N1", "N2", "N3", "R"]}
    unk = 0
    for s in stages:
        if s in counts:
            counts[s] += 1
        else:
            unk += 1
            counts["W"] += 1  # conservative: unknown -> Wake

    sleep_epochs = n - counts["W"]
    sleep_min = sleep_epochs * epoch_min
    wake_min = counts["W"] * epoch_min
    sleep_eff = (sleep_min / duration_min) if duration_min > 0 else None

    # sleep onset latency (minutes until first non-W)
    sol_min = None
    for i, s in enumerate(stages):
        if s != "W":
            sol_min = i * epoch_min
            break
    if sol_min is None:
        sol_min = duration_min

    awakenings = 0
    for i in range(1, n):
        if stages[i] == "W" and stages[i - 1] != "W":
            awakenings += 1

    transitions = 0
    for i in range(1, n):
        if stages[i] != stages[i - 1]:
            transitions += 1
    tph = (transitions / (duration_min / 60.0)) if duration_min > 0 else None

    conf_mean = None
    if "PRED_CONFIDENCE" in df.columns:
        conf = pd.to_numeric(df["PRED_CONFIDENCE"], errors="coerce").dropna()
        if not conf.empty:
            conf_mean = float(conf.mean())

    hr_mean = hr_min = hr_max = None
    if "HR_BPM_SMOOTH" in df.columns:
        hr = pd.to_numeric(df["HR_BPM_SMOOTH"], errors="coerce").dropna()
        if not hr.empty:
            hr_mean = float(hr.mean())
            hr_min = float(hr.min())
            hr_max = float(hr.max())

    rmssd_mean = None
    if "RMSSD_MS" in df.columns:
        rm = pd.to_numeric(df["RMSSD_MS"], errors="coerce").dropna()
        if not rm.empty:
            rmssd_mean = float(rm.mean())

    q_good = q_ok = q_bad = None
    if "PPG_QUALITY" in df.columns:
        q = df["PPG_QUALITY"].astype(str).str.strip().str.title()
        total = len(q)
        if total > 0:
            q_good = float((q == "Good").sum()) / total
            q_ok = float((q == "Ok").sum()) / total
            q_bad = float((q == "Bad").sum()) / total

    start_dt = end_dt = None
    if "ISO_TIME" in df.columns:
        start_dt = _safe_parse_iso(df["ISO_TIME"].iloc[0])
        end_dt = _safe_parse_iso(df["ISO_TIME"].iloc[-1])

    return dict(
        ok=True,
        n_epochs=n,
        epoch_seconds=epoch_seconds,
        duration_min=duration_min,
        sleep_min=sleep_min,
        wake_min=wake_min,
        sleep_eff=sleep_eff,
        sleep_onset_min=sol_min,
        awakenings=awakenings,
        transitions=transitions,
        transitions_per_hr=tph,
        counts=counts,
        unknown_epochs=unk,
        conf_mean=conf_mean,
        hr_mean=hr_mean,
        hr_min=hr_min,
        hr_max=hr_max,
        rmssd_mean=rmssd_mean,
        q_good=q_good,
        q_ok=q_ok,
        q_bad=q_bad,
        start_dt=start_dt,
        end_dt=end_dt,
    )


def build_sleep_recommendations(metrics: dict) -> list[str]:
    """
    Rule-based, personalized recommendations based on THIS run's metrics.
    General wellness guidance only.
    """
    recs: list[str] = []
    if not metrics.get("ok"):
        return ["Not enough data to generate recommendations."]

    n_epochs = int(metrics.get("n_epochs", 0))
    dur_min = float(metrics.get("duration_min", 0.0))
    sleep_min = float(metrics.get("sleep_min", 0.0))
    eff = metrics.get("sleep_eff", None)
    sol = float(metrics.get("sleep_onset_min", 0.0))
    awak = int(metrics.get("awakenings", 0))
    tph = metrics.get("transitions_per_hr", None)

    hr_mean = metrics.get("hr_mean", None)
    rmssd_mean = metrics.get("rmssd_mean", None)
    q_bad = metrics.get("q_bad", None)
    q_good = metrics.get("q_good", None)

    end_dt = metrics.get("end_dt", None)

    target_sleep_min = 8.0 * 60.0
    min_good_sleep_min = 7.0 * 60.0
    max_typical_sleep_min = 9.5 * 60.0

    if n_epochs < 60 or dur_min < 60:
        recs.append(
            "This was a short session (≤30–60 min). Treat the staging as a **sensor / setup check**, not a full-night sleep evaluation."
        )
        recs.append(
            "For real sleep-quality insights, run overnight (ideally 6–9+ hours) and stop in the morning to generate the full report."
        )
        if q_bad is not None and q_bad > 0.40:
            recs.append("PPG quality was often Bad—adjust fit, reduce motion, warm hands, and re-seat sensors for better HR/HRV.")
        return recs[:6]

    if sleep_min < min_good_sleep_min:
        deficit = target_sleep_min - sleep_min
        add_min = int(round(deficit / 15.0)) * 15
        if add_min > 0:
            recs.append(
                f"Sleep duration looks short. Aim for ~7–9 hours; tonight add about **{add_min} minutes** of sleep opportunity "
                f"(earlier bedtime or later wake)."
            )
    elif sleep_min > max_typical_sleep_min:
        recs.append("Sleep opportunity is very long. If you feel groggy, try a slightly shorter but more consistent sleep window.")
    else:
        recs.append("Sleep duration is within a typical 7–9 hour range—keep the schedule consistent.")

    if eff is not None:
        eff_pct = 100.0 * float(eff)
        if eff_pct < 80.0:
            recs.append(
                f"Sleep efficiency was lower (~{eff_pct:.0f}%). Try to reduce disruptions: cooler/darker room, steady noise, and avoid late caffeine/alcohol."
            )
        elif eff_pct < 88.0:
            recs.append(
                f"Sleep efficiency was moderate (~{eff_pct:.0f}%). Small wins: consistent wind-down, stable temperature, and limit late fluids."
            )
        else:
            recs.append(f"Nice: sleep efficiency was high (~{eff_pct:.0f}%). Keep your current routine and wake time consistent.")

    if sol >= 30:
        recs.append("Sleep onset took a while (≥30 min). Try starting your wind-down earlier (dim lights, no phone, relaxing breathing) and keep bedtime consistent.")
    elif sol >= 20:
        recs.append("Sleep onset was a bit slow (≥20 min). A 20–30 minute wind-down (low light + no screens) often helps.")

    if awak >= 4:
        recs.append("Several awakenings were detected. Check comfort factors (temperature, light leaks, noise) and consider reducing fluids right before bed.")
    elif awak >= 2:
        recs.append("A couple awakenings were detected. If you’re not feeling rested, look at temperature/noise and late meals as common triggers.")

    if tph is not None:
        try:
            tphf = float(tph)
            if tphf > 12:
                recs.append("High stage switching suggests restless/unstable sleep (or movement). Stabilize the headband fit and try a calmer pre-sleep routine.")
            elif tphf > 8:
                recs.append("Moderate stage switching. If you wake up tired, prioritize consistency and reduce pre-bed stimulation (screens, intense activity).")
        except Exception:
            pass

    if q_bad is not None and q_bad > 0.40:
        recs.append("PPG quality was often Bad—HR/HRV insights may be unreliable. Adjust fit, reduce motion, warm hands, and re-seat sensors.")
    else:
        if hr_mean is not None:
            try:
                if float(hr_mean) >= 80:
                    recs.append("Average HR during the session was on the higher side. Common causes: stress, late meals, alcohol, warm room—try cooler temp and a lighter evening routine.")
            except Exception:
                pass

        if rmssd_mean is not None and (q_good is None or q_good >= 0.30):
            try:
                if float(rmssd_mean) < 25:
                    recs.append("HRV (RMSSD) was on the lower side for this session. Recovery is often better with consistent sleep timing, reduced late stress, and cooler temperature.")
            except Exception:
                pass

    if end_dt is not None:
        sol_buf = max(15.0, min(60.0, sol))
        target_time_in_bed = target_sleep_min + sol_buf
        suggested_bed = end_dt - timedelta(minutes=target_time_in_bed)
        recs.append(
            f"If you want ~8 hours of sleep and wake around **{end_dt.strftime('%I:%M %p').lstrip('0')}**, "
            f"aim for **lights out ~{suggested_bed.strftime('%I:%M %p').lstrip('0')}** "
            f"(includes ~{int(round(sol_buf))} min for falling asleep)."
        )

    return recs[:8]


def _render_hypnogram_png_from_df(df: pd.DataFrame, epoch_seconds: int = 30) -> bytes:
    """
    Create a dark-themed colored-block hypnogram as PNG bytes (no file needed).
    """
    stages = df["PRED_STAGE"].astype(str).str.strip().str.upper().replace({"REM": "R"}).tolist()
    stage_order = ["W", "R", "N1", "N2", "N3"]
    y_pos = {s: i for i, s in enumerate(stage_order)}
    stage_colors = {"W":"#f59e0b","N1":"#60a5fa","N2":"#22c55e","N3":"#8b5cf6","R":"#ef4444"}
    unknown_color = "#9ca3af"

    n = len(stages)
    epoch_min = epoch_seconds / 60.0

    fig = plt.figure(figsize=(10.6, 2.8), dpi=200, facecolor="#0E1117")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#111827")

    block_h = 0.85
    for i, s in enumerate(stages):
        s2 = s if s in y_pos else "W"
        yp = y_pos.get(s2, y_pos["W"])
        c = stage_colors.get(s2, unknown_color)
        x = i * epoch_min
        ax.add_patch(
            Rectangle(
                (x, yp - block_h / 2),
                epoch_min,
                block_h,
                facecolor=c,
                edgecolor=(1, 1, 1, 0.10),
                linewidth=0.8,
            )
        )

    total_min = n * epoch_min
    ax.set_xlim(0, total_min)
    ax.set_ylim(-0.75, len(stage_order) - 0.25)
    ax.invert_yaxis()

    ax.set_yticks([y_pos[s] for s in stage_order])
    ax.set_yticklabels(stage_order, fontsize=10, color="white")
    ax.set_xlabel("Time (minutes)", color="white", fontsize=10)
    ax.set_title("Hypnogram (Predicted Stages)", color="white", fontsize=12, pad=10)

    ax.grid(True, axis="x", alpha=0.12, color="white", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.15))
    ax.tick_params(colors="white")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _draw_panel(c, x, y, w, h, fill_color, radius=12):
    c.setFillColor(fill_color)
    c.roundRect(x, y, w, h, radius, stroke=0, fill=1)


def _wrap_text_to_width(text: str, max_width_pt: float, font_name: str, font_size: float) -> list[str]:
    """
    Wrap text to a true width in PDF points using ReportLab's stringWidth.
    """
    words = str(text).split()
    if not words:
        return [""]

    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        trial = cur + " " + w
        if stringWidth(trial, font_name, font_size) <= max_width_pt:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _wrap_draw(
    c,
    x,
    y,
    text,
    max_width_pt: float,
    line_h: float = 0.20 * inch,
    font: tuple[str, float] = ("Helvetica", 10),
    color=None,
    bullet: bool = False,
    bottom_y: float | None = None,
):
    """
    Draw wrapped text that will NOT overlap below bottom_y.
    Returns the new y after drawing (or after clipping stops).
    """
    if color is not None:
        c.setFillColor(color)
    c.setFont(font[0], font[1])

    prefix = "• " if bullet else ""
    lines = _wrap_text_to_width(prefix + str(text), max_width_pt, font[0], font[1])

    for ln in lines:
        # stop if we'd draw into the next panel (prevents overlap)
        if bottom_y is not None and y - line_h < bottom_y:
            return y
        c.drawString(x, y, ln)
        y -= line_h

    return y


def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _render_stage_distribution_png(df: pd.DataFrame) -> bytes:
    stage_order = ["W", "N1", "N2", "N3", "R"]
    stage_colors = {"W":"#f59e0b","N1":"#60a5fa","N2":"#22c55e","N3":"#8b5cf6","R":"#ef4444"}

    stages = df["PRED_STAGE"].astype(str).str.strip().str.upper().replace({"REM":"R"}).tolist()
    counts = {k: 0 for k in stage_order}
    for s in stages:
        if s in counts:
            counts[s] += 1
        else:
            counts["W"] += 1  # unknown -> Wake

    x = np.arange(len(stage_order))
    y = [counts[k] for k in stage_order]

    fig = plt.figure(figsize=(8.6, 3.2), dpi=200, facecolor="#0E1117")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#111827")

    ax.bar(x, y, color=[stage_colors[k] for k in stage_order], edgecolor=(1,1,1,0.15), linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_order, color="white")
    ax.tick_params(axis="y", colors="white")
    ax.grid(axis="y", alpha=0.15, color="white")
    for spine in ax.spines.values():
        spine.set_color((1,1,1,0.15))

    ax.set_title("Stage Distribution", color="white", fontsize=12, pad=10)
    ax.set_ylabel("Epoch count", color="white", fontsize=10)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def _render_hr_trend_png(df: pd.DataFrame, epoch_seconds: int = 30) -> bytes | None:
    if "HR_BPM_SMOOTH" not in df.columns:
        return None
    hr = pd.to_numeric(df["HR_BPM_SMOOTH"], errors="coerce").values
    if np.all(np.isnan(hr)):
        return None

    t_min = np.arange(len(df)) * (float(epoch_seconds) / 60.0)

    fig = plt.figure(figsize=(8.6, 3.2), dpi=200, facecolor="#0E1117")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#111827")

    ax.plot(t_min, hr, linewidth=2.0)
    ax.set_title("Heart Rate Trend (smooth)", color="white", fontsize=12, pad=10)
    ax.set_xlabel("Time (min)", color="white", fontsize=10)
    ax.set_ylabel("BPM", color="white", fontsize=10)
    ax.grid(True, alpha=0.15, color="white")
    for spine in ax.spines.values():
        spine.set_color((1,1,1,0.15))
    ax.tick_params(colors="white")

    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def _render_ppg_quality_png(df: pd.DataFrame, epoch_seconds: int = 30) -> bytes | None:
    if "PPG_QUALITY_SCORE" not in df.columns:
        return None
    q = pd.to_numeric(df["PPG_QUALITY_SCORE"], errors="coerce").values
    if np.all(np.isnan(q)):
        return None

    t_min = np.arange(len(df)) * (float(epoch_seconds) / 60.0)

    fig = plt.figure(figsize=(8.6, 3.2), dpi=200, facecolor="#0E1117")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#111827")

    ax.plot(t_min, q, linewidth=2.0)
    ax.set_ylim(0, 1)
    ax.set_title("PPG Signal Quality (0–1)", color="white", fontsize=12, pad=10)
    ax.set_xlabel("Time (min)", color="white", fontsize=10)
    ax.set_ylabel("Quality score", color="white", fontsize=10)
    ax.grid(True, alpha=0.15, color="white")
    for spine in ax.spines.values():
        spine.set_color((1,1,1,0.15))
    ax.tick_params(colors="white")

    fig.tight_layout()
    return _fig_to_png_bytes(fig)

def save_run_report_pdf(
    epochs_csv_path: Path,
    out_pdf_path: Path,
    title: str = "Muse Sleep Run Report",
    epoch_seconds: int = 30,
) -> Path:
    epochs_csv_path = Path(epochs_csv_path)
    out_pdf_path = Path(out_pdf_path)

    df = pd.read_csv(epochs_csv_path)
    m = compute_sleep_metrics_from_epochs(df, epoch_seconds=epoch_seconds)
    recs = build_sleep_recommendations(m)

    # images
    hyp_png = _render_hypnogram_png_from_df(df, epoch_seconds=epoch_seconds)
    dist_png = _render_stage_distribution_png(df)
    hr_png = _render_hr_trend_png(df, epoch_seconds=epoch_seconds)
    q_png = _render_ppg_quality_png(df, epoch_seconds=epoch_seconds)

    # style
    bg = colors.HexColor("#0E1117")
    panel = colors.HexColor("#111827")
    txt = colors.whitesmoke
    muted = colors.Color(1, 1, 1, alpha=0.78)

    # ALWAYS long version (letter)
    W, H = letter
    c = canvas.Canvas(str(out_pdf_path), pagesize=letter)

    # -----------------------------
    # Helpers: CLIPPING + WRAPPING
    # -----------------------------
    def _clip_round_rect(_c: canvas.Canvas, x, y, w, h, r):
        p = _c.beginPath()
        p.roundRect(x, y, w, h, r)
        _c.clipPath(p, stroke=0, fill=0)

    def _wrap_lines_to_width(text: str, font_name: str, font_size: float, max_w_pts: float):
        if not text:
            return []
        words = str(text).replace("\n", " ").split()
        lines = []
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            if stringWidth(test, font_name, font_size) <= max_w_pts:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                if stringWidth(w, font_name, font_size) <= max_w_pts:
                    cur = w
                else:
                    chunk = ""
                    for ch in w:
                        t2 = chunk + ch
                        if stringWidth(t2, font_name, font_size) <= max_w_pts:
                            chunk = t2
                        else:
                            if chunk:
                                lines.append(chunk)
                            chunk = ch
                    cur = chunk
        if cur:
            lines.append(cur)
        return lines

    def _sanitize_text(s: str) -> str:
        """
        ReportLab doesn't render markdown. Strip common markdown tokens so we don't
        print '**' literally and cause ugly wraps/overlaps.
        """
        if s is None:
            return ""
        t = str(s)
        t = t.replace("**", "")
        t = t.replace("__", "")
        t = t.replace("`", "")
        return t

    def _draw_wrapped_block(
        _c: canvas.Canvas,
        x_left: float,
        y_top: float,
        text: str,
        max_w_pts: float,
        min_y: float,
        font_name: str,
        font_size: float,
        line_h: float,
        color,
        bullet: bool = False,
        bullet_indent_pts: float = 10.0,
        gap_after_pts: float = 4.0,
        allow_ellipsis: bool = True,
    ) -> float:
        _c.setFont(font_name, font_size)
        _c.setFillColor(color)

        text = _sanitize_text(text)

        x_text = x_left
        wrap_w = max_w_pts
        if bullet:
            x_text = x_left + bullet_indent_pts
            wrap_w = max(10.0, max_w_pts - bullet_indent_pts)

        lines = _wrap_lines_to_width(text, font_name, font_size, wrap_w)

        max_lines = int(max(0, (y_top - min_y) // line_h))
        if max_lines <= 0:
            return y_top

        clipped = False
        if len(lines) > max_lines:
            clipped = True
            lines = lines[:max_lines]

        y = y_top
        for i, ln in enumerate(lines):
            y -= line_h
            if y < min_y:
                break
            if bullet and i == 0:
                _c.drawString(x_left, y, "•")
                _c.drawString(x_text, y, ln)
            else:
                _c.drawString(x_text if bullet else x_left, y, ln)

        if clipped and allow_ellipsis:
            y_last = y_top - max_lines * line_h
            if y_last >= min_y and lines:
                ell = "…"
                last = lines[-1]
                s = last
                while stringWidth(s + ell, font_name, font_size) > wrap_w and len(s) > 0:
                    s = s[:-1]
                _c.drawString(x_text if bullet else x_left, y_last, s + ell)

        return y - gap_after_pts

    def _draw_panel_and_clip(_c, x, y, w, h, fill_color, radius=12):
        _draw_panel(_c, x, y, w, h, fill_color, radius=radius)
        _c.saveState()
        _clip_round_rect(_c, x, y, w, h, radius)

    def _end_clip(_c):
        _c.restoreState()

    # -----------------------------
    # PAGE 1
    # -----------------------------
    c.setFillColor(bg)
    c.rect(0, 0, W, H, stroke=0, fill=1)

    # Header
    c.setFillColor(txt)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(0.75 * inch, H - 0.75 * inch, title)

    c.setFont("Helvetica", 10)
    when = f"Source: {epochs_csv_path.name}"
    if m.get("start_dt") and m.get("end_dt"):
        when = (
            f"{m['start_dt'].strftime('%Y-%m-%d %I:%M %p').lstrip('0')}"
            f"  →  {m['end_dt'].strftime('%Y-%m-%d %I:%M %p').lstrip('0')}"
        )
    c.setFillColor(muted)
    c.drawString(0.75 * inch, H - 1.02 * inch, when)

    # Badge (always "RUN REPORT")
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(colors.Color(1, 1, 1, alpha=0.10))
    c.roundRect(W - 2.45 * inch, H - 1.15 * inch, 1.70 * inch, 0.28 * inch, 7, stroke=0, fill=1)
    c.setFillColor(colors.Color(1, 1, 1, alpha=0.85))
    c.drawCentredString(W - 1.60 * inch, H - 1.07 * inch, "RUN REPORT")

    # Layout constants
    pad = 0.75 * inch
    gap = 0.32 * inch
    inner_pad = 0.25 * inch

    # Hypnogram panel
    hyp_x = pad
    hyp_w = W - 2 * pad
    hyp_h = 1.85 * inch
    hyp_y = H - pad - 1.55 * inch - hyp_h

    _draw_panel_and_clip(c, hyp_x, hyp_y, hyp_w, hyp_h, panel, radius=12)
    c.drawImage(
        ImageReader(io.BytesIO(hyp_png)),
        hyp_x + 0.18 * inch,
        hyp_y + 0.18 * inch,
        width=hyp_w - 0.36 * inch,
        height=hyp_h - 0.36 * inch,
        mask="auto",
        preserveAspectRatio=True,
        anchor="c",
    )
    _end_clip(c)

    # Metrics + Recs area (guaranteed to fit above footer)
    footer_y = 0.55 * inch
    met_h = 2.70 * inch
    met_y = max(footer_y + 0.35 * inch, hyp_y - gap - met_h)

    met_x = pad
    met_w = (W - 2 * pad) * 0.52

    rec_x = met_x + met_w + 0.30 * inch
    rec_w = (W - 2 * pad) - (met_w + 0.30 * inch)
    rec_y = met_y
    rec_h = met_h

    _draw_panel(c, met_x, met_y, met_w, met_h, panel, radius=12)
    _draw_panel(c, rec_x, rec_y, rec_w, rec_h, panel, radius=12)

    # -------- Metrics (CLIPPED so it can never spill into footer) --------
    c.setFillColor(txt)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(met_x + inner_pad, met_y + met_h - 0.45 * inch, "Key metrics")

    c.saveState()
    _clip_round_rect(c, met_x, met_y, met_w, met_h, 12)

    epoch_min = float(epoch_seconds) / 60.0
    n_epochs = int(m.get("n_epochs", len(df)))
    dur_min = float(m.get("duration_min", 0.0))
    sleep_min = float(m.get("sleep_min", 0.0))
    wake_min = float(m.get("wake_min", 0.0))
    eff = m.get("sleep_eff", None)
    counts = m.get("counts", {}) or {}

    lines = [
        f"Epochs: {n_epochs}   |   Duration: {dur_min:.1f} min",
        f"Sleep time: {sleep_min:.1f} min   |   Wake: {wake_min:.1f} min",
        f"Sleep efficiency: {'—' if eff is None else f'{100 * float(eff):.1f}%'}",
        f"Sleep onset: {float(m.get('sleep_onset_min', 0.0)):.1f} min   |   Awakenings: {int(m.get('awakenings', 0))}",
        (
            f"Stage mins: "
            f"W {counts.get('W', 0) * epoch_min:.1f} | "
            f"N1 {counts.get('N1', 0) * epoch_min:.1f} | "
            f"N2 {counts.get('N2', 0) * epoch_min:.1f} | "
            f"N3 {counts.get('N3', 0) * epoch_min:.1f} | "
            f"R {counts.get('R', 0) * epoch_min:.1f}"
        ),
    ]
    if m.get("hr_mean") is not None:
        lines.append(
            f"HR smooth: avg {float(m['hr_mean']):.0f} bpm  (min {float(m['hr_min']):.0f}, max {float(m['hr_max']):.0f})"
        )
    if m.get("rmssd_mean") is not None:
        lines.append(
            f"RMSSD: avg {float(m['rmssd_mean']):.0f} ms (best when PPG quality is Good)"
        )

    text_left = met_x + inner_pad
    text_top = met_y + met_h - 0.70 * inch
    text_min_y = met_y + 0.35 * inch
    text_w = met_w - 2 * inner_pad

    yy = text_top
    for ln in lines:
        yy2 = _draw_wrapped_block(
            c,
            text_left,
            yy,
            ln,
            max_w_pts=text_w,
            min_y=text_min_y,
            font_name="Helvetica",
            font_size=10,
            line_h=0.20 * inch,
            color=muted,
            bullet=False,
            gap_after_pts=3.0,
        )
        if yy2 == yy:
            break
        yy = yy2

    c.restoreState()

    # -------- Recommendations (CLIPPED + markdown stripped) --------
    c.setFillColor(txt)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(rec_x + inner_pad, rec_y + rec_h - 0.45 * inch, "Recommendations")

    c.saveState()
    _clip_round_rect(c, rec_x, rec_y, rec_w, rec_h, 12)

    r_left = rec_x + inner_pad
    r_top = rec_y + rec_h - 0.70 * inch
    r_min_y = rec_y + 0.35 * inch
    r_w = rec_w - 2 * inner_pad

    yy = r_top
    for r in recs:
        yy2 = _draw_wrapped_block(
            c,
            r_left,
            yy,
            r,
            max_w_pts=r_w,
            min_y=r_min_y,
            font_name="Helvetica",
            font_size=10,
            line_h=0.20 * inch,
            color=muted,
            bullet=True,
            bullet_indent_pts=12.0,
            gap_after_pts=6.0,
        )
        if yy2 == yy:
            break
        yy = yy2

    c.restoreState()

    # Footer (now guaranteed not to overlap because panels are clipped + met_y respects footer)
    c.setFont("Helvetica-Oblique", 8.5)
    c.setFillColor(colors.Color(1, 1, 1, alpha=0.55))
    c.drawString(
        pad,
        0.45 * inch,
        "General wellness guidance only (not medical advice).",
    )

    c.showPage()

    # -----------------------------
    # PAGE 2 (ALWAYS included)
    # -----------------------------
    c.setFillColor(bg)
    c.rect(0, 0, W, H, stroke=0, fill=1)

    c.setFillColor(txt)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(pad, H - 0.75 * inch, "Visual Summary")

    c.setFillColor(muted)
    c.setFont("Helvetica", 10)
    c.drawString(pad, H - 1.02 * inch, "Stage composition, heart rate trend, and signal quality (when available).")

    gap2 = 0.35 * inch
    card_w = (W - 2 * pad - gap2) / 2.0
    card_h = (H - 2 * pad - 1.25 * inch - gap2) / 2.0

    # top-left: stage distribution
    x1 = pad
    y1 = H - pad - 1.25 * inch - card_h
    _draw_panel_and_clip(c, x1, y1, card_w, card_h, panel, radius=12)
    c.drawImage(
        ImageReader(io.BytesIO(dist_png)),
        x1 + 0.18 * inch,
        y1 + 0.18 * inch,
        width=card_w - 0.36 * inch,
        height=card_h - 0.36 * inch,
        mask="auto",
        preserveAspectRatio=True,
        anchor="c",
    )
    _end_clip(c)

    # top-right: HR trend (or placeholder)
    x2 = pad + card_w + gap2
    y2 = y1
    _draw_panel(c, x2, y2, card_w, card_h, panel, radius=12)
    if hr_png is not None:
        _draw_panel_and_clip(c, x2, y2, card_w, card_h, panel, radius=12)
        c.drawImage(
            ImageReader(io.BytesIO(hr_png)),
            x2 + 0.18 * inch,
            y2 + 0.18 * inch,
            width=card_w - 0.36 * inch,
            height=card_h - 0.36 * inch,
            mask="auto",
            preserveAspectRatio=True,
            anchor="c",
        )
        _end_clip(c)
    else:
        c.setFillColor(muted)
        c.setFont("Helvetica", 11)
        c.drawString(x2 + inner_pad, y2 + card_h - 0.55 * inch, "Heart rate trend")
        c.setFont("Helvetica", 10)
        c.drawString(x2 + inner_pad, y2 + card_h - 0.85 * inch, "Not available (missing HR_BPM_SMOOTH).")

    # bottom-left: PPG quality (or placeholder)
    x3 = pad
    y3 = pad
    _draw_panel(c, x3, y3, card_w, card_h, panel, radius=12)
    if q_png is not None:
        _draw_panel_and_clip(c, x3, y3, card_w, card_h, panel, radius=12)
        c.drawImage(
            ImageReader(io.BytesIO(q_png)),
            x3 + 0.18 * inch,
            y3 + 0.18 * inch,
            width=card_w - 0.36 * inch,
            height=card_h - 0.36 * inch,
            mask="auto",
            preserveAspectRatio=True,
            anchor="c",
        )
        _end_clip(c)
    else:
        c.setFillColor(muted)
        c.setFont("Helvetica", 11)
        c.drawString(x3 + inner_pad, y3 + card_h - 0.55 * inch, "PPG quality trend")
        c.setFont("Helvetica", 10)
        c.drawString(x3 + inner_pad, y3 + card_h - 0.85 * inch, "Not available (missing PPG_QUALITY_SCORE).")

    # bottom-right: hypnogram thumbnail
    x4 = x2
    y4 = y3
    _draw_panel_and_clip(c, x4, y4, card_w, card_h, panel, radius=12)
    c.drawImage(
        ImageReader(io.BytesIO(hyp_png)),
        x4 + 0.18 * inch,
        y4 + 0.18 * inch,
        width=card_w - 0.36 * inch,
        height=card_h - 0.36 * inch,
        mask="auto",
        preserveAspectRatio=True,
        anchor="c",
    )
    _end_clip(c)

    c.showPage()

    c.save()
    return out_pdf_path

# -----------------------------
# Session state
# -----------------------------
if "eeg_receiver" not in st.session_state:
    st.session_state["eeg_receiver"] = LSLReceiver(stream_type="EEG", fs=EEG_FS, n_chan=len(EEG_CHANNEL_NAMES))
eeg_receiver: LSLReceiver = st.session_state["eeg_receiver"]

if "ppg_receiver" not in st.session_state:
    st.session_state["ppg_receiver"] = LSLReceiver(stream_type="PPG", fs=PPG_FS, n_chan=len(PPG_CHANNEL_NAMES))
ppg_receiver: LSLReceiver = st.session_state["ppg_receiver"]

def get_default_model_run_dir() -> Path:
    """
    Edit this to your local model folder path on your machine.
    For example:
        Path("F:\\! Senior Design Project\\XGBoost\\runs\\run_YYYYMMDD_HHMMSS")    """
    return Path(r"F:\! Senior Design Project\physionet\physionet.org\files\XGBoost\runs\run_20260210_025057")
    # Fallback: same folder as muse.py (useful if you copy model files next to muse.py)
    #return Path(__file__).resolve().parent

if "live_infer" not in st.session_state:
    st.session_state["live_infer"] = LiveSleepInference(get_default_model_run_dir())

live_infer: LiveSleepInference = st.session_state["live_infer"]

if "collector" not in st.session_state:
    st.session_state["collector"] = EpochCollector(eeg_receiver, ppg_receiver, cfg_store, infer_engine=live_infer)
collector: EpochCollector = st.session_state["collector"]
collector.cfg_store = cfg_store
collector.infer_engine = live_infer

if "muselsl_proc" not in st.session_state:
    st.session_state["muselsl_proc"] = None

if "scan_state" not in st.session_state:
    st.session_state["scan_state"] = {"scanning": False, "macs": [], "last_raw": ""}

if "ui_state" not in st.session_state:
    st.session_state["ui_state"] = {"muse_found": False, "lsl_connected_eeg": False, "lsl_connected_ppg": False}

# -----------------------------
# Controls (top)
# -----------------------------
st.subheader("Muse Control")
c_scan, c_conn, c_pause = st.columns([1, 2, 1], gap="large")

scan = st.session_state["scan_state"]
ui = st.session_state["ui_state"]

with c_scan:
    if st.button("🔎 Scan for Muse", use_container_width=True):
        scan["scanning"] = True
        scan["macs"] = []
        scan["last_raw"] = ""
        st.session_state["scan_state"] = scan

        with st.spinner("Scanning for Muse headbands..."):
            _, macs, raw = scan_for_muse(timeout_s=4)

        scan["scanning"] = False
        scan["macs"] = macs
        scan["last_raw"] = raw
        st.session_state["scan_state"] = scan

        ui["muse_found"] = bool(macs)
        st.session_state["ui_state"] = ui

        st.rerun()

    scan = st.session_state["scan_state"]
    ui = st.session_state["ui_state"]

    if scan.get("scanning", False):
        st.info("Scanning…")
    elif ui.get("lsl_connected_eeg", False) or ui.get("lsl_connected_ppg", False):
        st.success("Muse connected ✅")
    elif scan.get("macs"):
        st.success("Muse detected ✅")
    else:
        st.warning("No Muse detected yet")

with c_conn:
    if st.button("▶️ Start + Connect", use_container_width=True):
        started_new = False

        p = st.session_state.get("muselsl_proc", None)
        if p is None or p.poll() is not None:
            st.session_state["muselsl_proc"] = start_muselsl_stream()
            started_new = True

        # Warmup so streams actually exist before first fetch attempts
        if started_new:
            time.sleep(1.2)

        ok_eeg, ok_ppg = connect_with_retries(eeg_receiver, ppg_receiver)

        ui = st.session_state["ui_state"]
        ui["lsl_connected_eeg"] = bool(ok_eeg)
        ui["lsl_connected_ppg"] = bool(ok_ppg)
        st.session_state["ui_state"] = ui

        st.rerun()

    # side-by-side status banners
    ui = st.session_state["ui_state"]
    b_eeg, b_ppg = st.columns(2, gap="small")
    with b_eeg:
        if ui.get("lsl_connected_eeg", False):
            st.success("EEG ✅")
        else:
            st.warning("EEG not connected")
    with b_ppg:
        if ui.get("lsl_connected_ppg", False):
            st.success("PPG ✅")
        else:
            st.warning("PPG not connected")

with c_pause:
    paused_label = "⏸ Pause" if (eeg_receiver.running and not eeg_receiver.paused) else "▶ Resume"
    if st.button(paused_label, use_container_width=True):
        if not eeg_receiver.running:
            st.info("Connect first to pause/resume.")
        else:
            if eeg_receiver.paused:
                eeg_receiver.resume()
                ppg_receiver.resume()
            else:
                eeg_receiver.pause()
                ppg_receiver.pause()

        st.rerun()

with st.expander("Advanced", expanded=False):
    if st.button("⏹ Stop + Clear Data", use_container_width=True):
        collector.stop()
        eeg_receiver.stop_and_clear()
        ppg_receiver.stop_and_clear()
        stop_proc(st.session_state["muselsl_proc"])
        st.session_state["muselsl_proc"] = None
        st.session_state["ui_state"] = {"muse_found": False, "lsl_connected_eeg": False, "lsl_connected_ppg": False}
        st.session_state["scan_state"] = {"scanning": False, "macs": [], "last_raw": ""}
        st.warning("Stopped and cleared buffer.")
        st.rerun()

# MAC list
scan = st.session_state["scan_state"]
if scan.get("macs"):
    st.write("Detected MAC address(es):")
    for m in scan["macs"]:
        st.code(m)

# Running badge
if eeg_receiver.running and eeg_receiver.paused:
    st.markdown('<div class="run-badge"><span class="small-muted">Paused</span></div>', unsafe_allow_html=True)
elif eeg_receiver.running:
    st.markdown('<div class="run-badge"><div class="spinner"></div><span class="small-muted">Running</span></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="run-badge"><span class="small-muted">Idle</span></div>', unsafe_allow_html=True)

if eeg_receiver.running and eeg_receiver.sample_count > 50:
    st.success("EEG samples are flowing ✅")
elif eeg_receiver.running:
    st.info("EEG connected but waiting for samples…")
else:
    st.info("Not receiving EEG samples yet.")

if ppg_receiver.running and ppg_receiver.sample_count > 20:
    st.success("PPG samples are flowing ✅")
elif ppg_receiver.running:
    st.info("PPG connected but waiting for samples…")
else:
    st.info("Not receiving PPG samples yet.")

# Debug panel
if show_debug:
    with st.expander("Debug", expanded=True):
        st.write("CFG (current):", cfg_store.get())
        st.write("EEG running/paused:", eeg_receiver.running, eeg_receiver.paused)
        st.write("EEG sample_count:", eeg_receiver.sample_count, "last_ts:", eeg_receiver.last_ts, "err:", eeg_receiver.last_error)
        st.write("PPG running/paused:", ppg_receiver.running, ppg_receiver.paused)
        st.write("PPG sample_count:", ppg_receiver.sample_count, "last_ts:", ppg_receiver.last_ts, "err:", ppg_receiver.last_error)
        st.write("PPG meta:", ppg_receiver.stream_meta)
        st.write("collector:", collector.status())

st.divider()

st.markdown('<div id="dc-row-anchor"></div>', unsafe_allow_html=True)

# -----------------------------
# Main layout: Charts (left) + Data Collection panel (right)
# -----------------------------
left, right = st.columns([3, 1], gap="large")

@st.fragment(run_every=5.0)
def render_collection_status(collector: EpochCollector):
    col_stat = collector.status()
    st.write("Epoch rows written:", col_stat["rows_written"])
    if col_stat["last_write_iso"]:
        st.write("Last epoch write:", col_stat["last_write_iso"])
    if col_stat["csv_path"]:
        st.code(Path(col_stat["csv_path"]).name)
    if col_stat["last_err"]:
        st.error(col_stat["last_err"])


with right:
    dc_panel = st.container()

    with dc_panel:
        st.subheader("Data Collection")

        col_stat = collector.status()
        collecting = bool(col_stat["collecting"])

        st.write("")

        # ---- Snapshot ----
        if st.button("💾 Save Snapshot", key="dc_snapshot", use_container_width=True):
            if not eeg_receiver.running:
                st.info("Connect first to save a snapshot.")
            else:
                p = save_snapshot_csv(
                    eeg_receiver,
                    ppg_receiver,
                    float(cfg_store.get()["rolling_sec"]),
                )
                if p is None:
                    st.error("No EEG data available to snapshot yet.")
                else:
                    st.success(f"Saved snapshot:\n{p.name}")
        
        # ---- Start / Stop ----
        b1, b2 = st.columns(2, gap="small")
        
        with b1:
            start_clicked = st.button(
                "▶ Start",
                key="dc_start",
                use_container_width=True,
                disabled=collecting,
            )

        with b2:
            stop_clicked = st.button(
                "⏹ Stop",
                key="dc_stop",
                use_container_width=True,
                disabled=not collecting,
            )

        # Handle Start
        if start_clicked:
            try:
                ok = collector.start()
                if not ok:
                    st.error(
                        collector.status().get("last_err")
                        or "Could not start collecting."
                    )
                else:
                    st.success("Collecting started (30s epochs).")
                st.rerun()
            except Exception as e:
                st.error(f"Start crashed: {e!r}")
                st.exception(e)
                
        # Handle Stop (IMPORTANT: keep this inside the same panel/scope)
        
        if stop_clicked:
            # Stop + wait for thread
            collector.stop(reason="Stop button pressed")
            collector.join(timeout=15.0)

            out_dir = collector.session_dir if collector.session_dir is not None else ensure_sleep_data_dir()

            # Save prediction summary image into the same run folder
            img_path = None
            try:
                img_path = save_prediction_summary_image(live_infer, out_dir, max_epochs=120)
            except Exception as e:
                st.error(f"Stopped collection, but could not save prediction summary image: {e}")

            if img_path is not None:
                st.success(f"Saved prediction summary image: {img_path.name}")
            else:
                st.info("No prediction summary image saved (no predictions yet).")

            # Save hypnogram into the same run folder (from the epochs CSV)
            hyp_path = None
            try:
                # IMPORTANT: call the collector method that stops/joins then saves
                # (It already joined above; that's fine — it will just no-op quickly.)
                hyp_path = collector.finalize_and_save_hypnogram()
            except Exception as e:
                st.error(f"Stopped collection, but could not save hypnogram: {e}")

            if hyp_path is not None:
                st.success(f"Saved hypnogram: {hyp_path.name}")
            else:
                st.warning("Hypnogram not saved. Check collector_log.txt for the exact reason.")

            # --- NEW: Save combined PDF report ---
            try:
                # Find the epochs CSV for this run
                epochs_csv = collector.csv_path  # should be set for the run
                if epochs_csv and Path(epochs_csv).exists():
                    pdf_path = Path(out_dir) / "run_report.pdf"
                    save_run_report_pdf(Path(epochs_csv), pdf_path, title="Muse Sleep Run Report", epoch_seconds=int(EPOCH_SEC))
                    st.success(f"Saved run report PDF: {pdf_path.name}")
                else:
                    st.warning("Run report not saved (epochs CSV not found).")
            except Exception as e:
                st.error(f"Could not save run report PDF: {e!r}")

            # Show log location (super useful)
            stat = collector.status()
            if stat.get("log_path"):
                st.caption(f"Debug log: {Path(stat['log_path']).name} (in the run folder)")

            st.rerun()

        # ---- Status badge (inside panel) ----
        col_stat = collector.status()
        collecting = bool(col_stat["collecting"])

        if collecting:
            st.markdown(
                '<div class="run-badge"><div class="spinner"></div>'
                '<span class="small-muted">Collecting… (30s epochs)</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="run-badge"><span class="small-muted">Not collecting</span></div>',
                unsafe_allow_html=True,
            )

        st.caption("Files save to a folder named **live prediction runs** next to muse.py.")
        render_collection_status(collector)

    st.markdown("### Live Sleep Stage Inference")

    if "model_run_dir_str" not in st.session_state:
        st.session_state["model_run_dir_str"] = str(get_default_model_run_dir())

    new_model_dir = st.text_input(
        "Model run folder (contains model_bundle.joblib and/or stage JSONs)",
        key="model_run_dir_str",
    )

    c_reload_model, c_model_status = st.columns([1, 1], gap="small")
    with c_reload_model:
        if st.button("🔄 Reload Model", use_container_width=True):
            st.session_state["live_infer"] = LiveSleepInference(Path(new_model_dir.strip()))
            live_infer = st.session_state["live_infer"]
            collector.infer_engine = live_infer
            st.rerun()

    with c_model_status:
        inf_stat = live_infer.status()
        if inf_stat["ready"]:
            st.success("Model ready ✅")
        else:
            st.warning("Model not ready")

    @st.fragment(run_every=5.0)
    def render_live_predictions_panel(infer_engine: LiveSleepInference):
        st.markdown("#### Live Sleep Stage Predictions (30s epochs)")

        s = infer_engine.status()
        if not s["ready"]:
            st.info("Load your DreamT XGBoost model folder to enable live inference.")
            if s.get("last_err"):
                st.error(s["last_err"])
            return

        latest = s.get("latest")
        if latest is None:
            st.caption("No epoch predictions yet. Start data collection to generate 30s epochs.")
            return

        stage_colors = {
            "W":  ("#f59e0b", "rgba(245,158,11,0.14)", "rgba(245,158,11,0.40)"),
            "N1": ("#60a5fa", "rgba(96,165,250,0.14)", "rgba(96,165,250,0.40)"),
            "N2": ("#22c55e", "rgba(34,197,94,0.14)", "rgba(34,197,94,0.40)"),
            "N3": ("#8b5cf6", "rgba(139,92,246,0.14)", "rgba(139,92,246,0.40)"),
            "R":  ("#ef4444", "rgba(239,68,68,0.14)", "rgba(239,68,68,0.40)"),
            "S":  ("#a3a3a3", "rgba(163,163,163,0.12)", "rgba(163,163,163,0.30)"),
        }

        def _stage_style(stage: str):
            return stage_colors.get(stage, ("#d1d5db", "rgba(255,255,255,0.06)", "rgba(255,255,255,0.18)"))

        def _fmt_elapsed_sec(ts_s):
            if ts_s is None:
                return "—"
            try:
                total = int(round(float(ts_s)))
                m = total // 60
                sec = total % 60
                return f"{m:02d}:{sec:02d}"
            except Exception:
                return "—"

        stage = str(latest.get("pred_stage", "—"))
        conf = latest.get("confidence", None)
        conf_txt = "—" if conf is None else f"{100.0 * float(conf):.1f}%"
        epoch_idx = latest.get("epoch_index", "—")
        elapsed_txt = _fmt_elapsed_sec(latest.get("timestamp_s"))
        iso_txt = latest.get("iso_time", "")

        txt_c, bg_c, bd_c = _stage_style(stage)

        st.markdown(
            f"""
            <div style="
                border:1px solid {bd_c};
                border-radius:14px;
                padding:0.8rem 0.9rem;
                background:{bg_c};
                margin-bottom:0.6rem;
            ">
              <div style="font-size:0.82rem; opacity:0.75;">Current epoch prediction</div>
              <div style="display:flex; align-items:baseline; justify-content:space-between; gap:8px; margin-top:2px;">
                <div style="font-size:1.45rem; font-weight:800; color:{txt_c};">{stage}</div>
                <div style="font-size:0.90rem; opacity:0.85;">Conf: <b>{conf_txt}</b></div>
              </div>
              <div style="display:flex; gap:12px; flex-wrap:wrap; margin-top:6px; font-size:0.84rem; opacity:0.78;">
                <span>Epoch: <b>{epoch_idx}</b></span>
                <span>Elapsed: <b>{elapsed_txt}</b></span>
              </div>
              <div style="font-size:0.78rem; opacity:0.60; margin-top:4px;">{iso_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        hist = infer_engine.get_history(30)

        if hist:
            st.markdown("**Recent stage distribution (last 30 epochs)**")

            stage_order = ["W", "N1", "N2", "N3", "R"]
            counts = {k: 0 for k in stage_order}

            for h in hist:
                stg = h.get("pred_stage")
                if stg in counts:
                    counts[stg] += 1

            import plotly.graph_objects as go

            stage_colors = {
                "W": "#f59e0b",
                "N1": "#60a5fa",
                "N2": "#22c55e",
                "N3": "#8b5cf6",
                "R": "#ef4444",
            }

            x_vals = stage_order
            y_vals = [counts[k] for k in stage_order]
            bar_colors = [stage_colors[k] for k in stage_order]

            fig_hist = go.Figure(
                data=[
                    go.Bar(
                        x=x_vals,
                        y=y_vals,
                        marker_color=bar_colors,
                        text=y_vals,
                        textposition="outside",
                        hovertemplate="Stage %{x}<br>Count: %{y}<extra></extra>",
                    )
                ]
            )

            fig_hist.update_layout(
                height=220,
                margin=dict(l=20, r=10, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(color="rgba(255,255,255,0.92)"),
                xaxis=dict(
                    title=None,
                    showgrid=False,
                    zeroline=False,
                    tickfont=dict(size=12),
                ),
                yaxis=dict(
                    title=None,
                    rangemode="tozero",
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.08)",
                    zeroline=False,
                    dtick=1,
                ),
                showlegend=False,
            )

            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

            st.markdown("**Epoch strip (oldest → newest)**")

            strip_blocks = []
            for i, h in enumerate(hist):
                stg = str(h.get("pred_stage", "—"))
                idx = h.get("epoch_index", "")
                conf = h.get("confidence", None)

                color = stage_colors.get(stg, "#9ca3af")
                conf_txt = "—" if conf is None else f"{100.0 * float(conf):.1f}%"

                block = (
                    f'<div title="Epoch #{idx} | Stage: {stg} | Conf: {conf_txt}" '
                    f'style="width:12px;height:22px;border-radius:4px;'
                    f'background:{color};border:1px solid rgba(255,255,255,0.18);'
                    f'box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);"></div>'
                )
                strip_blocks.append(block)

            st.markdown(
                '<div style="display:flex;align-items:center;gap:4px;flex-wrap:nowrap;'
                'overflow-x:auto;padding:6px 2px 2px 2px;margin-bottom:0.25rem;">'
                + "".join(strip_blocks)
                + "</div>",
                unsafe_allow_html=True,
            )

            legend_html = []
            for k in stage_order:
                legend_html.append(
                    f'<span style="display:inline-flex;align-items:center;gap:6px;'
                    f'margin:2px 10px 2px 0;font-size:0.80rem;opacity:0.9;">'
                    f'<span style="width:10px;height:10px;border-radius:3px;display:inline-block;'
                    f'background:{stage_colors[k]};border:1px solid rgba(255,255,255,0.18);"></span>{k}</span>'
                )

            st.markdown(
                '<div style="margin-bottom:0.35rem;">' + "".join(legend_html) + "</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Details (probabilities + table)", expanded=False):
            st.write({
                "pA_S": latest.get("pA_S"),
                "pA_W": latest.get("pA_W"),
                "pB_NREM": latest.get("pB_NREM"),
                "pB_R": latest.get("pB_R"),
                "pC_N1": latest.get("pC_N1"),
                "pC_N2": latest.get("pC_N2"),
                "pC_N3": latest.get("pC_N3"),
            })

            st.markdown("**Inference diagnostics**")
            st.write({
                "expected_features": latest.get("diag_n_expected"),
                "present_before_align": latest.get("diag_n_present_before_align"),
                "missing_before_align": latest.get("diag_n_missing_before_align"),
                "nonzero_after_align": latest.get("diag_n_nonzero"),
                "nonzero_fraction": (
                    None if latest.get("diag_nonzero_frac") is None
                    else round(float(latest.get("diag_nonzero_frac")), 4)
                ),
                "nonzero_cols_preview": latest.get("diag_nonzero_cols_preview"),
            })

            if hist:
                dfh = pd.DataFrame(hist)
                if "confidence" in dfh.columns:
                    dfh["confidence"] = dfh["confidence"].apply(
                        lambda v: None if v is None else round(float(v), 3)
                    )

                show_cols = [
                    c for c in [
                        "epoch_index",
                        "timestamp_s",
                        "pred_stage",
                        "confidence",
                        "iso_time",
                        "diag_n_nonzero",
                        "diag_nonzero_frac",
                    ]
                    if c in dfh.columns
                ]
                st.dataframe(dfh.iloc[::-1][show_cols], use_container_width=True, height=220)

    render_live_predictions_panel(live_infer)

with left:
    port = ensure_data_server(eeg_receiver, ppg_receiver, cfg_store)

    html_template = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {
      margin:0; padding:0;
      background:__DARK_BG__;
      color:__DARK_TEXT__;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }
    .grid { display:grid; grid-template-columns:1fr; gap:40px; }

    .card {
      background:__DARK_BG__;
      border-radius:12px;
      overflow: visible;
      padding-top: 6px;
      padding-bottom: 8px;
    }
    #ppgcard { overflow: visible; position: relative; z-index: 20; }

    .ppghead{
      display:flex; align-items:center; justify-content:space-between;
      padding: 10px 12px 10px 12px;
      gap: 14px;
      flex-wrap: wrap;
      position: relative;
      z-index: 2000;
      overflow: visible;
    }
    .ppghead .left { font-size: 1.05rem; opacity:0.95; }
    .ppghead .right{
      font-size: 0.95rem;
      opacity:0.88;
      display:flex;
      gap:14px;
      align-items:baseline;
      flex-wrap:wrap;
      position: relative;
      z-index: 2000;
      overflow: visible;
    }
    .pill{
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 0.92rem;
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(255,255,255,0.06);
      white-space: nowrap;
    }
    .muted { opacity:0.85; }

    .charthead{
      padding: 12px 12px 4px 12px;
      font-size: 1.02rem;
      opacity: 0.95;
      position: relative;
      z-index: 2000;
      overflow: visible;
    }

    .tip {
      position: relative;
      display: inline-flex;
      align-items: baseline;
      gap: 6px;
      cursor: help;
      border-bottom: 1px dotted rgba(255,255,255,0.35);
      z-index: 3000;
      overflow: visible;
    }
    .tip .bubble {
      position: absolute;
      left: 0;
      top: 130%;
      min-width: 220px;
      max-width: 320px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(17,24,39,0.94);
      border: 1px solid rgba(255,255,255,0.14);
      box-shadow: 0 10px 22px rgba(0,0,0,0.35);
      color: rgba(255,255,255,0.92);
      font-size: 0.92rem;
      line-height: 1.25rem;
      z-index: 4000;
      opacity: 0;
      transform: translateY(-4px);
      pointer-events: none;
      transition: opacity 120ms ease, transform 120ms ease;
    }
    .tip:hover .bubble {
      opacity: 1;
      transform: translateY(0px);
    }

    .tip.tip-rmssd .bubble {
      left: auto !important;
      right: 0 !important;
    }

.dc-panel-hook { display: none; }

div[data-testid="stHorizontalBlock"],
div[data-testid="column"],
div[data-testid="stVerticalBlock"],
div[data-testid="stMainBlockContainer"],
div[data-testid="stBlock"],
div[data-testid="stElementContainer"],
div.block-container {
  overflow: visible !important;
}

div[data-testid="column"]:has(.dc-panel-hook) {
  position: sticky !important;
  top: 0.85rem !important;
  align-self: flex-start !important;
  z-index: 30 !important;
}

div[data-testid="column"]:has(.dc-panel-hook) > div[data-testid="stVerticalBlock"] {
  background: rgba(17, 24, 39, 0.72);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 0.85rem 0.9rem 0.9rem 0.9rem;
  box-shadow: 0 8px 20px rgba(0,0,0,0.22);

  max-height: calc(100vh - 1.7rem);
  overflow: auto !important;
}

div[data-testid="column"]:has(.dc-panel-hook) .element-container:first-of-type {
  margin-bottom: 0.2rem;
}

@media (max-width: 899px) {
  div[data-testid="column"]:has(.dc-panel-hook) {
    position: static !important;
    top: auto !important;
    z-index: auto !important;
  }

  div[data-testid="column"]:has(.dc-panel-hook) > div[data-testid="stVerticalBlock"] {
    max-height: none !important;
    overflow: visible !important;
  }
}

@media (max-width: 899px) {
  div[data-testid="column"]:has(.dc-panel-hook) > div[data-testid="stVerticalBlock"] {
    position: static;
    top: auto;
    max-height: none;
    overflow: visible;
  }
}
  </style>
</head>
<body>
  <div class="grid">
    <div class="card" id="eegcard" style="display:none;"><div id="eeg" style="height:440px;"></div></div>

    <div class="card" id="ppgcard" style="display:none;">
      <div class="ppghead">
        <div class="left">
          ❤️ <b>Heart Rate</b>
          <span class="muted">(inst / smooth):</span>
          <b><span id="hrInst">—</span></b> / <b><span id="hrSmooth">—</span></b>
          <span class="muted">BPM</span>
        </div>
        <div class="right">
          <span class="pill" id="qualPill">PPG: —</span>

          <span class="tip tip-rmssd">
            <span class="muted">RMSSD</span>
            <span class="bubble">
              <b>RMSSD</b> is an HRV metric: it measures short-term beat-to-beat variability
              (root mean square of successive IBI differences). Higher often means more
              parasympathetic “recovery”, but only if the PPG signal is clean.
            </span>
          </span>
          <b><span id="rmssd">—</span></b><span class="muted"> ms</span>

          <span class="muted" id="hint"></span>
        </div>
      </div>

      <div id="ppg" style="height:300px;"></div>

      <div class="charthead">
        <span class="tip">
          <span><b>IBI trend</b> <span class="muted">(ms)</span></span>
          <span class="bubble">
            <b>IBI</b> (inter-beat interval) is the time between detected beats.
            Stable, smooth IBI points usually mean good peak detection. Noisy spikes
            often mean motion/contact issues.
          </span>
        </span>
      </div>
      <div id="ibi" style="height:220px;"></div>
    </div>

    <div class="card" id="bandscard" style="display:none;"><div id="bands" style="height:320px;"></div></div>
    <div class="card" id="psdcard" style="display:none;"><div id="psd" style="height:360px;"></div></div>
  </div>

<script>
const HOST = "127.0.0.1";
const PORT = __PORT__;
let lastUpdateMs = 200;

function darkLayoutBase(title) {
  return {
    title: {text: title},
    paper_bgcolor: "__DARK_BG__",
    plot_bgcolor: "__DARK_PLOT__",
    font: {color: "__DARK_TEXT__"},
    margin: {l:80,r:25,t:45,b:55},
    xaxis: { gridcolor: "__DARK_GRID__", zerolinecolor: "__DARK_GRID__", linecolor: "__DARK_GRID__", tickcolor: "__DARK_GRID__" },
    yaxis: { gridcolor: "__DARK_GRID__", zerolinecolor: "__DARK_GRID__", linecolor: "__DARK_GRID__", tickcolor: "__DARK_GRID__" },
  };
}

function robustZ(arr) {
  const x = arr.map(v => (Number.isFinite(v) ? v : NaN));
  const finite = x.filter(v => Number.isFinite(v));
  if (finite.length < 10) return x.map(_ => 0);

  finite.sort((a,b) => a-b);
  const mid = Math.floor(finite.length/2);
  const med = (finite.length % 2) ? finite[mid] : 0.5*(finite[mid-1]+finite[mid]);

  const absdev = finite.map(v => Math.abs(v - med)).sort((a,b)=>a-b);
  const mad = (absdev.length % 2) ? absdev[mid] : 0.5*(absdev[mid-1]+absdev[mid]);
  const scale = (mad > 1e-9) ? mad : 1.0;

  return x.map(v => {
    if (!Number.isFinite(v)) return 0;
    let z = (v - med) / scale;
    if (z > 8) z = 8;
    if (z < -8) z = -8;
    return z;
  });
}

function setQualityPill(label, score) {
  const pill = document.getElementById("qualPill");
  let bg = "rgba(255,255,255,0.06)";
  let bd = "rgba(255,255,255,0.16)";
  if (label === "Good") {
    bg = "rgba(34,197,94,0.18)";
    bd = "rgba(34,197,94,0.45)";
  } else if (label === "OK") {
    bg = "rgba(234,179,8,0.18)";
    bd = "rgba(234,179,8,0.45)";
  } else {
    bg = "rgba(239,68,68,0.16)";
    bd = "rgba(239,68,68,0.45)";
  }
  pill.style.background = bg;
  pill.style.borderColor = bd;
  pill.textContent = `PPG: ${label} (${Math.round(100*score)}%)`;
}

async function pollOnce() {
  const url = "http://" + HOST + ":" + PORT + "/data";
  try {
    const resp = await fetch(url, {cache:"no-store"});
    const data = await resp.json();
    if (!data.ok) return;

    if (data.update_ms && Math.abs(data.update_ms - lastUpdateMs) > 1) {
      lastUpdateMs = data.update_ms;
      clearInterval(window.__timer);
      window.__timer = setInterval(pollOnce, lastUpdateMs);
    }

    const hasEEG = (data.t && data.t.length && data.y && data.y.length);
    const hasPPG = (data.ppg_t && data.ppg_t.length && data.ppg_y && data.ppg_y.length && data.ppg_channels && data.ppg_channels.length);
    const hasBands = (data.bp_frac && data.bp_frac.length && data.bands && data.bands.length && data.channels && data.channels.length);
    const hasPSD = (data.f && data.f.length && data.psd_db && data.psd_db.length);

    document.getElementById("eegcard").style.display = hasEEG ? "block" : "none";
    document.getElementById("ppgcard").style.display = hasPPG ? "block" : "none";
    document.getElementById("bandscard").style.display = (hasEEG && hasBands) ? "block" : "none";
    document.getElementById("psdcard").style.display = (hasEEG && hasPSD) ? "block" : "none";

    const isPaused = !!data.paused;
    document.getElementById("hint").innerText = (hasPPG || hasEEG) ? (isPaused ? "Paused — showing last captured data." : "") : "";

    document.getElementById("hrInst").innerText =
      (data.hr_bpm_inst === null || data.hr_bpm_inst === undefined) ? "—" : Math.round(data.hr_bpm_inst).toString();
    document.getElementById("hrSmooth").innerText =
      (data.hr_bpm_smooth === null || data.hr_bpm_smooth === undefined) ? "—" : Math.round(data.hr_bpm_smooth).toString();
    document.getElementById("rmssd").innerText =
      (data.rmssd_ms === null || data.rmssd_ms === undefined) ? "—" : Math.round(data.rmssd_ms).toString();
    setQualityPill(data.ppg_quality || "Bad", data.ppg_quality_score || 0);

    if (hasEEG) {
      const traces = data.channels.map((ch, i) => ({
        x: data.t, y: data.y[i], mode:"lines", name: ch,
        line: {width: 1.6},
      }));

      const anns = data.channels.map((ch, i) => ({
        x: data.t[0], xref: "x",
        y: data.base[i], yref: "y",
        text: ch,
        showarrow: false,
        xanchor: "right",
        font: {size: 12, color: "__DARK_TEXT__"},
      }));

      const eegLayout = darkLayoutBase("EEG (rolling / seismograph style)");
      eegLayout.showlegend = false;
      eegLayout.xaxis.title = "Time (s) (now = 0)";
      eegLayout.yaxis.title = "Amplitude (µV)  (offset display)";
      eegLayout.annotations = anns;
      Plotly.react("eeg", traces, eegLayout, {displayModeBar:false, responsive:true});
    }

    if (hasPPG) {
      const ppgTraces = data.ppg_channels.map((ch, i) => {
        const z = robustZ(data.ppg_y[i]);
        return {
          x: data.ppg_t,
          y: z,
          mode:"lines",
          name: ch + " (norm)",
          line: {width: 1.5},
        };
      });

      const bandOK = (data.ppg_band_t && data.ppg_band_y && data.ppg_band_t.length && data.ppg_band_y.length);
      if (bandOK) {
        ppgTraces.unshift({
          x: data.ppg_band_t,
          y: data.ppg_band_y,
          mode: "lines",
          name: "Bandpassed (HR)",
          line: {width: 2.2},
          opacity: 0.95,
        });

        if (data.ppg_peak_t && data.ppg_peak_t.length) {
          ppgTraces.unshift({
            x: data.ppg_peak_t,
            y: data.ppg_peak_t.map(_ => 0),
            mode: "markers",
            name: "Peaks",
            marker: {size: 7, symbol: "circle"},
            opacity: 0.95
          });
        }
      }

      const ppgLayout = darkLayoutBase("PPG (rolling): normalized raw + bandpassed + peaks");
      ppgLayout.xaxis.title = "Time (s) (now = 0)";
      ppgLayout.yaxis.title = "Robust z-score (a.u.)";
      ppgLayout.showlegend = true;
      ppgLayout.margin = {l:80,r:25,t:45,b:55};
      Plotly.react("ppg", ppgTraces, ppgLayout, {displayModeBar:false, responsive:true});

      const ibiLayout = darkLayoutBase("");
      ibiLayout.xaxis.title = "Time (s) (now = 0)";
      ibiLayout.yaxis.title = "IBI (ms)";
      ibiLayout.margin = {l:80,r:25,t:20,b:55};

      if (data.ibi_t && data.ibi_ms && data.ibi_t.length && data.ibi_ms.length) {
        const ibiTrace = {
          x: data.ibi_t,
          y: data.ibi_ms,
          mode: "lines+markers",
          name: "IBI",
          line: {width: 2.0},
          marker: {size: 6}
        };
        Plotly.react("ibi", [ibiTrace], ibiLayout, {displayModeBar:false, responsive:true});
      } else {
        Plotly.react("ibi", [], ibiLayout, {displayModeBar:false, responsive:true});
      }
    }

    if (hasEEG && hasBands && data.band_colors) {
      const bands = data.bands;
      const bandTraces = bands.map((b, bi) => ({
        type:"bar",
        name:b,
        x:data.channels,
        y:data.bp_frac.map(row => row[bi]),
        marker:{color: data.band_colors[b] || "#999"}
      }));
      const bandLayout = darkLayoutBase("Band power fraction (per channel)");
      bandLayout.barmode = "stack";
      bandLayout.margin = {l:70,r:25,t:45,b:45};
      bandLayout.yaxis = {range:[0,1], title:"Fraction", gridcolor:"__DARK_GRID__", zerolinecolor:"__DARK_GRID__"};
      Plotly.react("bands", bandTraces, bandLayout, {displayModeBar:false, responsive:true});
    }

    if (hasEEG && hasPSD && data.band_edges && data.bands && data.band_colors) {
      const psdTrace = { x:data.f, y:data.psd_db, mode:"lines", name:"Mean PSD", line:{width:2.2} };
      const shapes = [];
      const annotations = [];
      const ymax = Math.max(...data.psd_db);

      for (const b of data.bands) {
        const [f1, f2] = data.band_edges[b];
        const c = data.band_colors[b] || "#999";
        shapes.push({
          type:"rect", xref:"x", yref:"paper",
          x0:f1, x1:f2, y0:0, y1:1,
          fillcolor:c, opacity:0.14, line:{width:0}
        });
        annotations.push({
          x:(f1+f2)/2, y:ymax,
          xref:"x", yref:"y",
          text:b,
          showarrow:false,
          yanchor:"bottom",
          opacity:0.95,
          font: {size: 12, color: c}
        });
      }

      const psdLayout = darkLayoutBase("PSD (mean across channels)");
      psdLayout.xaxis.title = "Frequency (Hz)";
      psdLayout.yaxis.title = "PSD (dB)";
      psdLayout.margin = {l:80, r:25, t:45, b:70};
      psdLayout.shapes = shapes;
      psdLayout.annotations = annotations;
      Plotly.react("psd", [psdTrace], psdLayout, {displayModeBar:false, responsive:true});
    }
  } catch (e) {}
}

pollOnce();
window.__timer = setInterval(pollOnce, lastUpdateMs);
</script>
</body>
</html>
"""

    html = (
        html_template
        .replace("__PORT__", str(int(port)))
        .replace("__DARK_BG__", DARK_BG)
        .replace("__DARK_PLOT__", DARK_PLOT)
        .replace("__DARK_GRID__", DARK_GRID)
        .replace("__DARK_TEXT__", DARK_TEXT)
    )


    components.html(html, height=1900, scrolling=False)
