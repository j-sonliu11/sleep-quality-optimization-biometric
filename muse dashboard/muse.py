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
CONNECT_RETRY_TOTAL_S = 25.0   # <-- increased (Windows LSL publish can be slow)
CONNECT_RETRY_STEP_S = 0.75

# Collector: allow a short wait for first samples
COLLECTOR_WAIT_FOR_SAMPLES_S = 4.0

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
        self.lock = threading.Lock()

        self.running = False
        self.paused = False
        self.thread = None
        self.inlet = None
        self.last_error = None
        self.stream_meta = None

        self.sample_count = 0
        self.last_ts = None

    def start(self, timeout_s: float = 2.0, prefer_name_contains: str = "muse") -> bool:
        """
        Attempts to connect within timeout_s. Designed to be called repeatedly
        (retry loop) to eliminate “two-click” behavior.
        """
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
                        self.inlet = StreamInlet(best, max_buflen=60)
                        self.stream_meta = {
                            "name": best.name(),
                            "type": best.type(),
                            "channels": best.channel_count(),
                            "fs": best.nominal_srate(),
                        }
                        self.running = True
                        self.last_error = None
                        self.thread = threading.Thread(target=self._run, daemon=True)
                        self.thread.start()
                        return True

            except Exception as e:
                self.last_error = repr(e)

            if time.time() >= deadline:
                return False

    def _run(self):
        try:
            while self.running:
                if self.paused:
                    time.sleep(0.05)
                    continue
                sample, ts = self.inlet.pull_sample(timeout=0.5)
                if sample is None:
                    continue
                if len(sample) < self.n_chan:
                    continue

                sample = sample[: self.n_chan]
                try:
                    sample = [float(v) for v in sample]
                    ts = float(ts)
                except Exception:
                    continue

                with self.lock:
                    self.ts.append(ts)
                    self.buf.append(sample)
                    self.sample_count += 1
                    self.last_ts = ts
        except Exception as e:
            self.running = False
            self.last_error = repr(e)

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
        if last_ts is None:
            return np.array([]), np.array([[]], dtype=float)
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
            if self.thread:
                self.thread.join(timeout=1)
        except Exception:
            pass
        with self.lock:
            self.ts.clear()
            self.buf.clear()
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

    ts_e, X_e = eeg.get_window(rolling)
    eeg_ok = X_e.size != 0

    ts_p_roll, X_p_roll = ppg.get_window(rolling)
    ppg_ok = X_p_roll.size != 0

    # IMPORTANT: keep ok=True for the “one click” behavior, but we will HIDE graphs until data exists (JS).
    if not eeg_ok and not ppg_ok:
        return {
            "ok": True,
            "waiting_for_samples": True,
            "update_ms": float(cfg.get("update_ms", 200.0)),
            "paused": bool(eeg.paused),
            "running": bool(eeg.running),
            "ppg_running": bool(ppg.running),

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

            "ppg_t": [],
            "ppg_y": [],
            "ppg_channels": PPG_CHANNEL_NAMES,
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

    payload = {
        "ok": True,
        "update_ms": float(cfg["update_ms"]),
        "paused": bool(eeg.paused),
        "running": bool(eeg.running),
        "ppg_running": bool(ppg.running),
    }

    # ---- EEG payload
    if eeg_ok:
        do_notch = bool(int(cfg["use_notch"]))
        Xf = apply_filters(X_e, EEG_FS, float(cfg["bp_low"]), float(cfg["bp_high"]), do_notch, float(cfg["notch_freq"]))
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
            for bi, (lbl, (f1, f2)) in enumerate(BANDS.items()):
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

    # ---- PPG payload + HR/HRV/Quality + bandpassed plot + peaks + IBI trend
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

        if X_smooth.size and X_smooth.shape[1] >= 2:
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
    do_notch = bool(int(cfg["use_notch"]))
    Xf = apply_filters(X_epoch, EEG_FS, float(cfg["bp_low"]), float(cfg["bp_high"]), do_notch, float(cfg["notch_freq"]))

    row = {}
    for i, ch in enumerate(EEG_CHANNEL_NAMES):
        row[ch] = float(np.mean(Xf[:, i]))
        row[f"{ch}__std"] = float(np.std(Xf[:, i]))

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

    row["Sleep_Stage"] = ""
    return row


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
        self.lock = threading.Lock()

        self.ready = False
        self.last_err = None

        self.xgb = None
        self.boosterA = None
        self.boosterB = None
        self.boosterC = None
        self.expected_cols: list[str] | None = None

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

        X2 = X.copy()
        missing = [c for c in expected_cols if c not in X2.columns]
        extra = [c for c in X2.columns if c not in expected_cols]

        for c in missing:
            X2[c] = 0.0
        if extra:
            X2 = X2.drop(columns=extra)

        return X2[expected_cols]

    def _coerce_numeric(self, X: pd.DataFrame):
        Xn = X.apply(pd.to_numeric, errors="coerce")
        Xn = Xn.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return Xn

    def _predict_cascade_from_boosters(self, X: pd.DataFrame):
        dmat = self.xgb.DMatrix(X, feature_names=list(X.columns))

        # Stage A: ["S","W"]  => indices [0,1]
        probaA = self.boosterA.predict(dmat)
        isW = probaA[:, 1] >= 0.5
        pred = np.array(["S"] * len(X), dtype=object)
        pred[isW] = "W"

        # Stage B: ["NREM","R"] => indices [0,1]
        probaB = np.full((len(X), 2), np.nan, dtype=float)
        idxS = np.where(~isW)[0]
        if len(idxS) > 0:
            dS = self.xgb.DMatrix(X.iloc[idxS], feature_names=list(X.columns))
            pb = self.boosterB.predict(dS)
            probaB[idxS] = pb
            isR = pb[:, 1] >= 0.5
            pred[idxS[isR]] = "R"
            pred[idxS[~isR]] = "NREM"

            # Stage C: ["N1","N2","N3"] => indices [0,1,2]
            probaC = np.full((len(X), 3), np.nan, dtype=float)
            idxNREM = idxS[~isR]
            if len(idxNREM) > 0:
                dN = self.xgb.DMatrix(X.iloc[idxNREM], feature_names=list(X.columns))
                pc = self.boosterC.predict(dN)
                probaC[idxNREM] = pc
                c_idx = np.argmax(pc, axis=1)
                pred[idxNREM[c_idx == 0]] = "N1"
                pred[idxNREM[c_idx == 1]] = "N2"
                pred[idxNREM[c_idx == 2]] = "N3"
        else:
            probaC = np.full((len(X), 3), np.nan, dtype=float)

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
            # model features only (drop labels/time-ish columns)
            drop = {"Sleep_Stage", "TIMESTAMP", "ISO_TIME", "pred_stage"}
            feat = {k: v for k, v in row.items() if k not in drop}

            X = pd.DataFrame([feat])
            X = self._coerce_numeric(X)
            X = self._align_columns(X, self.expected_cols)
            X = self._coerce_numeric(X)

            pred, probaA, probaB, probaC = self._predict_cascade_from_boosters(X)

            label = str(pred[0])

            out = {
                "epoch_index": None,  # filled by caller if desired
                "timestamp_s": row.get("TIMESTAMP", None),
                "iso_time": row.get("ISO_TIME", None),
                "pred_stage": label,
                "pA_S": float(probaA[0, 0]),
                "pA_W": float(probaA[0, 1]),
                "pB_NREM": (None if np.isnan(probaB[0, 0]) else float(probaB[0, 0])),
                "pB_R": (None if np.isnan(probaB[0, 1]) else float(probaB[0, 1])),
                "pC_N1": (None if np.isnan(probaC[0, 0]) else float(probaC[0, 0])),
                "pC_N2": (None if np.isnan(probaC[0, 1]) else float(probaC[0, 1])),
                "pC_N3": (None if np.isnan(probaC[0, 2]) else float(probaC[0, 2])),
            }

            # simple "display confidence" based on final stage path
            if label == "W":
                out["confidence"] = out["pA_W"]
            elif label == "R":
                out["confidence"] = out["pB_R"] if out["pB_R"] is not None else None
            elif label in {"N1", "N2", "N3"}:
                key = f"pC_{label}"
                out["confidence"] = out.get(key, None)
            else:
                out["confidence"] = None

            with self.lock:
                self.history.append(out)
                # keep last 300 epochs in memory (~2.5 hours)
                if len(self.history) > 300:
                    self.history = self.history[-300:]

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

    def get_history(self, n: int = 20):
        with self.lock:
            return list(self.history[-int(max(1, n)):])


class EpochCollector:
    def __init__(self, eeg: LSLReceiver, ppg: LSLReceiver, cfg_store: ConfigStore, infer_engine: LiveSleepInference | None = None):
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
        self.session_dir: Path | None = None  # NEW: per-collection run folder
        self.rows_written = 0
        self.last_write_iso = None
        self.last_err = None

        self.infer_engine = infer_engine

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

        with self.lock:
            ts_w, X_w = self.eeg.get_window(1.0)
            if ts_w.size == 0:
                self.last_err = "No EEG samples in buffer yet (wait a moment after connecting)."
                return False

            # Root folder + per-run session folder
            out_root = ensure_sleep_data_dir()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = out_root / f"run_{stamp}"
            self.session_dir.mkdir(parents=True, exist_ok=True)

            self.csv_path = self.session_dir / f"muse_epochs_{stamp}.csv"

            self.running = True
            self.rows_written = 0
            self.last_write_iso = None
            self.last_err = None

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
            return True

    def stop(self):
        with self.lock:
            self.running = False

    def status(self):
        with self.lock:
            return {
                "collecting": self.running,
                "csv_path": str(self.csv_path) if self.csv_path else None,
                "session_dir": str(self.session_dir) if self.session_dir else None,  # NEW: helpful for saving PNGs in same folder
                "rows_written": int(self.rows_written),
                "last_write_iso": self.last_write_iso,
                "last_err": self.last_err,
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
            while True:
                with self.lock:
                    if not self.running:
                        return

                ts_new, X_new = self.eeg.get_since(self.last_seen_ts_eeg)
                if ts_new.size == 0:
                    time.sleep(0.10)
                else:
                    self.last_seen_ts_eeg = float(ts_new[-1])
                    for t, x in zip(ts_new.tolist(), X_new.tolist()):
                        self.epoch_ts.append(float(t))
                        self.epoch_X.append(x)

                if self.ppg.running and self.last_seen_ts_ppg is not None:
                    tsp_new, Xp_new = self.ppg.get_since(self.last_seen_ts_ppg)
                    if tsp_new.size:
                        self.last_seen_ts_ppg = float(tsp_new[-1])
                        for t, x in zip(tsp_new.tolist(), Xp_new.tolist()):
                            self.ppg_ts.append(float(t))
                            self.ppg_X.append(x)

                if self.epoch_start_ts is None and self.epoch_ts:
                    self.epoch_start_ts = float(self.epoch_ts[0])

                if not self.epoch_ts:
                    continue

                if (float(self.epoch_ts[-1]) - float(self.epoch_start_ts)) >= EPOCH_SEC:
                    epoch_start = float(self.epoch_start_ts)
                    epoch_end = epoch_start + EPOCH_SEC

                    ts_arr = np.asarray(self.epoch_ts, dtype=float)
                    X_arr = np.asarray(self.epoch_X, dtype=float)

                    m = (ts_arr >= epoch_start) & (ts_arr < epoch_end)
                    ts_epoch = ts_arr[m]
                    X_epoch = X_arr[m]

                    self.epoch_start_ts = epoch_end

                    keep = ts_arr >= epoch_end
                    self.epoch_ts = ts_arr[keep].tolist()
                    self.epoch_X = X_arr[keep].tolist()

                    if ts_epoch.size < int(EEG_FS * EPOCH_SEC * 0.70):
                        continue

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

                    # ---- Live sleep stage inference (DreamT XGBoost cascade)
                    # Wrapped so inference issues don't kill data collection
                    if self.infer_engine is not None and self.infer_engine.ready:
                        try:
                            pred_out = self.infer_engine.predict_epoch_row(row)
                            if pred_out is not None:
                                pred_out["epoch_index"] = int(self.rows_written + 1)

                                # Optional: also save prediction columns into the CSV row itself
                                row["PRED_STAGE"] = pred_out.get("pred_stage", "")
                                row["PRED_CONFIDENCE"] = "" if pred_out.get("confidence") is None else float(pred_out["confidence"])
                                row["P_A_W"] = "" if pred_out.get("pA_W") is None else float(pred_out["pA_W"])
                                row["P_B_R"] = "" if pred_out.get("pB_R") is None else float(pred_out["pB_R"])
                                row["P_C_N1"] = "" if pred_out.get("pC_N1") is None else float(pred_out["pC_N1"])
                                row["P_C_N2"] = "" if pred_out.get("pC_N2") is None else float(pred_out["pC_N2"])
                                row["P_C_N3"] = "" if pred_out.get("pC_N3") is None else float(pred_out["pC_N3"])
                        except Exception as e:
                            self.last_err = f"Live inference error (collector kept running): {e!r}"

                    # CSV append wrapped so you get a clearer error if it fails
                    try:
                        self._append_row(row)
                        self.rows_written += 1
                        self.last_write_iso = datetime.now().isoformat(timespec="seconds")
                    except Exception as e:
                        self.last_err = f"CSV write error: {e!r}"
                        # Keep collector alive and retry next epoch
                        continue

        except Exception as e:
            with self.lock:
                self.last_err = repr(e)
                self.running = False
                
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

        # y-axis integer ticks look nicer for epoch counts
        ymax = max(y) if y else 0
        ax1.set_ylim(0, max(1, ymax + 1))

        ax1.set_title(
            f"Predicted Sleep Stage Distribution (last {len(hist)} epochs)",
            color="white",
            fontsize=12,
            pad=10,
        )

        # Bar labels
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
        # Draw colored blocks
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

        # sparse x tick labels for readability
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

        # Legend
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
      Path(r"F:\! Senior Design Project\XGBoost\runs\run_YYYYMMDD_HHMMSS")
    """
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

        # Fix “two-click label” (force immediate redraw)
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
left, right = st.columns([3, 1], gap="large")

# -----------------------------
# Main layout: Charts (left) + Data Collection panel (right)
# -----------------------------
left, right = st.columns([3, 1], gap="large")

@st.fragment(run_every=1.0)
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
    # Real Streamlit container (widgets live inside this)
    dc_panel = st.container()

    with dc_panel:

        st.subheader("Data Collection")

        col_stat = collector.status()
        collecting = bool(col_stat["collecting"])

        st.write("")

        if st.button("💾 Save Snapshot", use_container_width=True):
            if not eeg_receiver.running:
                st.info("Connect first to save a snapshot.")
            else:
                p = save_snapshot_csv(eeg_receiver, ppg_receiver, float(cfg_store.get()["rolling_sec"]))
                if p is None:
                    st.error("No EEG data available to snapshot yet.")
                else:
                    st.success(f"Saved snapshot:\\n{p.name}")

        b1, b2 = st.columns(2, gap="small")
        with b1:
            start_clicked = st.button("▶ Start", use_container_width=True, disabled=collecting)
        with b2:
            stop_clicked = st.button("⏹ Stop", use_container_width=True, disabled=not collecting)

        if start_clicked:
            ok = collector.start()
            if not ok:
                st.error(collector.status().get("last_err") or "Could not start collecting.")
            else:
                st.success("Collecting started (30s epochs).")
            st.rerun()

        if stop_clicked:
            collector.stop()

            # Save prediction summary image (into the same run_... folder as the CSV)
            img_path = None
            try:
                img_path = save_prediction_summary_image(
                live_infer,
                collector.session_dir if collector.session_dir is not None else ensure_sleep_data_dir(),
                max_epochs=120,
                )
            except Exception as e:
                st.error(f"Stopped collection, but could not save prediction summary image: {e}")

            if img_path is not None:
                t.warning(f"Collecting stopped (file kept). Saved prediction summary image: {img_path.name}")
            else:
                st.warning("Collecting stopped (file kept). No prediction summary image was saved (no predictions yet).")

            st.rerun()  # one-click label/state update    

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

        st.caption("Files save to a folder named **Sleep Data** next to muse.py.")
        render_collection_status(collector)

    st.markdown("### Live Sleep Stage Inference")

    # Optional path override (persists in session)
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

    @st.fragment(run_every=1.0)
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

        # -------- helpers --------
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

        # -------- latest prediction card --------
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

        # -------- mini histogram + epoch strip (recent epochs) --------
        hist = infer_engine.get_history(30)  # last 30 epochs (~15 min)

        if hist:
            st.markdown("**Recent stage distribution (last 30 epochs)**")

            stage_order = ["W", "N1", "N2", "N3", "R"]
            counts = {k: 0 for k in stage_order}

            for h in hist:
                stg = h.get("pred_stage")
                if stg in counts:
                    counts[stg] += 1

            # Plotly mini histogram
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

            # ---------- Epoch strip (temporal order) ----------
            st.markdown("**Epoch strip (oldest → newest)**")

            strip_blocks = []
            # hist is oldest -> newest (assuming your history append order)
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

            # tiny legend row
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

        # -------- optional details --------
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

            # Inference diagnostics (optional but very useful)
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

    # NOTE: raw template (NOT an f-string)
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

    /* Allow tooltips to overflow plot containers */
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

    /* Header above IBI graph (tooltip lives here) */
    .charthead{
      padding: 12px 12px 4px 12px;
      font-size: 1.02rem;
      opacity: 0.95;
      position: relative;
      z-index: 2000;
      overflow: visible;
    }

    /* Soft tooltip (hover) */
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

    /* RMSSD tooltip: anchor to the RIGHT so it shifts LEFT (prevents clipping) */
    .tip.tip-rmssd .bubble {
      left: auto !important;
      right: 0 !important;
    }

    /* Invisible marker */
.dc-panel-hook { display: none; }

/* IMPORTANT: let sticky work (ancestors cannot clip) */
div[data-testid="stHorizontalBlock"],
div[data-testid="column"],
div[data-testid="stVerticalBlock"],
div[data-testid="stMainBlockContainer"],
div[data-testid="stBlock"],
div[data-testid="stElementContainer"],
div.block-container {
  overflow: visible !important;
}

/* Make the RIGHT COLUMN (the one containing our marker) sticky */
div[data-testid="column"]:has(.dc-panel-hook) {
  position: sticky !important;
  top: 0.85rem !important;
  align-self: flex-start !important;   /* key for columns/flex layouts */
  z-index: 30 !important;
}

/* Style the actual panel content inside that column */
div[data-testid="column"]:has(.dc-panel-hook) > div[data-testid="stVerticalBlock"] {
  background: rgba(17, 24, 39, 0.72);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 0.85rem 0.9rem 0.9rem 0.9rem;
  box-shadow: 0 8px 20px rgba(0,0,0,0.22);

  max-height: calc(100vh - 1.7rem);
  overflow: auto !important;
}

/* Optional: tighten first element spacing inside panel */
div[data-testid="column"]:has(.dc-panel-hook) .element-container:first-of-type {
  margin-bottom: 0.2rem;
}

/* Mobile: disable sticky */
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

/* Mobile / narrow screens: disable sticky so layout behaves naturally */
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
    <!-- Start hidden (professional “clean” boot). JS unhides once data exists. -->
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

      <!-- IBI tooltip is on the HEADER of the IBI graph -->
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

    // Clean boot: hide cards until data exists
    document.getElementById("eegcard").style.display = hasEEG ? "block" : "none";
    document.getElementById("ppgcard").style.display = hasPPG ? "block" : "none";
    document.getElementById("bandscard").style.display = (hasEEG && hasBands) ? "block" : "none";
    document.getElementById("psdcard").style.display = (hasEEG && hasPSD) ? "block" : "none";

    // Only show hint once a card is visible
    const isPaused = !!data.paused;
    document.getElementById("hint").innerText = (hasPPG || hasEEG) ? (isPaused ? "Paused — showing last captured data." : "") : "";

    // HR + quality + RMSSD
    document.getElementById("hrInst").innerText =
      (data.hr_bpm_inst === null || data.hr_bpm_inst === undefined) ? "—" : Math.round(data.hr_bpm_inst).toString();
    document.getElementById("hrSmooth").innerText =
      (data.hr_bpm_smooth === null || data.hr_bpm_smooth === undefined) ? "—" : Math.round(data.hr_bpm_smooth).toString();
    document.getElementById("rmssd").innerText =
      (data.rmssd_ms === null || data.rmssd_ms === undefined) ? "—" : Math.round(data.rmssd_ms).toString();
    setQualityPill(data.ppg_quality || "Bad", data.ppg_quality_score || 0);

    // EEG plot
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

    // PPG plot
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

      // IBI trend (title removed; header above plot carries it)
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

    // Bands
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

    // PSD
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

    # No internal iframe scrollbar; page scroll only.
    components.html(html, height=1900, scrolling=False)