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
from scipy.signal import butter, sosfiltfilt, iirnotch, welch, filtfilt
import streamlit as st
import streamlit.components.v1 as components
from pylsl import StreamInlet, resolve_byprop

# -----------------------------
# Config
# -----------------------------
EEG_FS = 256.0
EEG_CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]
MAX_BUF_SECONDS = 120.0

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

EPOCH_SEC = 30.0  # <-- requested 30-second epochs

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

/* Hide Streamlit's built-in status widget (top-right flashing stop/running UI) */
div[data-testid="stStatusWidget"] { display: none !important; }

/* --- Tight spacing ONLY in the Data Collection panel --- */
.collect-panel .element-container { margin-bottom: 0.12rem !important; }
.collect-panel .stButton { margin-bottom: 0.10rem !important; }
.collect-panel hr { margin: 0.35rem 0 !important; }
.collect-panel div.stButton > button { padding-top: 0.55rem !important; padding-bottom: 0.55rem !important; }
.collect-panel .stCaption { margin-top: 0.15rem !important; }

.collect-panel hr.tight-hr {
  margin: 0.15rem 0 !important;   /* <-- adjust smaller/larger */
  border: none;
  border-top: 1px solid rgba(255,255,255,0.15);
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧠 Muse 2 Live Dashboard")

# -----------------------------
# Thread-safe CONFIG for server thread
# (Server thread must NOT touch st.session_state)
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

# Ensure we have a stable cfg store early (used all over)
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
    st.caption("Tip: Muse must be ON + Bluetooth enabled.")

# Write settings into persistent config (FIXED: you had invalid set_config(...) syntax)
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
# Filtering
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

# -----------------------------
# LSL Receiver
# -----------------------------
class LSLReceiver:
    def __init__(self, fs=EEG_FS, n_chan=4, max_seconds=MAX_BUF_SECONDS):
        self.fs = fs
        self.n_chan = n_chan
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

    def start(self, timeout_s: float = 12.0) -> bool:
        self.paused = False
        deadline = time.time() + max(0.0, float(timeout_s))
        while True:
            try:
                streams = resolve_byprop("type", "EEG", timeout=0.5)
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
                        if "muse" in n: score += 4
                        if ch == self.n_chan: score += 3
                        if fs > 0 and abs(fs - self.fs) < 10: score += 2
                        if "eeg" in str(typ).lower(): score += 1

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
        """Return (ts, X) with ts > last_ts. If last_ts is None, returns empty."""
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

# -----------------------------
# muselsl helpers
# -----------------------------
def start_muselsl_stream():
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(
        [sys.executable, "-m", "muselsl", "stream"],
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
            timeout=max(5, int(timeout_s) + 5),
        )
        raw = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        macs = re.findall(r"([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})", raw)
        macs = list(dict.fromkeys([m.upper() for m in macs]))
        return (len(macs) > 0), macs, raw
    except Exception as e:
        return False, [], repr(e)

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

def build_payload(receiver: LSLReceiver, cfg_store: ConfigStore):
    cfg = cfg_store.get()
    rolling = float(cfg["rolling_sec"])
    ts, X = receiver.get_window(rolling)
    if X.size == 0:
        return {"ok": False, "reason": "no_data"}

    do_notch = bool(int(cfg["use_notch"]))
    Xf = apply_filters(X, EEG_FS, float(cfg["bp_low"]), float(cfg["bp_high"]), do_notch, float(cfg["notch_freq"]))
    t = (ts - ts[-1]).astype(float)

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

    return {
        "ok": True,
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
        "update_ms": float(cfg["update_ms"]),
        "paused": bool(receiver.paused),
        "running": bool(receiver.running),
    }

class DataHandler(BaseHTTPRequestHandler):
    receiver_ref = None
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
                if self.receiver_ref is None or self.cfg_store_ref is None:
                    payload = {"ok": False, "reason": "server_not_ready"}
                else:
                    payload = build_payload(self.receiver_ref, self.cfg_store_ref)
                self._send(200, json.dumps(payload).encode("utf-8"))
            except Exception as e:
                self._send(200, json.dumps({"ok": False, "err": repr(e)}).encode("utf-8"))
        else:
            self._send(404, json.dumps({"ok": False}).encode("utf-8"))

    def log_message(self, format, *args):
        return

def ensure_data_server(receiver: LSLReceiver, cfg_store: ConfigStore):
    # If server already exists, update the ORIGINAL handler class refs
    if "data_server_port" in st.session_state and "data_server_obj" in st.session_state:
        handler_cls = st.session_state.get("data_server_handler_cls", None)
        if handler_cls is not None:
            handler_cls.receiver_ref = receiver
            handler_cls.cfg_store_ref = cfg_store
        return st.session_state["data_server_port"]

    port = find_free_port(8765)

    # store handler class used by the server (so reruns can update its refs)
    DataHandler.receiver_ref = receiver
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
    out_dir = Path(__file__).resolve().parent / "Sleep Data"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def epoch_features(ts_epoch: np.ndarray, X_epoch: np.ndarray, cfg: dict):
    """
    Return dict for ONE 30-second epoch row.
    Format: PSG-like columns: TIMESTAMP + channels + derived columns + Sleep_Stage blank.
    """
    do_notch = bool(int(cfg["use_notch"]))
    Xf = apply_filters(X_epoch, EEG_FS, float(cfg["bp_low"]), float(cfg["bp_high"]), do_notch, float(cfg["notch_freq"]))

    row = {}

    # Simple per-channel stats
    for i, ch in enumerate(EEG_CHANNEL_NAMES):
        row[ch] = float(np.mean(Xf[:, i]))
        row[f"{ch}__std"] = float(np.std(Xf[:, i]))

    # Bandpowers per channel (absolute) + fractions
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

class EpochCollector:
    """
    Background thread that watches receiver samples and appends ONE CSV row per 30s epoch.
    """
    def __init__(self, receiver: LSLReceiver, cfg_store: ConfigStore):
        self.receiver = receiver
        self.cfg_store = cfg_store
        self.thread = None
        self.running = False
        self.lock = threading.Lock()

        self.start_ts = None
        self.last_seen_ts = None
        self.epoch_start_ts = None

        self.epoch_ts = []
        self.epoch_X = []

        self.csv_path: Path | None = None
        self.rows_written = 0
        self.last_write_iso = None
        self.last_err = None

    def start(self) -> bool:
        with self.lock:
            if self.running:
                return True
            if not self.receiver.running:
                self.last_err = "Receiver not running (connect to LSL first)."
                return False

            out_dir = ensure_sleep_data_dir()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_path = out_dir / f"muse_epochs_{stamp}.csv"

            self.running = True
            self.rows_written = 0
            self.last_write_iso = None
            self.last_err = None

            ts_w, X_w = self.receiver.get_window(1.0)
            if ts_w.size == 0:
                self.last_err = "No samples in buffer yet."
                self.running = False
                return False

            self.start_ts = float(ts_w[-1])
            self.last_seen_ts = float(ts_w[-1])
            self.epoch_start_ts = float(ts_w[-1])
            self.epoch_ts = []
            self.epoch_X = []

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

                ts_new, X_new = self.receiver.get_since(self.last_seen_ts)
                if ts_new.size == 0:
                    time.sleep(0.10)
                    continue

                self.last_seen_ts = float(ts_new[-1])

                for t, x in zip(ts_new.tolist(), X_new.tolist()):
                    self.epoch_ts.append(float(t))
                    self.epoch_X.append(x)

                if self.epoch_start_ts is None and self.epoch_ts:
                    self.epoch_start_ts = float(self.epoch_ts[0])

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

                    if self.start_ts is None:
                        self.start_ts = epoch_start
                    row["TIMESTAMP"] = float(epoch_start - float(self.start_ts))
                    row["ISO_TIME"] = datetime.now().isoformat(timespec="seconds")

                    self._append_row(row)

                    self.rows_written += 1
                    self.last_write_iso = datetime.now().isoformat(timespec="seconds")

        except Exception as e:
            with self.lock:
                self.last_err = repr(e)
                self.running = False

# -----------------------------
# Snapshot save (time-series window to CSV)
# -----------------------------
def save_snapshot_csv(receiver: LSLReceiver, seconds: float) -> Path | None:
    ts, X = receiver.get_window(seconds)
    if X.size == 0:
        return None
    out_dir = ensure_sleep_data_dir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = out_dir / f"muse_snapshot_{stamp}.csv"

    t0 = float(ts[0])
    rel_t = ts - t0

    import csv
    with p.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["TIMESTAMP"] + EEG_CHANNEL_NAMES + ["Sleep_Stage"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(X.shape[0]):
            row = {"TIMESTAMP": float(rel_t[i]), "Sleep_Stage": ""}
            for ci, ch in enumerate(EEG_CHANNEL_NAMES):
                row[ch] = float(X[i, ci])
            w.writerow(row)
    return p

# -----------------------------
# Session state
# -----------------------------
if "receiver" not in st.session_state:
    st.session_state["receiver"] = LSLReceiver(n_chan=len(EEG_CHANNEL_NAMES))
receiver: LSLReceiver = st.session_state["receiver"]

if "collector" not in st.session_state:
    st.session_state["collector"] = EpochCollector(receiver, cfg_store)
collector: EpochCollector = st.session_state["collector"]

# Keep collector's cfg_store reference fresh across reruns (important!)
collector.cfg_store = cfg_store

if "muselsl_proc" not in st.session_state:
    st.session_state["muselsl_proc"] = None

if "scan_state" not in st.session_state:
    st.session_state["scan_state"] = {"scanning": False, "macs": [], "last_raw": ""}

if "ui_state" not in st.session_state:
    st.session_state["ui_state"] = {"muse_found": False, "lsl_connected": False}

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
            _, macs, raw = scan_for_muse(timeout_s=10)

        scan["scanning"] = False
        scan["macs"] = macs
        scan["last_raw"] = raw
        st.session_state["scan_state"] = scan

        ui["muse_found"] = bool(macs)
        st.session_state["ui_state"] = ui

    scan = st.session_state["scan_state"]
    if scan.get("scanning", False):
        st.info("Scanning…")
    elif scan.get("macs"):
        st.success("Muse detected ✅")
    else:
        st.warning("No Muse detected yet")

with c_conn:
    if st.button("▶️ Start + Connect", use_container_width=True):
        p = st.session_state["muselsl_proc"]
        if p is None or p.poll() is not None:
            st.session_state["muselsl_proc"] = start_muselsl_stream()

        ok = receiver.start(timeout_s=12.0)
        ui = st.session_state["ui_state"]
        ui["lsl_connected"] = bool(ok)
        st.session_state["ui_state"] = ui

    ui = st.session_state["ui_state"]
    if ui.get("lsl_connected", False):
        st.success("Connected to LSL ✅")
    else:
        st.warning("Not connected to LSL")

with c_pause:
    paused_label = "⏸ Pause" if (receiver.running and not receiver.paused) else "▶ Resume"
    if st.button(paused_label, use_container_width=True, disabled=not receiver.running):
        if receiver.paused:
            receiver.resume()
        else:
            receiver.pause()

with st.expander("Advanced", expanded=False):
    if st.button("⏹ Stop + Clear Data", use_container_width=True):
        collector.stop()
        receiver.stop_and_clear()
        stop_proc(st.session_state["muselsl_proc"])
        st.session_state["muselsl_proc"] = None
        st.session_state["ui_state"] = {"muse_found": False, "lsl_connected": False}
        st.session_state["scan_state"] = {"scanning": False, "macs": [], "last_raw": ""}
        st.warning("Stopped and cleared buffer.")

# MAC list
scan = st.session_state["scan_state"]
if scan.get("macs"):
    st.write("Detected MAC address(es):")
    for m in scan["macs"]:
        st.code(m)

# Running badge
if receiver.running and receiver.paused:
    st.markdown('<div class="run-badge"><span class="small-muted">Paused</span></div>', unsafe_allow_html=True)
elif receiver.running:
    st.markdown('<div class="run-badge"><div class="spinner"></div><span class="small-muted">Running</span></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="run-badge"><span class="small-muted">Idle</span></div>', unsafe_allow_html=True)

if receiver.running and receiver.sample_count > 50:
    st.success("EEG samples are flowing ✅")
elif receiver.running:
    st.info("Connected but waiting for samples…")
else:
    st.info("Not receiving EEG samples yet.")

# Debug panel
if show_debug:
    with st.expander("Debug", expanded=True):
        st.write("CFG (current):", cfg_store.get())
        st.write("receiver.running:", receiver.running)
        st.write("receiver.paused:", receiver.paused)
        st.write("receiver.sample_count:", receiver.sample_count)
        st.write("receiver.last_ts:", receiver.last_ts)
        st.write("receiver.last_error:", receiver.last_error)

        col_stat = collector.status()
        st.write("collector:", col_stat)

        p = st.session_state.get("muselsl_proc", None)
        st.write("muselsl running:", (p is not None and p.poll() is None))
        if p is not None:
            st.write("muselsl returncode:", p.poll())
        ts_dbg, X_dbg = receiver.get_window(5.0)
        st.write("buffer samples (last 5s):", int(X_dbg.shape[0]))
        if X_dbg.size:
            st.write("X shape/dtype:", X_dbg.shape, str(X_dbg.dtype))
            st.write("last sample:", X_dbg[-1].tolist())

st.divider()

# -----------------------------
# Main layout: Charts (left) + Data Collection panel (right)
# -----------------------------
left, right = st.columns([3, 1], gap="large")

@st.fragment(run_every=1.0)  # refresh this block every 1 second
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
    st.markdown('<div class="collect-panel">', unsafe_allow_html=True)
    st.subheader("Data Collection")

    col_stat = collector.status()
    collecting = bool(col_stat["collecting"])

    st.write("")

    # Snapshot
    if st.button("💾 Save Snapshot", use_container_width=True, disabled=not receiver.running):
        p = save_snapshot_csv(receiver, float(cfg_store.get()["rolling_sec"]))
        if p is None:
            st.error("No data available to snapshot yet.")
        else:
            st.success(f"Saved snapshot:\n{p.name}")

    # Buttons (capture clicks first)
    b1, b2 = st.columns(2, gap="small")

    with b1:
        start_clicked = st.button("▶ Start", use_container_width=True, disabled=collecting)

    with b2:
        stop_clicked = st.button("⏹ Stop", use_container_width=True, disabled=not collecting)

    # Handle clicks AFTER columns (so indicator spans full width)
    if start_clicked:
        ok = collector.start()
        if not ok:
            st.error(collector.status().get("last_err") or "Could not start collecting.")
        else:
            st.success("Collecting started (30s epochs).")

    if stop_clicked:
        collector.stop()
        st.warning("Collecting stopped (file kept).")

    # ✅ Full-width running indicator (not inside a column)
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

    st.markdown('</div>', unsafe_allow_html=True)

with left:
    # -----------------------------
    # Live plots (iframe polling; no Streamlit reruns -> no scroll jumping)
    # -----------------------------
    port = ensure_data_server(receiver, cfg_store)

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{
      margin:0; padding:0;
      background:{DARK_BG};
      color:{DARK_TEXT};
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    }}
    .grid {{ display:grid; grid-template-columns:1fr; gap:18px; }}
    .card {{ background:{DARK_BG}; border-radius:12px; }}
    .muted {{ opacity:0.85; font-size: 0.95rem; }}
  </style>
</head>
<body>
  <div class="grid">
    <div class="card"><div id="eeg" style="height:440px;"></div></div>
    <div class="card"><div id="bands" style="height:320px;"></div></div>
    <div class="card"><div id="psd" style="height:360px;"></div></div>
    <div class="muted" id="hint"></div>
  </div>

<script>
const HOST = "127.0.0.1";
const PORT = {port};
let lastUpdateMs = 200;

function darkLayoutBase(title) {{
  return {{
    title: {{text: title}},
    paper_bgcolor: "{DARK_BG}",
    plot_bgcolor: "{DARK_PLOT}",
    font: {{color: "{DARK_TEXT}"}},
    margin: {{l:80,r:25,t:45,b:55}},
    xaxis: {{ gridcolor: "{DARK_GRID}", zerolinecolor: "{DARK_GRID}", linecolor: "{DARK_GRID}", tickcolor: "{DARK_GRID}" }},
    yaxis: {{ gridcolor: "{DARK_GRID}", zerolinecolor: "{DARK_GRID}", linecolor: "{DARK_GRID}", tickcolor: "{DARK_GRID}" }},
  }};
}}

async function pollOnce() {{
  const url = `http://${{HOST}}:${{PORT}}/data`;
  try {{
    const resp = await fetch(url, {{cache:"no-store"}});
    const data = await resp.json();
    if (!data.ok) {{
      document.getElementById("hint").innerText = "";
      return;
    }}

    if (data.update_ms && Math.abs(data.update_ms - lastUpdateMs) > 1) {{
      lastUpdateMs = data.update_ms;
      clearInterval(window.__timer);
      window.__timer = setInterval(pollOnce, lastUpdateMs);
    }}

    const isPaused = !!data.paused;
    document.getElementById("hint").innerText = isPaused ? "Paused — showing last captured data." : "";

    const traces = data.channels.map((ch, i) => ({{
      x: data.t, y: data.y[i], mode:"lines", name: ch,
      line: {{width: 1.6}},
    }}));

    const anns = data.channels.map((ch, i) => ({{
      x: data.t[0], xref: "x",
      y: data.base[i], yref: "y",
      text: ch,
      showarrow: false,
      xanchor: "right",
      font: {{size: 12, color: "{DARK_TEXT}"}},
    }}));

    const eegLayout = darkLayoutBase("EEG (rolling / seismograph style)");
    eegLayout.showlegend = false;
    eegLayout.xaxis.title = "Time (s) (now = 0)";
    eegLayout.yaxis.title = "Amplitude (µV)  (offset display)";
    eegLayout.annotations = anns;
    Plotly.react("eeg", traces, eegLayout, {{displayModeBar:false, responsive:true}});

    const bands = data.bands;
    const bandTraces = bands.map((b, bi) => ({{
      type:"bar",
      name:b,
      x:data.channels,
      y:data.bp_frac.map(row => row[bi]),
      marker:{{color: data.band_colors[b] || "#999"}}
    }}));
    const bandLayout = darkLayoutBase("Band power fraction (per channel)");
    bandLayout.barmode = "stack";
    bandLayout.margin = {{l:70,r:25,t:45,b:45}};
    bandLayout.yaxis = {{range:[0,1], title:"Fraction", gridcolor:"{DARK_GRID}", zerolinecolor:"{DARK_GRID}"}};
    Plotly.react("bands", bandTraces, bandLayout, {{displayModeBar:false, responsive:true}});

    const psdTrace = {{ x:data.f, y:data.psd_db, mode:"lines", name:"Mean PSD", line:{{width:2.2}} }};
    const shapes = [];
    const annotations = [];
    const ymax = Math.max(...data.psd_db);

    for (const b of data.bands) {{
      const [f1, f2] = data.band_edges[b];
      const c = data.band_colors[b] || "#999";
      shapes.push({{
        type:"rect", xref:"x", yref:"paper",
        x0:f1, x1:f2, y0:0, y1:1,
        fillcolor:c, opacity:0.14, line:{{width:0}}
      }});
      annotations.push({{
        x:(f1+f2)/2, y:ymax,
        xref:"x", yref:"y",
        text:b,
        showarrow:false,
        yanchor:"bottom",
        opacity:0.95,
        font: {{size: 12, color: c}}
      }});
    }}

    const psdLayout = darkLayoutBase("PSD (mean across channels)");
    psdLayout.xaxis.title = "Frequency (Hz)";
    psdLayout.yaxis.title = "PSD (dB)";
    psdLayout.shapes = shapes;
    psdLayout.annotations = annotations;
    Plotly.react("psd", [psdTrace], psdLayout, {{displayModeBar:false, responsive:true}});
  }} catch (e) {{}}
}}

pollOnce();
window.__timer = setInterval(pollOnce, lastUpdateMs);
</script>
</body>
</html>
"""
    components.html(html, height=1200, scrolling=True)
