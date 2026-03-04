from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from muse_config import (
    EEG_FS, EEG_CHANNEL_NAMES, PPG_FS, PPG_CHANNEL_NAMES,
    BANDS, BAND_COLORS,
    HR_INST_WINDOW_SEC, HR_SMOOTH_WINDOW_SEC,
)
from muse_filters import apply_filters, bandpower_welch, robust_z
from muse_ppg import _detect_beats, estimate_bpm_from_ibi, rmssd_ms_from_ibi, ppg_quality_score
from muse_lsl import LSLReceiver
from muse_session import ConfigStore

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
    ts_p_roll, X_p_roll = ppg.get_window(rolling)

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

    eeg_ok = (X_e.size != 0) and (ts_e.size >= 5) and (X_e.shape[0] >= 5)
    ppg_ok = (X_p_roll.size != 0) and (ts_p_roll.size >= 5) and (X_p_roll.shape[0] >= 5)

    waiting_for_samples = not eeg_ok and not ppg_ok

    payload = {
        "ok": True,
        "waiting_for_samples": bool(waiting_for_samples),
        "update_ms": float(cfg.get("update_ms", 200.0)),
        "paused": bool(eeg.paused),
        "running": bool(eeg.running),
        "ppg_running": bool(ppg.running),
        "eeg_status": eeg_meta,
        "ppg_status": ppg_meta,
    }

    # ---- EEG
    if eeg_ok:
        do_notch = bool(int(cfg["use_notch"]))
        Xf = apply_filters(
            X_e, EEG_FS,
            float(cfg["bp_low"]), float(cfg["bp_high"]),
            do_notch, float(cfg["notch_freq"]),
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

        from scipy.signal import welch
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
                "t": [], "y": [], "base": [],
                "channels": EEG_CHANNEL_NAMES,
                "bands": list(BANDS.keys()),
                "band_edges": {k: list(v) for k, v in BANDS.items()},
                "band_colors": BAND_COLORS,
                "bp_frac": [], "f": [], "psd_db": [],
            }
        )

    # ---- PPG
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
                "ppg_t": [], "ppg_y": [], "ppg_channels": [],
                "hr_bpm_inst": None, "hr_bpm_smooth": None,
                "ppg_quality": "Bad", "ppg_quality_score": 0.0,
                "ppg_quality_details": {},
                "ppg_band_t": [], "ppg_band_y": [],
                "ppg_peak_t": [], "ibi_t": [], "ibi_ms": [],
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

def ensure_data_server(eeg: LSLReceiver, ppg: LSLReceiver, cfg_store: ConfigStore, st_session_state: dict):
    if "data_server_port" in st_session_state and "data_server_obj" in st_session_state:
        handler_cls = st_session_state.get("data_server_handler_cls", None)
        if handler_cls is not None:
            handler_cls.eeg_ref = eeg
            handler_cls.ppg_ref = ppg
            handler_cls.cfg_store_ref = cfg_store
        return st_session_state["data_server_port"]

    port = find_free_port(8765)

    DataHandler.eeg_ref = eeg
    DataHandler.ppg_ref = ppg
    DataHandler.cfg_store_ref = cfg_store
    st_session_state["data_server_handler_cls"] = DataHandler

    httpd = HTTPServer(("127.0.0.1", port), DataHandler)

    def _serve():
        try:
            httpd.serve_forever()
        except Exception:
            pass

    t = threading.Thread(target=_serve, daemon=True)
    t.start()

    st_session_state["data_server_port"] = port
    st_session_state["data_server_thread"] = t
    st_session_state["data_server_obj"] = httpd
    return port
