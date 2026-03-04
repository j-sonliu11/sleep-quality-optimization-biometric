from __future__ import annotations

import time
import threading
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from muse_config import (
    EEG_FS, EPOCH_SEC,
    PPG_FS,
    HR_SMOOTH_WINDOW_SEC,
    COLLECTOR_WAIT_FOR_SAMPLES_S,
)
from muse_lsl import LSLReceiver
from muse_session import ConfigStore
from muse_features import ensure_sleep_data_dir, epoch_features
from muse_ppg import _detect_beats, estimate_bpm_from_ibi, rmssd_ms_from_ibi, ppg_quality_score

class EpochCollector:
    def __init__(
        self,
        eeg: LSLReceiver,
        ppg: LSLReceiver,
        cfg_store: ConfigStore,
        infer_engine=None,
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

        self.log_path: Path | None = None
        self.last_stop_reason: str | None = None
        self.last_stop_iso: str | None = None
        self.last_heartbeat_iso: str | None = None

        self.infer_engine = infer_engine

    def _log(self, msg: str):
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

        t0 = time.time()
        while time.time() - t0 < COLLECTOR_WAIT_FOR_SAMPLES_S:
            ts_w, X_w = self.eeg.get_window(1.0)
            if ts_w.size and X_w.size:
                break
            time.sleep(0.15)

        csv_name = None

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

    def status(self):
        with self.lock:
            return {
                "collecting": self.running,
                "csv_path": str(self.csv_path) if self.csv_path else None,
                "session_dir": str(self.session_dir) if self.session_dir else None,
                "rows_written": int(self.rows_written),
                "last_write_iso": self.last_write_iso,
                "last_err": self.last_err,
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
            last_eeg_progress_wall = time.time()

            while True:
                with self.lock:
                    if not self.running:
                        self._log("THREAD | exit (running=False)")
                        return
                    self.last_heartbeat_iso = datetime.now().isoformat(timespec="seconds")

                ts_new, X_new = self.eeg.get_since(self.last_seen_ts_eeg)

                if ts_new.size == 0:
                    time.sleep(0.10)
                else:
                    last_eeg_progress_wall = time.time()

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

                if self.ppg.running and self.last_seen_ts_ppg is not None:
                    tsp_new, Xp_new = self.ppg.get_since(self.last_seen_ts_ppg)
                    if tsp_new.size:
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

                if time.time() - last_eeg_progress_wall > 20.0:
                    self._log("EEG | stalled >20s (waiting for samples / reconnect)")
                    last_eeg_progress_wall = time.time()

                if self.epoch_start_ts is None and self.epoch_ts:
                    self.epoch_start_ts = float(self.epoch_ts[0])

                if not self.epoch_ts or self.epoch_start_ts is None:
                    continue

                if (float(self.epoch_ts[-1]) - float(self.epoch_start_ts)) < EPOCH_SEC:
                    continue

                epoch_start = float(self.epoch_start_ts)
                epoch_end = epoch_start + EPOCH_SEC

                ts_arr = np.asarray(self.epoch_ts, dtype=float)
                X_arr = np.asarray(self.epoch_X, dtype=float)

                m = (ts_arr >= epoch_start) & (ts_arr < epoch_end)
                ts_epoch = ts_arr[m]
                X_epoch = X_arr[m]

                min_samples = int(EEG_FS * EPOCH_SEC * 0.70)
                if ts_epoch.size < min_samples:
                    self._log(f"EPOCH | skipped (too few EEG samples: {ts_epoch.size}/{min_samples})")
                    newest = float(ts_arr[-1])
                    self.epoch_start_ts = newest
                    keep = ts_arr >= newest
                    self.epoch_ts = ts_arr[keep].tolist()
                    self.epoch_X = X_arr[keep].tolist()
                    continue

                self.epoch_start_ts = epoch_end
                keep = ts_arr >= epoch_end
                self.epoch_ts = ts_arr[keep].tolist()
                self.epoch_X = X_arr[keep].tolist()

                cfg = self.cfg_store.get()
                row = epoch_features(ts_epoch, X_epoch, cfg)

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

                if self.infer_engine is not None and getattr(self.infer_engine, "ready", False):
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