from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from muse_config import (
    EEG_FS, EEG_CHANNEL_NAMES, BANDS,
    MUSE_TO_DREAMT_ADAPT,
)
from muse_filters import apply_filters, bandpower_welch
from muse_eeg_adapt import ensure_eeg_uv, adapt_muse_eeg_toward_dreamt

def ensure_sleep_data_dir() -> Path:
    out_dir = Path(__file__).resolve().parent / "live prediction runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def epoch_features(ts_epoch: np.ndarray, X_epoch: np.ndarray, cfg: dict):
    if MUSE_TO_DREAMT_ADAPT:
        Xf, adapt_diag = adapt_muse_eeg_toward_dreamt(X_epoch, cfg, eeg_fs=EEG_FS)
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

    for i, ch in enumerate(EEG_CHANNEL_NAMES):
        xi = np.nan_to_num(Xf[:, i], nan=0.0, posinf=0.0, neginf=0.0)
        row[ch] = float(np.mean(xi))
        row[f"{ch}__std"] = float(np.std(xi))
        row[f"{ch}__rms"] = float(np.sqrt(np.mean(xi * xi)))
        row[f"{ch}__ptp"] = float(np.max(xi) - np.min(xi))
        row[f"{ch}__absmean"] = float(np.mean(np.abs(xi)))

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

    try:
        row["EEG_GLOBAL_STD_MEAN"] = float(np.mean([row[f"{ch}__std"] for ch in EEG_CHANNEL_NAMES]))
        row["EEG_GLOBAL_RMS_MEAN"] = float(np.mean([row[f"{ch}__rms"] for ch in EEG_CHANNEL_NAMES]))
        row["EEG_GLOBAL_PTP_MEAN"] = float(np.mean([row[f"{ch}__ptp"] for ch in EEG_CHANNEL_NAMES]))
    except Exception:
        row["EEG_GLOBAL_STD_MEAN"] = 0.0
        row["EEG_GLOBAL_RMS_MEAN"] = 0.0
        row["EEG_GLOBAL_PTP_MEAN"] = 0.0

    row["EEG_UNITS_MODE"] = str(adapt_diag.get("units_detected", "unknown"))
    row["EEG_P95_ABS_PRE"] = "" if adapt_diag.get("pre_p95_abs") is None else float(adapt_diag["pre_p95_abs"])
    row["EEG_P95_ABS_POST"] = "" if adapt_diag.get("post_p95_abs") is None else float(adapt_diag["post_p95_abs"])
    row["EEG_USED_CAR"] = int(bool(adapt_diag.get("used_car", False)))
    row["EEG_USED_ROBUST_STD"] = int(bool(adapt_diag.get("used_robust_std", False)))
    row["EEG_USED_STD_MATCH"] = int(bool(adapt_diag.get("used_std_match", False)))

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

        if self.impute_mode == "mean":
            for f, s in self.stats.items():
                if f in X2.columns:
                    mu = float(s.get("mean", 0.0))
                    X2[f] = X2[f].astype(float)
                    X2[f] = X2[f].where(np.isfinite(X2[f]), np.nan)
                    X2[f] = X2[f].fillna(mu)
        else:
            X2 = X2.replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
