from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from muse_config import (
    EEG_CHANNEL_NAMES, PPG_CHANNEL_NAMES, PPG_FS,
    HR_SMOOTH_WINDOW_SEC, HR_INST_WINDOW_SEC
)
from muse_features import ensure_sleep_data_dir
from muse_ppg import _detect_beats, estimate_bpm_from_ibi, rmssd_ms_from_ibi, ppg_quality_score
from muse_server import _align_nearest
from muse_lsl import LSLReceiver

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
