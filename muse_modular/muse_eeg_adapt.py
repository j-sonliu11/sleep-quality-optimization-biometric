from __future__ import annotations

import numpy as np

from muse_config import (
    EEG_CHANNEL_NAMES,
    EEG_CLIP_UV,
    EEG_SOFTSCALE_UV,
    EEG_USE_CAR,
    EEG_USE_ROBUST_STD,
    EEG_REMOVE_DC_PER_EPOCH,
    EEG_MATCH_DREAMT_STD,
    MUSE_TARGET_STD_UV,
    EEG_STD_GAIN_MIN,
    EEG_STD_GAIN_MAX,
    ENABLE_MUSE_DREAMT_FEATURE_ALIAS,
)

from muse_filters import apply_filters

def ensure_eeg_uv(X: np.ndarray) -> tuple[np.ndarray, str]:
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
    X = np.asarray(X, dtype=float)
    s = max(1e-6, float(scale_uv))
    return s * np.tanh(X / s)

def robust_channel_standardize(X: np.ndarray) -> np.ndarray:
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
        g = 1.0 if s <= 1e-9 else float(tgt / s)
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
    muse_chs = ["TP9", "AF7", "AF8", "TP10"]
    suffix_map = {ch: {} for ch in muse_chs}
    for k, v in feat.items():
        for ch in muse_chs:
            if k == ch:
                suffix_map[ch][""] = v
            elif k.startswith(ch + "__"):
                suffix = k[len(ch):]
                suffix_map[ch][suffix] = v

    def get(ch: str, suffix: str):
        return suffix_map.get(ch, {}).get(suffix, None)

    all_suffixes = set()
    for ch in muse_chs:
        all_suffixes.update(suffix_map[ch].keys())

    aliases = {}
    for suffix in all_suffixes:
        tp9 = get("TP9", suffix)
        af7 = get("AF7", suffix)
        af8 = get("AF8", suffix)
        tp10 = get("TP10", suffix)

        frontal = _safe_mean([af7, af8])
        posterior = _safe_mean([tp9, tp10])
        right_temp = _safe_mean([tp10, af8])
        left_temp = _safe_mean([tp9, af7])

        aliases[f"F4-M1{suffix}"] = frontal
        aliases[f"C4-M1{suffix}"] = right_temp
        aliases[f"O2-M1{suffix}"] = posterior
        aliases[f"Fp1-O2{suffix}"] = _safe_mean([af7, posterior])
        aliases[f"CZ-T4{suffix}"] = tp10 if (tp10 is not None and np.isfinite(tp10)) else right_temp
        aliases[f"T3-CZ{suffix}"] = tp9 if (tp9 is not None and np.isfinite(tp9)) else left_temp

    if "EEG_GLOBAL_STD_MEAN" in feat:
        aliases["GLOBAL_STD_MEAN"] = feat["EEG_GLOBAL_STD_MEAN"]
    if "EEG_GLOBAL_RMS_MEAN" in feat:
        aliases["GLOBAL_RMS_MEAN"] = feat["EEG_GLOBAL_RMS_MEAN"]
    if "EEG_GLOBAL_PTP_MEAN" in feat:
        aliases["GLOBAL_PTP_MEAN"] = feat["EEG_GLOBAL_PTP_MEAN"]

    return aliases

def adapt_muse_eeg_toward_dreamt(X_epoch_raw: np.ndarray, cfg: dict, eeg_fs: float):
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

    X, unit_mode = ensure_eeg_uv(X)
    diag["units_detected"] = unit_mode

    do_notch = bool(int(cfg["use_notch"]))
    X = apply_filters(X, eeg_fs, float(cfg["bp_low"]), float(cfg["bp_high"]), do_notch, float(cfg["notch_freq"]))

    if EEG_REMOVE_DC_PER_EPOCH and X.ndim == 2:
        X = X - np.mean(X, axis=0, keepdims=True)

    if EEG_USE_CAR and X.ndim == 2 and X.shape[1] >= 2:
        X = common_average_reference(X)

    X = winsorize_clip(X, -EEG_CLIP_UV, EEG_CLIP_UV)
    X = soft_compress_tanh(X, EEG_SOFTSCALE_UV)

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

    if EEG_USE_ROBUST_STD:
        X = robust_channel_standardize(X)

    diag["post_p95_abs"] = float(np.percentile(np.abs(X), 95))
    return X, diag
