from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from muse_features import FeatureCalibrator
from muse_eeg_adapt import build_muse_to_dreamt_feature_aliases
from muse_config import ENABLE_MUSE_DREAMT_FEATURE_ALIAS

class LiveSleepInference:
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
        return X.reindex(columns=expected_cols, fill_value=np.nan).copy()

    def _coerce_numeric(self, X: pd.DataFrame):
        Xn = X.apply(pd.to_numeric, errors="coerce")
        Xn = Xn.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return Xn

    def _augment_with_feature_aliases(self, feat: dict) -> tuple[dict, dict]:
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
        arr = np.asarray(p, dtype=float)

        if arr.ndim == 2 and arr.shape[1] == 2:
            s = np.sum(arr, axis=1, keepdims=True)
            s[s <= 0] = 1.0
            return arr / s

        if arr.ndim == 1 and arr.size % 2 == 0:
            arr = arr.reshape(-1, 2)
            s = np.sum(arr, axis=1, keepdims=True)
            s[s <= 0] = 1.0
            return arr / s

        if arr.ndim == 1:
            pos = np.clip(arr, 0.0, 1.0)
            neg = 1.0 - pos
            return np.column_stack([neg, pos])

        if arr.ndim == 2 and arr.shape[1] == 1:
            pos = np.clip(arr[:, 0], 0.0, 1.0)
            neg = 1.0 - pos
            return np.column_stack([neg, pos])

        raise ValueError(f"Unexpected binary prediction shape: {arr.shape}")

    def _normalize_multiclass_proba(self, p, n_classes: int):
        arr = np.asarray(p, dtype=float)

        if arr.ndim == 2 and arr.shape[1] == n_classes:
            s = np.sum(arr, axis=1, keepdims=True)
            s[s <= 0] = 1.0
            return arr / s

        if arr.ndim == 1 and arr.size % n_classes == 0:
            arr = arr.reshape(-1, n_classes)
            s = np.sum(arr, axis=1, keepdims=True)
            s[s <= 0] = 1.0
            return arr / s

        if arr.ndim == 1 and arr.size == n_classes:
            arr = arr.reshape(1, n_classes)
            s = np.sum(arr, axis=1, keepdims=True)
            s[s <= 0] = 1.0
            return arr / s

        raise ValueError(f"Unexpected multiclass prediction shape: {arr.shape}, expected (?, {n_classes})")

    def _predict_cascade_from_boosters(self, X: pd.DataFrame):
        dmat = self.xgb.DMatrix(X, feature_names=list(X.columns))

        rawA = self.boosterA.predict(dmat)
        probaA = self._normalize_binary_proba(rawA)

        pA_S = probaA[:, 0]
        pA_W = probaA[:, 1]

        isW = pA_W >= 0.80
        pred = np.array(["S"] * len(X), dtype=object)
        pred[isW] = "W"

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

            if boosterA is None or boosterB is None or boosterC is None:
                jsonA = self.run_dir / "xgb_stageA_W_vs_S.json"
                jsonB = self.run_dir / "xgb_stageB_R_vs_NREM.json"
                jsonC = self.run_dir / "xgb_stageC_N1_vs_N2_vs_N3.json"
                if not (jsonA.exists() and jsonB.exists() and jsonC.exists()):
                    raise RuntimeError(f"Missing model files in {self.run_dir}. Need model_bundle.joblib or stage JSONs.")
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
        if not self.ready:
            return None

        try:
            drop = {"Sleep_Stage", "TIMESTAMP", "ISO_TIME", "pred_stage", "EEG_UNITS_MODE", "PPG_QUALITY"}
            feat_base = {k: v for k, v in row.items() if k not in drop}

            feat, alias_diag = self._augment_with_feature_aliases(feat_base)

            X_raw = pd.DataFrame([feat])
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

            X = self.calibrator.transform_df(X)
            X = self._coerce_numeric(X)

            vals = X.iloc[0].astype(float).values
            nonzero_mask = np.abs(vals) > 0.0
            nonzero_cols = [c for c, nz in zip(X.columns.tolist(), nonzero_mask.tolist()) if nz]
            nonzero_vals = [float(v) for v, nz in zip(vals.tolist(), nonzero_mask.tolist()) if nz]

            n_nonzero = len(nonzero_cols)
            n_expected = len(self.expected_cols) if self.expected_cols else len(X.columns)
            nonzero_frac = (n_nonzero / n_expected) if n_expected > 0 else None

            top_pairs = sorted(
                zip(nonzero_cols, nonzero_vals),
                key=lambda kv: abs(float(kv[1])),
                reverse=True
            )[:12]

            pred, probaA, probaB, probaC = self._predict_cascade_from_boosters(X)
            label = str(pred[0])

            out = {
                "epoch_index": None,
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

                "diag_n_expected": int(n_expected),
                "diag_n_present_before_align": int(n_present_before),
                "diag_n_missing_before_align": int(len(missing_before_align)),
                "diag_n_extra_before_align": int(len(extra_before_align)),
                "diag_n_nonzero": int(n_nonzero),
                "diag_nonzero_frac": (None if nonzero_frac is None else float(nonzero_frac)),
                "diag_nonzero_cols_preview": nonzero_cols[:12],
                "diag_missing_cols_preview": missing_before_align[:12],
                "diag_extra_cols_preview": extra_before_align[:12],

                "diag_alias_enabled": int(alias_diag.get("alias_enabled", 0)),
                "diag_alias_added_total": int(alias_diag.get("alias_added_total", 0)),
                "diag_alias_added_matching_expected": int(alias_diag.get("alias_added_matching_expected", 0)),
                "diag_alias_matching_expected_preview": alias_diag.get("alias_matching_expected_preview", []),

                "diag_nonzero_vals_preview": [{"feature": k, "value": round(float(v), 6)} for k, v in top_pairs],
            }

            if label == "W":
                out["confidence"] = out["pA_W"]
            elif label == "R":
                out["confidence"] = out["pB_R"] if out["pB_R"] is not None else None
            elif label in {"N1", "N2", "N3"}:
                out["confidence"] = out.get(f"pC_{label}", None)
            else:
                out["confidence"] = None

            MAX_PRED_HISTORY = 2880
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
