# predict_muse.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def _safe_import_xgboost():
    try:
        import xgboost as xgb
        return xgb
    except Exception as e:
        raise SystemExit(
            "\n[ERROR] xgboost is not installed in this environment.\n"
            "Run: pip install xgboost\n"
            f"Details: {e}\n"
        )

def _try_load_joblib_bundle(bundle_path: Path):
    """
    Best-case: your training code saved *everything* needed (models + expected columns).
    If loading fails, we fall back to JSON models.
    """
    try:
        import joblib
        bundle = joblib.load(bundle_path)
        return bundle
    except Exception as e:
        print(f"[WARN] Could not load joblib bundle: {bundle_path}")
        print(f"       Reason: {e}")
        return None

def _load_booster_from_json(xgb, json_path: Path):
    booster = xgb.Booster()
    booster.load_model(str(json_path))
    return booster

def _infer_feature_cols(df: pd.DataFrame):
    # Common non-feature columns to drop
    drop = {"Sleep_Stage", "TIMESTAMP", "ISO_TIME", "time", "label", "y"}
    cols = [c for c in df.columns if c not in drop]
    return cols

def _align_columns(X: pd.DataFrame, expected_cols: list[str] | None):
    """
    Ensure X has exactly expected_cols in that order.
    - adds missing cols as 0
    - drops extras
    """
    if not expected_cols:
        return X

    X2 = X.copy()
    missing = [c for c in expected_cols if c not in X2.columns]
    extra = [c for c in X2.columns if c not in expected_cols]

    if missing:
        print(f"[WARN] Missing {len(missing)} expected columns. Filling with 0.")
        # Fill missing with zeros (better than NaN for trees)
        for c in missing:
            X2[c] = 0.0

    if extra:
        print(f"[INFO] Dropping {len(extra)} extra columns not used by model.")
        X2 = X2.drop(columns=extra)

    X2 = X2[expected_cols]
    return X2

def _coerce_numeric(X: pd.DataFrame):
    Xn = X.apply(pd.to_numeric, errors="coerce")
    # Replace inf, NaN
    Xn = Xn.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return Xn

def _predict_cascade_from_boosters(xgb, boosterA, boosterB, boosterC, X: pd.DataFrame):
    """
    Uses raw Booster.predict on DMatrix.
    Class order (from your metrics):
      Stage A classes: ["S","W"]
      Stage B classes: ["NREM","R"]
      Stage C classes: ["N1","N2","N3"]
    """
    dmat = xgb.DMatrix(X, feature_names=list(X.columns))

    probaA = boosterA.predict(dmat)  # (n,2)
    # indices: 0="S", 1="W"
    isW = probaA[:, 1] >= 0.5
    pred = np.array(["S"] * len(X), dtype=object)
    pred[isW] = "W"

    # Only for those predicted S: run stage B
    probaB = np.full((len(X), 2), np.nan, dtype=float)
    idxS = np.where(~isW)[0]
    if len(idxS) > 0:
        dS = xgb.DMatrix(X.iloc[idxS], feature_names=list(X.columns))
        pb = boosterB.predict(dS)  # (m,2) 0="NREM",1="R"
        probaB[idxS] = pb
        isR = pb[:, 1] >= 0.5
        pred[idxS[isR]] = "R"
        pred[idxS[~isR]] = "NREM"

        # Only for those predicted NREM: run stage C
        probaC = np.full((len(X), 3), np.nan, dtype=float)
        idxNREM = idxS[~isR]
        if len(idxNREM) > 0:
            dN = xgb.DMatrix(X.iloc[idxNREM], feature_names=list(X.columns))
            pc = boosterC.predict(dN)  # (k,3) 0=N1,1=N2,2=N3
            probaC[idxNREM] = pc
            c_idx = np.argmax(pc, axis=1)
            pred[idxNREM[c_idx == 0]] = "N1"
            pred[idxNREM[c_idx == 1]] = "N2"
            pred[idxNREM[c_idx == 2]] = "N3"
    else:
        probaC = np.full((len(X), 3), np.nan, dtype=float)

    return pred, probaA, probaB, probaC

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muse_csv", required=True, help="Path to Muse epoch-feature CSV")
    ap.add_argument("--run_dir", required=True, help="Path to your DreamT run folder (contains model_bundle.joblib and/or stage JSONs)")
    ap.add_argument("--out_csv", default="", help="Optional output CSV path")
    args = ap.parse_args()

    muse_csv = Path(args.muse_csv)
    run_dir = Path(args.run_dir)

    # ---- Load Muse data ----
    df = pd.read_csv(muse_csv)
    feat_cols = _infer_feature_cols(df)
    X = df[feat_cols].copy()
    X = _coerce_numeric(X)

    # ---- Try joblib bundle first ----
    bundle_path = run_dir / "model_bundle.joblib"
    bundle = _try_load_joblib_bundle(bundle_path)

    expected_cols = None
    if isinstance(bundle, dict):
        # Try a few common keys your training code might have used
        for k in ["feature_cols", "feature_columns", "expected_cols", "columns"]:
            if k in bundle and isinstance(bundle[k], (list, tuple)):
                expected_cols = list(bundle[k])
                print(f"[INFO] Found expected columns in bundle key: {k} ({len(expected_cols)} cols)")
                break

    X_aligned = _align_columns(X, expected_cols)
    X_aligned = _coerce_numeric(X_aligned)

    # ---- Load models (bundle OR JSON fallback) ----
    xgb = _safe_import_xgboost()

    boosterA = boosterB = boosterC = None

    # Bundle case: try to find models in dict
    if isinstance(bundle, dict):
        # common keys
        boosterA = bundle.get("stageA") or bundle.get("model_stageA") or bundle.get("modelA") or bundle.get("xgb_stageA")
        boosterB = bundle.get("stageB") or bundle.get("model_stageB") or bundle.get("modelB") or bundle.get("xgb_stageB")
        boosterC = bundle.get("stageC") or bundle.get("model_stageC") or bundle.get("modelC") or bundle.get("xgb_stageC")

        # If they’re XGBClassifier objects, we can extract booster
        def to_booster(m):
            if m is None:
                return None
            if hasattr(m, "get_booster"):
                return m.get_booster()
            if isinstance(m, xgb.Booster):
                return m
            return None

        boosterA, boosterB, boosterC = map(to_booster, [boosterA, boosterB, boosterC])

    # Fallback: load from JSON files (these filenames match what you uploaded)
    if boosterA is None or boosterB is None or boosterC is None:
        print("[INFO] Using JSON model fallback.")
        jsonA = run_dir / "xgb_stageA_W_vs_S.json"
        jsonB = run_dir / "xgb_stageB_R_vs_NREM.json"
        jsonC = run_dir / "xgb_stageC_N1_vs_N2_vs_N3.json"
        if not (jsonA.exists() and jsonB.exists() and jsonC.exists()):
            raise SystemExit(
                "\n[ERROR] Could not find required model files.\n"
                f"Looked for:\n  {jsonA}\n  {jsonB}\n  {jsonC}\n"
                "Make sure your run_dir contains these.\n"
            )
        boosterA = _load_booster_from_json(xgb, jsonA)
        boosterB = _load_booster_from_json(xgb, jsonB)
        boosterC = _load_booster_from_json(xgb, jsonC)

    # ---- Predict ----
    pred, probaA, probaB, probaC = _predict_cascade_from_boosters(xgb, boosterA, boosterB, boosterC, X_aligned)

    out = df.copy()
    out["pred_stage"] = pred
    # Stage A probs: S, W
    out["pA_S"] = probaA[:, 0]
    out["pA_W"] = probaA[:, 1]
    # Stage B probs: NREM, R
    out["pB_NREM"] = probaB[:, 0]
    out["pB_R"] = probaB[:, 1]
    # Stage C probs: N1, N2, N3
    out["pC_N1"] = probaC[:, 0]
    out["pC_N2"] = probaC[:, 1]
    out["pC_N3"] = probaC[:, 2]

    # ---- Save ----
    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        out_csv = muse_csv.with_name(muse_csv.stem + "__PREDICTED.csv")

    out.to_csv(out_csv, index=False)
    print(f"\n[DONE] Wrote predictions to:\n  {out_csv}\n")

if __name__ == "__main__":
    main()
