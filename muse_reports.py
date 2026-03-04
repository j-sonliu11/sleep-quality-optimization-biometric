from __future__ import annotations

import io
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth

from muse_features import ensure_sleep_data_dir
from muse_config import EPOCH_SEC


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