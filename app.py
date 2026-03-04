from __future__ import annotations

import time
import traceback
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go

from muse_config import (
    EEG_FS,
    EEG_CHANNEL_NAMES,
    PPG_FS,
    PPG_CHANNEL_NAMES,
    EPOCH_SEC,
)
from muse_ui_css import inject_css
from muse_session import get_config_store, synced_slider_number
from muse_lsl import LSLReceiver
from muse_muselsl import (
    start_muselsl_stream,
    stop_proc,
    scan_for_muse,
    connect_with_retries,
)
from muse_server import ensure_data_server
from muse_snapshot import save_snapshot_csv
from muse_infer import LiveSleepInference
from muse_collect import EpochCollector
from muse_features import ensure_sleep_data_dir
from muse_reports import save_prediction_summary_image, save_run_report_pdf
from muse_ui_html import render_html


def _show_fatal(e: Exception):
    st.error("❌ App error (see details below)")
    st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))


def get_default_model_run_dir() -> Path:
    return Path(
        r"F:\! Senior Design Project\physionet\physionet.org\files\XGBoost\runs\run_20260210_025057"
    )


def _restart_stream_and_reconnect(eeg_receiver, ppg_receiver, collector):
    # Optional: do NOT stop collector; it can tolerate stalls/restarts.
    # If you prefer, uncomment the stop/join lines.
    # collector.stop(reason="Auto-restart stream")
    # collector.join(timeout=5.0)

    # stop muselsl process if running
    stop_proc(st.session_state.get("muselsl_proc", None))
    st.session_state["muselsl_proc"] = None

    # clear receivers so they don't cling to dead stream metadata/source_id
    eeg_receiver.stop_and_clear()
    ppg_receiver.stop_and_clear()

    # restart muselsl
    st.session_state["muselsl_proc"] = start_muselsl_stream()
    time.sleep(1.2)  # give LSL outlets time to appear

    # reconnect receivers
    ok_eeg, ok_ppg = connect_with_retries(eeg_receiver, ppg_receiver)

    ui = st.session_state.get("ui_state", {})
    ui["lsl_connected_eeg"] = bool(ok_eeg)
    ui["lsl_connected_ppg"] = bool(ok_ppg)
    st.session_state["ui_state"] = ui

    # log throttling info
    st.session_state["auto_restart_last_ts"] = time.time()
    try:
        st.toast("🔁 Auto-reconnected Muse stream", icon="🔁")
    except Exception:
        # older streamlit versions may not have toast
        pass


def _auto_reconnect_watchdog(eeg_receiver, ppg_receiver, collector):
    """
    Detect stream stalls (sample_count not increasing) and automatically
    restart muselsl + reconnect.

    This runs inside Streamlit (main thread), so keep it fast and throttle.
    """
    now = time.time()

    # init state
    if "auto_restart_last_ts" not in st.session_state:
        st.session_state["auto_restart_last_ts"] = 0.0
    if "wd_prev_eeg_sc" not in st.session_state:
        st.session_state["wd_prev_eeg_sc"] = int(getattr(eeg_receiver, "sample_count", 0))
    if "wd_prev_ppg_sc" not in st.session_state:
        st.session_state["wd_prev_ppg_sc"] = int(getattr(ppg_receiver, "sample_count", 0))
    if "wd_eeg_stall_s" not in st.session_state:
        st.session_state["wd_eeg_stall_s"] = 0.0
    if "wd_ppg_stall_s" not in st.session_state:
        st.session_state["wd_ppg_stall_s"] = 0.0
    if "wd_last_check_ts" not in st.session_state:
        st.session_state["wd_last_check_ts"] = now

    dt = max(0.0, now - float(st.session_state["wd_last_check_ts"]))
    st.session_state["wd_last_check_ts"] = now

    # Only auto-reconnect when collecting (overnight behavior)
    collecting = bool(collector.status().get("collecting", False))
    if not collecting:
        # reset stall timers when not collecting
        st.session_state["wd_eeg_stall_s"] = 0.0
        st.session_state["wd_ppg_stall_s"] = 0.0
        st.session_state["wd_prev_eeg_sc"] = int(getattr(eeg_receiver, "sample_count", 0))
        st.session_state["wd_prev_ppg_sc"] = int(getattr(ppg_receiver, "sample_count", 0))
        return

    # Update stall timers based on sample_count changes
    eeg_sc = int(getattr(eeg_receiver, "sample_count", 0))
    ppg_sc = int(getattr(ppg_receiver, "sample_count", 0))

    if eeg_receiver.running and (not eeg_receiver.paused) and eeg_sc == int(st.session_state["wd_prev_eeg_sc"]):
        st.session_state["wd_eeg_stall_s"] += dt
    else:
        st.session_state["wd_eeg_stall_s"] = 0.0

    if ppg_receiver.running and (not ppg_receiver.paused) and ppg_sc == int(st.session_state["wd_prev_ppg_sc"]):
        st.session_state["wd_ppg_stall_s"] += dt
    else:
        st.session_state["wd_ppg_stall_s"] = 0.0

    st.session_state["wd_prev_eeg_sc"] = eeg_sc
    st.session_state["wd_prev_ppg_sc"] = ppg_sc

    # Thresholds
    STALL_TRIGGER_S = 12.0     # how long with no new samples before restart
    COOLDOWN_S = 45.0          # minimum time between restarts

    stalled = (st.session_state["wd_eeg_stall_s"] >= STALL_TRIGGER_S) or (
        st.session_state["wd_ppg_stall_s"] >= STALL_TRIGGER_S
    )

    if stalled and (now - float(st.session_state["auto_restart_last_ts"]) >= COOLDOWN_S):
        _restart_stream_and_reconnect(eeg_receiver, ppg_receiver, collector)


def main():
    st.set_page_config(page_title="Muse 2 Live", layout="wide")
    inject_css()

    try:
        st.title("🧠 Muse 2 Live Dashboard")

        # ---- Config store
        cfg_store = get_config_store()

        # ---- Sidebar
        with st.sidebar:
            st.header("Settings")
            rolling_sec = synced_slider_number(
                "Rolling window (s)", 2.0, 30.0, 10.0, 1.0, "rolling_sec", fmt="%.2f"
            )
            bp_low = synced_slider_number(
                "Band-pass low (Hz)", 0.1, 30.0, 1.0, 0.1, "bp_low", fmt="%.2f"
            )
            bp_high = synced_slider_number(
                "Band-pass high (Hz)", 10.0, 60.0, 45.0, 0.5, "bp_high", fmt="%.2f"
            )

            use_notch = st.checkbox("Notch filter", value=True)
            notch_label = st.selectbox(
                "Line freq (Hz)",
                ["60.0 Hz (NA)", "50.0 Hz (EU, AS, AF)"],
                index=0,
            )
            notch_freq = 60.0 if notch_label.startswith("60") else 50.0

            psd_len = synced_slider_number(
                "PSD window (s)", 1.0, 8.0, 4.0, 1.0, "psd_len", fmt="%.2f"
            )
            update_ms = synced_slider_number(
                "Graph update (ms)", 50.0, 1000.0, 200.0, 50.0, "update_ms", fmt="%.0f"
            )
            offset_uv = synced_slider_number(
                "Trace vertical offset (µV)", 50.0, 800.0, 320.0, 10.0, "offset_uv", fmt="%.0f"
            )
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

        # ---- Session state: receivers
        if "eeg_receiver" not in st.session_state:
            st.session_state["eeg_receiver"] = LSLReceiver(
                "EEG", EEG_FS, len(EEG_CHANNEL_NAMES)
            )
        if "ppg_receiver" not in st.session_state:
            st.session_state["ppg_receiver"] = LSLReceiver(
                "PPG", PPG_FS, len(PPG_CHANNEL_NAMES)
            )
        eeg_receiver = st.session_state["eeg_receiver"]
        ppg_receiver = st.session_state["ppg_receiver"]

        # ---- Inference + collector
        if "live_infer" not in st.session_state:
            st.session_state["live_infer"] = LiveSleepInference(get_default_model_run_dir())
        live_infer = st.session_state["live_infer"]

        if "collector" not in st.session_state:
            st.session_state["collector"] = EpochCollector(
                eeg_receiver, ppg_receiver, cfg_store, infer_engine=live_infer
            )
        collector = st.session_state["collector"]
        collector.cfg_store = cfg_store
        collector.infer_engine = live_infer

        # ---- misc UI state
        if "muselsl_proc" not in st.session_state:
            st.session_state["muselsl_proc"] = None
        if "scan_state" not in st.session_state:
            st.session_state["scan_state"] = {"scanning": False, "macs": [], "last_raw": ""}
        if "ui_state" not in st.session_state:
            st.session_state["ui_state"] = {
                "muse_found": False,
                "lsl_connected_eeg": False,
                "lsl_connected_ppg": False,
            }

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

                if started_new:
                    time.sleep(1.2)

                ok_eeg, ok_ppg = connect_with_retries(eeg_receiver, ppg_receiver)

                ui = st.session_state["ui_state"]
                ui["lsl_connected_eeg"] = bool(ok_eeg)
                ui["lsl_connected_ppg"] = bool(ok_ppg)
                st.session_state["ui_state"] = ui
                st.rerun()

        # Connection badges
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
                st.rerun()

        with st.expander("Advanced", expanded=False):
            if st.button("⏹ Stop + Clear Data", use_container_width=True):
                collector.stop()
                eeg_receiver.stop_and_clear()
                ppg_receiver.stop_and_clear()
                stop_proc(st.session_state["muselsl_proc"])
                st.session_state["muselsl_proc"] = None
                st.session_state["ui_state"] = {
                    "muse_found": False,
                    "lsl_connected_eeg": False,
                    "lsl_connected_ppg": False,
                }
                st.session_state["scan_state"] = {"scanning": False, "macs": [], "last_raw": ""}
                st.warning("Stopped and cleared buffer.")
                st.rerun()

        scan = st.session_state["scan_state"]
        if scan.get("macs"):
            st.write("Detected MAC address(es):")
            for m in scan["macs"]:
                st.code(m)

        # Running badge
        if eeg_receiver.running and eeg_receiver.paused:
            st.markdown(
                '<div class="run-badge"><span class="small-muted">Paused</span></div>',
                unsafe_allow_html=True,
            )
        elif eeg_receiver.running:
            st.markdown(
                '<div class="run-badge"><div class="spinner"></div><span class="small-muted">Running</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="run-badge"><span class="small-muted">Idle</span></div>',
                unsafe_allow_html=True,
            )

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

        if show_debug:
            with st.expander("Debug", expanded=True):
                st.write("CFG (current):", cfg_store.get())
                st.write("EEG running/paused:", eeg_receiver.running, eeg_receiver.paused)
                st.write(
                    "EEG sample_count:",
                    eeg_receiver.sample_count,
                    "last_ts:",
                    eeg_receiver.last_ts,
                    "err:",
                    eeg_receiver.last_error,
                )
                st.write("PPG running/paused:", ppg_receiver.running, ppg_receiver.paused)
                st.write(
                    "PPG sample_count:",
                    ppg_receiver.sample_count,
                    "last_ts:",
                    ppg_receiver.last_ts,
                    "err:",
                    ppg_receiver.last_error,
                )
                st.write("PPG meta:", ppg_receiver.stream_meta)
                st.write("collector:", collector.status())

        st.divider()
        st.markdown('<div id="dc-row-anchor"></div>', unsafe_allow_html=True)

        # -----------------------------
        # Main layout: Charts (left) + Panel (right)
        # -----------------------------
        left, right = st.columns([3, 1], gap="large")

        @st.fragment(run_every=5.0)
        def render_collection_status():
            col_stat = collector.status()
            st.write("Epoch rows written:", col_stat["rows_written"])
            if col_stat["last_write_iso"]:
                st.write("Last epoch write:", col_stat["last_write_iso"])
            if col_stat["csv_path"]:
                st.code(Path(col_stat["csv_path"]).name)
            if col_stat["last_err"]:
                st.error(col_stat["last_err"])

        @st.fragment(run_every=5.0)
        def render_live_predictions_panel():
            st.markdown("#### Live Sleep Stage Predictions (30s epochs)")

            infer_engine = st.session_state["live_infer"]  # always fresh
            s = infer_engine.status()

            if not s["ready"]:
                st.info("Load your DreamT XGBoost model folder to enable live inference.")
                if s.get("last_err"):
                    st.error(s["last_err"])
                return

            hist_all = infer_engine.get_history(None)
            if not hist_all:
                st.caption("No epoch predictions yet. Start data collection to generate 30s epochs.")
                return

            latest = hist_all[-1]

            stage_order = ["W", "N1", "N2", "N3", "R"]
            stage_colors = {
                "W": "#f59e0b",
                "N1": "#60a5fa",
                "N2": "#22c55e",
                "N3": "#8b5cf6",
                "R": "#ef4444",
                "S": "#9ca3af",
            }
            stage_card_colors = {
                "W": ("#f59e0b", "rgba(245,158,11,0.14)", "rgba(245,158,11,0.40)"),
                "N1": ("#60a5fa", "rgba(96,165,250,0.14)", "rgba(96,165,250,0.40)"),
                "N2": ("#22c55e", "rgba(34,197,94,0.14)", "rgba(34,197,94,0.40)"),
                "N3": ("#8b5cf6", "rgba(139,92,246,0.14)", "rgba(139,92,246,0.40)"),
                "R": ("#ef4444", "rgba(239,68,68,0.14)", "rgba(239,68,68,0.40)"),
                "S": ("#a3a3a3", "rgba(163,163,163,0.12)", "rgba(163,163,163,0.30)"),
            }

            def _stage_style(stage: str):
                return stage_card_colors.get(
                    stage, ("#d1d5db", "rgba(255,255,255,0.06)", "rgba(255,255,255,0.18)")
                )

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

            # ---------- card ----------
            stage = str(latest.get("pred_stage", "—")).upper().strip()
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

            # show last ~2h for performance
            max_show = 240  # 2 hours @ 30s
            hist = hist_all[-max_show:] if len(hist_all) > max_show else hist_all

            # ---------- stage distribution ----------
            counts = {k: 0 for k in stage_order}
            for h in hist:
                stg = str(h.get("pred_stage", "")).upper().strip()
                if stg in counts:
                    counts[stg] += 1

            fig_hist = go.Figure(
                data=[
                    go.Bar(
                        x=stage_order,
                        y=[counts[k] for k in stage_order],
                        marker_color=[stage_colors[k] for k in stage_order],
                        text=[counts[k] for k in stage_order],
                        textposition="outside",
                        hovertemplate="Stage %{x}<br>Count: %{y}<extra></extra>",
                    )
                ]
            )
            fig_hist.update_layout(
                height=230,
                margin=dict(l=20, r=10, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(color="rgba(255,255,255,0.92)"),
                xaxis=dict(showgrid=False, zeroline=False, title=None, tickfont=dict(size=12)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, title=None, dtick=1),
                showlegend=False,
            )
            st.markdown("**Stage distribution (recent)**")
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

            # ---------- epoch timeline strip ----------
            st.markdown("**Epoch timeline (oldest → newest)**")
            strip_blocks = []
            for h in hist:
                stg = str(h.get("pred_stage", "—")).upper().strip()
                idx = h.get("epoch_index", "")
                c = stage_colors.get(stg, "#9ca3af")
                conf2 = h.get("confidence", None)
                conf_txt2 = "—" if conf2 is None else f"{100.0 * float(conf2):.1f}%"
                strip_blocks.append(
                    f'<div title="Epoch #{idx} | Stage: {stg} | Conf: {conf_txt2}" '
                    f'style="width:12px;height:22px;border-radius:4px;'
                    f'background:{c};border:1px solid rgba(255,255,255,0.18);'
                    f'box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);"></div>'
                )

            st.markdown(
                '<div style="display:flex;align-items:center;gap:4px;flex-wrap:nowrap;'
                'overflow-x:auto;padding:6px 2px 2px 2px;margin-bottom:0.25rem;">'
                + "".join(strip_blocks)
                + "</div>",
                unsafe_allow_html=True,
            )

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

            # ---------- live hypnogram ----------
            st.markdown("**Live hypnogram**")

            ys = []
            colors = []
            hover = []

            for h in hist:
                stg = str(h.get("pred_stage", "W")).upper().strip()
                if stg not in stage_order:
                    stg = "W"
                ys.append(stg)
                colors.append(stage_colors.get(stg, "#9ca3af"))
                hover.append(f"Epoch {h.get('epoch_index','')}: {stg}")

            # Each epoch block is 0.5 minutes (30s)
            fig_hyp = go.Figure()
            fig_hyp.add_trace(
                go.Bar(
                    x=[0.5] * len(ys),
                    y=ys,
                    orientation="h",
                    marker=dict(color=colors),
                    hovertext=hover,
                    hoverinfo="text",
                )
            )
            fig_hyp.update_layout(
                height=240,
                barmode="stack",
                margin=dict(l=35, r=10, t=10, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(color="rgba(255,255,255,0.92)"),
                showlegend=False,
                xaxis=dict(
                    title="Time (min) (recent window)",
                    showgrid=True,
                    gridcolor="rgba(255,255,255,0.08)",
                    zeroline=False,
                ),
                yaxis=dict(
                    title=None,
                    categoryorder="array",
                    categoryarray=stage_order[::-1],
                    showgrid=False,
                ),
            )
            st.plotly_chart(fig_hyp, use_container_width=True, config={"displayModeBar": False})

            with st.expander("Details (probabilities + table)", expanded=False):
                st.write(
                    {
                        "pA_S": latest.get("pA_S"),
                        "pA_W": latest.get("pA_W"),
                        "pB_NREM": latest.get("pB_NREM"),
                        "pB_R": latest.get("pB_R"),
                        "pC_N1": latest.get("pC_N1"),
                        "pC_N2": latest.get("pC_N2"),
                        "pC_N3": latest.get("pC_N3"),
                    }
                )
                dfh = pd.DataFrame(hist)
                if "confidence" in dfh.columns:
                    dfh["confidence"] = dfh["confidence"].apply(
                        lambda v: None if v is None else round(float(v), 3)
                    )
                show_cols = [c for c in ["epoch_index", "timestamp_s", "pred_stage", "confidence", "iso_time"] if c in dfh.columns]
                st.dataframe(dfh.iloc[::-1][show_cols], use_container_width=True, height=260)

        with right:
            # Watchdog tick: runs only while collecting (see watchdog function)
            @st.fragment(run_every=5.0)
            def _watchdog_tick():
                _auto_reconnect_watchdog(eeg_receiver, ppg_receiver, collector)

            _watchdog_tick()

            st.subheader("Data Collection")
            col_stat = collector.status()
            collecting = bool(col_stat["collecting"])

            if st.button("💾 Save Snapshot", key="dc_snapshot", use_container_width=True):
                if not eeg_receiver.running:
                    st.info("Connect first to save a snapshot.")
                else:
                    p = save_snapshot_csv(
                        eeg_receiver,
                        ppg_receiver,
                        float(cfg_store.get()["rolling_sec"]),
                    )
                    if p is None:
                        st.error("No EEG data available to snapshot yet.")
                    else:
                        st.success(f"Saved snapshot:\n{Path(p).name}")

            b1, b2 = st.columns(2, gap="small")
            with b1:
                start_clicked = st.button(
                    "▶ Start",
                    key="dc_start",
                    use_container_width=True,
                    disabled=collecting,
                )
            with b2:
                stop_clicked = st.button(
                    "⏹ Stop",
                    key="dc_stop",
                    use_container_width=True,
                    disabled=not collecting,
                )

            if start_clicked:
                ok = collector.start()
                if not ok:
                    st.error(collector.status().get("last_err") or "Could not start collecting.")
                else:
                    st.success("Collecting started (30s epochs).")
                st.rerun()

            if stop_clicked:
                collector.stop(reason="Stop button pressed")
                collector.join(timeout=15.0)

                out_dir = collector.session_dir if collector.session_dir is not None else ensure_sleep_data_dir()

                img_path = save_prediction_summary_image(st.session_state["live_infer"], out_dir, max_epochs=120)
                if img_path is not None:
                    st.success(f"Saved prediction summary image: {Path(img_path).name}")
                else:
                    st.info("No prediction summary image saved (no predictions yet).")

                try:
                    epochs_csv = collector.csv_path
                    if epochs_csv and Path(epochs_csv).exists():
                        pdf_path = Path(out_dir) / "run_report.pdf"
                        save_run_report_pdf(
                            Path(epochs_csv),
                            pdf_path,
                            title="Muse Sleep Run Report",
                            epoch_seconds=int(EPOCH_SEC),
                        )
                        st.success(f"Saved run report PDF: {pdf_path.name}")
                    else:
                        st.warning("Run report not saved (epochs CSV not found).")
                except Exception as e:
                    st.error(f"Could not save run report PDF: {e!r}")

                stat = collector.status()
                if stat.get("log_path"):
                    st.caption(f"Debug log: {Path(stat['log_path']).name} (in the run folder)")

                st.rerun()

            collecting = bool(collector.status()["collecting"])
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

            st.caption("Files save to a folder named **live prediction runs** next to the modular files.")
            render_collection_status()

            st.markdown("### Live Sleep Stage Inference")

            if "model_run_dir_str" not in st.session_state:
                st.session_state["model_run_dir_str"] = str(get_default_model_run_dir())

            new_model_dir = st.text_input("Model run folder", key="model_run_dir_str")

            c_reload_model, c_model_status = st.columns([1, 1], gap="small")
            with c_reload_model:
                if st.button("🔄 Reload Model", use_container_width=True):
                    st.session_state["live_infer"] = LiveSleepInference(Path(new_model_dir.strip()))
                    # IMPORTANT: re-wire collector to the NEW model instance
                    st.session_state["collector"].infer_engine = st.session_state["live_infer"]
                    st.rerun()

            with c_model_status:
                inf_stat = st.session_state["live_infer"].status()
                if inf_stat["ready"]:
                    st.success("Model ready ✅")
                else:
                    st.warning("Model not ready")

            # Live prediction visuals under the model section
            render_live_predictions_panel()

        with left:
            port = ensure_data_server(eeg_receiver, ppg_receiver, cfg_store, st.session_state)
            html = render_html(port)
            components.html(html, height=1900, scrolling=False)

    except Exception as e:
        _show_fatal(e)


if __name__ == "__main__":
    main()