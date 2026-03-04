from __future__ import annotations

import threading
from dataclasses import dataclass

import streamlit as st

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
