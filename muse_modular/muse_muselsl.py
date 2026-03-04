from __future__ import annotations

import os
import re
import sys
import time
import subprocess

from muse_config import CONNECT_RETRY_TOTAL_S, CONNECT_RETRY_STEP_S
from muse_lsl import LSLReceiver

def start_muselsl_stream():
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(
        [sys.executable, "-m", "muselsl", "stream", "--ppg"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

def stop_proc(p):
    if p is None:
        return
    try:
        p.terminate()
    except Exception:
        pass

def scan_for_muse(timeout_s: int = 10):
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "muselsl", "list"],
            capture_output=True,
            text=True,
            timeout=max(5, int(timeout_s) + 2),
        )
        raw = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        macs = re.findall(r"([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})", raw)
        macs = list(dict.fromkeys([m.upper() for m in macs]))
        return (len(macs) > 0), macs, raw
    except Exception as e:
        return False, [], repr(e)

def connect_with_retries(eeg: LSLReceiver, ppg: LSLReceiver) -> tuple[bool, bool]:
    t0 = time.time()
    ok_eeg = bool(eeg.running)
    ok_ppg = bool(ppg.running)

    while time.time() - t0 < CONNECT_RETRY_TOTAL_S:
        if not ok_eeg:
            ok_eeg = eeg.start(timeout_s=CONNECT_RETRY_STEP_S)
        if not ok_ppg:
            ok_ppg = ppg.start(timeout_s=CONNECT_RETRY_STEP_S)

        if ok_eeg and ok_ppg:
            break

        time.sleep(0.15)

    return ok_eeg, ok_ppg
