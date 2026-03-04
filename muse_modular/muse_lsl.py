from __future__ import annotations

import time
import threading
from collections import deque

import numpy as np
from pylsl import StreamInlet, resolve_byprop

from muse_config import MAX_BUF_SECONDS

class LSLReceiver:
    def __init__(self, stream_type: str, fs: float, n_chan: int, max_seconds: float = MAX_BUF_SECONDS):
        self.stream_type = str(stream_type)
        self.fs = float(fs)
        self.n_chan = int(n_chan)
        self.maxlen = int(max_seconds * fs)

        self.ts = deque(maxlen=self.maxlen)
        self.buf = deque(maxlen=self.maxlen)
        self.lock = threading.RLock()

        self.running = False
        self.paused = False
        self.thread = None
        self.inlet = None
        self.last_error = None
        self.stream_meta = None

        self.sample_count = 0
        self.last_ts = None

        self._reconnecting = False

    def start(self, timeout_s: float = 2.0, prefer_name_contains: str = "muse") -> bool:
        self.paused = False
        deadline = time.time() + max(0.0, float(timeout_s))

        type_candidates = [self.stream_type]
        if self.stream_type.upper() == "PPG":
            type_candidates = ["PPG", "AUX"]

        while True:
            try:
                streams = []
                for tval in type_candidates:
                    streams = resolve_byprop("type", tval, timeout=0.35)
                    if streams:
                        break

                if streams:
                    best, best_score = None, -1
                    for s in streams:
                        try:
                            ch = s.channel_count()
                            fs = s.nominal_srate()
                            name = s.name()
                            typ = s.type()
                        except Exception:
                            continue

                        score = 0
                        n = str(name).lower()
                        if prefer_name_contains and prefer_name_contains in n:
                            score += 4
                        if ch == self.n_chan:
                            score += 3
                        if fs > 0 and abs(fs - self.fs) < max(6.0, 0.25 * self.fs):
                            score += 2
                        if str(typ).lower() in [x.lower() for x in type_candidates]:
                            score += 1

                        if score > best_score:
                            best_score, best = score, s

                    if best is not None:
                        new_inlet = StreamInlet(best, max_buflen=60)

                        try:
                            source_id = best.source_id()
                        except Exception:
                            source_id = None

                        with self.lock:
                            self.inlet = new_inlet
                            self.stream_meta = {
                                "name": best.name(),
                                "type": best.type(),
                                "source_id": source_id,
                                "channels": best.channel_count(),
                                "fs": best.nominal_srate(),
                            }
                            self.running = True
                            self.paused = False
                            self.last_error = None
                            self.ts.clear()
                            self.buf.clear()
                            self.sample_count = 0
                            self.last_ts = None

                        self.thread = threading.Thread(target=self._run, daemon=True)
                        self.thread.start()
                        return True

            except Exception as e:
                self.last_error = repr(e)

            if time.time() >= deadline:
                return False

    def _run(self):
        last_ok = time.time()
        backoff = 0.6

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            try:
                with self.lock:
                    inlet = self.inlet

                if inlet is None:
                    self.last_error = "Inlet is None -> reconnect"
                    self._reconnect_inlet(backoff=backoff)
                    backoff = min(8.0, backoff * 1.5)
                    last_ok = time.time()
                    continue

                sample, ts = inlet.pull_sample(timeout=0.5)

            except Exception as e:
                self.last_error = f"pull_sample error: {e!r} -> reconnect"
                self._reconnect_inlet(backoff=backoff)
                backoff = min(8.0, backoff * 1.5)
                last_ok = time.time()
                continue

            if sample is None or ts is None:
                if time.time() - last_ok > 10.0:
                    self.last_error = "No samples for 10s -> reconnect"
                    self._reconnect_inlet(backoff=backoff)
                    backoff = min(8.0, backoff * 1.5)
                    last_ok = time.time()
                continue

            backoff = 0.6
            last_ok = time.time()

            try:
                s = np.asarray(sample, dtype=float).reshape(-1)

                if s.size < self.n_chan:
                    s = np.pad(s, (0, self.n_chan - s.size), constant_values=np.nan)
                elif s.size > self.n_chan:
                    s = s[: self.n_chan]

                with self.lock:
                    self.ts.append(float(ts))
                    self.buf.append(s.tolist())
                    self.sample_count += 1
                    self.last_ts = float(ts)
            except Exception as e:
                self.last_error = f"Buffer append error: {e!r}"
                continue

    def _reconnect_inlet(self, backoff: float = 1.0):
        if self._reconnecting:
            return
        self._reconnecting = True

        try:
            time.sleep(float(backoff))

            target_type = getattr(self, "stream_type", None)
            target_name = None
            target_source_id = None

            meta = getattr(self, "stream_meta", None)
            if meta:
                target_name = meta.get("name")
                target_source_id = meta.get("source_id")

            streams = []

            if target_source_id:
                try:
                    streams.extend(resolve_byprop("source_id", target_source_id, timeout=1.0))
                except Exception:
                    pass

            if not streams and target_name:
                try:
                    streams.extend(resolve_byprop("name", target_name, timeout=1.0))
                except Exception:
                    pass

            if not streams and target_type:
                type_candidates = [str(target_type)]
                if str(target_type).upper() == "PPG":
                    type_candidates = ["PPG", "AUX"]

                for tval in type_candidates:
                    try:
                        streams = resolve_byprop("type", tval, timeout=1.0)
                        if streams:
                            break
                    except Exception:
                        continue

            if not streams:
                self.last_error = (
                    f"Reconnect failed: no LSL streams found "
                    f"(source_id={target_source_id!r}, name={target_name!r}, type={target_type!r})"
                )
                return

            best, best_score = None, -1
            for s in streams:
                try:
                    ch = s.channel_count()
                    fs = s.nominal_srate()
                    name = s.name()
                    typ = s.type()
                    sid = getattr(s, "source_id", lambda: None)()
                except Exception:
                    continue

                score = 0
                if target_source_id and sid == target_source_id:
                    score += 10
                if target_name and str(name) == str(target_name):
                    score += 6
                if ch == self.n_chan:
                    score += 3
                if fs > 0 and abs(fs - self.fs) < max(6.0, 0.25 * self.fs):
                    score += 2
                if target_type and str(typ).lower() == str(target_type).lower():
                    score += 1

                if score > best_score:
                    best_score, best = score, s

            if best is None:
                self.last_error = "Reconnect failed: streams found but none scored as usable."
                return

            new_inlet = StreamInlet(best, max_buflen=60)

            ok = False
            for _ in range(2):
                samp, tss = new_inlet.pull_sample(timeout=0.25)
                if samp is not None and tss is not None:
                    ok = True
                    break

            with self.lock:
                self.inlet = new_inlet
                self.ts.clear()
                self.buf.clear()
                self.sample_count = 0
                self.last_ts = None

                try:
                    self.stream_meta = {
                        "name": best.name(),
                        "type": best.type(),
                        "source_id": getattr(best, "source_id", lambda: None)(),
                        "channels": best.channel_count(),
                        "fs": best.nominal_srate(),
                    }
                except Exception:
                    pass

            self.last_error = "Reconnected." if ok else "Reconnected (waiting for samples...)"

        except Exception as e:
            self.last_error = f"Reconnect exception: {e!r}"
        finally:
            self._reconnecting = False

    def get_window(self, seconds: float):
        with self.lock:
            if not self.ts:
                return np.array([]), np.array([[]], dtype=float)
            ts = np.array(self.ts, dtype=float)
            X = np.array(self.buf, dtype=float)
        t_end = ts[-1]
        mask = ts >= (t_end - seconds)
        return ts[mask], X[mask]

    def get_since(self, last_ts: float | None):
        with self.lock:
            if not self.ts:
                return np.array([]), np.array([[]], dtype=float)
            ts = np.array(self.ts, dtype=float)
            X = np.array(self.buf, dtype=float)

        if last_ts is None or ts.size == 0:
            return np.array([]), np.array([[]], dtype=float)

        if ts[-1] < float(last_ts):
            return ts, X

        mask = ts > float(last_ts)
        return ts[mask], X[mask]

    def pause(self):
        self.paused = True

    def resume(self):
        if self.running:
            self.paused = False

    def stop_and_clear(self):
        self.running = False
        self.paused = False
        try:
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=2)
        except Exception:
            pass

        with self.lock:
            self.ts.clear()
            self.buf.clear()
            self.inlet = None
            self.stream_meta = None
            self.sample_count = 0
            self.last_ts = None
