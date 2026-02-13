""""
muse_to_cpb_led.py

- Connects to Muse EEG via LSL (from `python -m muselsl stream`)
- Connects to Circuit Playground Bluefruit via BLE Nordic UART (RX UUID)
- Computes rolling RMS amplitude in microvolts over last WINDOW_SEC seconds
- Maps:
      0 uV   -> RED   (255, 0, 0)
      900 uV -> BLUE  (0, 0, 255)
- Sends "rgb R G B\n" continuously (only when color changes enough)

Run:
  Terminal 1: python -m muselsl stream
  Terminal 2: python muse_to_cpb_led.py
"""

import asyncio
import time
import numpy as np
from pylsl import resolve_byprop, StreamInlet
from bleak import BleakScanner, BleakClient

# -----------------------------
# BLE (Circuit Playground)
# -----------------------------
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
TARGET_NAME_CONTAINS = "CIRCUITPY"   # change if you renamed the board

# -----------------------------
# Muse / LSL
# -----------------------------
EEG_TYPE = "EEG"
EEG_CHANNELS = 4  # Muse 2 with muselsl typically provides 4 EEG chans

# -----------------------------
# Voltage -> Color mapping
# -----------------------------
UV_MIN = 0.0
UV_MAX = 900.0

UPDATE_HZ = 6.0      # how often to update LED
WINDOW_SEC = 1.0     # RMS computed over last N seconds

# send only if color changed enough (reduces BLE spam)
COLOR_CHANGE_THRESHOLD = 3  # smaller = more responsive


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def uv_to_rgb(uv_rms: float) -> tuple[int, int, int]:
    """
    Linear map uv_rms in [UV_MIN..UV_MAX] to red->blue.
    0 uV   -> (255,0,0)
    900 uV -> (0,0,255)
    """
    if UV_MAX <= UV_MIN:
        t = 0.0
    else:
        t = (uv_rms - UV_MIN) / (UV_MAX - UV_MIN)
    t = clamp01(t)

    r = int(round(255 * (1.0 - t)))
    g = 0
    b = int(round(255 * t))
    return (r, g, b)


def rms_uv(samples: np.ndarray) -> float:
    """
    samples: shape (N, C) in microvolts (typically muselsl outputs uV)
    RMS computed after per-channel DC removal.
    """
    if samples.size == 0:
        return 0.0
    x = samples.astype(np.float64)

    # remove DC per channel (mean over window)
    x = x - np.mean(x, axis=0, keepdims=True)

    # RMS across time+channels
    return float(np.sqrt(np.mean(x * x)))


def color_dist(c1, c2) -> int:
    return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1]), abs(c1[2] - c2[2]))


def connect_lsl_inlet(timeout_s: float = 15.0) -> StreamInlet:
    print("Resolving LSL EEG stream...")
    deadline = time.time() + float(timeout_s)

    while time.time() < deadline:
        streams = resolve_byprop("type", EEG_TYPE, timeout=1.0)
        if streams:
            # Prefer streams whose name contains "Muse"
            streams_sorted = sorted(
                streams,
                key=lambda s: ("muse" not in str(s.name()).lower(), -s.channel_count()),
            )
            s = streams_sorted[0]
            print(f"Found LSL stream: name={s.name()} type={s.type()} ch={s.channel_count()} fs={s.nominal_srate()}")
            return StreamInlet(s, max_buflen=60)

    raise RuntimeError("Could not find an LSL EEG stream. Did you run: python -m muselsl stream ?")


async def find_ble_device():
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover(timeout=6.0)

    for d in devices:
        if d.name and TARGET_NAME_CONTAINS in d.name:
            print(f"Found BLE board: {d.name}  {d.address}")
            return d

    print("Board not found. Named devices seen:")
    for d in devices:
        if d.name:
            print(f"  name={d.name}  addr={d.address}")
    return None


async def main():
    # 1) Connect to LSL Muse stream
    inlet = connect_lsl_inlet()

    # 2) Find BLE board
    device = await find_ble_device()
    if not device:
        print("ERROR: BLE board not found.")
        return

    # 3) Connect BLE and run loop
    async with BleakClient(device.address) as client:
        print("BLE connected.")
        print(f"Mapping RMS uV -> color: red@{UV_MIN}uV  blue@{UV_MAX}uV")
        print("Press Ctrl-C to stop.\n")

        last_color = (0, 0, 0)
        period = 1.0 / float(UPDATE_HZ)

        # Rolling sample buffer over WINDOW_SEC
        buf = []
        ts_buf = []

        while True:
            # Pull samples for a short burst (keeps loop responsive)
            t_pull_start = time.time()
            while (time.time() - t_pull_start) < 0.08:  # pull up to 80 ms each cycle
                sample, ts = inlet.pull_sample(timeout=0.0)
                if sample is None:
                    break
                if len(sample) >= EEG_CHANNELS:
                    buf.append(sample[:EEG_CHANNELS])
                    ts_buf.append(float(ts))

            # Trim buffer to last WINDOW_SEC based on LSL timestamps
            if ts_buf:
                t_end = ts_buf[-1]
                cutoff = t_end - float(WINDOW_SEC)

                # find first index >= cutoff
                i0 = 0
                for i, t in enumerate(ts_buf):
                    if t >= cutoff:
                        i0 = i
                        break
                buf = buf[i0:]
                ts_buf = ts_buf[i0:]

            # Compute RMS uV
            if buf:
                X = np.asarray(buf, dtype=np.float64)
                uv = rms_uv(X)
            else:
                uv = 0.0

            # Map to color
            rgb = uv_to_rgb(uv)

            # Send if changed enough
            if color_dist(rgb, last_color) >= COLOR_CHANGE_THRESHOLD:
                msg = f"rgb {rgb[0]} {rgb[1]} {rgb[2]}\n".encode("utf-8")
                await client.write_gatt_char(UART_RX_UUID, msg)
                last_color = rgb

            # Print a live status line occasionally
            # (prints ~once/sec)
            if int(time.time()) % 1 == 0:
                print(f"RMS={uv:7.2f} uV  -> rgb={rgb}      ", end="\r")

            await asyncio.sleep(period)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
