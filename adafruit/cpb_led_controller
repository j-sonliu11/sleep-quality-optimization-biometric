import time
import board
import neopixel

from adafruit_ble import BLERadio
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement
from adafruit_ble.services.nordic import UARTService

# ----------------------------
# NeoPixels (CPB has 10 pixels)
# ----------------------------
PIXEL_COUNT = 10
pixels = neopixel.NeoPixel(board.NEOPIXEL, PIXEL_COUNT, brightness=0.2, auto_write=True)
pixels.fill((0, 0, 0))

def clamp(x):
    try:
        x = int(x)
    except Exception:
        return 0
    return 0 if x < 0 else 255 if x > 255 else x

def set_rgb(r, g, b):
    pixels.fill((clamp(r), clamp(g), clamp(b)))

# ----------------------------
# BLE UART
# ----------------------------
ble = BLERadio()
ble.name = "SLEEP_LED_1"   # New BLE name
uart = UARTService()
adv = ProvideServicesAdvertisement(uart)

# Optional: show "alive" indicator (dim green)
set_rgb(0, 10, 0)

ble.start_advertising(adv)

buf = ""  # line buffer

def handle_line(line: str):
    line = line.strip()
    if not line:
        return

    low = line.lower()

    if low == "on":
        set_rgb(255, 255, 255)
        uart.write(b"OK\n")
        return

    if low == "off":
        set_rgb(0, 0, 0)
        uart.write(b"OK\n")
        return

    # rgb 255 0 128
    if low.startswith("rgb"):
        parts = line.split()
        if len(parts) == 4:
            set_rgb(parts[1], parts[2], parts[3])
            uart.write(b"OK\n")
        else:
            uart.write(b"ERR:usage rgb R G B\n")
        return

    uart.write(b"ERR:unknown\n")

while True:
    # Keep advertising when disconnected
    if not ble.connected:
        if not ble.advertising:
            ble.start_advertising(adv)
        time.sleep(0.05)
        continue

    try:
        # IMPORTANT: BLE writes often arrive in chunks, not 1 byte at a time.
        n = uart.in_waiting
        if n:
            data = uart.read(n)  # read all available bytes
            if data:
                for b in data:
                    ch = chr(b)
                    if ch == "\n":
                        handle_line(buf)
                        buf = ""
                    elif ch != "\r":
                        # prevent runaway buffer
                        if len(buf) < 256:
                            buf += ch
                        else:
                            buf = ""
                            uart.write(b"ERR:overflow\n")
        time.sleep(0.005)

    except Exception as e:
        # Don't crash-loop: report and keep running
        try:
            uart.write(("ERR:" + repr(e) + "\n").encode("utf-8"))
        except Exception:
            pass
        buf = ""
        time.sleep(0.1)
