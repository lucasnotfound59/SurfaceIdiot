"""
SurfaceIdiot - Glove Calibration Script

Sends calibration commands to the ESP32 over serial and saves the result
to calibration.json for use by collect.py and the training pipeline.

Usage:
    python calibrate.py --port /dev/ttyUSB0
    python calibrate.py --port COM3          # Windows
"""

import argparse
import json
import time
import serial
import sys


def wait_for_done(ser: serial.Serial, keyword: str, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode(errors="replace").strip()
            if line:
                print(f"  ESP32: {line}")
            if keyword in line:
                return True
    return False


def collect_normalized_readings(ser: serial.Serial, n: int = 150):
    """Read n valid JSON frames from the glove and return averaged finger values."""
    readings = []
    print(f"  Collecting {n} frames", end="", flush=True)
    while len(readings) < n:
        if ser.in_waiting:
            raw = ser.readline().decode(errors="replace").strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                data = json.loads(raw)
                f = data["fingers"]
                readings.append([
                    f["thumb"], f["index"], f["middle"], f["ring"], f["pinky"]
                ])
                if len(readings) % 30 == 0:
                    print(".", end="", flush=True)
            except (json.JSONDecodeError, KeyError):
                pass
    print()
    avg = [sum(r[i] for r in readings) / n for i in range(5)]
    return avg


def main():
    parser = argparse.ArgumentParser(description="Glove calibration")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port of ESP32")
    parser.add_argument("--baud", default=115200, type=int)
    parser.add_argument("--output", default="calibration.json", help="Output calibration file")
    args = parser.parse_args()

    print(f"Connecting to {args.port} @ {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=2)
    except serial.SerialException as e:
        print(f"ERROR: Cannot open port: {e}")
        sys.exit(1)

    time.sleep(2)  # Let ESP32 boot
    ser.reset_input_buffer()

    print("\n=== Step 1: Straight baseline ===")
    print("Fully extend ALL fingers and hold for 5 seconds.")
    input("Press Enter when ready...")

    ser.write(b"CAL_STRAIGHT\n")
    if not wait_for_done(ser, "CAL_STRAIGHT_DONE", timeout=12):
        print("ERROR: Timed out waiting for straight calibration.")
        sys.exit(1)

    print("\n=== Step 2: Fist baseline ===")
    print("Make a TIGHT fist and hold for 5 seconds.")
    input("Press Enter when ready...")

    ser.write(b"CAL_FIST\n")
    if not wait_for_done(ser, "CAL_FIST_DONE", timeout=12):
        print("ERROR: Timed out waiting for fist calibration.")
        sys.exit(1)

    # Read back current live values for verification
    print("\n=== Verification: live readings ===")
    print("Open and close your hand a few times. Values should range 0.0–1.0.")
    print("Press Ctrl+C to stop and save.\n")

    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    try:
        while True:
            if ser.in_waiting:
                raw = ser.readline().decode(errors="replace").strip()
                if not raw or raw.startswith("#"):
                    continue
                try:
                    data = json.loads(raw)
                    f = data["fingers"]
                    vals = "  ".join(f"{fn}: {f[fn]:.2f}" for fn in fingers)
                    print(f"\r{vals}", end="", flush=True)
                except (json.JSONDecodeError, KeyError):
                    pass
    except KeyboardInterrupt:
        print()

    # Save minimal calibration metadata (ESP32 stores the raw ADC values itself in NVS,
    # but we also save a human-readable JSON for reference and for Python-side use)
    ser.write(b"STATUS\n")
    time.sleep(0.5)
    status_lines = []
    while ser.in_waiting:
        line = ser.readline().decode(errors="replace").strip()
        if line:
            status_lines.append(line)

    calibration = {
        "port": args.port,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fingers": fingers,
        "notes": "Raw ADC min/max are stored on ESP32 NVS. This file records verification info.",
        "status_dump": status_lines,
    }
    with open(args.output, "w") as fp:
        json.dump(calibration, fp, indent=2)

    print(f"\nCalibration saved to {args.output}")
    ser.close()


if __name__ == "__main__":
    main()
