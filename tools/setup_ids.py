"""
tools/setup_ids.py  -  First-time Servo ID Assignment Wizard
=============================================================

All Feetech HLS3915M servos ship from the factory with ID=1.
Connecting multiple factory-default servos to the same bus causes
bus collisions — that's why you see "Incorrect status packet!" errors.

This wizard guides you to assign unique IDs (1-16) one servo at a time:

  Step 1: Connect ONLY ONE servo to the bus.
  Step 2: Run this script — it pings ID 1, confirms the servo is online.
  Step 3: You tell the script which joint this servo controls.
  Step 4: Script assigns the correct ID for that joint.
  Step 5: Disconnect that servo; connect the next one. Repeat.

Joint → ID mapping (Orca Hand v1):
  thumb_abd  = 1    index_abd  = 5    middle_abd = 8
  thumb_mcp  = 2    index_mcp  = 6    middle_mcp = 9
  thumb_pip  = 3    index_pip  = 7    middle_pip = 10
  thumb_dip  = 4
  ring_abd   = 11   pinky_abd  = 14
  ring_mcp   = 12   pinky_mcp  = 15
  ring_pip   = 13   pinky_pip  = 16

Usage:
    python tools/setup_ids.py
    python tools/setup_ids.py --port /dev/cu.usbmodem5B790315271
    python tools/setup_ids.py --joint index_mcp   # assign ID for one specific joint
    python tools/setup_ids.py --scan              # just scan what IDs are online
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hardware.hls3915m_bus import (
    HLSBus, CommError,
    REG_ID, BROADCAST_ID,
    INST_WRITE, HEADER,
)
from hardware.hand_controller import JOINT_CONFIG, ALL_JOINT_NAMES

# ─── Joint → Servo ID table (from JOINT_CONFIG) ───────────────────────────────

JOINT_TO_ID = {name: cfg.servo_id for name, cfg in JOINT_CONFIG.items()}
ID_TO_JOINT = {cfg.servo_id: name for name, cfg in JOINT_CONFIG.items()}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sep(char="─", width=58):
    print(char * width)


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "q"


def _scan_all(bus: HLSBus) -> list[int]:
    """Ping IDs 1-16, return list of responding IDs."""
    found = []
    print("  Scanning IDs 1-16 ...", end="", flush=True)
    for sid in range(1, 17):
        if bus.ping(sid):
            found.append(sid)
    print(f"  found: {found if found else 'none'}")
    return found


def _write_id_broadcast(bus: HLSBus, current_id: int, new_id: int) -> bool:
    """
    Change servo ID.  Uses a direct WRITE to REG_ID (EEPROM).
    The servo must be the ONLY device on the bus (or you know its current ID).
    Power-cycle required for the change to take effect on most firmware.
    """
    try:
        # Write new ID to EEPROM register 0x05
        bus.write(current_id, REG_ID, bytes([new_id]))
        return True
    except CommError as e:
        print(f"  [!] ID write failed: {e}")
        return False


# ─── Scan mode ────────────────────────────────────────────────────────────────

def do_scan(bus: HLSBus):
    _sep("=")
    print("  Scan Mode — checking which IDs respond")
    _sep("=")
    found = _scan_all(bus)
    print()
    _sep()
    print(f"  {'ID':>4}  {'Joint':<16}  Status")
    _sep()
    for sid in range(1, 17):
        joint = ID_TO_JOINT.get(sid, "???")
        if sid in found:
            print(f"  #{sid:>2}   {joint:<16}  [ONLINE]")
        else:
            print(f"  #{sid:>2}   {joint:<16}  offline")
    _sep()
    print(f"\n  Online: {len(found)}/16")
    if len(found) > 1:
        print("\n  [Note] Multiple servos online.")
        print("  For ID assignment, connect ONE servo at a time.")
    elif len(found) == 0:
        print("\n  [Note] No servos found. Check wiring and power.")


# ─── Single-joint assignment ──────────────────────────────────────────────────

def assign_one(bus: HLSBus, joint_name: str) -> bool:
    """
    Assign the correct ID to a single servo already connected as ID=1 (or any ID).
    Returns True on success.
    """
    target_id = JOINT_TO_ID[joint_name]

    _sep()
    print(f"  Joint   : {joint_name}")
    print(f"  Target ID: {target_id}")
    _sep()

    # Find the servo — it's either at ID=1 (factory) or already has target_id
    print("  Looking for servo ...")
    online = _scan_all(bus)

    if not online:
        print("  [!] No servo found. Check wiring and power, then try again.")
        return False

    if target_id in online and len(online) == 1:
        print(f"  Servo already has correct ID={target_id}. Nothing to do.")
        return True

    if len(online) > 1:
        print(f"\n  [!] Multiple servos detected: {online}")
        print("  Disconnect all but ONE servo, then run again.")
        return False

    current_id = online[0]
    print(f"  Found servo at ID={current_id}.")

    if current_id == target_id:
        print("  Already correct. Nothing to do.")
        return True

    ans = _ask(
        f"\n  Change ID {current_id} -> {target_id}  for joint '{joint_name}'? [y/n]: "
    )
    if ans != "y":
        print("  Skipped.")
        return False

    print(f"  Writing ID {target_id} to servo ...", end="", flush=True)
    ok = _write_id_broadcast(bus, current_id, target_id)
    if not ok:
        return False
    print("  done")

    # Verify by pinging the new ID (some firmware needs power-cycle first)
    print("  Waiting 0.5s for EEPROM write to settle ...", end="", flush=True)
    time.sleep(0.5)
    print()

    if bus.ping(target_id):
        print(f"  [OK] Servo now responds at ID={target_id}.")
        print(f"  Safe to disconnect this servo and connect the next one.")
        return True
    else:
        print(
            f"  [Warn] ID {target_id} did not respond yet.\n"
            f"  Some servos require a power-cycle before the new ID takes effect.\n"
            f"  Power-cycle the servo, then run:  python tools/setup_ids.py --scan"
        )
        return True   # Write likely succeeded; just needs power-cycle


# ─── Interactive wizard ────────────────────────────────────────────────────────

MENU_TEXT = """
  Orca Hand v1 — Joint → ID Table
  ─────────────────────────────────────
  [1]  thumb_abd   (ID 1)   [9]  middle_mcp (ID 9)
  [2]  thumb_mcp   (ID 2)   [10] middle_pip (ID 10)
  [3]  thumb_pip   (ID 3)   [11] ring_abd   (ID 11)
  [4]  thumb_dip   (ID 4)   [12] ring_mcp   (ID 12)
  [5]  index_abd   (ID 5)   [13] ring_pip   (ID 13)
  [6]  index_mcp   (ID 6)   [14] pinky_abd  (ID 14)
  [7]  index_pip   (ID 7)   [15] pinky_mcp  (ID 15)
  [8]  middle_abd  (ID 8)   [16] pinky_pip  (ID 16)
  ─────────────────────────────────────
  [s]  scan bus              [q]  quit
"""

NUM_TO_JOINT = {str(cfg.servo_id): name for name, cfg in JOINT_CONFIG.items()}


def do_wizard(bus: HLSBus):
    _sep("=")
    print("  SurfaceIdiot — Servo ID Setup Wizard")
    _sep("=")
    print("""
  IMPORTANT: Connect ONE servo at a time.
  All factory servos start at ID=1.  Connecting multiple at once
  causes bus collisions and communication errors.

  Workflow:
    1. Connect one servo
    2. Choose its joint from the menu
    3. Script assigns the correct ID
    4. Disconnect that servo; connect next one
""")
    input("  Press Enter to start...")

    while True:
        print(MENU_TEXT)
        ans = _ask("  Choose joint number (1-16), 's' to scan, 'q' to quit: ")

        if ans == "q":
            break
        if ans == "s":
            _scan_all(bus)
            continue

        joint = NUM_TO_JOINT.get(ans)
        if joint is None:
            print("  Invalid choice. Enter a number 1-16.")
            continue

        assign_one(bus, joint)
        print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Servo ID setup wizard (Orca Hand v1)")
    parser.add_argument("--port",  default="/dev/cu.usbmodem5B790315271")
    parser.add_argument("--baud",  type=int, default=1_000_000)
    parser.add_argument("--scan",  action="store_true",
                        help="Scan bus and show which IDs respond")
    parser.add_argument("--joint", default=None,
                        help="Assign ID for one specific joint, e.g. index_mcp")
    args = parser.parse_args()

    if args.joint and args.joint not in JOINT_TO_ID:
        print(f"Unknown joint '{args.joint}'. Valid: {ALL_JOINT_NAMES}")
        sys.exit(1)

    try:
        with HLSBus(args.port, args.baud) as bus:
            if args.scan:
                do_scan(bus)
            elif args.joint:
                assign_one(bus, args.joint)
            else:
                do_wizard(bus)
    except Exception as e:
        print(f"\n[Fatal] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
