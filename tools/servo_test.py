"""
tools/servo_test.py  -  Sequential Servo Test
==============================================

Moves servo #1 through #16 one at a time so you can visually verify:
  1. The servo responds (online check)
  2. The correct physical joint moves
  3. The direction of movement makes sense

Each servo:
  neutral (0°) -> test position -> back to neutral -> user confirms

Controls after each servo moves:
  Enter / y   = correct, continue to next
  n           = wrong assignment (recorded in summary)
  s           = skip (servo offline or already tested)
  q           = quit

Usage:
    python tools/servo_test.py
    python tools/servo_test.py --port /dev/cu.usbmodem5B790315271
    python tools/servo_test.py --id 5          # test only servo #5
    python tools/servo_test.py --range 1 4     # test servos 1-4
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hardware.hls3915m_bus import HLSBus, deg_to_pos, pos_to_deg, CommError
from hardware.hand_controller import JOINT_CONFIG

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_PORT = "/dev/cu.usbmodem5B790315271"
DEFAULT_BAUD = 1_000_000

FLEX_TEST_DEG = 30.0   # test angle for MCP/PIP/DIP joints
ABD_TEST_DEG  = 12.0   # smaller range for abduction joints
MOVE_SPEED    = 150    # 0=max speed, bigger=slower; 150 is gentle
HOLD_SECS     = 1.5    # how long to hold the test position

ABD_JOINTS = {"thumb_abd", "index_abd", "middle_abd",
              "ring_abd",  "pinky_abd"}

# Servo ID → joint name lookup
ID_TO_JOINT = {cfg.servo_id: name for name, cfg in JOINT_CONFIG.items()}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _test_deg(joint_name: str) -> float:
    return ABD_TEST_DEG if joint_name in ABD_JOINTS else FLEX_TEST_DEG


def _separator(char="─", width=58):
    print(char * width)


def _move_and_hold(bus: HLSBus, servo_id: int, deg: float) -> float | None:
    """Move servo to deg, hold HOLD_SECS, return actual angle. None on error."""
    pos = deg_to_pos(deg)
    try:
        bus.set_position(servo_id, pos, speed=MOVE_SPEED)
        time.sleep(HOLD_SECS)
        actual = pos_to_deg(bus.get_position(servo_id))
        return actual
    except CommError as e:
        print(f"   [!] Move error: {e}")
        return None


def _return_neutral(bus: HLSBus, servo_id: int):
    try:
        bus.set_position(servo_id, 2048, speed=MOVE_SPEED)
        time.sleep(0.6)
    except CommError:
        pass


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "q"


# ─── Single servo test ────────────────────────────────────────────────────────

def test_one(bus: HLSBus, servo_id: int, return_neutral: bool = True) -> str:
    """
    Run the test sequence for one servo.
    return_neutral=False: servo stays at test angle after moving (use n<id> to reset).
    Returns: "ok" | "wrong" | "offline" | "skip" | "quit"
    """
    joint_name = ID_TO_JOINT.get(servo_id, f"??? (not in config)")
    deg        = _test_deg(joint_name)

    print(f"\n  Servo #{servo_id:2d}  |  joint: {joint_name}")
    print(f"  Expected body part: {_describe(joint_name)}")

    # ── Online check ──────────────────────────────────────────────────────────
    if not bus.ping(servo_id):
        print("  Status : OFFLINE (no response)")
        ans = _ask("  [Enter=skip / q=quit]: ")
        return "quit" if ans == "q" else "offline"

    print("  Status : online")

    # ── Enable torque & go neutral first ─────────────────────────────────────
    try:
        bus.enable_torque(servo_id)
    except CommError as e:
        print(f"  Torque enable failed: {e}")
        return "offline"

    print(f"  Neutral : going to 0° first ...", end="", flush=True)
    _return_neutral(bus, servo_id)
    print("  ready")

    # ── Move to test angle ────────────────────────────────────────────────────
    print(f"  Moving  : 0° -> {deg:+.0f}° (holding {HOLD_SECS}s) ...", end="", flush=True)
    actual = _move_and_hold(bus, servo_id, deg)
    if actual is None:
        _return_neutral(bus, servo_id)
        bus.disable_torque(servo_id)
        return "offline"
    print(f"  actual {actual:+.1f}°")

    # ── Return to neutral (optional) ──────────────────────────────────────────
    if return_neutral:
        print(f"  Returning to neutral ...", end="", flush=True)
        _return_neutral(bus, servo_id)
        bus.disable_torque(servo_id)
        print("  done")
    else:
        print(f"  Staying at {deg:+.0f}°  (type n{servo_id} to return neutral)")

    # ── User confirm ──────────────────────────────────────────────────────────
    ans = _ask("  Correct joint moved? [Enter/y=yes  n=wrong  s=skip  q=quit]: ")
    if ans == "q":
        return "quit"
    if ans == "n":
        print("  Marked: WRONG")
        return "wrong"
    if ans == "s":
        return "skip"
    print("  Marked: OK")
    return "ok"


def _describe(joint_name: str) -> str:
    """Human-readable description of what should move."""
    descriptions = {
        "thumb_abd":  "Thumb abduction  (thumb spreads away from index)",
        "thumb_mcp":  "Thumb MCP flex   (thumb base knuckle bends)",
        "thumb_pip":  "Thumb PIP flex   (thumb middle joint bends)",
        "thumb_dip":  "Thumb DIP flex   (thumb tip joint bends)",
        "index_abd":  "Index abduction  (index spreads away from middle)",
        "index_mcp":  "Index MCP flex   (index base knuckle bends)",
        "index_pip":  "Index PIP flex   (index middle joint bends)",
        "middle_abd": "Middle abduction (middle spreads from ring)",
        "middle_mcp": "Middle MCP flex  (middle base knuckle bends)",
        "middle_pip": "Middle PIP flex  (middle middle joint bends)",
        "ring_abd":   "Ring abduction   (ring spreads from pinky)",
        "ring_mcp":   "Ring MCP flex    (ring base knuckle bends)",
        "ring_pip":   "Ring PIP flex    (ring middle joint bends)",
        "pinky_abd":  "Pinky abduction  (pinky spreads outward)",
        "pinky_mcp":  "Pinky MCP flex   (pinky base knuckle bends)",
        "pinky_pip":  "Pinky PIP flex   (pinky middle joint bends)",
    }
    return descriptions.get(joint_name, joint_name)


# ─── Main ─────────────────────────────────────────────────────────────────────

def _print_summary(results: dict, id_list: list):
    print()
    _separator("=")
    print("  SUMMARY")
    _separator("=")
    icons = {"ok": "[OK]", "wrong": "[!!]", "offline": "[--]",
             "skip": "[  ]", "quit": "[  ]"}
    for sid in id_list:
        s = results.get(sid, "untested")
        joint = ID_TO_JOINT.get(sid, "?")
        print(f"  {icons.get(s, '[?]')}  #{sid:2d}  {joint:<14}  {s}")

    wrong   = [sid for sid, s in results.items() if s == "wrong"]
    offline = [sid for sid, s in results.items() if s == "offline"]
    ok      = [sid for sid, s in results.items() if s == "ok"]

    print()
    print(f"  Passed  : {len(ok)}")
    print(f"  Wrong   : {len(wrong)}  {wrong if wrong else ''}")
    print(f"  Offline : {len(offline)}  {offline if offline else ''}")
    _separator("=")


# ─── Interactive mode ─────────────────────────────────────────────────────────

def run_interactive(bus: HLSBus):
    """
    交互模式：随时输入舵机编号，立刻测试那个舵机。
    支持：
      数字 1-16  → 测试对应舵机
      a          → 顺序测试所有 1-16
      n <id>     → 让指定舵机回中立位
      s          → 显示关节-ID 对照表
      q / Enter  → 退出
    """
    results: dict[int, str] = {}
    tested: list[int] = []

    _separator("=")
    print("  Interactive Servo Test  (type servo ID to test, q=quit)")
    _separator("=")
    print()
    print("  ID map:")
    for name, cfg in sorted(JOINT_CONFIG.items(), key=lambda kv: kv[1].servo_id):
        print(f"    #{cfg.servo_id:>2}  {name}")
    print()

    while True:
        try:
            raw = input("  > Servo ID [1-16 / a=all / n<id>=neutral / s=map / q=quit]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if raw in ("q", ""):
            break

        # Show map
        if raw == "s":
            for name, cfg in sorted(JOINT_CONFIG.items(), key=lambda kv: kv[1].servo_id):
                print(f"    #{cfg.servo_id:>2}  {name}")
            continue

        # Return one servo to neutral: "n5" or "n 5"
        if raw.startswith("n"):
            try:
                sid = int(raw[1:].strip())
                print(f"  Sending #{sid} to neutral ...", end="", flush=True)
                _return_neutral(bus, sid)
                print("  done")
            except (ValueError, IndexError):
                print("  Usage: n<id>  e.g. n5")
            continue

        # Run all sequentially
        if raw == "a":
            for sid in range(1, 17):
                _separator()
                status = test_one(bus, sid)
                results[sid] = status
                if sid not in tested:
                    tested.append(sid)
                if status == "quit":
                    break
            continue

        # Single servo by number
        try:
            sid = int(raw)
        except ValueError:
            print("  Enter a number 1-16, 'a', 'n<id>', 's', or 'q'.")
            continue

        if not 1 <= sid <= 16:
            print("  ID must be between 1 and 16.")
            continue

        _separator()
        status = test_one(bus, sid, return_neutral=False)
        results[sid] = status
        if sid not in tested:
            tested.append(sid)
        if status == "quit":
            break

    _print_summary(results, tested if tested else list(range(1, 17)))


# ─── Sequential mode ──────────────────────────────────────────────────────────

def run_sequential(bus: HLSBus, id_list: list):
    results: dict[int, str] = {}
    for servo_id in id_list:
        _separator()
        status = test_one(bus, servo_id)
        results[servo_id] = status
        if status == "quit":
            print("\nTest aborted by user.")
            break
    _print_summary(results, id_list)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global MOVE_SPEED

    parser = argparse.ArgumentParser(description="Sequential servo test")
    parser.add_argument("--port",  default=DEFAULT_PORT)
    parser.add_argument("--baud",  type=int, default=DEFAULT_BAUD)
    parser.add_argument("--id",    type=int, default=None,
                        help="Test only this one servo ID")
    parser.add_argument("--range", type=int, nargs=2, metavar=("FROM", "TO"),
                        help="Test servo IDs FROM..TO inclusive")
    parser.add_argument("--speed", type=int, default=MOVE_SPEED,
                        help="Servo speed 0-4095 (smaller=faster, 0=max)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode: type servo ID to test on demand")
    args = parser.parse_args()

    MOVE_SPEED = args.speed

    _separator("=")
    print("  SurfaceIdiot -- Servo Test")
    print(f"  Port : {args.port}  @  {args.baud//1000}k baud")
    print(f"  Test : flex {FLEX_TEST_DEG}°  |  abduction {ABD_TEST_DEG}°  "
          f"|  hold {HOLD_SECS}s  |  speed {MOVE_SPEED}")
    _separator("=")
    print()

    try:
        with HLSBus(args.port, args.baud) as bus:
            if args.interactive or (not args.id and not args.range):
                # 默认进入交互模式
                run_interactive(bus)
            else:
                # 指定了 --id 或 --range，顺序测试
                if args.id:
                    id_list = [args.id]
                else:
                    id_list = list(range(args.range[0], args.range[1] + 1))
                print(f"  IDs  : {id_list}")
                print()
                input("Press Enter to start the test sequence...")
                run_sequential(bus, id_list)
    except Exception as e:
        print(f"\n[Fatal] {e}")


if __name__ == "__main__":
    main()
