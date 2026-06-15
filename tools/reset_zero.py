"""
tools/reset_zero.py — 零位校准 / Reset
========================================

把手【当前的物理姿态】设为所有关节的 0° 基准：
    · 所有手指左右居中
    · 所有屈伸关节完全伸直

校准后，控制器（MediaPipe 入口 / 肌电入口都通用）会把这个姿态当作 0°。
偏移存到 hardware/zero_offsets.json，跨程序重启保留。

━━━ 标准流程 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. 运行本程序，它会松开力矩（手指可以用手扳动）
    2. 把手摆成「全部伸直、左右居中」的姿态
    3. 按 Enter —— 程序读取当前位置并保存为 0°
    4. （可选）保持力矩，锁定在这个零位姿态
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用法:
    python tools/reset_zero.py                 # 标准校准流程
    python tools/reset_zero.py --keep-torque   # 不松力矩（手已在正确姿态）
    python tools/reset_zero.py --show          # 查看当前已保存的校准
    python tools/reset_zero.py --clear         # 清除校准，恢复出厂中点(2048)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hardware.hls3915m_bus import HLSBus, pos_to_deg, CommError
from hardware.joint_map import JOINT_CONFIG, LIVE_JOINT_NAMES, is_live
from hardware.calibration import (
    load_offsets, save_offsets, clear_offsets, calib_path,
)

DEFAULT_PORT = "/dev/cu.usbmodem5B790315271"
DEFAULT_BAUD = 1_000_000


def _sep(char="─", width=58):
    print(char * width)


# ─── 查看 / 清除 ──────────────────────────────────────────────────────────────

def do_show():
    offsets = load_offsets()
    _sep("=")
    print("  当前零位校准")
    print(f"  文件: {calib_path()}")
    _sep("=")
    if not offsets:
        print("  （无校准）所有关节 0° = 出厂中点 2048")
        return
    print(f"  {'关节':<14} {'home位置':>8}  {'≈角度':>7}")
    _sep()
    for joint in JOINT_CONFIG:
        if joint in offsets:
            raw = offsets[joint]
            print(f"  {joint:<14} {raw:>8d}  {pos_to_deg(raw):>6.1f}°")
    _sep()
    print(f"  共 {len(offsets)} 个关节已校准")


def do_clear():
    if clear_offsets():
        print(f"[校准] 已删除 {calib_path()}，恢复出厂中点(2048)为 0°")
    else:
        print("[校准] 没有校准文件，无需清除")


# ─── 校准 ─────────────────────────────────────────────────────────────────────

def do_calibrate(args):
    _sep("=")
    print("  零位校准 — 把当前姿态设为所有关节的 0°")
    print(f"  端口: {args.port}  @  {args.baud//1000}k")
    _sep("=")

    with HLSBus(args.port, args.baud, echo=args.echo) as bus:

        # 1) 找出在线的「真实」关节（虚空映射的坏舵机自动排除）
        live_online = []
        for joint in LIVE_JOINT_NAMES:
            sid = JOINT_CONFIG[joint].servo_id
            if bus.ping(sid):
                live_online.append(joint)

        if not live_online:
            print("\n[校准] 没有检测到任何在线舵机，请检查接线和电源。")
            return

        print(f"\n  在线可校准关节 {len(live_online)} 个:")
        for j in live_online:
            print(f"    · {j}  (ID {JOINT_CONFIG[j].servo_id})")

        # 2) 松力矩，方便手动摆姿势
        if not args.keep_torque:
            print("\n  松开力矩，现在可以用手把手指扳成目标姿态 ...")
            for j in live_online:
                try:
                    bus.disable_torque(JOINT_CONFIG[j].servo_id)
                except CommError:
                    pass

        # 3) 等用户摆好姿态
        print("\n  请把手摆成【所有手指伸直、左右居中】的姿态。")
        try:
            input("  摆好后按 Enter 采集零位（Ctrl+C 取消）... ")
        except (EOFError, KeyboardInterrupt):
            print("\n  已取消，未保存。")
            return

        # 4) 读取当前所有在线关节的位置
        offsets = {}
        print("\n  采集当前位置:")
        for j in live_online:
            sid = JOINT_CONFIG[j].servo_id
            try:
                raw = bus.get_position(sid)
                offsets[j] = raw
                print(f"    {j:<14} ID{sid:>2}  home={raw}")
            except CommError as e:
                print(f"    {j:<14} ID{sid:>2}  读取失败: {e}")

        if not offsets:
            print("\n[校准] 没有成功读取任何位置，未保存。")
            return

        # 5) 与已有校准合并保存（保留这次没在线的关节的旧值）
        merged = load_offsets()
        merged.update(offsets)
        meta = {
            "note": "home = 该关节 0°(伸直/居中) 对应的舵机原始位置",
            "joints": {j: JOINT_CONFIG[j].servo_id for j in offsets},
        }
        save_offsets(merged, meta=meta)
        print(f"\n  ✓ 已保存 {len(offsets)} 个关节零位到:")
        print(f"    {calib_path()}")

        # 6) 可选：保持力矩，锁定零位姿态
        if not args.no_hold:
            print("\n  开启力矩，锁定当前零位姿态 ...")
            for j, raw in offsets.items():
                sid = JOINT_CONFIG[j].servo_id
                try:
                    bus.enable_torque(sid)
                    bus.set_position(sid, raw, speed=150)
                except CommError:
                    pass
            print("  完成。手已锁定在 0° 姿态。")
        else:
            print("\n  （--no-hold）力矩保持关闭，手指仍可自由扳动。")

    print("\n校准结束。下次运行 MediaPipe / 肌电入口时会自动套用这个零位。")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="零位校准：把当前姿态设为 0°")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--echo", action="store_true",
                        help="3 线半双工接法时加此项")
    parser.add_argument("--keep-torque", action="store_true",
                        help="不松力矩（手已经在正确姿态时用）")
    parser.add_argument("--no-hold", action="store_true",
                        help="校准后不开力矩锁定（保持可自由扳动）")
    parser.add_argument("--show", action="store_true",
                        help="查看当前已保存的校准并退出")
    parser.add_argument("--clear", action="store_true",
                        help="清除校准，恢复出厂中点(2048) 为 0°")
    args = parser.parse_args()

    if args.show:
        do_show()
        return
    if args.clear:
        do_clear()
        return

    try:
        do_calibrate(args)
    except Exception as e:
        print(f"\n[Fatal] {e}")


if __name__ == "__main__":
    main()
