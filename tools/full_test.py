"""
tools/full_test.py  -  整体测试 (Full Hand Test)
=================================================

让每一根手指依次【轻微弯曲一下，再恢复原状】，用来快速确认整只手能动、
腱绳没卡死、方向正常。和 servo_test.py（逐个舵机精确测试）不同，
这里是面向“整只手”的健康检查。

测试流程:
    1. 连接总线，自动检测在线舵机
    2. 全部回中立位
    3. 逐根手指: 拇指 → 食指 → 中指 → 无名指 → 小指
         · 该手指的屈伸关节一起轻微弯曲 (GENTLE_DEG)
         · 保持一会儿
         · 该手指回到原状 (0°)
    4. （可选 --wave）所有手指一起轻微弯曲再松开，做个“握一下”的动作

舵机 ↔ 自由度 映射在 hardware/joint_map.py 中维护，改那里即可。

用法:
    python tools/full_test.py
    python tools/full_test.py --port /dev/cu.usbmodem5B790315271
    python tools/full_test.py --deg 25          # 弯曲幅度调大
    python tools/full_test.py --loops 3         # 循环 3 遍
    python tools/full_test.py --wave            # 结尾加一个整手握合动作
    python tools/full_test.py --finger index    # 只测某一根手指
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hardware.hand_controller import HandController
from hardware.joint_map import FINGERS, FLEX_JOINTS

# ─── 默认参数 ─────────────────────────────────────────────────────────────────

DEFAULT_PORT = "/dev/cu.usbmodem5B790315271"
DEFAULT_BAUD = 1_000_000

GENTLE_DEG   = 20.0    # 每根手指屈伸关节的轻微弯曲角度
MOVE_SPEED   = 200     # 运动速度（0=最快，越大越慢）—— 整体测试用慢速更安全
HOLD_SECS    = 0.8     # 弯曲后保持时间
SETTLE_SECS  = 0.6     # 回到原状后停顿时间

FINGER_LABEL = {
    "thumb":  "拇指 Thumb",
    "index":  "食指 Index",
    "middle": "中指 Middle",
    "ring":   "无名指 Ring",
    "pinky":  "小指 Pinky",
}


# ─── 工具 ─────────────────────────────────────────────────────────────────────

def _sep(char="─", width=56):
    print(char * width)


def _bend_finger(hand: HandController, finger: str, deg: float, speed: int):
    """让一根手指的屈伸关节一起弯曲 deg 度（abduction 不动）。"""
    pose = {j: deg for j in FLEX_JOINTS[finger]}
    hand.set_all(pose, speed=speed)


def _relax_finger(hand: HandController, finger: str, speed: int):
    """让一根手指回到原状 (0°)。"""
    pose = {j: 0.0 for j in FLEX_JOINTS[finger]}
    hand.set_all(pose, speed=speed)


def _finger_online_count(hand: HandController, finger: str) -> int:
    """这根手指有几个关节在线。"""
    return sum(1 for j in FINGERS[finger] if j in hand._online)


# ─── 单根手指动作 ─────────────────────────────────────────────────────────────

def test_finger(hand: HandController, finger: str, deg: float, speed: int):
    label = FINGER_LABEL.get(finger, finger)
    n_on  = _finger_online_count(hand, finger)
    n_all = len(FINGERS[finger])

    print(f"\n  {label:<14}  ({n_on}/{n_all} 关节在线)")

    if n_on == 0:
        print("    跳过：该手指无在线舵机")
        return

    flex_online = [j for j in FLEX_JOINTS[finger] if j in hand._online]
    print(f"    弯曲 {deg:.0f}°  →  {flex_online}")
    _bend_finger(hand, finger, deg, speed)
    time.sleep(HOLD_SECS)

    print(f"    恢复原状")
    _relax_finger(hand, finger, speed)
    time.sleep(SETTLE_SECS)


# ─── 整手握合（wave）─────────────────────────────────────────────────────────

def test_wave(hand: HandController, deg: float, speed: int):
    print("\n  整手握合 → 张开")
    # 所有手指屈伸关节一起弯
    pose = {}
    for finger in FINGERS:
        for j in FLEX_JOINTS[finger]:
            pose[j] = deg
    hand.set_all(pose, speed=speed)
    time.sleep(HOLD_SECS + 0.4)

    # 全部松开
    for j in pose:
        pose[j] = 0.0
    hand.set_all(pose, speed=speed)
    time.sleep(SETTLE_SECS + 0.4)


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="整体测试：每根手指轻微动一下再恢复")
    parser.add_argument("--port",   default=DEFAULT_PORT)
    parser.add_argument("--baud",   type=int, default=DEFAULT_BAUD)
    parser.add_argument("--deg",    type=float, default=GENTLE_DEG,
                        help="每根手指弯曲角度 (°)")
    parser.add_argument("--speed",  type=int, default=MOVE_SPEED,
                        help="运动速度 0-4095（越大越慢）")
    parser.add_argument("--loops",  type=int, default=1, help="整体循环遍数")
    parser.add_argument("--wave",   action="store_true",
                        help="每遍结尾加一个整手握合动作")
    parser.add_argument("--finger", default=None,
                        choices=list(FINGERS.keys()),
                        help="只测某一根手指")
    args = parser.parse_args()

    _sep("=")
    print("  SurfaceIdiot — 整体测试 (Full Hand Test)")
    print(f"  端口  : {args.port}  @  {args.baud//1000}k")
    print(f"  幅度  : {args.deg:.0f}°   速度: {args.speed}   循环: {args.loops}")
    _sep("=")

    targets = [args.finger] if args.finger else list(FINGERS.keys())

    hand = HandController(args.port, args.baud, default_speed=args.speed)
    try:
        hand.connect()

        if not hand._online:
            print("\n[整体测试] 没有检测到任何在线舵机，请检查接线和电源。")
            return

        print("\n  先回中立位 ...")
        hand.set_neutral(speed=args.speed)
        time.sleep(1.2)

        for loop in range(args.loops):
            if args.loops > 1:
                _sep()
                print(f"  第 {loop + 1}/{args.loops} 遍")

            for finger in targets:
                test_finger(hand, finger, args.deg, args.speed)

            if args.wave and not args.finger:
                test_wave(hand, args.deg, args.speed)

        print("\n  全部完成，回中立位 ...")
        hand.set_neutral(speed=args.speed)
        time.sleep(1.2)

    except KeyboardInterrupt:
        print("\n[中断] 正在回中立位 ...")
    except Exception as e:
        print(f"\n[错误] {e}")
    finally:
        hand.disconnect(go_neutral=True)
        print("\n整体测试结束。")


if __name__ == "__main__":
    main()
