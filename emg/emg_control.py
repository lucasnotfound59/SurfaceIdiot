"""
emg/emg_control.py — 肌电手环 → HLS3915M 控制入口（第二入口）
==============================================================

与 handTracking/hls_control.py（MediaPipe 入口）并列的【第二个入口】。
两者共用同一套后端 HandController + hardware/joint_map.py，可任选其一驱动灵巧手。

数据流：
    EMGSource(16 通道) → EMGMapper → HandController.set_all() → HLS3915M 舵机

━━━ 接入真实手环 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
只改 build_source() 一个函数：返回你的 EMGSource 子类实例
（或用 CallbackEMGSource 包装现成取数函数）。其余代码无需改动。
默认用 MockEMGSource（无硬件也能跑通整条链路）。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

用法：
    python emg/emg_control.py --mock                 # 模拟数据，不连舵机
    python emg/emg_control.py                         # 模拟数据 → 真实舵机
    python emg/emg_control.py --port /dev/cu.usbmodem5B790315271
    python emg/emg_control.py --rate 50 --alpha 0.2   # 控制频率/平滑系数
"""

import argparse
import os
import signal
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from emg.emg_source import EMGSource, MockEMGSource, CallbackEMGSource, NUM_CHANNELS
from emg.emg_mapping import EMGMapper, EMG_CHANNEL_MAP
from hardware.joint_map import JOINT_CONFIG, LIVE_JOINT_NAMES, is_live


# ─── 在这里接入真实手环 ───────────────────────────────────────────────────────

def build_source(args) -> EMGSource:
    """
    返回肌电数据源实例。

    ⭐ 真实手环对接，把这里替换成：
        return MyArmbandSource(port=args.emg_port, ...)
    或包装现成函数：
        return CallbackEMGSource(read_fn=my_read_16_channels)

    现在默认返回模拟数据源。
    """
    # TODO: 真实手环数据源在此 return
    return MockEMGSource()


# ─── 文本 HUD ─────────────────────────────────────────────────────────────────

def _print_status(angles: dict, online_set, fps: float):
    """打印一行紧凑的可用关节角度（只显示有真实舵机的关节）。"""
    parts = []
    for joint in LIVE_JOINT_NAMES:
        on = (online_set is None) or (joint in online_set)
        mark = "" if online_set is None else ("" if on else "x")
        parts.append(f"{joint}:{angles.get(joint, 0.0):5.1f}{mark}")
    line = "  ".join(parts)
    # \r 原地刷新
    sys.stdout.write(f"\r[{fps:4.0f}Hz] {line}   ")
    sys.stdout.flush()


# ─── 主循环 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="肌电手环 → HLS3915M 灵巧手控制")
    parser.add_argument("--port",  default="/dev/cu.usbmodem5B790315271",
                        help="舵机总线串口")
    parser.add_argument("--baud",  type=int, default=1_000_000)
    parser.add_argument("--speed", type=int, default=200,
                        help="舵机运动速度 0-4095（越大越慢）")
    parser.add_argument("--rate",  type=float, default=30.0,
                        help="控制频率 Hz（每秒发送多少次指令）")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="EMA 平滑系数 0-1（越小越平滑、越滞后）")
    parser.add_argument("--mock",  action="store_true",
                        help="模拟模式：不连接舵机，只打印映射出的角度")
    parser.add_argument("--echo",  action="store_true",
                        help="启用 echo（3 线半双工接法）")
    args = parser.parse_args()

    # ── Ctrl+C 干净退出 ───────────────────────────────────────────────────────
    _running = [True]
    def _sig_handler(sig, frame):
        print("\n[Ctrl+C] Shutting down...")
        _running[0] = False
    signal.signal(signal.SIGINT, _sig_handler)

    # ── 初始化舵机后端 ────────────────────────────────────────────────────────
    hand = None
    online_set = None   # None = mock 模式
    if not args.mock:
        from hardware.hand_controller import HandController
        hand = HandController(args.port, args.baud, echo=args.echo,
                              default_speed=args.speed)
        hand.connect()
        online_set = hand._online
        if not online_set:
            print("[警告] 没有检测到任何舵机。切换到 mock 模式继续运行。")
            hand = None
    else:
        print("[EMG] mock 模式：仅显示映射角度，不发送串口命令")

    # ── 初始化肌电数据源 + 映射器 ─────────────────────────────────────────────
    source = build_source(args)
    mapper = EMGMapper(channel_map=EMG_CHANNEL_MAP, alpha=args.alpha)

    print(f"\n通道→关节映射（共 {NUM_CHANNELS} 路，可用舵机关节 {len(LIVE_JOINT_NAMES)} 个）:")
    for ch, joint in enumerate(EMG_CHANNEL_MAP):
        flag = "✓" if is_live(joint) else "·(无舵机)"
        print(f"  ch{ch:>2} → {joint:<12} {flag}")
    print("\nCtrl+C 退出。\n")

    period = 1.0 / max(args.rate, 1.0)
    fps    = 0.0
    t_prev = time.monotonic()

    try:
        source.connect()
        while _running[0]:
            # 1) 读 16 路肌电
            channels = source.read_channels()

            # 2) 映射成关节角度（含平滑）
            angles = mapper.map(channels)

            # 3) 下发到舵机（离线/虚空关节自动跳过）
            if hand is not None:
                hand.set_all(angles, speed=args.speed)

            # 4) 状态显示
            _print_status(angles, online_set, fps)

            # 5) 控频
            time.sleep(period)
            t_now  = time.monotonic()
            fps    = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now

    except Exception as e:
        print(f"\n[Error] {e}")
    finally:
        try:
            source.disconnect()
        except Exception:
            pass
        if hand is not None:
            hand.disconnect()
        print("\nExited.")


if __name__ == "__main__":
    main()
