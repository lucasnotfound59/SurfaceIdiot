"""
control.py — 灵巧手统一控制入口（双入口分发器）
================================================

一个入口，两种数据源，共用同一套舵机后端（HandController + hardware/joint_map.py）：

    --source mediapipe   摄像头 + MediaPipe 手部追踪   → handTracking/hls_control.py
    --source emg         肌电手环 16 通道              → emg/emg_control.py

两个子入口也可以单独直接运行，本文件只是把它们汇总到一处、方便切换。
未知参数会原样透传给对应子入口。

用法：
    python control.py --source emg --mock
    python control.py --source mediapipe --port /dev/cu.usbmodem5B790315271
    python control.py emg --rate 50          # source 也可作为第一个位置参数
    python control.py mediapipe --mock
"""

import os
import runpy
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))


USAGE = (
    "用法: python control.py --source {mediapipe|emg} [子入口参数...]\n"
    "  mediapipe : 摄像头 + MediaPipe 手部追踪\n"
    "  emg       : 肌电手环 16 通道\n"
    "示例:\n"
    "  python control.py --source emg --mock\n"
    "  python control.py mediapipe --port /dev/cu.usbmodem5B790315271\n"
)


def _extract_source(argv):
    """从参数里取出 source，剩余参数透传给子入口。"""
    args = list(argv)
    # 形式一：--source X
    if "--source" in args:
        i = args.index("--source")
        try:
            src = args[i + 1]
        except IndexError:
            return None, args
        del args[i:i + 2]
        return src, args
    # 形式二：第一个位置参数就是 source
    if args and not args[0].startswith("-"):
        return args[0], args[1:]
    return None, args


def main():
    source, rest = _extract_source(sys.argv[1:])

    if source in (None, "help", "-h", "--help"):
        print(USAGE)
        sys.exit(0 if source in ("help", "-h", "--help") else 1)

    source = source.lower()

    if source in ("mediapipe", "mp", "camera", "vision"):
        script = os.path.join(_ROOT, "handTracking", "hls_control.py")
    elif source in ("emg", "armband", "myo"):
        script = os.path.join(_ROOT, "emg", "emg_control.py")
    else:
        print(f"未知数据源: '{source}'\n")
        print(USAGE)
        sys.exit(1)

    # 把剩余参数交给子入口的 argparse，再以 __main__ 方式运行它
    sys.argv = [script] + rest
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
