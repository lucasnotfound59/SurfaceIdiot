#!/usr/bin/env bash
# run.sh — 用正确的 Python 环境跑项目脚本，免去记 conda 环境的烦恼
#
# 用法:
#   ./run.sh port            # 查找串口 (portFinder.py)
#   ./run.sh test            # 单舵机交互测试 (servo_test.py)
#   ./run.sh full            # 整体测试 (full_test.py)
#   ./run.sh control         # 主入口 MediaPipe 实时控制 (hls_control.py)
#   ./run.sh map             # 打印舵机↔自由度映射表 (joint_map.py)
#   ./run.sh <相对路径.py> [参数...]   # 直接跑任意脚本
#
# 例:
#   ./run.sh full --deg 25 --loops 2
#   ./run.sh test --id 5

set -e

# handtrack 环境的 Python（装了 vassar_feetech_servo_sdk / pyserial / mediapipe）
PY="/opt/anaconda3/envs/handtrack/bin/python"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -x "$PY" ]; then
    echo "❌ 找不到 handtrack 环境的 Python: $PY"
    echo "   请确认 conda 环境 handtrack 已创建，或修改本脚本顶部的 PY 变量。"
    exit 1
fi

cmd="${1:-help}"
shift || true

case "$cmd" in
    port|portfinder)  exec "$PY" "$ROOT/tools/portFinder.py" "$@" ;;
    test|servo)       exec "$PY" "$ROOT/tools/servo_test.py" "$@" ;;
    full|fulltest)    exec "$PY" "$ROOT/tools/full_test.py" "$@" ;;
    setup|setupids)   exec "$PY" "$ROOT/tools/setup_ids.py" "$@" ;;
    reset|zero)       exec "$PY" "$ROOT/tools/reset_zero.py" "$@" ;;
    idcheck)          exec "$PY" "$ROOT/tools/id_check.py" "$@" ;;
    control|hls)      exec "$PY" "$ROOT/handTracking/hls_control.py" "$@" ;;
    emg)              exec "$PY" "$ROOT/emg/emg_control.py" "$@" ;;
    hand)             exec "$PY" "$ROOT/control.py" "$@" ;;
    map)              exec "$PY" "$ROOT/hardware/joint_map.py" "$@" ;;
    help|-h|--help)
        echo "用法: ./run.sh <命令> [参数...]"
        echo "  port     查找串口"
        echo "  test     单舵机交互测试"
        echo "  full     整体测试（每根手指动一下）"
        echo "  setup    舵机 ID 设置向导"
        echo "  reset    零位校准（当前姿态设为 0°）"
        echo "  control  入口①：MediaPipe 实时控制"
        echo "  emg      入口②：肌电手环实时控制"
        echo "  hand     统一分发器 (control.py --source mediapipe|emg)"
        echo "  map      打印舵机↔自由度映射表"
        echo "  <脚本.py> [参数]   直接跑任意脚本"
        ;;
    *)
        # 当成相对脚本路径直接跑
        exec "$PY" "$ROOT/$cmd" "$@"
        ;;
esac
