"""
SurfaceIdiot — MediaPipe → HLS3915M 统一控制入口
=================================================

摄像头捕捉人手 → MediaPipe 计算 15 个关节角度 → HLS3915M 舵机总线 → 灵巧手

与之前的 orca_control.py 的区别:
  - 后端换成 HandController（HLS3915M TTL 总线），不再依赖 orca_core
  - 自动适配缺失舵机（初期只有几个舵机时正常工作）
  - --mock 模式：无硬件也能跑，只看相机和关节角度

MediaPipe → HandController 关节名称映射（15/16 DOF）:
  thumb_mcp_flex  ← MediaPipe 拇指 MCP 屈伸
  thumb_mcp_abd   ← MediaPipe 拇指外展
  thumb_ip        ← MediaPipe 拇指 IP（用 PIP 近似）
  {index/middle/ring/pinky}_mcp  ← 各手指 MCP
  {index/middle/ring/pinky}_pip  ← 各手指 PIP
  {index/middle/ring}_abd        ← 相邻手指展开角
  wrist_flex      ← 手腕左右倾斜
  （wrist_abd 暂无 MediaPipe 数据源，保持 0°）

用法:
    python handTracking/hls_control.py --mock
    python handTracking/hls_control.py --port /dev/tty.usbserial-0001
    python handTracking/hls_control.py --port /dev/tty.usbserial-0001 --alpha 0.25
"""

import argparse
import os
import signal
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np

# ── 路径：让 handTracking/ 内的脚本也能 import hardware/ ─────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── MediaPipe 模型 ────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print("首次运行：下载 hand_landmarker.task (~9MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("下载完成")

# ── 手部骨架连线 ──────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

# ── Joint display order & labels (Orca Hand v1) ───────────────────────────────
JOINT_DISPLAY = [
    ("thumb_abd",  "Th.Abd "),   # ID 1
    ("thumb_mcp",  "Th.MCP "),   # ID 2
    ("thumb_pip",  "Th.PIP "),   # ID 3
    ("thumb_dip",  "Th.DIP "),   # ID 4
    ("index_abd",  "Idx.Abd"),   # ID 5
    ("index_mcp",  "Idx.MCP"),   # ID 6
    ("index_pip",  "Idx.PIP"),   # ID 7
    ("middle_abd", "Mid.Abd"),   # ID 8
    ("middle_mcp", "Mid.MCP"),   # ID 9
    ("middle_pip", "Mid.PIP"),   # ID 10
    ("ring_abd",   "Rng.Abd"),   # ID 11
    ("ring_mcp",   "Rng.MCP"),   # ID 12
    ("ring_pip",   "Rng.PIP"),   # ID 13
    ("pinky_abd",  "Pky.Abd"),   # ID 14
    ("pinky_mcp",  "Pky.MCP"),   # ID 15
    ("pinky_pip",  "Pky.PIP"),   # ID 16
]

# ─── 数学工具（从 orca_control.py 复用）──────────────────────────────────────

def _angle3(a, b, c) -> float:
    """三点夹角（°），b 为顶点。"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def _abduction(lm, base_idx, a_idx, b_idx) -> float:
    """相邻手指外展角（°），带符号：正 = 展开。"""
    origin = np.array([lm[base_idx].x, lm[base_idx].y])
    pa     = np.array([lm[a_idx].x,    lm[a_idx].y]) - origin
    pb     = np.array([lm[b_idx].x,    lm[b_idx].y]) - origin
    cross  = pa[0]*pb[1] - pa[1]*pb[0]
    return float(np.degrees(np.arctan2(cross, np.dot(pa, pb))))

def _remap(v, src_min, src_max, dst_min, dst_max) -> float:
    """线性映射并 clamp。"""
    t = (v - src_min) / (src_max - src_min + 1e-6)
    return float(np.clip(dst_min + t * (dst_max - dst_min), dst_min, dst_max))

# ─── MediaPipe → HLS 关节角度 ────────────────────────────────────────────────

def landmarks_to_hls(lm) -> dict:
    """
    将 MediaPipe 21 个地标转换为 Orca Hand v1 关节角度字典（单位：度）。
    关节名与 ORCA_ROM / HandController.JOINT_CONFIG 完全一致。

    MediaPipe 地标索引:
      手腕=0
      拇指  CMC=1 MCP=2  IP=3  TIP=4
      食指  MCP=5  PIP=6  DIP=7  TIP=8
      中指  MCP=9  PIP=10 DIP=11 TIP=12
      无名  MCP=13 PIP=14 DIP=15 TIP=16
      小指  MCP=17 PIP=18 DIP=19 TIP=20
    """
    def pt(i):
        return [lm[i].x, lm[i].y, lm[i].z]

    # 屈伸角：MediaPipe 关节约 160°(伸直) ~ 20°(弯曲) → 映射到 0°-90°
    def flex(a, b, c, src=(160, 20)) -> float:
        return _remap(_angle3(pt(a), pt(b), pt(c)), src[0], src[1], 0, 90)

    joints = {
        # 大拇指（4 DOF）— 拇指轴方向特殊，屈伸范围稍小
        "thumb_mcp": flex(1, 2, 3, src=(150, 30)),
        "thumb_pip": flex(2, 3, 4, src=(170, 60)),
        "thumb_dip": flex(2, 3, 4, src=(170, 60)),  # MediaPipe 无 DIP，用 IP 近似

        # 食指（3 DOF）
        "index_mcp": flex(5,  6,  7),
        "index_pip": flex(6,  7,  8),

        # 中指（3 DOF）
        "middle_mcp": flex(9,  10, 11),
        "middle_pip": flex(10, 11, 12),

        # 无名指（3 DOF）
        "ring_mcp": flex(13, 14, 15),
        "ring_pip": flex(14, 15, 16),

        # 小指（3 DOF）
        "pinky_mcp": flex(17, 18, 19),
        "pinky_pip": flex(18, 19, 20),

        # 外展角（所有手指，含小指）
        "thumb_abd":  _remap(_abduction(lm, 0,  2,  5), -40, 40, -30, 60),
        "index_abd":  _remap(_abduction(lm, 0,  5,  9), -30, 30, -20, 20),
        "middle_abd": _remap(_abduction(lm, 0,  9, 13), -30, 30, -20, 20),
        "ring_abd":   _remap(_abduction(lm, 0, 13, 17), -30, 30, -20, 20),
        "pinky_abd":  _remap(_abduction(lm, 0, 17, 20), -30, 30, -20, 20),
    }
    return joints

# ─── 低通滤波（平滑摄像头抖动）──────────────────────────────────────────────

class JointSmoother:
    """指数移动平均，alpha 越小越平滑（但响应越慢）。"""

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self._state: dict = {}

    def update(self, joints: dict) -> dict:
        if not self._state:
            self._state = dict(joints)
            return dict(joints)
        for k, v in joints.items():
            self._state[k] = self.alpha * v + (1 - self.alpha) * self._state[k]
        return dict(self._state)

# ─── OpenCV 可视化工具 ────────────────────────────────────────────────────────

def draw_skeleton(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(l.x * w), int(l.y * h)) for l in lm]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (0, 220, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 6, (0, 0, 255), -1)
        cv2.circle(frame, p, 7, (255, 255, 255), 1)


def draw_joint_hud(frame, joints: dict, online_set: set | None = None):
    """
    右侧浮层：每个关节一行，显示名称 + 角度条 + 数值 + 在线状态。
    online_set=None 表示 mock 模式（不显示在线状态）。
    """
    h, w = frame.shape[:2]
    panel_w = 230
    x0 = w - panel_w - 5
    row_h = 22
    top = 10

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 4, top - 4),
                  (w - 4, top + row_h * len(JOINT_DISPLAY) + 6),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    bar_w = 100
    for i, (jname, label) in enumerate(JOINT_DISPLAY):
        y = top + i * row_h
        val   = joints.get(jname, 0.0)
        cfg   = _joint_range(jname)
        norm  = (val - cfg[0]) / max(cfg[1] - cfg[0], 1e-3)
        norm  = max(0.0, min(1.0, norm))

        # 在线/离线/mock 颜色
        if online_set is None:
            color = (100, 200, 100)   # mock: 绿
        elif jname in online_set:
            color = (80,  180, 255)   # 在线: 蓝
        else:
            color = (60,  60,  60)    # 离线: 暗灰

        # 进度条
        cv2.rectangle(frame, (x0, y + 6), (x0 + bar_w, y + 16), (50, 50, 50), -1)
        cv2.rectangle(frame, (x0, y + 6), (x0 + int(bar_w * norm), y + 16), color, -1)

        # 标签 + 数值
        suffix = "" if online_set is None else (" ✓" if jname in online_set else " -")
        cv2.putText(frame,
                    f"{label}: {val:5.1f}{suffix}",
                    (x0 + bar_w + 4, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)


def _joint_range(jname: str):
    """返回关节 (min, max) 角度范围（用于进度条归一化，与 Orca Hand v1 ROM 一致）。"""
    _ranges = {
        # 大拇指
        "thumb_abd": (-30, 60), "thumb_mcp": (0, 90),
        "thumb_pip": (0, 90),   "thumb_dip": (0, 70),
        # 食指
        "index_abd": (-20, 20), "index_mcp": (0, 90), "index_pip": (0, 90),
        # 中指
        "middle_abd": (-20, 20), "middle_mcp": (0, 90), "middle_pip": (0, 90),
        # 无名指
        "ring_abd": (-20, 20), "ring_mcp": (0, 90), "ring_pip": (0, 90),
        # 小指
        "pinky_abd": (-20, 20), "pinky_mcp": (0, 90), "pinky_pip": (0, 90),
    }
    return _ranges.get(jname, (0, 90))


def draw_fps_bar(frame, fps: float, n_online: int | None, n_total: int):
    h, w = frame.shape[:2]
    if n_online is None:
        hw_str = "mock mode"
    else:
        hw_str = f"servos {n_online}/{n_total} online"
    cv2.putText(frame,
                f"FPS:{fps:.0f}  {hw_str}  Ctrl+C=quit  N=neutral  R=rescan",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

# ─── 主循环 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MediaPipe → HLS3915M 灵巧手控制")
    parser.add_argument("--port",   default="/dev/cu.usbmodem5B790315271",
                        help="舵机总线串口（macOS: /dev/cu.usbmodem... 用 tools/portFinder.py 查询）")
    parser.add_argument("--baud",   type=int, default=1_000_000)
    parser.add_argument("--cam",    type=int, default=0,   help="摄像头索引")
    parser.add_argument("--alpha",  type=float, default=0.35,
                        help="低通滤波系数 0-1（越小越平滑，越滞后）")
    parser.add_argument("--speed",  type=int, default=200,
                        help="舵机运动速度 0-4095（越大越慢）")
    parser.add_argument("--mock",   action="store_true",
                        help="模拟模式：不连接硬件，只显示相机和角度")
    parser.add_argument("--echo",   action="store_true",
                        help="启用 echo（3 线半双工接法）")
    args = parser.parse_args()

    # ── Ctrl+C in terminal → clean shutdown ───────────────────────────────────
    _running = [True]
    def _sig_handler(sig, frame):
        print("\n[Ctrl+C] Shutting down...")
        _running[0] = False
    signal.signal(signal.SIGINT, _sig_handler)

    # ── 初始化硬件 ────────────────────────────────────────────────────────────
    hand = None
    online_set = None  # None = mock 模式

    if not args.mock:
        from hardware.hand_controller import HandController
        hand = HandController(args.port, args.baud, echo=args.echo,
                              default_speed=args.speed)
        hand.connect()
        online_set = hand._online   # 引用（connect 后已填充）

        if not online_set:
            print("[警告] 没有检测到任何舵机。切换到 mock 模式继续运行。")
            hand = None
    else:
        print("[HLS] mock 模式：仅显示 MediaPipe 角度，不发送串口命令")
        from hardware.hand_controller import JOINT_CONFIG
        online_set = None

    # ── 初始化 MediaPipe ──────────────────────────────────────────────────────
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    det_opts  = vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    # ── 打开摄像头 ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smoother   = JointSmoother(alpha=args.alpha)
    last_joints: dict = {jn: 0.0 for jn, _ in JOINT_DISPLAY}
    frame_idx  = 0
    fps        = 0.0
    t_prev     = time.monotonic()

    n_total = 16  # 本项目最终目标 16 DOF

    print("\nReady.  Q (in camera window) or Ctrl+C (terminal) to quit.  N = neutral.\n")

    try:
        with vision.HandLandmarker.create_from_options(det_opts) as detector:
            while cap.isOpened() and _running[0]:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                # ── MediaPipe 推理 ──────────────────────────────────────────
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms  = int(cap.get(cv2.CAP_PROP_POS_MSEC)) or frame_idx * 33
                result = detector.detect_for_video(mp_img, ts_ms)

                if result.hand_landmarks:
                    lm = result.hand_landmarks[0]

                    # 骨架
                    draw_skeleton(frame, lm)

                    # 计算并平滑关节角度
                    raw_joints    = landmarks_to_hls(lm)
                    smooth_joints = smoother.update(raw_joints)
                    last_joints   = smooth_joints

                    # 发送到舵机（只有在线的才会实际执行）
                    if hand is not None:
                        hand.set_all(smooth_joints, speed=args.speed)
                else:
                    cv2.putText(frame, "No hand detected",
                            (50, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 80, 255), 2)

                # ── HUD ────────────────────────────────────────────────────
                draw_joint_hud(frame, last_joints, online_set)
                draw_fps_bar(frame, fps,
                             len(online_set) if online_set is not None else None,
                             n_total)

                cv2.imshow("SurfaceIdiot | MediaPipe -> HLS3915M", frame)

                # ── 按键处理 ───────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n') and hand is not None:
                    hand.set_neutral(speed=150)
                elif key == ord('r') and hand is not None:
                    print("[手] 重新检测舵机...")
                    hand.redetect()
                    online_set = hand._online

                # ── FPS 计算 ───────────────────────────────────────────────
                t_now  = time.monotonic()
                fps    = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
                t_prev = t_now
                frame_idx += 1

    except Exception as e:
        print(f"\n[Error] {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if hand is not None:
            hand.disconnect()
        print("Exited.")


if __name__ == "__main__":
    main()
