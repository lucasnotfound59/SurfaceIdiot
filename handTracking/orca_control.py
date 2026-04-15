"""
SurfaceIdiot - MediaPipe → Orca Hand 实时控制

通过摄像头捕捉人手姿态，用 MediaPipe 解析关节角度，
映射到 Orca Hand 的 17 个自由度并实时发送。

依赖:
    pip install orca_core mediapipe opencv-python numpy Pillow

用法:
    # 真实硬件
    python orca_control.py --port /dev/tty.usbserial-XXXXXXXX

    # 模拟模式（无硬件也能跑，打印指令）
    python orca_control.py --mock
"""

import argparse
import collections
import os
import queue
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── 模型文件 ────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("正在下载 hand_landmarker.task …")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("下载完成")

# ── 骨架连线 ────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

# ── Orca Hand 关节名称（17 DOF） ─────────────────────────────
# ROM 参考值（度），来自 orca_core 默认配置
# 格式: joint_name → (min_deg, max_deg)
ORCA_ROM = {
    "thumb_abd":   (-30,  60),
    "thumb_mcp":   (  0,  90),
    "thumb_pip":   (  0,  90),
    "thumb_dip":   (  0,  70),
    "index_abd":   (-20,  20),
    "index_mcp":   (  0,  90),
    "index_pip":   (  0,  90),
    "middle_abd":  (-20,  20),
    "middle_mcp":  (  0,  90),
    "middle_pip":  (  0,  90),
    "ring_abd":    (-20,  20),
    "ring_mcp":    (  0,  90),
    "ring_pip":    (  0,  90),
    "pinky_abd":   (-20,  20),
    "pinky_mcp":   (  0,  90),
    "pinky_pip":   (  0,  90),
    "wrist":       (-30,  30),
}

# 控制频率
CONTROL_HZ = 20
SMOOTH_ALPHA = 0.35   # 低通滤波系数（越小越平滑但越滞后）


# ── 中文渲染 ────────────────────────────────────────────────
_FONT_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
]
_font_cache: dict = {}

def _get_font(size: int):
    if size not in _font_cache:
        for p in _FONT_PATHS:
            if os.path.exists(p):
                _font_cache[size] = ImageFont.truetype(p, size)
                return _font_cache[size]
        _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

def put_cn(frame, text, pos, size=18, color=(220, 220, 220)):
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(pos, text, font=_get_font(size),
                             fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ── 数学工具 ────────────────────────────────────────────────

def angle3(a, b, c):
    """三点夹角（度），以 b 为顶点"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def abduction_angle(lm, base_idx, a_idx, b_idx):
    """
    相邻手指外展角：计算两根手指 MCP 连线相对于手掌轴的偏转。
    返回带符号角度（度），正值 = 向外展开。
    """
    origin = np.array([lm[base_idx].x, lm[base_idx].y])
    pa     = np.array([lm[a_idx].x,    lm[a_idx].y]) - origin
    pb     = np.array([lm[b_idx].x,    lm[b_idx].y]) - origin
    cross  = pa[0]*pb[1] - pa[1]*pb[0]  # z 分量
    dot    = np.dot(pa, pb)
    return float(np.degrees(np.arctan2(cross, dot)))

def remap(v, src_min, src_max, dst_min, dst_max):
    """线性映射并 clamp"""
    t = (v - src_min) / (src_max - src_min + 1e-6)
    return float(np.clip(dst_min + t * (dst_max - dst_min), dst_min, dst_max))

def lm_pt(lm, i):
    return [lm[i].x, lm[i].y, lm[i].z]


# ── MediaPipe → Orca 关节角度映射 ────────────────────────────

def landmarks_to_orca(lm) -> dict:
    """
    将 21 个 MediaPipe 地标转换为 Orca Hand 关节角度字典（单位：度）。

    MediaPipe 索引：
      手腕=0, 拇指 CMC=1 MCP=2 IP=3 TIP=4
      食指 MCP=5 PIP=6 DIP=7 TIP=8
      中指 MCP=9 PIP=10 DIP=11 TIP=12
      无名 MCP=13 PIP=14 DIP=15 TIP=16
      小指 MCP=17 PIP=18 DIP=19 TIP=20
    """
    def pt(i): return lm_pt(lm, i)

    # ─ 弯曲角（MCP/PIP）：关节角越小 = 手指越弯 ─
    # 映射规则：mediapipe 角约 160°(伸直) → 20°(握拳)
    #           Orca MCP/PIP 约 0°(伸直) → 90°(握拳)
    def flex(a, b, c, src=(160, 20)):
        raw = angle3(pt(a), pt(b), pt(c))
        return remap(raw, src[0], src[1], 0, 90)

    joints = {
        # 拇指（MediaPipe 拇指轴与其他手指不同，范围稍小）
        "thumb_mcp": flex(1, 2, 3, src=(150, 30)),
        "thumb_pip": flex(2, 3, 4, src=(170, 60)),
        "thumb_dip": flex(2, 3, 4, src=(170, 60)),   # mediapipe 无 DIP，用 IP 近似

        # 食指
        "index_mcp": flex(5,  6,  7),
        "index_pip": flex(6,  7,  8),

        # 中指
        "middle_mcp": flex(9,  10, 11),
        "middle_pip": flex(10, 11, 12),

        # 无名指
        "ring_mcp": flex(13, 14, 15),
        "ring_pip": flex(14, 15, 16),

        # 小指
        "pinky_mcp": flex(17, 18, 19),
        "pinky_pip": flex(18, 19, 20),
    }

    # ─ 外展角（ABduction） ─
    # 用相邻手指 MCP 之间的夹角来估计展开程度
    joints["thumb_abd"]  = remap(abduction_angle(lm, 0, 2, 5),  -40, 40, -30, 60)
    joints["index_abd"]  = remap(abduction_angle(lm, 0, 5, 9),  -30, 30, -20, 20)
    joints["middle_abd"] = remap(abduction_angle(lm, 0, 9, 13), -30, 30, -20, 20)
    joints["ring_abd"]   = remap(abduction_angle(lm, 0, 13, 17),-30, 30, -20, 20)
    joints["pinky_abd"]  = remap(abduction_angle(lm, 0, 17, 20),-30, 30, -20, 20)

    # ─ 手腕（用手腕到中指MCP的向量倾斜估计） ─
    wrist = np.array([lm[0].x, lm[0].y])
    mid_mcp = np.array([lm[9].x, lm[9].y])
    vec = mid_mcp - wrist
    wrist_angle = float(np.degrees(np.arctan2(vec[0], -vec[1])))  # 相对竖直
    joints["wrist"] = remap(wrist_angle, -45, 45, -30, 30)

    # ─ clamp 到 ROM ─
    for name, val in joints.items():
        lo, hi = ORCA_ROM[name]
        joints[name] = float(np.clip(val, lo, hi))

    return joints


# ── 低通滤波器（平滑抖动） ────────────────────────────────────

class JointSmoother:
    def __init__(self, alpha: float = SMOOTH_ALPHA):
        self.alpha = alpha
        self.state: dict | None = None

    def update(self, joints: dict) -> dict:
        if self.state is None:
            self.state = dict(joints)
            return dict(joints)
        for k, v in joints.items():
            self.state[k] = self.alpha * v + (1 - self.alpha) * self.state[k]
        return dict(self.state)


# ── Orca Hand 控制线程 ────────────────────────────────────────

class OrcaController(threading.Thread):
    """
    后台线程：从队列取关节指令，发送给 Orca Hand。
    主线程只需 put() 最新的关节字典，控制线程自动以 CONTROL_HZ 执行。
    """

    def __init__(self, port: str, mock: bool = False):
        super().__init__(daemon=True)
        self.mock      = mock
        self.port      = port
        self.cmd_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop     = threading.Event()
        self.connected = False
        self.hand      = None
        self.last_joints: dict = {}

    def run(self):
        if not self.mock:
            try:
                from orca_core import OrcaHand
                self.hand = OrcaHand()          # 使用默认 v1 右手配置
                ok, msg = self.hand.connect(self.port)
                if not ok:
                    print(f"[OrcaController] 连接失败: {msg}")
                    return
                self.hand.enable_torque()
                self.hand.set_neutral_position()
                self.connected = True
                print(f"[OrcaController] 已连接 {self.port}")
            except Exception as e:
                print(f"[OrcaController] 初始化失败: {e}")
                return
        else:
            self.connected = True
            print("[OrcaController] 模拟模式，不连接真实硬件")

        interval = 1.0 / CONTROL_HZ
        while not self._stop.is_set():
            t0 = time.time()
            try:
                joints = self.cmd_queue.get_nowait()
                self.last_joints = joints
                if self.mock:
                    # 模拟模式：打印指令摘要
                    summary = "  ".join(
                        f"{k.split('_')[0][0]}{k.split('_')[1][0].upper()}: {v:5.1f}°"
                        for k, v in list(joints.items())[:6]
                    )
                    print(f"\r{summary}", end="", flush=True)
                else:
                    self.hand.set_joint_positions(
                        joints, num_steps=3, step_size=0.002
                    )
            except queue.Empty:
                pass
            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def send(self, joints: dict):
        """非阻塞发送（丢弃旧的未发指令）"""
        try:
            self.cmd_queue.put_nowait(joints)
        except queue.Full:
            try:
                self.cmd_queue.get_nowait()
            except queue.Empty:
                pass
            self.cmd_queue.put_nowait(joints)

    def stop(self):
        self._stop.set()
        if self.hand and not self.mock:
            try:
                self.hand.set_neutral_position()
                self.hand.disable_torque()
                self.hand.disconnect()
            except Exception:
                pass


# ── HUD 绘制 ─────────────────────────────────────────────────

def draw_skeleton(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(l.x * w), int(l.y * h)) for l in lm]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 6, (0, 0, 255), -1)
        cv2.circle(frame, p, 7, (255, 255, 255), 1)

def draw_joint_hud(frame, joints: dict, connected: bool):
    """右侧显示所有关节当前角度"""
    h, w = frame.shape[:2]
    x0 = w - 210

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 5, 5), (w - 5, h - 5), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    status_color = (0, 220, 0) if connected else (0, 60, 220)
    status_text  = "已连接" if connected else "未连接"
    put_cn(frame, f"Orca Hand  {status_text}", (x0, 10), size=15, color=status_color)

    bar_w = 150
    for i, (name, val) in enumerate(joints.items()):
        lo, hi = ORCA_ROM[name]
        norm   = (val - lo) / (hi - lo + 1e-6)
        y      = 36 + i * 22

        cv2.rectangle(frame, (x0, y + 4), (x0 + bar_w, y + 14), (60, 60, 60), -1)
        cv2.rectangle(frame, (x0, y + 4),
                      (x0 + int(bar_w * norm), y + 14), (80, 180, 120), -1)
        cv2.putText(frame, f"{name}: {val:5.1f}", (x0, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)


# ── 主循环 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MediaPipe → Orca Hand 控制")
    parser.add_argument("--port",    default="/dev/tty.usbserial-0001",
                        help="Orca Hand 串口（macOS: /dev/tty.usbserial-XXX）")
    parser.add_argument("--cam",     type=int, default=0, help="摄像头索引")
    parser.add_argument("--mock",    action="store_true", help="模拟模式（无硬件）")
    parser.add_argument("--alpha",   type=float, default=SMOOTH_ALPHA,
                        help="低通滤波系数 0-1（越小越平滑）")
    args = parser.parse_args()

    # 启动 Orca 控制线程
    orca = OrcaController(port=args.port, mock=args.mock)
    orca.start()
    time.sleep(1.5)

    # 初始化 MediaPipe
    base_opts = python.BaseOptions(model_asset_path=MODEL_PATH)
    det_opts  = vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smoother   = JointSmoother(alpha=args.alpha)
    frame_idx  = 0
    last_joints: dict = {k: 0.0 for k in ORCA_ROM}

    print("启动完成，按 Q 退出，按 N 让机械手回中立位")

    with vision.HandLandmarker.create_from_options(det_opts) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms    = int(cap.get(cv2.CAP_PROP_POS_MSEC)) or frame_idx * 33
            results  = detector.detect_for_video(mp_img, ts_ms)

            if results.hand_landmarks:
                lm = results.hand_landmarks[0]
                draw_skeleton(frame, lm)

                raw_joints   = landmarks_to_orca(lm)
                smooth_joints = smoother.update(raw_joints)
                last_joints   = smooth_joints
                orca.send(smooth_joints)
            else:
                put_cn(frame, "未检测到手部", (50, 50), size=32, color=(0, 100, 255))

            draw_joint_hud(frame, last_joints, orca.connected)

            # FPS
            cv2.putText(frame, f"FPS: {cap.get(cv2.CAP_PROP_FPS):.0f}",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow("MediaPipe → Orca Hand", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n') and orca.hand:
                orca.hand.set_neutral_position()

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    orca.stop()
    print("\n退出")


if __name__ == "__main__":
    main()
