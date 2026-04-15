import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
from PIL import Image, ImageDraw, ImageFont

# ── 中文字体（macOS 自带）──────────────────────────────────
_FONT_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]
_font_cache: dict = {}

def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        for path in _FONT_PATHS:
            if os.path.exists(path):
                _font_cache[size] = ImageFont.truetype(path, size)
                break
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

def cv2_put_chinese(frame, text: str, pos, font_size: int = 18,
                    color=(220, 220, 220)):
    """在 OpenCV frame 上渲染中文（通过 Pillow 中转）"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=_get_font(font_size), fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ── 下载模型文件（首次运行自动下载）────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("首次运行，正在下载 hand_landmarker.task 模型（约 9MB）...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("模型下载完成")

# ── 手部骨架连接关系（与旧版 HAND_CONNECTIONS 相同）────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # 拇指
    (0,5),(5,6),(6,7),(7,8),           # 食指
    (5,9),(9,10),(10,11),(11,12),      # 中指
    (9,13),(13,14),(14,15),(15,16),    # 无名指
    (13,17),(0,17),(17,18),(18,19),(19,20),  # 小指 + 手掌
]

# ── 自定义颜色 ──────────────────────────────────────────────
DOT_COLOR      = (0, 0, 255)
LINE_COLOR     = (0, 255, 0)
DOT_RADIUS     = 8
LINE_THICKNESS = 3


def draw_custom_landmarks(frame, landmarks):
    """绘制关键点和骨架连线"""
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], LINE_COLOR, LINE_THICKNESS)

    for point in points:
        cv2.circle(frame, point, DOT_RADIUS, DOT_COLOR, -1)
        cv2.circle(frame, point, DOT_RADIUS + 1, (255, 255, 255), 1)


def calculate_finger_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def get_finger_angles(landmarks):
    def pt(i):
        return [landmarks[i].x, landmarks[i].y, landmarks[i].z]

    return {
        "thumb":  calculate_finger_angle(pt(1),  pt(2),  pt(3)),
        "index":  calculate_finger_angle(pt(5),  pt(6),  pt(7)),
        "middle": calculate_finger_angle(pt(9),  pt(10), pt(11)),
        "ring":   calculate_finger_angle(pt(13), pt(14), pt(15)),
        "pinky":  calculate_finger_angle(pt(17), pt(18), pt(19)),
    }


def angle_to_normalized(angle, min_angle=30, max_angle=170):
    return float(np.clip(1.0 - (angle - min_angle) / (max_angle - min_angle), 0.0, 1.0))


def draw_finger_bars(frame, angles):
    """左侧实时弯曲度进度条"""
    labels = {"thumb":"拇指","index":"食指","middle":"中指","ring":"无名","pinky":"小指"}
    colors = {
        "thumb":  (100, 100, 255),
        "index":  (100, 255, 100),
        "middle": (255, 100, 100),
        "ring":   (255, 255, 100),
        "pinky":  (255, 100, 255),
    }

    x, y0 = 10, 30
    bar_w, bar_h, gap = 140, 16, 30

    overlay = frame.copy()
    cv2.rectangle(overlay, (x-5, y0-20), (x+bar_w+65, y0+gap*5+5), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (finger, angle) in enumerate(angles.items()):
        y     = y0 + i * gap
        norm  = angle_to_normalized(angle)
        color = colors[finger]

        cv2.rectangle(frame, (x+38, y), (x+38+bar_w, y+bar_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (x+38, y), (x+38+int(bar_w*norm), y+bar_h), color, -1)
        cv2_put_chinese(frame, labels[finger], (x, y), font_size=15, color=color)
        cv2.putText(frame, f"{norm:.2f}", (x+38+bar_w+4, y+13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)


def main():
    # ── 初始化 HandLandmarker（新版 Tasks API）──────────────
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("启动中... 按 Q 退出")

    with vision.HandLandmarker.create_from_options(options) as detector:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # 新版 API 需要传入时间戳（毫秒）
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms == 0:
                timestamp_ms = frame_idx * 33  # fallback: 30fps

            results = detector.detect_for_video(mp_image, timestamp_ms)

            if results.hand_landmarks:
                for landmarks in results.hand_landmarks:
                    draw_custom_landmarks(frame, landmarks)
                    angles = get_finger_angles(landmarks)
                    draw_finger_bars(frame, angles)
            else:
                cv2_put_chinese(frame, "未检测到手部", (50, 50),
                               font_size=32, color=(0, 100, 255))

            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
