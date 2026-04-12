"""
SurfaceIdiot - Real-time Inference

Loads a trained GraspPolicy checkpoint and runs it in a loop:
  1. Grabs the latest frame from the wrist camera
  2. Reads the latest glove joint history (or uses zeros if no glove connected)
  3. Predicts target joint angles
  4. Optionally sends angles over serial to the robot controller

Usage:
    # Dry-run (camera only, print predictions)
    python infer.py --checkpoint ../checkpoints/best_model.pth

    # With robot serial output
    python infer.py --checkpoint ../checkpoints/best_model.pth --robot_port /dev/ttyUSB1
"""

import argparse
import collections
import json
import time
import threading
import queue

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from model import GraspPolicy

FINGERS    = ["thumb", "index", "middle", "ring", "pinky"]
JOINT_DIM  = 5


# ─── Wrist camera reader ──────────────────────────────────────────────────────

class CameraReader(threading.Thread):
    def __init__(self, index: int):
        super().__init__(daemon=True)
        self.index = index
        self.frame = None
        self._stop = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while not self._stop.is_set():
            ret, frame = cap.read()
            if ret:
                self.frame = frame
        cap.release()

    def stop(self):
        self._stop.set()


# ─── Glove reader (reused from collect.py logic) ──────────────────────────────

class GloveReader(threading.Thread):
    def __init__(self, port: str, baud: int = 115200):
        super().__init__(daemon=True)
        self.port      = port
        self.last_data = None
        self._stop     = threading.Event()

    def run(self):
        import serial
        try:
            ser = serial.Serial(self.port, 115200, timeout=1)
            while not self._stop.is_set():
                if ser.in_waiting:
                    raw = ser.readline().decode(errors="replace").strip()
                    if raw and not raw.startswith("#"):
                        try:
                            self.last_data = json.loads(raw)
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            print(f"Glove error: {e}")

    def get_joints(self):
        if self.last_data is None:
            return None
        try:
            f = self.last_data["fingers"]
            return np.array([f[k] for k in FINGERS], dtype=np.float32)
        except (KeyError, TypeError):
            return None

    def stop(self):
        self._stop.set()


# ─── Image transform (matches training val transform) ─────────────────────────

IMG_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ─── Robot serial output ──────────────────────────────────────────────────────

def send_joint_command(ser, joints: np.ndarray):
    """
    Send a JSON command to the robot micro-controller.
    Format: {"joints": [t, i, m, r, p]}\n
    """
    cmd = json.dumps({"joints": joints.tolist()}) + "\n"
    ser.write(cmd.encode())


# ─── Main inference loop ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SurfaceIdiot inference")
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--cam_wrist",    type=int, default=0)
    parser.add_argument("--glove_port",   default=None, help="Glove serial port (optional)")
    parser.add_argument("--robot_port",   default=None, help="Robot serial port (optional)")
    parser.add_argument("--history_len",  type=int, default=6)
    parser.add_argument("--fps",          type=int, default=30, help="Target inference FPS")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ──
    model = GraspPolicy(history_len=args.history_len).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, "
          f"val_loss={ckpt.get('val_loss', '?')})")

    # ── Start camera ──
    cam = CameraReader(args.cam_wrist)
    cam.start()
    time.sleep(0.5)

    # ── Start glove ──
    glove = None
    if args.glove_port:
        glove = GloveReader(args.glove_port)
        glove.start()
        time.sleep(1)

    # ── Robot serial ──
    robot_ser = None
    if args.robot_port:
        import serial
        robot_ser = serial.Serial(args.robot_port, 115200)
        print(f"Robot connected on {args.robot_port}")

    # Joint history buffer (initialised to all-zeros = hand open)
    joint_buf = collections.deque(
        [np.zeros(JOINT_DIM, dtype=np.float32)] * args.history_len,
        maxlen=args.history_len,
    )

    interval = 1.0 / args.fps
    print(f"\nRunning at {args.fps} FPS. Press Q in the preview window to quit.\n")

    try:
        while True:
            t0 = time.time()

            frame = cam.frame
            if frame is None:
                time.sleep(0.01)
                continue

            # ── Preprocess image ──
            img_tensor = IMG_TRANSFORM(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = img_tensor.unsqueeze(0).to(device)   # (1, 3, 224, 224)

            # ── Build history tensor ──
            history = np.stack(list(joint_buf))                # (H, 5)
            hist_tensor = torch.from_numpy(history).unsqueeze(0).to(device)  # (1, H, 5)

            # ── Inference ──
            with torch.no_grad():
                pred = model(img_tensor, hist_tensor).squeeze(0).cpu().numpy()  # (5,)

            # Update history with latest glove reading (if available) or prediction
            if glove:
                g = glove.get_joints()
                joint_buf.append(g if g is not None else pred)
            else:
                joint_buf.append(pred)

            # ── Send to robot ──
            if robot_ser:
                send_joint_command(robot_ser, pred)

            # ── Display ──
            disp = frame.copy()
            for i, (name, val) in enumerate(zip(FINGERS, pred)):
                bar_w = int(val * 150)
                cv2.rectangle(disp, (10, 20 + i*28), (10 + bar_w, 40 + i*28), (0, 200, 100), -1)
                cv2.putText(disp, f"{name}: {val:.2f}", (170, 36 + i*28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.putText(disp, f"FPS: {1/(time.time()-t0+1e-6):.0f}",
                        (10, disp.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("SurfaceIdiot — Inference", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Rate limit
            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    except KeyboardInterrupt:
        pass

    cam.stop()
    if glove:
        glove.stop()
    if robot_ser:
        robot_ser.close()
    cv2.destroyAllWindows()
    print("Inference stopped.")


if __name__ == "__main__":
    main()
