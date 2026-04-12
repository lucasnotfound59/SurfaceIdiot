"""
SurfaceIdiot - Data Collection Script

Synchronously records:
  - Side-view camera frames (external perspective)
  - Wrist-view camera frames (what the robot will see)
  - Glove sensor data (finger angles + IMU)

Each episode is saved as:
  data/<object>/<YYYYMMDD_HHMMSS>/
      side_000000.jpg
      wrist_000000.jpg
      ...
      metadata.json   <- frame timestamps + glove readings

Usage:
    python collect.py --object ball --glove_port /dev/ttyUSB0
    python collect.py --object cup  --cam_side 0 --cam_wrist 1
    python collect.py --list_objects  # show collected stats
"""

import argparse
import json
import os
import sys
import time
import threading
import queue
from datetime import datetime

import cv2
import serial


# ─── Glove reader (runs in background thread) ──────────────────────────────

class GloveReader(threading.Thread):
    """Continuously reads JSON frames from the ESP32 into a queue."""

    def __init__(self, port: str, baud: int = 115200):
        super().__init__(daemon=True)
        self.port  = port
        self.baud  = baud
        self.queue: queue.Queue = queue.Queue(maxsize=10)
        self._stop = threading.Event()
        self.connected = False
        self.last_data = None

    def run(self):
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
            self.connected = True
            while not self._stop.is_set():
                if ser.in_waiting:
                    raw = ser.readline().decode(errors="replace").strip()
                    if not raw or raw.startswith("#"):
                        continue
                    try:
                        data = json.loads(raw)
                        self.last_data = data
                        # Non-blocking put (drop if consumer is slow)
                        try:
                            self.queue.put_nowait(data)
                        except queue.Full:
                            try:
                                self.queue.get_nowait()
                            except queue.Empty:
                                pass
                            self.queue.put_nowait(data)
                    except json.JSONDecodeError:
                        pass
        except serial.SerialException as e:
            print(f"\nGlove serial error: {e}")
            self.connected = False

    def stop(self):
        self._stop.set()

    def get_latest(self):
        """Return the most recent frame (non-blocking)."""
        return self.last_data


# ─── Camera setup ──────────────────────────────────────────────────────────

def open_camera(index: int, width: int = 640, height: int = 480, fps: int = 30):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


# ─── Episode recording ──────────────────────────────────────────────────────

def collect_episode(
    object_name: str,
    cam_side: cv2.VideoCapture,
    cam_wrist: cv2.VideoCapture,
    glove: GloveReader,
    data_root: str = "data",
    preview: bool = True,
) -> int:
    """
    Record one grasp episode.
    Controls:
        SPACE  – start / stop recording
        Q      – quit without saving
    Returns number of frames saved, or 0 if aborted.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(data_root, object_name, stamp)
    os.makedirs(save_dir, exist_ok=True)

    recording = False
    frames_meta = []
    frame_idx   = 0

    WINDOW = "SurfaceIdiot — press SPACE to start, Q to quit"

    while True:
        ret_s, frame_side  = cam_side.read()
        ret_w, frame_wrist = cam_wrist.read()

        if not ret_s or not ret_w:
            print("Camera read error. Check camera indices.")
            break

        glove_data = glove.get_latest()

        if recording and glove_data is not None:
            # Save images
            cv2.imwrite(
                os.path.join(save_dir, f"side_{frame_idx:06d}.jpg"),
                frame_side,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            cv2.imwrite(
                os.path.join(save_dir, f"wrist_{frame_idx:06d}.jpg"),
                frame_wrist,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            frames_meta.append({
                "frame_id":  frame_idx,
                "timestamp": time.time(),
                "glove":     glove_data,
            })
            frame_idx += 1

        if preview:
            # Overlay status
            status = "● REC" if recording else "○ READY"
            color  = (0, 0, 255) if recording else (0, 255, 0)
            disp   = frame_side.copy()
            cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, color, 2)
            if recording:
                cv2.putText(disp, f"Frames: {frame_idx}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if glove_data:
                f = glove_data.get("fingers", {})
                hud = "  ".join(
                    f"{k[0].upper()}: {v:.2f}"
                    for k, v in f.items()
                )
                cv2.putText(disp, hud, (10, disp.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(WINDOW, disp)
            cv2.imshow("Wrist View", frame_wrist)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if not recording:
                print(f"  Recording started → {save_dir}")
                recording = True
            else:
                break  # stop recording, save episode
        elif key == ord("q"):
            print("  Aborted, discarding episode.")
            import shutil
            shutil.rmtree(save_dir, ignore_errors=True)
            return 0

    cv2.destroyAllWindows()

    if not frames_meta:
        print("  No frames captured.")
        return 0

    # Save metadata
    metadata = {
        "object":     object_name,
        "episode_id": stamp,
        "num_frames": len(frames_meta),
        "frames":     frames_meta,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"  Saved {len(frames_meta)} frames to {save_dir}")
    return len(frames_meta)


# ─── Stats ──────────────────────────────────────────────────────────────────

def list_objects(data_root: str = "data"):
    if not os.path.isdir(data_root):
        print("No data directory found.")
        return
    for obj in sorted(os.listdir(data_root)):
        obj_dir = os.path.join(data_root, obj)
        if not os.path.isdir(obj_dir):
            continue
        episodes = [
            ep for ep in os.listdir(obj_dir)
            if os.path.isfile(os.path.join(obj_dir, ep, "metadata.json"))
        ]
        total_frames = 0
        for ep in episodes:
            try:
                with open(os.path.join(obj_dir, ep, "metadata.json")) as f:
                    total_frames += json.load(f).get("num_frames", 0)
            except Exception:
                pass
        print(f"  {obj:20s}  {len(episodes):4d} episodes  {total_frames:6d} frames")


# ─── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SurfaceIdiot data collection")
    parser.add_argument("--object",      default="ball",       help="Object being grasped")
    parser.add_argument("--glove_port",  default="/dev/ttyUSB0")
    parser.add_argument("--baud",        default=115200, type=int)
    parser.add_argument("--cam_side",    default=0, type=int,  help="Side camera index")
    parser.add_argument("--cam_wrist",   default=1, type=int,  help="Wrist camera index")
    parser.add_argument("--data_root",   default="data")
    parser.add_argument("--episodes",    default=500, type=int, help="Target episode count")
    parser.add_argument("--no_preview",  action="store_true")
    parser.add_argument("--list_objects",action="store_true",  help="Show data stats and exit")
    args = parser.parse_args()

    if args.list_objects:
        list_objects(args.data_root)
        return

    # Connect glove
    print(f"Connecting to glove on {args.glove_port}...")
    glove = GloveReader(args.glove_port, args.baud)
    glove.start()
    time.sleep(2)
    if not glove.connected:
        print("WARNING: Glove not connected. Continuing without glove data.")

    # Open cameras
    try:
        cam_side  = open_camera(args.cam_side)
        cam_wrist = open_camera(args.cam_wrist)
    except RuntimeError as e:
        print(f"Camera error: {e}")
        glove.stop()
        sys.exit(1)

    # Count existing episodes
    obj_dir = os.path.join(args.data_root, args.object)
    existing = 0
    if os.path.isdir(obj_dir):
        existing = sum(
            1 for ep in os.listdir(obj_dir)
            if os.path.isfile(os.path.join(obj_dir, ep, "metadata.json"))
        )

    print(f"\nObject: '{args.object}'  |  Existing episodes: {existing}  |  Target: {args.episodes}")
    print("─" * 60)
    print("SPACE = start/stop recording   Q = quit session\n")

    ep_count = existing
    try:
        while ep_count < args.episodes:
            print(f"Episode {ep_count + 1}/{args.episodes}:")
            n = collect_episode(
                object_name=args.object,
                cam_side=cam_side,
                cam_wrist=cam_wrist,
                glove=glove,
                data_root=args.data_root,
                preview=not args.no_preview,
            )
            if n > 0:
                ep_count += 1
            # After Q is pressed inside episode the loop would break, but
            # pressing Q only aborts one episode. To quit the session the
            # user can Ctrl-C here.
    except KeyboardInterrupt:
        print("\nSession ended by user.")

    cam_side.release()
    cam_wrist.release()
    cv2.destroyAllWindows()
    glove.stop()
    print(f"\nTotal episodes for '{args.object}': {ep_count}")


if __name__ == "__main__":
    main()
