"""
SurfaceIdiot — 16 DOF 灵巧手控制器（Orca Hand v1，支持部分舵机缺失）
======================================================================

将 16 个关节角度映射到 HLS3915M 舵机位置，并通过 TTL 总线同步发送。
缺失的舵机会被自动跳过，不影响在线舵机的正常控制。

16 DOF 配置（Orca Hand v1，无手腕）:
    大拇指 (4 DOF) : thumb_abd, thumb_mcp, thumb_pip, thumb_dip   → ID  1-4
    食指   (3 DOF) : index_abd, index_mcp, index_pip              → ID  5-7
    中指   (3 DOF) : middle_abd, middle_mcp, middle_pip           → ID  8-10
    无名指 (3 DOF) : ring_abd, ring_mcp, ring_pip                 → ID 11-13
    小指   (3 DOF) : pinky_abd, pinky_mcp, pinky_pip              → ID 14-16

关节命名与 orca_control.py / ORCA_ROM 完全一致，便于直接对接。

用法:
    from hardware.hand_controller import HandController

    hand = HandController("/dev/cu.usbmodem5B790315271")
    hand.connect()
    hand.set_neutral()
    hand.set_joint("index_mcp", 60.0)
    hand.set_all({"index_mcp": 60, "index_pip": 50, "thumb_mcp": 40})
    hand.disconnect()
"""

import time
from typing import Dict, List, Optional

from hardware.hls3915m_bus import (
    HLSBus, deg_to_pos, pos_to_deg, CommError,
    UNITS_PER_DEG, CENTER_POS,
)
from hardware.calibration import load_offsets

# ─── 关节配置 ─────────────────────────────────────────────────────────────────
# 舵机 ↔ 自由度 映射统一在 hardware/joint_map.py 中维护（唯一可编辑来源）。
# 这里 re-export，保持对外接口不变。

from hardware.joint_map import (
    JointConfig,
    JOINT_CONFIG,
    ALL_JOINT_NAMES,
    ALL_SERVO_IDS,
    ID_TO_JOINT,
    JOINT_TO_ID,
    FINGERS,
    FLEX_JOINTS,
)


# ─── 预设手型 ─────────────────────────────────────────────────────────────────

PRESET_POSES: Dict[str, Dict[str, float]] = {
    "neutral": {k: 0.0 for k in ALL_JOINT_NAMES},

    "fist": {
        "thumb_abd": 20,  "thumb_mcp": 60,  "thumb_pip": 60,  "thumb_dip": 50,
        "index_abd": 0,   "index_mcp": 80,  "index_pip": 80,
        "middle_abd": 0,  "middle_mcp": 80, "middle_pip": 80,
        "ring_abd": 0,    "ring_mcp": 80,   "ring_pip": 80,
        "pinky_abd": 0,   "pinky_mcp": 80,  "pinky_pip": 80,
    },

    "pinch": {  # 拇食指捏握（捏笔/薄片）
        "thumb_abd": 30,  "thumb_mcp": 35,  "thumb_pip": 30,  "thumb_dip": 20,
        "index_abd": -10, "index_mcp": 50,  "index_pip": 40,
        "middle_abd": 0,  "middle_mcp": 70, "middle_pip": 70,
        "ring_abd": 0,    "ring_mcp": 75,   "ring_pip": 75,
        "pinky_abd": 0,   "pinky_mcp": 75,  "pinky_pip": 75,
    },

    "open": {k: 0.0 for k in ALL_JOINT_NAMES},
}


# ─── 主类 ─────────────────────────────────────────────────────────────────────

class HandController:
    """
    16 DOF 灵巧手高层控制接口。

    用法:
        hand = HandController("/dev/tty.usbserial-0001")
        hand.connect()
        hand.set_neutral()
        hand.set_all({"index_mcp": 60, "index_pip": 50})
        hand.set_pose("fist")
        status = hand.read_all()
        hand.disconnect()
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 1_000_000,
        default_speed: int = 300,
        echo: bool = False,
    ):
        self.bus           = HLSBus(port, baudrate, echo=echo)
        self.default_speed = default_speed
        # 关节当前目标角度缓存（°）
        self._targets: Dict[str, float] = {k: 0.0 for k in ALL_JOINT_NAMES}
        # 在线/离线关节集合（connect 后填充）
        self._online:  set = set()   # 已检测到的关节名
        self._offline: set = set()   # ping 无响应的关节名
        # 零位校准：{关节名: home_raw_position}，0° 对应该位置（缺省 2048 中点）
        self._home: Dict[str, int] = load_offsets()
        if self._home:
            print(f"[Hand] 已加载零位校准（{len(self._home)} 个关节）")

    def reload_calibration(self):
        """重新读取零位校准文件（reset_zero 后无需重启即可生效）。"""
        self._home = load_offsets()

    def _home_pos(self, joint_name: str) -> int:
        """该关节 0° 对应的舵机原始位置（有校准用校准，否则用出厂中点）。"""
        return self._home.get(joint_name, CENTER_POS)

    # ── 在线检测 ──────────────────────────────────────────────────────────────

    def _detect_online(self):
        """
        逐一 ping 所有舵机，将结果分入 _online / _offline。
        打印彩色汇总表，方便调试阶段快速确认哪些舵机已接入。
        """
        id_to_joint = {cfg.servo_id: name for name, cfg in JOINT_CONFIG.items()}
        self._online.clear()
        self._offline.clear()

        print("\n[Hand] 检测舵机在线状态 ...")
        print(f"  {'关节':<20} {'ID':>4}  状态")
        print("  " + "─" * 38)

        for name, cfg in JOINT_CONFIG.items():
            ok = self.bus.ping(cfg.servo_id)
            if ok:
                self._online.add(name)
                mark = "✓ 在线"
            else:
                self._offline.add(name)
                mark = "✗ 缺失"
            print(f"  {name:<20} #{cfg.servo_id:>2}   {mark}")

        n_on  = len(self._online)
        n_off = len(self._offline)
        print("  " + "─" * 38)
        print(
            f"  在线: {n_on}/{n_on + n_off} 个舵机"
            + (f"，缺失: {sorted(self._offline)}" if self._offline else "，全部就绪")
        )
        print()

    # ── 连接 ──────────────────────────────────────────────────────────────────

    def connect(self, enable_torque: bool = True):
        """连接总线，自动检测在线舵机，仅对在线舵机开力矩。"""
        self.bus.connect()
        self._detect_online()

        if not self._online:
            print("[Hand] ⚠ 没有检测到任何舵机，请检查接线和电源。")
            return

        if enable_torque:
            print("[Hand] 正在开启在线关节力矩...")
            for name in sorted(self._online):
                cfg = JOINT_CONFIG[name]
                try:
                    self.bus.enable_torque(cfg.servo_id)
                except CommError as e:
                    print(f"  ! {name} (ID {cfg.servo_id}) 力矩开启失败: {e}")
                    # 开启失败时从在线集移除，避免后续发送命令
                    self._online.discard(name)
                    self._offline.add(name)
            print(f"[Hand] ✓ 力矩已开启（{len(self._online)} 个关节）\n")

    def disconnect(self, go_neutral: bool = True):
        """断开连接，可选先回中立位（仅在线关节）。"""
        if go_neutral and self._online:
            print("[Hand] 回中立位...")
            try:
                self.set_neutral(speed=150)
                time.sleep(2.0)
            except Exception:
                pass
        # 关闭所有在线舵机的力矩
        for name in self._online:
            try:
                self.bus.disable_torque(JOINT_CONFIG[name].servo_id)
            except CommError:
                pass
        self.bus.disconnect()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ── 角度转换 ──────────────────────────────────────────────────────────────

    def _angle_to_pos(self, joint_name: str, angle_deg: float) -> int:
        """
        将关节角度 (°) 转换为舵机原始位置值。
        以「零位校准位置」为 0° 原点（未校准则用出厂中点 2048）。
        """
        cfg = JOINT_CONFIG[joint_name]
        # 限位保护
        angle_deg = max(cfg.angle_min, min(cfg.angle_max, angle_deg))
        base  = self._home_pos(joint_name)
        delta = (angle_deg - cfg.neutral) * cfg.direction * UNITS_PER_DEG
        pos   = int(round(base + delta))
        return max(0, min(4095, pos))

    def _pos_to_angle(self, joint_name: str, position: int) -> float:
        """将舵机位置值转换回关节角度 (°)，以零位校准位置为原点。"""
        cfg = JOINT_CONFIG[joint_name]
        base    = self._home_pos(joint_name)
        raw_deg = (position - base) / UNITS_PER_DEG
        return raw_deg * cfg.direction + cfg.neutral

    # ── 单关节控制 ────────────────────────────────────────────────────────────

    def set_joint(
        self,
        joint_name: str,
        angle_deg: float,
        speed: Optional[int] = None,
    ):
        """
        设置单个关节目标角度。
        若该关节对应的舵机不在线，命令被静默丢弃（目标值仍会缓存）。

        Args:
            joint_name : 关节名，见 JOINT_CONFIG
            angle_deg  : 目标角度 (°)
            speed      : 运动速度 (0-4095)，None 则使用默认速度
        """
        if joint_name not in JOINT_CONFIG:
            raise ValueError(f"未知关节: '{joint_name}'。可用: {ALL_JOINT_NAMES}")

        # 缓存目标值（无论是否在线）
        self._targets[joint_name] = angle_deg

        if joint_name not in self._online:
            return   # 舵机缺失，静默跳过

        cfg = JOINT_CONFIG[joint_name]
        spd = speed if speed is not None else self.default_speed
        pos = self._angle_to_pos(joint_name, angle_deg)
        self.bus.set_position(cfg.servo_id, pos, speed=spd)

    def get_joint(self, joint_name: str) -> float:
        """读取单个关节当前角度 (°)。"""
        if joint_name not in JOINT_CONFIG:
            raise ValueError(f"未知关节: '{joint_name}'")
        cfg = JOINT_CONFIG[joint_name]
        pos = self.bus.get_position(cfg.servo_id)
        return self._pos_to_angle(joint_name, pos)

    # ── 多关节同步控制 ────────────────────────────────────────────────────────

    def set_all(
        self,
        angles: Dict[str, float],
        speed: Optional[int] = None,
    ):
        """
        同步设置多个关节（单包 SYNC_WRITE，在线舵机同时启动）。
        缺失的舵机自动跳过，不影响其余关节运动。
        未指定的关节保持当前目标。

        Args:
            angles : {joint_name: angle_deg, ...}
            speed  : 全部关节使用同一速度；None = 使用各关节默认速度
        """
        commands  = []
        skipped   = []

        for joint_name, angle_deg in angles.items():
            if joint_name not in JOINT_CONFIG:
                print(f"  [警告] 未知关节 '{joint_name}'，跳过")
                continue

            # 无论是否在线都缓存目标值
            self._targets[joint_name] = angle_deg

            if joint_name not in self._online:
                skipped.append(joint_name)
                continue

            cfg = JOINT_CONFIG[joint_name]
            spd = speed if speed is not None else cfg.speed
            pos = self._angle_to_pos(joint_name, angle_deg)
            commands.append((cfg.servo_id, pos, spd))

        if commands:
            self.bus.sync_set_positions(commands)

    def set_neutral(self, speed: Optional[int] = None):
        """所有关节回中立位 (0°)。"""
        self.set_all(
            {k: cfg.neutral for k, cfg in JOINT_CONFIG.items()},
            speed=speed if speed is not None else self.default_speed,
        )

    def set_pose(self, pose_name: str, speed: Optional[int] = None):
        """
        执行预设手型。

        Args:
            pose_name : "neutral" / "fist" / "pinch" / "open"
        """
        if pose_name not in PRESET_POSES:
            raise ValueError(
                f"未知预设: '{pose_name}'。可用: {list(PRESET_POSES.keys())}"
            )
        self.set_all(PRESET_POSES[pose_name], speed=speed)

    # ── 运行时重新检测 ────────────────────────────────────────────────────────

    def redetect(self, enable_torque: bool = True):
        """
        重新 ping 所有舵机并更新在线状态。
        用于热插拔场景：接上新舵机后无需重启程序，直接调用此方法即可。
        """
        # 先关掉当前已知在线舵机的力矩（安全起见）
        for name in list(self._online):
            try:
                self.bus.disable_torque(JOINT_CONFIG[name].servo_id)
            except CommError:
                pass

        self._detect_online()

        if enable_torque:
            for name in self._online:
                try:
                    self.bus.enable_torque(JOINT_CONFIG[name].servo_id)
                except CommError as e:
                    print(f"  ! {name} 力矩开启失败: {e}")
                    self._online.discard(name)
                    self._offline.add(name)

    @property
    def online_joints(self) -> List[str]:
        """返回当前在线关节名列表（按 JOINT_CONFIG 顺序）。"""
        return [n for n in ALL_JOINT_NAMES if n in self._online]

    @property
    def offline_joints(self) -> List[str]:
        """返回当前离线关节名列表。"""
        return [n for n in ALL_JOINT_NAMES if n in self._offline]

    # ── MediaPipe → 手部关节角度映射 ─────────────────────────────────────────

    def from_mediapipe(self, mp_angles: Dict[str, float], speed: Optional[int] = None):
        """
        将 MediaPipe 输出的角度字典映射到本手的关节空间。
        mp_angles 的 key 格式与 orca_control.py 中相同（如 "index_mcp"）。

        MediaPipe 角度 (°) → 本手角度 (°) 直接透传（范围由限位保护）。
        如果坐标系不同，在此处添加换算。
        """
        # 字段映射：MediaPipe 名 → 本手名（名称完全一致时可直接透传）
        mapped = {}
        for mp_key, angle in mp_angles.items():
            if mp_key in JOINT_CONFIG:
                mapped[mp_key] = angle
        self.set_all(mapped, speed=speed)

    # ── 状态读取 ──────────────────────────────────────────────────────────────

    def read_all(self) -> Dict[str, dict]:
        """
        读取所有关节状态。
        在线关节：实时从总线读取位置、电压、温度。
        离线关节：返回缓存的目标值，并标记 online=False。

        返回格式:
            {joint_name: {"angle_deg": float, "online": bool, ...}, ...}
        """
        status: Dict[str, dict] = {}

        # ── 在线关节：从总线读 ──
        online_ids = [JOINT_CONFIG[n].servo_id for n in self._online]
        if online_ids:
            raw = self.bus.read_all_status(online_ids)
            id_to_joint = {cfg.servo_id: name for name, cfg in JOINT_CONFIG.items()}
            for sid, info in raw.items():
                name = id_to_joint.get(sid, f"servo_{sid}")
                if info.get("ok"):
                    angle = self._pos_to_angle(name, info["position"])
                    status[name] = {
                        "angle_deg": round(angle, 2),
                        "position":  info["position"],
                        "voltage_V": info["voltage_V"],
                        "temp_C":    info["temp_C"],
                        "online":    True,
                        "ok":        True,
                    }
                else:
                    status[name] = {
                        "angle_deg": self._targets.get(name, 0.0),
                        "online":    True,
                        "ok":        False,
                        "error":     info.get("error"),
                    }

        # ── 离线关节：用目标缓存补全 ──
        for name in self._offline:
            status[name] = {
                "angle_deg": self._targets.get(name, 0.0),
                "online":    False,
                "ok":        False,
                "error":     "舵机未连接",
            }

        # 按 JOINT_CONFIG 顺序返回
        return {name: status[name] for name in ALL_JOINT_NAMES if name in status}

    def print_status(self):
        """打印所有关节状态（调试用），离线关节以不同样式标出。"""
        status = self.read_all()
        n_on  = len(self._online)
        n_all = len(ALL_JOINT_NAMES)

        print(f"\n  灵巧手状态  [{n_on}/{n_all} 个舵机在线]")
        print(f"  {'关节':<20} {'ID':>4}  {'角度(°)':>8}  {'位置':>6}  {'电压':>6}  {'温度':>5}  状态")
        print("  " + "─" * 70)

        for name, info in status.items():
            cfg = JOINT_CONFIG[name]
            if info.get("online") and info.get("ok"):
                print(
                    f"  {name:<20} #{cfg.servo_id:>2}  "
                    f"{info['angle_deg']:>8.1f}  "
                    f"{info['position']:>6d}  "
                    f"{info['voltage_V']:>5.1f}V  "
                    f"{info['temp_C']:>4d}°C  ✓"
                )
            elif info.get("online"):
                print(
                    f"  {name:<20} #{cfg.servo_id:>2}  "
                    f"{'---':>8}  {'---':>6}  {'---':>6}  {'---':>5}  "
                    f"✗ {info.get('error', '')}"
                )
            else:
                # 离线关节：显示缓存目标值，颜色暗示缺失
                print(
                    f"  {name:<20} #{cfg.servo_id:>2}  "
                    f"{info['angle_deg']:>7.1f}° "
                    f"{'':>6}  {'---':>6}  {'---':>5}  "
                    f"— 缺失"
                )
        print()


# ─── 直接运行：交互式测试 ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="灵巧手控制器测试")
    p.add_argument("--port",   default="/dev/tty.usbserial-0001")
    p.add_argument("--baud",   type=int, default=1_000_000)
    p.add_argument("--pose",   default=None,
                   help="执行预设手型: neutral / fist / pinch / open")
    p.add_argument("--joint",  default=None, help="关节名")
    p.add_argument("--angle",  type=float, default=None, help="目标角度 (°)")
    p.add_argument("--speed",  type=int,   default=300,  help="运动速度")
    p.add_argument("--status", action="store_true",      help="打印当前状态")
    p.add_argument("--echo",   action="store_true")
    args = p.parse_args()

    hand = HandController(args.port, args.baud, echo=args.echo)
    hand.connect()

    try:
        if args.pose:
            print(f"执行预设手型: {args.pose}")
            hand.set_pose(args.pose, speed=args.speed)
            time.sleep(2)

        if args.joint and args.angle is not None:
            print(f"  {args.joint} → {args.angle}°")
            hand.set_joint(args.joint, args.angle, speed=args.speed)
            time.sleep(2)

        if args.status:
            hand.print_status()

        if not (args.pose or args.joint or args.status):
            # 默认演示: 握拳 → 松开
            print("演示: 中立 → 握拳 → 松开")
            hand.set_neutral(speed=200);  time.sleep(2)
            hand.set_pose("fist", speed=200); time.sleep(2)
            hand.set_neutral(speed=200);  time.sleep(2)

    except KeyboardInterrupt:
        print("\n中断")
    finally:
        hand.disconnect(go_neutral=False)
