"""
hardware/joint_map.py — 舵机 ↔ 自由度 映射表（唯一可编辑来源）
================================================================

⭐ 这是整个项目里【唯一】需要修改舵机-关节对应关系的地方。
   改这里，hand_controller / servo_test / full_test / hls_control 全部自动生效。

每个关节用一个 JointConfig 描述：
    servo_id  : 舵机 ID（总线地址，1-16）
    angle_min : 最小允许角度 (°)
    angle_max : 最大允许角度 (°)
    direction : +1 或 -1 —— 装好手指后若该关节方向反了，改这里
    neutral   : 中立位偏移 (°) —— 装好后 0° 不是理想初始位置时，在这里补偿
    speed     : 默认运动速度（0=最快，越大越慢）

────────────────────────────────────────────────────────────────
当前配置：Orca Hand v1，16 DOF，无手腕
    大拇指 (4 DOF) : thumb_abd, thumb_mcp, thumb_pip, thumb_dip   → ID  1-4
    食指   (3 DOF) : index_abd, index_mcp, index_pip              → ID  5-7
    中指   (3 DOF) : middle_abd, middle_mcp, middle_pip           → ID  8-10
    无名指 (3 DOF) : ring_abd, ring_mcp, ring_pip                 → ID 11-13
    小指   (3 DOF) : pinky_abd, pinky_mcp, pinky_pip              → ID 14-16
────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass
from typing import Dict, List


# ─── 关节描述 ─────────────────────────────────────────────────────────────────

@dataclass
class JointConfig:
    """单个关节的完整描述。"""
    servo_id:  int          # 舵机 ID（总线地址）
    angle_min: float        # 最小允许角度 (°)
    angle_max: float        # 最大允许角度 (°)
    direction: int   = 1    # +1 或 -1（根据装配后实际方向）
    neutral:   float = 0.0  # 中立位偏移 (°)
    speed:     int   = 300  # 默认运动速度（0=最大，越大越慢）


# ─── 16 DOF 映射表 ────────────────────────────────────────────────────────────
# ⚠️ 修改舵机-关节对应，只动这张表即可。
#
# 🔧 当前硬件状态（2026-06）：只有 8 个舵机可用，其余坏掉。
#    坏掉的关节使用「虚空映射」—— servo_id 设为 DUMMY_ID_BASE+ 的占位值，
#    这些 ID 在总线上 ping 不到，程序会自动判为离线并跳过，绝不发指令。
#    等舵机修好/换新后，把对应行的 servo_id 改回真实 ID 即可。
#
#    可用舵机 ↔ 关节（来自实测接线）：
#       ID 12 → thumb_abd   (大拇指左右)
#       ID 9  → thumb_mcp   (大拇指旋转，基部，暂定 mcp)
#       ID 11 → thumb_dip   (大拇指远端屈伸)
#       ID 4  → middle_abd  (中指左右)
#       ID 3  → middle_mcp  (中指近端屈伸)
#       ID 5  → middle_pip  (中指远端屈伸)
#       ID 13 → pinky_abd   (小指左右)
#       ID 14 → pinky_mcp   (小指近端屈伸)

DUMMY_ID_BASE = 90   # 虚空映射占位 ID 起点；>= 此值表示该关节当前无可用舵机

JOINT_CONFIG: Dict[str, JointConfig] = {

    # ── 大拇指 (4 DOF) ───────────────────────────────────────────────────────
    "thumb_abd": JointConfig(servo_id=12, angle_min=-30, angle_max=60,  direction=1),  # 大拇指左右
    "thumb_mcp": JointConfig(servo_id=9,  angle_min=0,   angle_max=90,  direction=1),  # 大拇指旋转(基部)
    "thumb_pip": JointConfig(servo_id=90, angle_min=0,   angle_max=90,  direction=1),  # 坏：虚空映射
    "thumb_dip": JointConfig(servo_id=11, angle_min=0,   angle_max=70,  direction=1),  # 大拇指远端屈伸

    # ── 食指 (3 DOF)  全部坏 ─────────────────────────────────────────────────
    "index_abd": JointConfig(servo_id=91, angle_min=-20, angle_max=20,  direction=1),  # 坏：虚空映射
    "index_mcp": JointConfig(servo_id=92, angle_min=0,   angle_max=90,  direction=1),  # 坏：虚空映射
    "index_pip": JointConfig(servo_id=93, angle_min=0,   angle_max=90,  direction=1),  # 坏：虚空映射

    # ── 中指 (3 DOF)  全部可用 ───────────────────────────────────────────────
    "middle_abd": JointConfig(servo_id=4,  angle_min=-20, angle_max=20, direction=1),  # 中指左右
    "middle_mcp": JointConfig(servo_id=3,  angle_min=0,   angle_max=90, direction=1),  # 中指近端屈伸
    "middle_pip": JointConfig(servo_id=5,  angle_min=0,   angle_max=90, direction=1),  # 中指远端屈伸

    # ── 无名指 (3 DOF)  全部坏 ───────────────────────────────────────────────
    "ring_abd": JointConfig(servo_id=94, angle_min=-20, angle_max=20,   direction=1),  # 坏：虚空映射
    "ring_mcp": JointConfig(servo_id=95, angle_min=0,   angle_max=90,   direction=1),  # 坏：虚空映射
    "ring_pip": JointConfig(servo_id=96, angle_min=0,   angle_max=90,   direction=1),  # 坏：虚空映射

    # ── 小指 (3 DOF)  2/3 可用 ───────────────────────────────────────────────
    "pinky_abd": JointConfig(servo_id=13, angle_min=-20, angle_max=20,  direction=1),  # 小指左右
    "pinky_mcp": JointConfig(servo_id=14, angle_min=0,   angle_max=90,  direction=1),  # 小指近端屈伸
    "pinky_pip": JointConfig(servo_id=97, angle_min=0,   angle_max=90,  direction=1),  # 坏：虚空映射
}


def is_live(joint_name: str) -> bool:
    """该关节是否有真实可用舵机（servo_id < DUMMY_ID_BASE）。"""
    return JOINT_CONFIG[joint_name].servo_id < DUMMY_ID_BASE


LIVE_JOINT_NAMES: List[str] = [n for n in JOINT_CONFIG if is_live(n)]


# ─── 手指分组（整体测试 / 逐指动作用）────────────────────────────────────────
# 每根手指包含的关节名，顺序：abduction 在前，flex（mcp→pip→dip）在后。

FINGERS: Dict[str, List[str]] = {
    "thumb":  ["thumb_abd",  "thumb_mcp",  "thumb_pip", "thumb_dip"],
    "index":  ["index_abd",  "index_mcp",  "index_pip"],
    "middle": ["middle_abd", "middle_mcp", "middle_pip"],
    "ring":   ["ring_abd",   "ring_mcp",   "ring_pip"],
    "pinky":  ["pinky_abd",  "pinky_mcp",  "pinky_pip"],
}

# 每根手指的“屈伸”关节（不含 abduction），整体测试时让它们轻微弯曲。
FLEX_JOINTS: Dict[str, List[str]] = {
    finger: [j for j in joints if not j.endswith("_abd")]
    for finger, joints in FINGERS.items()
}


# ─── 派生查找表（自动生成，勿手动改）─────────────────────────────────────────

ALL_JOINT_NAMES: List[str]      = list(JOINT_CONFIG.keys())
ALL_SERVO_IDS:   List[int]      = [cfg.servo_id for cfg in JOINT_CONFIG.values()]
ID_TO_JOINT:     Dict[int, str] = {cfg.servo_id: name for name, cfg in JOINT_CONFIG.items()}
JOINT_TO_ID:     Dict[str, int] = {name: cfg.servo_id for name, cfg in JOINT_CONFIG.items()}


# ─── 直接运行：打印当前映射表 ────────────────────────────────────────────────

if __name__ == "__main__":
    print("当前舵机 ↔ 自由度 映射表")
    print("=" * 60)
    print(f"  {'ID':>4}  {'关节名':<14} {'范围(°)':>12}  {'方向':>4}  状态")
    print("  " + "─" * 54)
    # 先按关节顺序（拇→食→中→无名→小）展示，更直观
    for name, cfg in JOINT_CONFIG.items():
        rng = f"[{cfg.angle_min:>4g}, {cfg.angle_max:>4g}]"
        if is_live(name):
            status = f"✓ 可用 (ID {cfg.servo_id})"
            id_str = f"{cfg.servo_id:>4}"
        else:
            status = "✗ 坏/虚空映射"
            id_str = "  --"
        print(f"  {id_str}  {name:<14} {rng:>12}  {cfg.direction:>+4d}  {status}")
    print("  " + "─" * 54)
    print(f"  共 {len(JOINT_CONFIG)} 个自由度，可用 {len(LIVE_JOINT_NAMES)} 个，"
          f"虚空 {len(JOINT_CONFIG) - len(LIVE_JOINT_NAMES)} 个")
    print(f"  可用关节: {LIVE_JOINT_NAMES}")
