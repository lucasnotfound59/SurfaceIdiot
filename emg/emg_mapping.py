"""
emg/emg_mapping.py — 16 路肌电通道 → 16 关节角度
================================================

把肌电手环的 16 路激活度 (0-1) 转换成灵巧手的 {关节名: 角度(°)} 字典。
相当于 MediaPipe 路径里的 landmarks_to_hls()，但输入换成肌电信号。

⭐ 修改「哪一路通道驱动哪个关节」，只动 EMG_CHANNEL_MAP。
   关节的角度范围 / 方向沿用 hardware/joint_map.py，不在这里重复定义。

激活度 → 角度 约定：
   · 屈伸关节 (angle_min = 0)   : 0.0=伸直(0°)，1.0=最大屈
   · 外展关节 (angle_min < 0)   : 0.0=最小，0.5=中立(0°)，1.0=最大
"""

from typing import Dict, List

from hardware.joint_map import JOINT_CONFIG, ALL_JOINT_NAMES
from emg.emg_source import NUM_CHANNELS

# ─── 通道 → 关节 映射表 ───────────────────────────────────────────────────────
# 第 i 个元素 = 第 i 路肌电通道驱动的关节名。
# 默认按 16 DOF 顺序排列；按你手环实际电极位置调整即可。

EMG_CHANNEL_MAP: List[str] = [
    "thumb_abd",    # ch 0
    "thumb_mcp",    # ch 1
    "thumb_pip",    # ch 2
    "thumb_dip",    # ch 3
    "index_abd",    # ch 4
    "index_mcp",    # ch 5
    "index_pip",    # ch 6
    "middle_abd",   # ch 7
    "middle_mcp",   # ch 8
    "middle_pip",   # ch 9
    "ring_abd",     # ch 10
    "ring_mcp",     # ch 11
    "ring_pip",     # ch 12
    "pinky_abd",    # ch 13
    "pinky_mcp",    # ch 14
    "pinky_pip",    # ch 15
]

# 启动时自检：映射长度与关节名必须合法
assert len(EMG_CHANNEL_MAP) == NUM_CHANNELS, \
    f"EMG_CHANNEL_MAP 应有 {NUM_CHANNELS} 项，实际 {len(EMG_CHANNEL_MAP)}"
for _j in EMG_CHANNEL_MAP:
    assert _j in JOINT_CONFIG, f"EMG_CHANNEL_MAP 含未知关节: {_j}"


# ─── 映射器 ───────────────────────────────────────────────────────────────────

class EMGMapper:
    """
    16 路激活度 → 关节角度字典，内置指数移动平均(EMA)平滑。

    用法:
        mapper = EMGMapper(alpha=0.3)
        angles = mapper.map([0.1, 0.8, ...])   # 16 个 0-1 的值
        # angles = {"thumb_abd": 12.3, "thumb_mcp": 70.1, ...}
    """

    def __init__(self, channel_map: List[str] = None, alpha: float = 0.3):
        """
        Args:
            channel_map : 通道→关节映射，默认 EMG_CHANNEL_MAP
            alpha       : EMA 平滑系数 0-1（越小越平滑、越滞后）
        """
        self.channel_map = channel_map if channel_map is not None else EMG_CHANNEL_MAP
        self.alpha       = alpha
        self._state: Dict[str, float] = {}

    @staticmethod
    def _act_to_angle(joint_name: str, act: float) -> float:
        """单通道激活度 (0-1) → 关节角度 (°)，落在该关节 ROM 内。"""
        cfg = JOINT_CONFIG[joint_name]
        act = max(0.0, min(1.0, act))
        lo, hi = cfg.angle_min, cfg.angle_max
        # 线性映射到 [lo, hi]；屈伸关节 lo=0 时 0 激活=伸直，外展关节 0.5 激活≈中立
        return lo + act * (hi - lo)

    def map(self, channels: List[float]) -> Dict[str, float]:
        """把 16 路激活度转成平滑后的关节角度字典。"""
        if len(channels) != len(self.channel_map):
            raise ValueError(
                f"通道数不匹配：映射需要 {len(self.channel_map)}，收到 {len(channels)}"
            )

        raw: Dict[str, float] = {}
        for ch, joint in enumerate(self.channel_map):
            raw[joint] = self._act_to_angle(joint, channels[ch])

        # ── EMA 平滑 ──
        if not self._state:
            self._state = dict(raw)
        else:
            a = self.alpha
            for k, v in raw.items():
                self._state[k] = a * v + (1 - a) * self._state[k]

        return dict(self._state)

    def reset(self):
        """清空平滑状态（重新开始时调用）。"""
        self._state.clear()
