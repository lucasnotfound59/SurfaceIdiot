"""
emg/emg_source.py — 肌电手环数据源接口
========================================

本文件定义【肌电（EMG）数据源】的统一接口，是真实手环对接的边界。

⭐ 真实手环对接，只需三选一：
   方式 A：继承 EMGSource，实现 connect / read_channels / disconnect
   方式 B：用 CallbackEMGSource 包装你已有的取数函数
   方式 C：先用 MockEMGSource（模拟数据）把整条链路跑通

然后把数据源实例交给 emg/emg_control.py，控制链路其余部分无需改动。

━━━ 接口约定 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
read_channels() 返回长度为 16 的 list[float]，
每个元素是该通道的【归一化激活度】，范围 0.0(放松) ~ 1.0(最大收缩)。
通道顺序对应的关节见 emg/emg_mapping.py 的 EMG_CHANNEL_MAP。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import math
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

NUM_CHANNELS = 16   # 肌电通道数（与灵巧手 16 DOF 一一对应）


# ─── 抽象接口 ─────────────────────────────────────────────────────────────────

class EMGSource(ABC):
    """肌电数据源抽象基类。真实手环继承本类即可接入。"""

    @abstractmethod
    def connect(self) -> None:
        """打开手环连接（串口 / 蓝牙 / SDK 初始化等）。"""
        ...

    @abstractmethod
    def read_channels(self) -> List[float]:
        """
        返回当前 16 路通道激活度，list[float]，每个 0.0-1.0。
        必须返回恰好 NUM_CHANNELS 个值。
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """关闭手环连接。"""
        ...

    @property
    def num_channels(self) -> int:
        return NUM_CHANNELS

    # 上下文管理：with EMGSource() as src: ...
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()


# ─── 包装已有取数函数 ─────────────────────────────────────────────────────────

class CallbackEMGSource(EMGSource):
    """
    把一个【现成的取数函数】包装成 EMGSource。

    适合你已经有读手环数据的代码：
        def my_read() -> list[float]:   # 返回 16 个 0-1 的值
            ...
        source = CallbackEMGSource(read_fn=my_read,
                                   connect_fn=my_open, disconnect_fn=my_close)
    """

    def __init__(
        self,
        read_fn: Callable[[], List[float]],
        connect_fn: Optional[Callable[[], None]] = None,
        disconnect_fn: Optional[Callable[[], None]] = None,
    ):
        self._read       = read_fn
        self._connect    = connect_fn
        self._disconnect = disconnect_fn

    def connect(self) -> None:
        if self._connect:
            self._connect()

    def read_channels(self) -> List[float]:
        vals = list(self._read())
        if len(vals) != NUM_CHANNELS:
            raise ValueError(
                f"EMG 通道数应为 {NUM_CHANNELS}，实际收到 {len(vals)}"
            )
        return [float(v) for v in vals]

    def disconnect(self) -> None:
        if self._disconnect:
            self._disconnect()


# ─── 模拟数据源（无硬件测试用）────────────────────────────────────────────────

class MockEMGSource(EMGSource):
    """
    模拟 16 路肌电数据，用于无硬件时跑通整条链路。
    每个通道是相位错开的正弦波，0-1 之间缓慢起伏，
    看起来像手指依次轻微收缩-放松。
    """

    def __init__(self, period_s: float = 4.0, step: float = 0.05):
        self.period = period_s
        self.step   = step     # 每次 read 推进的虚拟时间
        self._t     = 0.0

    def connect(self) -> None:
        print(f"[EMG] MockEMGSource 启动（模拟 {NUM_CHANNELS} 通道，周期 {self.period}s）")

    def read_channels(self) -> List[float]:
        self._t += self.step
        out = []
        for ch in range(NUM_CHANNELS):
            phase = 2 * math.pi * (ch / NUM_CHANNELS)
            v = 0.5 + 0.5 * math.sin(2 * math.pi * self._t / self.period + phase)
            out.append(v)
        return out

    def disconnect(self) -> None:
        print("[EMG] MockEMGSource 停止")
