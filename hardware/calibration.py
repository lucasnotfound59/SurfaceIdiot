"""
hardware/calibration.py — 零位校准（home offset）读写
======================================================

保存/读取每个关节的「零位」原始位置：
即手摆成「全部伸直、左右居中」时，各舵机的当前 raw position。

控制器（HandController）启动时自动读取这里的偏移，把该姿态当作 0°。
校准数据存在 hardware/zero_offsets.json，跨程序重启保留。

用 tools/reset_zero.py 来生成/更新这个文件。
"""

import json
import os
from typing import Dict, Optional

_CALIB_PATH = os.path.join(os.path.dirname(__file__), "zero_offsets.json")


def calib_path() -> str:
    """校准文件的绝对路径。"""
    return _CALIB_PATH


def load_offsets() -> Dict[str, int]:
    """
    读取零位偏移 {关节名: home_raw_position}。
    文件不存在或损坏时返回空字典（控制器会退回出厂中点 2048）。
    """
    if not os.path.exists(_CALIB_PATH):
        return {}
    try:
        with open(_CALIB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: int(v) for k, v in data.get("home", {}).items()}
    except Exception as e:
        print(f"[校准] 读取 {_CALIB_PATH} 失败: {e}")
        return {}


def save_offsets(offsets: Dict[str, int], meta: Optional[dict] = None):
    """
    写入零位偏移。
    Args:
        offsets : {关节名: home_raw_position}
        meta    : 可选元信息（如时间、舵机ID对照），仅供人看
    """
    data = {"home": {k: int(v) for k, v in offsets.items()}}
    if meta:
        data["meta"] = meta
    with open(_CALIB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def clear_offsets() -> bool:
    """删除校准文件，恢复出厂中点(2048)为 0°。返回是否删除了文件。"""
    if os.path.exists(_CALIB_PATH):
        os.remove(_CALIB_PATH)
        return True
    return False
