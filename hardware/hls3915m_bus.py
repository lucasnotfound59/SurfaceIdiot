"""
SurfaceIdiot — HLS3915M TTL 总线底层驱动
=========================================

硬件规格:
    型号    : Feetech HLS3915M
    电压    : 12 V
    扭矩    : 14 kg·cm
    通信    : TTL 半双工 UART（Feetech 标准协议）
    分辨率  : 4096 steps / 360°（≈ 0.088°/step）
    接口    : 双轴输出，空心杯电机，有绝对位置反馈

接线方式:
    推荐使用 Feetech FE-URT-1 调试器（USB→TTL，自动方向控制，无 echo）
    也可使用普通 USB-UART 芯片 + 1kΩ 电阻半双工合并（此时 echo=True）

         FE-URT-1
    USB ──────────── DATA ──┬── HLS3915M #1 DATA
                            ├── HLS3915M #2 DATA  (菊花链)
                            └── ...
                     GND  ──── 共地
                    (12V 单独供电给舵机，不走 FE-URT-1)

通信帧格式 (Feetech 标准):
    发送: [0xFF, 0xFF, ID, LENGTH, INSTRUCTION, PARAMS..., CHECKSUM]
    应答: [0xFF, 0xFF, ID, LENGTH, ERROR,       DATA...,  CHECKSUM]
    校验: CHECKSUM = (~(ID + LENGTH + INSTR + sum(PARAMS))) & 0xFF

⚠️  寄存器地址基于 Feetech 标准 TTL 协议（与 SMS/STS 系列相同族）。
    如与 HLS3915M 实际数据手册有出入，只需修改本文件顶部的 REG_* 常量。

用法:
    bus = HLSBus("/dev/tty.usbserial-0001")
    bus.connect()
    online = bus.scan()
    bus.enable_torque(1)
    bus.set_position(1, position=2048, speed=300)   # 回中心位
    pos = bus.get_position(1)
    bus.disconnect()
"""

import struct
import time
from typing import Dict, List, Optional, Tuple

import serial

# ─── 协议常量 ─────────────────────────────────────────────────────────────────

HEADER       = bytes([0xFF, 0xFF])
BROADCAST_ID = 0xFE  # 广播到总线上所有舵机

INST_PING       = 0x01  # 查询在线
INST_READ       = 0x02  # 读寄存器
INST_WRITE      = 0x03  # 写寄存器（立即执行）
INST_REG_WRITE  = 0x04  # 异步写（等待 ACTION 后执行）
INST_ACTION     = 0x05  # 触发所有已暂存的 REG_WRITE
INST_SYNC_WRITE = 0x83  # 同步写多舵机（单包，最低延迟）

# ─── 寄存器地址 ───────────────────────────────────────────────────────────────
# ⚠️ 如实际手册与此不同，仅修改下方常量，驱动逻辑无需改动

REG_MODEL_L            = 0x03  # 型号低字节           (EEPROM, 只读)
REG_ID                 = 0x05  # 舵机 ID              (EEPROM, 默认=1)
REG_BAUD_RATE          = 0x06  # 波特率               (EEPROM, 4=1Mbps)
REG_RETURN_DELAY       = 0x0B  # 应答延迟时间         (EEPROM, unit=2μs)
REG_MIN_ANGLE_L        = 0x15  # 最小角度限位低字节   (EEPROM, 2 bytes)
REG_MAX_ANGLE_L        = 0x17  # 最大角度限位低字节   (EEPROM, 2 bytes)
REG_MAX_TORQUE_L       = 0x26  # 最大力矩限制低字节   (EEPROM, 2 bytes)
REG_TORQUE_ENABLE      = 0x28  # 力矩使能             (RAM, 0=关 / 1=开)
REG_GOAL_POSITION_L    = 0x2A  # 目标位置低字节       (RAM, 2 bytes, 0-4095)
REG_GOAL_SPEED_L       = 0x2C  # 目标速度低字节       (RAM, 2 bytes, 0-4095)
REG_PRESENT_POSITION_L = 0x38  # 当前位置低字节       (RAM, 2 bytes, 只读)
REG_PRESENT_SPEED_L    = 0x3A  # 当前速度低字节       (RAM, 2 bytes, 只读)
REG_PRESENT_LOAD_L     = 0x3C  # 当前负载低字节       (RAM, 2 bytes, 只读)
REG_PRESENT_VOLTAGE    = 0x3E  # 当前电压             (RAM, 1 byte, unit=0.1V)
REG_PRESENT_TEMP       = 0x3F  # 当前温度             (RAM, 1 byte, °C)

# ─── 位置 / 角度换算 ──────────────────────────────────────────────────────────

RESOLUTION    = 4096           # 0-4095 对应 0-360°
UNITS_PER_DEG = RESOLUTION / 360.0   # ≈ 11.378 units/°
CENTER_POS    = RESOLUTION // 2      # 2048 = 180°（以此为 0° 参考点）


def deg_to_pos(deg: float) -> int:
    """角度 (°) → 原始位置值，以 180°(pos=2048) 为零点。"""
    return int(round(CENTER_POS + deg * UNITS_PER_DEG))


def pos_to_deg(pos: int) -> float:
    """原始位置值 → 角度 (°)，以 2048 为零点。"""
    return (pos - CENTER_POS) / UNITS_PER_DEG


# ─── 异常类 ───────────────────────────────────────────────────────────────────

class CommError(Exception):
    """总线通信错误基类"""


class BusTimeoutError(CommError):
    """读取超时"""


class ChecksumError(CommError):
    """收到包校验和错误"""


class ServoError(CommError):
    """舵机返回错误状态字节"""

    _FLAG_NAMES = {
        0x01: "电压异常",
        0x02: "角度越限",
        0x04: "过热",
        0x08: "范围错误",
        0x10: "校验失败",
        0x20: "过载",
        0x40: "指令错误",
    }

    def __init__(self, servo_id: int, error_byte: int):
        self.servo_id   = servo_id
        self.error_byte = error_byte
        flags = [name for bit, name in self._FLAG_NAMES.items() if error_byte & bit]
        super().__init__(
            f"舵机 {servo_id} 错误: {', '.join(flags) or f'未知(0x{error_byte:02X})'}"
        )


# ─── 主类 ─────────────────────────────────────────────────────────────────────

class HLSBus:
    """
    Feetech HLS3915M TTL 总线控制器

    支持菊花链连接多达 253 个舵机（单根信号线）。
    所有 ID 1-16 的舵机在同一条总线上通过 SYNC_WRITE 同步控制。
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 1_000_000,
        timeout: float = 0.05,
        echo: bool = False,
    ):
        """
        Args:
            port     : 串口路径
                        macOS : /dev/tty.usbserial-XXXXXXXX
                        Linux : /dev/ttyUSB0
            baudrate : 波特率（必须与舵机 EEPROM 中设置一致，默认 1 Mbps）
            timeout  : 单次读取超时（秒）
            echo     : 是否有 echo（半双工 3 线接法=True，FE-URT-1=False）
        """
        self.port     = port
        self.baudrate = baudrate
        self.timeout  = timeout
        self.echo     = echo
        self._ser: Optional[serial.Serial] = None

    # ── 连接管理 ──────────────────────────────────────────────────────────────

    def connect(self):
        """打开串口并清空缓冲区。"""
        self._ser = serial.Serial(
            self.port,
            self.baudrate,
            timeout=self.timeout,
            write_timeout=1.0,
        )
        self._ser.reset_input_buffer()
        print(f"[HLSBus] ✓ 已连接 {self.port}  ({self.baudrate // 1000} kbps)")

    def disconnect(self):
        """关闭串口。"""
        if self._ser and self._ser.is_open:
            self._ser.close()
        print("[HLSBus] 已断开连接")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    def _require_connected(self):
        if self._ser is None or not self._ser.is_open:
            raise CommError("未连接，请先调用 connect()")

    # ── 底层协议 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _checksum(body: bytes) -> int:
        """Feetech 校验: ~(sum of bytes from ID to last param) & 0xFF"""
        return (~sum(body)) & 0xFF

    def _build_packet(
        self, servo_id: int, instruction: int, params: bytes
    ) -> bytes:
        length = len(params) + 2  # instruction(1) + params(n) + checksum(1)
        body   = bytes([servo_id, length, instruction]) + params
        return HEADER + body + bytes([self._checksum(body)])

    def _send(self, packet: bytes):
        self._require_connected()
        self._ser.reset_input_buffer()
        self._ser.write(packet)
        self._ser.flush()
        if self.echo:
            # 半双工 3 线接法：TX 和 RX 共线，发出的字节会被自己读到，丢弃即可
            self._ser.read(len(packet))

    def _recv_status(self) -> Tuple[int, bytes]:
        """
        读取一个舵机状态包。
        返回 (servo_id, payload_bytes)，payload 不含 error 字节和校验字节。
        """
        # 1. 找包头 0xFF 0xFF
        buf = bytearray()
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            b = self._ser.read(1)
            if not b:
                continue
            buf.append(b[0])
            if len(buf) >= 2 and buf[-2] == 0xFF and buf[-1] == 0xFF:
                break
        else:
            raise BusTimeoutError("等待包头 0xFF 0xFF 超时")

        # 2. 读 ID + LENGTH
        hdr = self._ser.read(2)
        if len(hdr) < 2:
            raise BusTimeoutError("读取 ID/LENGTH 超时")
        servo_id, length = hdr[0], hdr[1]

        # 3. 读剩余（ERROR + DATA... + CHECKSUM）
        body = self._ser.read(length)
        if len(body) < length:
            raise BusTimeoutError(
                f"数据包不完整: 期望 {length} 字节，收到 {len(body)} 字节"
            )

        # 4. 校验
        chk_data  = bytes([servo_id, length]) + body[:-1]
        expected  = self._checksum(chk_data)
        if body[-1] != expected:
            raise ChecksumError(
                f"校验和错误: 收到 0x{body[-1]:02X}, 期望 0x{expected:02X}"
            )

        # 5. 错误字节检查
        error_byte = body[0]
        if error_byte:
            raise ServoError(servo_id, error_byte)

        payload = bytes(body[1:-1])  # 去掉 error 和 checksum
        return servo_id, payload

    # ── 基础指令 ──────────────────────────────────────────────────────────────

    def ping(self, servo_id: int) -> bool:
        """检查舵机是否在线。"""
        try:
            pkt = self._build_packet(servo_id, INST_PING, b"")
            self._send(pkt)
            self._recv_status()
            return True
        except CommError:
            return False

    def read(self, servo_id: int, start_addr: int, length: int) -> bytes:
        """从指定地址读取 length 字节。"""
        params = bytes([start_addr, length])
        pkt    = self._build_packet(servo_id, INST_READ, params)
        self._send(pkt)
        _, data = self._recv_status()
        return data

    def write(self, servo_id: int, start_addr: int, data: bytes):
        """向指定地址写入字节。广播 ID 不等待应答。"""
        params = bytes([start_addr]) + data
        pkt    = self._build_packet(servo_id, INST_WRITE, params)
        self._send(pkt)
        if servo_id != BROADCAST_ID:
            self._recv_status()

    def sync_write(
        self,
        start_addr: int,
        data_len: int,
        commands: List[Tuple[int, bytes]],
    ):
        """
        同步写多舵机（单包广播，无应答，延迟最低）。

        Args:
            start_addr : 起始寄存器地址
            data_len   : 每个舵机的数据字节数
            commands   : [(servo_id, data_bytes), ...]
        """
        payload = bytes([start_addr, data_len])
        for sid, d in commands:
            if len(d) != data_len:
                raise ValueError(
                    f"舵机 {sid} 数据长度错误: 期望 {data_len}, 实际 {len(d)}"
                )
            payload += bytes([sid]) + d

        # SYNC_WRITE 使用广播 ID，LENGTH = instruction(1) + payload(n) + chk(1)
        length = len(payload) + 2
        body   = bytes([BROADCAST_ID, length, INST_SYNC_WRITE]) + payload
        pkt    = HEADER + body + bytes([self._checksum(body)])
        self._send(pkt)
        # sync_write 不返回状态包

    # ── 便捷读写 ──────────────────────────────────────────────────────────────

    def read_u8(self, servo_id: int, addr: int) -> int:
        return self.read(servo_id, addr, 1)[0]

    def read_u16(self, servo_id: int, addr: int) -> int:
        data = self.read(servo_id, addr, 2)
        return struct.unpack_from("<H", data)[0]

    def write_u8(self, servo_id: int, addr: int, value: int):
        self.write(servo_id, addr, bytes([value & 0xFF]))

    def write_u16(self, servo_id: int, addr: int, value: int):
        self.write(servo_id, addr, struct.pack("<H", value & 0xFFFF))

    # ── 力矩控制 ──────────────────────────────────────────────────────────────

    def enable_torque(self, servo_id: int):
        """开启力矩（舵机可受控运动）。"""
        self.write_u8(servo_id, REG_TORQUE_ENABLE, 1)

    def disable_torque(self, servo_id: int):
        """关闭力矩（舵机可被手动转动，调试时使用）。"""
        self.write_u8(servo_id, REG_TORQUE_ENABLE, 0)

    # ── 位置控制 ──────────────────────────────────────────────────────────────

    def set_position(
        self,
        servo_id: int,
        position: int,
        speed: int = 0,
    ):
        """
        设置单个舵机目标位置。

        Args:
            servo_id : 舵机 ID
            position : 0-4095（0°-360°，中心 2048=180°）
            speed    : 0-4095（0=最大速度，数值越大越慢）
        """
        position = max(0, min(4095, int(position)))
        speed    = max(0, min(4095, int(speed)))
        # 位置 + 速度一次性写入（4 字节，小端）
        self.write(
            servo_id,
            REG_GOAL_POSITION_L,
            struct.pack("<HH", position, speed),
        )

    def get_position(self, servo_id: int) -> int:
        """读取当前位置（0-4095）。"""
        return self.read_u16(servo_id, REG_PRESENT_POSITION_L)

    def get_voltage(self, servo_id: int) -> float:
        """读取当前电压（V）。"""
        return self.read_u8(servo_id, REG_PRESENT_VOLTAGE) * 0.1

    def get_temperature(self, servo_id: int) -> int:
        """读取当前温度（°C）。"""
        return self.read_u8(servo_id, REG_PRESENT_TEMP)

    def sync_set_positions(
        self,
        commands: List[Tuple[int, int, int]],
    ):
        """
        同步设置多个舵机位置（单包发送，所有舵机同时启动运动）。

        Args:
            commands : [(servo_id, position, speed), ...]
                       position: 0-4095, speed: 0-4095
        """
        formatted = [
            (
                sid,
                struct.pack(
                    "<HH",
                    max(0, min(4095, int(pos))),
                    max(0, min(4095, int(spd))),
                ),
            )
            for sid, pos, spd in commands
        ]
        # 每个舵机 4 字节数据（position_L, position_H, speed_L, speed_H）
        self.sync_write(REG_GOAL_POSITION_L, 4, formatted)

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    def scan(self, id_range: range = range(1, 254)) -> List[int]:
        """
        扫描总线，返回在线舵机 ID 列表。
        默认扫描 1-253；对于本项目只需 range(1, 17)。
        """
        online = []
        print(f"[HLSBus] 扫描 ID {id_range.start}-{id_range.stop - 1} ...")
        for sid in id_range:
            if self.ping(sid):
                online.append(sid)
                print(f"  ✓ 舵机 ID {sid:3d}  在线")
        print(f"[HLSBus] 共发现 {len(online)} 个舵机: {online}")
        return online

    def read_all_status(self, ids: List[int]) -> Dict[int, dict]:
        """批量读取状态（逐个轮询）。"""
        result: Dict[int, dict] = {}
        for sid in ids:
            try:
                pos  = self.get_position(sid)
                volt = self.get_voltage(sid)
                temp = self.get_temperature(sid)
                result[sid] = {
                    "position":  pos,
                    "angle_deg": round(pos_to_deg(pos), 2),
                    "voltage_V": volt,
                    "temp_C":    temp,
                    "ok":        True,
                }
            except CommError as e:
                result[sid] = {"ok": False, "error": str(e)}
        return result


# ─── 直接运行：快速诊断 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="HLS3915M 总线诊断工具")
    p.add_argument("--port",    default="/dev/tty.usbserial-0001", help="串口路径")
    p.add_argument("--baud",    type=int, default=1_000_000,       help="波特率")
    p.add_argument("--scan",    action="store_true",               help="扫描总线")
    p.add_argument("--id",      type=int, default=None,            help="指定舵机 ID")
    p.add_argument("--goto",    type=float, default=None,          help="运动到指定角度 (°)")
    p.add_argument("--speed",   type=int,   default=200,           help="运动速度 0-4095")
    p.add_argument("--echo",    action="store_true",               help="启用 echo 模式（3 线接法）")
    args = p.parse_args()

    with HLSBus(args.port, args.baud, echo=args.echo) as bus:
        if args.scan:
            bus.scan(range(1, 17))

        if args.id is not None:
            if args.goto is not None:
                pos = deg_to_pos(args.goto)
                print(f"  舵机 {args.id}: {args.goto}° → position={pos}")
                bus.enable_torque(args.id)
                bus.set_position(args.id, pos, speed=args.speed)
                time.sleep(2)
            else:
                pos  = bus.get_position(args.id)
                volt = bus.get_voltage(args.id)
                temp = bus.get_temperature(args.id)
                print(
                    f"  舵机 {args.id}: pos={pos}  "
                    f"angle={pos_to_deg(pos):.2f}°  "
                    f"voltage={volt}V  temp={temp}°C"
                )
