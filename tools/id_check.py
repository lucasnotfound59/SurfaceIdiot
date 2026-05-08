# check_ids.py
from vassar_feetech_servo_sdk import ServoController

# 把你期望在线的 ID 全列出来
expected_ids = list(range(1, 17))  # [1, 2, 3, ..., 16]

controller = ServoController(
    servo_ids=expected_ids,
    servo_type="hls",
    port="/dev/cu.usbmodem5B790315271"
)
controller.connect()

positions = controller.read_all_positions()

print("=== 扫描结果 ===")
for sid in expected_ids:
    if sid in positions:
        print(f"✅ ID={sid:2d}  位置={positions[sid]}")
    else:
        print(f"❌ ID={sid:2d}  未响应")

print(f"\n在线: {len(positions)}/16 个")

controller.disconnect()