# set_id.py
from vassar_feetech_servo_sdk import ServoController

controller = ServoController(
    servo_ids=[1],
    servo_type="hls",
    port="/dev/cu.usbmodem5B790315271" #use portFinder.py to find the correct port
)
controller.connect()

success = controller.set_motor_id(
    current_id=1, ## current_id
    new_id=15, ## new_id
    confirm=True
)

if success:
    print("✅ ID 修改成功,断电重启舵机生效")
else:
    print("❌ 修改失败,检查连接和供电")

controller.disconnect()