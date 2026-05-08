# find_port.py - 先跑这个确认连接没问题
from vassar_feetech_servo_sdk import find_servo_port

ports = find_servo_port(return_all=True)
print(f"找到的串口: {ports}")