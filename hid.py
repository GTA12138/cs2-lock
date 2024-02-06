# USB HID鼠标报告的格式是4字节的16进制值。在前面添加90作为头部。
# 例如：90 01 0A F6 00  代表按下鼠标左键，同时右移10像素，下移-10像素
# 第一位：头部；
# 第二位：鼠标按键的状态，00代表不按下任何键，01代表按下左键，02代表按下右键
# 第三、四位：X轴和Y轴的移动量，二进制补码形式
# 此处有补码计算器：https://www.23bei.com/tool/56.html

import serial
import time

# 串口配置
serialPort = 'COM5'  # 请根据实际情况修改串口号
baudRate = 57600  # 波特率
ser = serial.Serial(serialPort, baudRate, timeout=0.5)

try:
    ser.write(bytearray([0x90, 0x00, 0x10, 0x00, 0x00]))
    ser.write(bytearray([0x90, 0x00, 0x00, 0x00, 0x00]))

    ser.write(bytearray([0x90, 0x00, 0x00, 0x10, 0x00]))
    ser.write(bytearray([0x90, 0x00, 0x00, 0x00, 0x00]))

    ser.close()

except Exception as e:
    print(e)