import cv2
import torch
import numpy as np
import time
import ctypes
import math
import pyautogui
import serial
import multiprocessing
from multiprocessing import Process, Queue
import os

#to_twos_complement_hex
def to_hex(value, bit_width):
    """
    将一个整数转换为其十六进制补码形式。
    :param value: 要转换的整数
    :param bit_width: 补码的位宽
    :return: 十六进制补码形式的字符串
    """
    # 计算最大值和最小值
    max_val = (1 << bit_width) - 1
    min_val = -(1 << (bit_width - 1))

    # 检查值是否在允许的范围内
    if not (min_val <= value <= max_val):
        raise ValueError(f"Value out of range for {bit_width}-bit: {value}")

    # 对于负数，计算补码
    if value < 0:
        value = (1 << bit_width) + value

    return (value & max_val)

def worker_SerialFlow(Queue_coordinate):
    """线程工作函数"""
    print(f"worker_SerialFlow:已启动, PID: {os.getpid()}")
    # 串口配置
    serialPort = 'COM5'  # 请根据实际情况修改串口号
    baudRate = 57600  # 波特率
    ser = serial.Serial(serialPort, baudRate, timeout=0.5)

    # 获取屏幕尺寸
    screen_width, screen_height = pyautogui.size()
    # error_Accumulation_X = 0
    # error_Accumulation_Y = 0
    # x_target = 0    # 鼠标距离目标的X轴像素量
    # y_target = 0    # 鼠标距离目标的Y轴像素量
    width_X = 50  # 目标区域X轴范围
    width_Y = 60  # 目标区域Y轴范围
    MAX_MOVE_VALUE = 125  # 最大移动像素点
    Min_Speed = 40
    while True:
        if not Queue_coordinate.empty():
            x_target,y_target= Queue_coordinate.get()
            start_time = time.time()  # 记录开始时间
            while Queue_coordinate.empty():
                end_time = time.time()  # 记录结束时间
                duration = (end_time - start_time) * 1000  # 计算持续时间并转换为毫秒
                if duration > 50:
                    print(f"识别超时 {duration} 毫秒")
                    break
                # 获取鼠标当前位置
                x_mouse, y_mouse = pyautogui.position()
                # print(f"当前鼠标位置: ({x_mouse}, {y_mouse})")
                # print(f"目标位置:{x_target, y_target}")

                if (((x_target - x_mouse) > width_X) or ((x_target - x_mouse) < (-width_X))
                    or ((y_target - y_mouse) > width_Y)) or ((y_target - y_mouse) < (-width_Y)):

                    P_x = (((x_target - x_mouse) / (screen_width)) * (MAX_MOVE_VALUE - Min_Speed))
                    P_y = (((y_target - y_mouse) / (screen_height)) * (MAX_MOVE_VALUE - Min_Speed))

                    if (x_target - x_mouse) < (-Min_Speed):
                        P_x = P_x - Min_Speed
                    elif (x_target - x_mouse) > (Min_Speed):
                        P_x = P_x + Min_Speed

                    if (y_target - y_mouse) < (-Min_Speed):
                        P_y = P_y - Min_Speed
                    elif (y_target - y_mouse) > (Min_Speed):
                        P_y = P_y + Min_Speed

                    distance_X = P_x
                    distance_Y = P_y

                    ser.write(bytearray([0x90, 0x00, to_hex((int)(distance_X), 8), to_hex((int)(distance_Y), 8), 0x00]))
                    # set_cursor_pos(*closest_box)
                    time.sleep(0.015)


# 定义SetCursorPos函数
def set_cursor_pos(x, y):
    ctypes.windll.user32.SetCursorPos(x, y)

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def worker_TargetDetection(Queue_frame,
                            Queue_return):

    Queue_frame_internal = Queue()
    Queue_return_internal = Queue()

    # 加载YOLOv5模型
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'./best.pt')  # 替换为你的.pt模型文件路径
    cv2.namedWindow('YOLOv5 Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLOv5 Detection', 800, 600)
    while True:
        # time.sleep(0.001)
        # if not Queue_frame.empty():
        frame  = Queue_frame.get()
        # Queue_frame_internal.put(frame)
        # 将接收到的字节串还原为图像
        # frame = cv2.imdecode(np.frombuffer(frame_encoded, np.uint8), cv2.IMREAD_COLOR)
        results = model(frame)
        # while not Queue_return.empty():
        #     continue
        Queue_return.put(results)
        print(results)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':

    Queue_coordinate = Queue()
    Queue_frame = Queue()
    Queue_return = Queue()

    processes = []

    p_SerialFlow = multiprocessing.Process(target=worker_SerialFlow, args=(Queue_coordinate,))
    processes.append(p_SerialFlow)
    p_SerialFlow.start()

    # p_TargetDetection = multiprocessing.Process(target=worker_TargetDetection, args=(Queue_frame,Queue_return,))
    # processes.append(p_TargetDetection)
    # p_TargetDetection.start()

    # 加载YOLOv5模型
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'./best.pt')  # 替换为你的.pt模型文件路径

    # 创建一个名为'YOLOv5 Detection'的窗口，并设置大小
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', 800, 600)

    count_i = 0
    while True:
        # 全屏截图
        try:
            screen_img = pyautogui.screenshot()
            frame = np.array(screen_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"捕获屏幕内容时出错: {e}")
            continue

        # 应用对象检测
        try:
            results = model(frame)
            print(results)
            # for detection in results:
                # 打印每个检测的属性，以便理解其结构
                # print(detection)
            # 设置置信度阈值
            # confidence_threshold = 0.8
            # detections = results.xywh[0]
            # 筛选置信度大于阈值的检测结果
            # filtered_detections = detections[detections[:, 4] > confidence_threshold]
            # results = filtered_detections
        # Queue_frame.put(frame)
        # while True:
            # if not Queue_return.empty():
                # results =Queue_return.get()
                # print(results)
                # break

        except Exception as e:
            print(f"应用对象检测时出错: {e}")
            continue

        # 处理检测结果
        try:
            # 获取鼠标当前位置
            current_mouse_pos = pyautogui.position()

            # 计算每个方框中心点与鼠标当前位置的距离，选择最近的方框
            closest_box = None
            closest_distance = float('inf')

            # print(" results.xyxy[0]:", results.xyxy[0])
            detections = results.xyxy[0]
            # 设置置信度阈值
            confidence_threshold = 0.9
            # 筛选置信度高于阈值的检测结果
            filtered_detections = detections[detections[:, 4] > confidence_threshold]

            # for *xyxy, conf, cls in results.xyxy[0]:
            for *xyxy, conf, cls in filtered_detections:
                # 获取方框中心坐标
                x_center = int((xyxy[0] + xyxy[2]) / 2)
                y_center = int((xyxy[1] + xyxy[3]) / 2)

                # 计算欧氏距离
                distance = euclidean_distance(current_mouse_pos, (x_center, y_center))

                # 更新最近的方框信息
                if distance < closest_distance:
                    closest_distance = distance
                    closest_box = (x_center, y_center)

            # 将鼠标移动到最近的方框中心
            # if closest_box:

                Queue_coordinate.put(closest_box)

                # x_target,y_target = closest_box

            if count_i >= 0:
                count_i = 0
                cv2.imshow('Detection', np.squeeze(results.render()))
            else:
                count_i += 1

        except Exception as e:
            print(f"处理检测结果时出错: {e}")
            continue

        # 按'q'退出循环
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

