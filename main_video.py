import argparse
import os
import sys
# import onnxruntime as ort
import cv2
import torch
import numpy as np
import time
import math
import subprocess
import threading
# import serial
import struct
from threading import Thread
import matplotlib.pyplot as plt
# import UART
from setting import *
from all_function import *
from all_type import *
from pre_armor import Tracker  # 跟踪器类
from detect_armor import ArmorDetector  # 模型推理类
from get_armor_points_cv import armor_getter  # 初始化装甲板检测类
from light_detector import LightDetector  # 导入灯条解算类
from armor_chose import TargetSelector  # 导入目标选择类
from pnp_solver import PnPSolver  # 导入PnP解算类

# from exceptiongroup import catch

CUDA = True
USE_OAK = False
USE_DH = True
FPS_TIME = 3
ROTATE = True

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

PORT = -1
BPS = 115200
TIMEOUT = 5

# communication
# vision = VisionData_t(PORT, BPS, TIMEOUT)

# 初始化3D绘图
if is_show_3d:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 3)
    ax.set_zlim(0, 2)
    plt.ion()
    plt.show()


def update_3d_fig(pre_amror):
    plt.cla()
    # 提取数据
    x, y, z, yaw, r = pre_amror.x, pre_amror.y, pre_amror.z, pre_amror.yaw, 0.26

    cx = x + r * np.cos(yaw)
    cy = y
    cz = z + r * np.sin(yaw)

    for i in range(3):
        angle = yaw + (i + 1) * np.pi / 2
        x_i = cx - r * np.cos(angle)
        y_i = cy
        z_i = cz - r * np.sin(angle)
        ax.scatter(x_i, z_i, y_i, c='red', s=50, label='Armor Point')

    # 绘制装甲板点和圆心
    ax.scatter(0, 0, 0, c='green', s=50, label='Car Point')
    ax.scatter(x, z, y, c='red', s=50, label='Armor Point')
    ax.scatter(cx, cz, cy, c='blue', s=50, marker='x', label='Circle Center')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 3)
    ax.set_zlim(0, 2)

    plt.pause(0.0001)
    plt.draw()


def write1(x, y, z):
    with open('data.txt', 'a') as file:
        file.write(f"{x} {y} {z}\n")


def run(video_path):
    """默认通信发送的pitch和yaw角度为0"""
    # 测试敌方颜色
    test_color = Color.RED
    if video_path.find("red") != -1:
        test_color = Color.RED
    elif video_path.find("blue") != -1:
        test_color = Color.BLUE
    if test_color == Color.RED:
        test_color = Color.BLUE
    else:
        test_color = Color.RED
    output_file = video_path[:-4] + "_output.mp4"
    cap = cv2.VideoCapture(video_path)
    ret = cap.isOpened()
    if not ret:
        print("Error: Unable to open video file:", video_path)
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器（MP4格式）
    fps = 30  # 帧率
    ret, orig_frame = cap.read()
    frame_size = (orig_frame.shape[1], orig_frame.shape[0])  # 视频帧大小（宽度, 高度）
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
    if used_yolo:
        # 初始化模型推断类
        armor_de = ArmorDetector(model_path, model_name, CUDA, test_color, ".pt")  # 我方颜色
    else:
        # 初始化CV类
        armor_de = armor_getter(test_color)
    # 初始化灯条解算类
    light_pos = LightDetector()
    # 初始化PnP解算类
    pnp_solver = PnPSolver()
    # 初始化目标选择类
    target_selector = TargetSelector()
    # #初始化预测类
    tra = Tracker()

    t = time.time()  # 初始化时间
    time1 = time.time()
    cnt = 0
    last_vision_yaw = 0
    armor = None
    predict_armor = None
    predicted_armor_yaw = 0

    print("Start processing...")
    while True:
        # 读取视频流的一帧
        ret, orig_frame = cap.read()
        # orig_frame = cv2.flip(orig_frame, -1)
        if not ret:
            video_writer.release()
            cap.release()
            cv2.destroyAllWindows()
            print("video write to", output_file, "over")
            break
        detected_point = []  # 初始化装甲板中心点结果列表
        if used_yolo:
            all_detect_armor, out_img = armor_de.detect_armor(orig_frame)
        else:
            ret, all_detect_armor, out_img = armor_de.get_armors_by_img(orig_frame)
        # print(tra.state)
        is_find = False

        for detected_armor_box in all_detect_armor:  # 遍历所有检测到的装甲板
            if used_yolo:
                # 提取灯条角点
                ret_detected, detected_armor, out_img = light_pos.extract_light_points(orig_frame, detected_armor_box,
                                                                                       out_img)
            else:
                ret_detected = True
                detected_armor = detected_armor_box
            if ret_detected:  # 如果灯条角点提取成功
                # 计算装甲板中心3D坐标
                ret_pnp, armor, out_img = pnp_solver.get_armor_target(detected_armor, out_img, 0, 0)
                if ret_pnp:  # 如果PnP解算成功, 将装甲板中心点添加到结果列表
                    detected_point.append(armor)

        if len(detected_point) > 0:
            # 选择最佳目标
            armor = target_selector.select_best_target(detected_point)
            if armor is None:
                continue
            # 标记显示识别到的装甲板
            found_pos2d = camera2xy(gimbal2camera(rotate_around_y(armor.gimbal_pos, -0), 0))
            cv2.circle(out_img, found_pos2d, 11, (0, 200, 200), 4)
            ax, ay, az = armor.gimbal_pos
            cv2.putText(out_img,
                        f"detecting x:{ax:<8.3f} y:{ay:<8.3f} z:{az:<8.3f} pos:{armor.camera_yaw * 180.0 / math.pi:<8.2f} yaw:{armor.yaw * 180.0 / math.pi:<8.2f}",
                        (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

            is_find = True
            # print(armor)
            t_n = time.time()

            # 初始化跟踪器
            if tra.state == TracState.LOST:
                tra.initial(armor)
                t = t_n
                continue

            # 更新跟踪器
            dt = t_n - t
            predict_armor, out_img = tra.update(armor, dt, out_img)
            t = t_n

            # 使用预测结果
            if predict_armor is not None:
                predicted_armor_yaw = predict_armor.yaw
                # update_3d_fig(current)
            else:
                continue
        else:
            target_selector.add_empty_entry()  # 更新历史记录

        # 处理目标丢失的情况
        if not is_find and tra.state != TracState.LOST:
            t_n = time.time()
            dt = t_n - t
            predict_armor, out_img = tra.update(None, dt, out_img)
            t = t_n

        if tra.state == TracState.TRACKING:
            last_vision_yaw = 0

        # 处理跟踪状态下的目标
        if tra.state == TracState.TRACKING or tra.state == TracState.TEMP_LOST:
            angle_pitch = 0
            # 重新将坐标转换为运动云台坐标系
            re_transform_pos = rotate_around_y(predict_armor.gimbal_pos, -0)
            predict_armor.gimbal_pos = re_transform_pos
            # 用运动云台坐标系计算弹道
            change_angle = ballistic_compensation(predict_armor.gimbal_pos)
            ax, ay, az = predict_armor.gimbal_pos
            cv2.putText(out_img,
                        f"predicted x:{ax:<9.3f} y:{ay:<9.3f} z:{az:<9.3f} yaw:{predicted_armor_yaw * 180.0 / math.pi:<9.3f}",
                        (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)
            angle_yoz = -(change_angle - angle_pitch)
            if az < 0.1:  # 距离过近
                continue
            angle_xoz = math.atan(ax / az) + 0
            if tra.state == TracState.TEMP_LOST:
                angle_xoz = angle_xoz - (0 - last_vision_yaw)
            if str(angle_xoz) == "nan":
                continue
            # if angle_xoz > 0.1 or angle_yoz > 0.1:
            #     vision.set_data(angle_xoz, angle_yoz, math.sqrt(az * az + ax * ax), 1, 0)
            # else:
            #     vision.set_data(angle_xoz, angle_yoz, math.sqrt(az * az + ax * ax), 1, 1)
            # 标记显示预测后的装甲板
            predicted_pos2d = camera2xy(gimbal2camera(predict_armor.gimbal_pos, 0))
            cv2.circle(out_img, predicted_pos2d, 14, (174, 29, 128), 4)
            cv2.putText(out_img,
                        f"sending pitch:{angle_yoz * 180 / math.pi:<9.3f}  yaw:{angle_xoz * 180 / math.pi:<9.3f}",
                        (50, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            # print(f"需要在水平方向旋转{angle_xoz * 180 / math.pi}°,需要在竖直方向旋转{angle_yoz * 180 / math.pi}°")
            # vision.send()
        else:
            # vision.set_data(vision.yaw, 0, 0, 0, 0)
            pass
        # cv2.putText(out_img,
        #             f"received pitch:{(vision.pitch * 180 / math.pi) if vision.pitch is not None else 0:<9.3f} yaw:{(vision.yaw * 180 / math.pi) if vision.yaw is not None else 0:<9.3f} ",
        #             (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        #
        cv2.putText(out_img, f"state:{tra.state}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)

        video_writer.write(out_img)
        if is_show_video:
            cv2.imshow("vision output", out_img)
            cv2.waitKey(1)

        cnt += 1
        if cnt == 20:
            fps = 20 / (time.time() - time1)
            time1 = time.time()
            cnt = 0
            print("fps", fps)


if __name__ == "__main__":
    # 根据视频文件名自动选择颜色，文件名中包含"red"或"blue"
    run(video_path=r"./test_data/0325blue.mp4")
    # run(video_path=r"./test_data/small_blue.avi")
    # run(video_path=r"./test_data/small_red.avi")
    # run(video_path=r"./test_data/big_red.avi")
    # run(video_path=r"./test_data/big_blue.avi")
    # run(video_path="./test_data/0323blue1.mp4")
    # run(video_path="./test_data/0323blue2.mp4")
    # run(video_path=r"./test_data/0325blue.mp4")
