import argparse
import os
import sys
import onnxruntime as ort
import cv2
import torch
import numpy as np
import time
import math
import subprocess
import threading
import serial
import struct
from threading import Thread
import matplotlib.pyplot as plt
import UART
from setting import *
from all_function import *
from all_type import *
from pre_armor import Tracker  # 跟踪器类
from detect_armor import ArmorDetector  # 模型推理类
from get_armor_points_cv import armor_getter  # 初始化装甲板检测类
from UART import VisionData_t  # 通信类
from camera_get_photo import InitCamera  # 相机类
from light_detector import LightDetector  # 导入灯条解算类
from armor_chose import TargetSelector  # 导入目标选择类
from pnp_solver import PnPSolver  # 导入PnP解算类
from guardRobot import GuardRobot

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
vision = VisionData_t(PORT, BPS, TIMEOUT)

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


def run():
    # 初始化相机类
    print("Camera type:", cameraType, "    ID:", cameraID)
    camera = InitCamera(cameraType)
    print(cameraID, "init success.")
    if used_yolo:
        # 初始化模型推断类
        print("model:", model_path + model_name, "   use_cuda:", CUDA)
        armor_de = ArmorDetector(model_path, model_name, CUDA, friend_color)  # 我方颜色
        print("armor detector init success.")
        print("Troop type:", my_TroopType, "   Friend color:", friend_color)
        print("Is show video:", is_show_video, "   Save video times:", save_video_time)
    else:
        # 初始化CV类
        armor_de = armor_getter(friend_color)
    # 初始化灯条解算类
    light_pos = LightDetector()
    # 初始化PnP解算类
    pnp_solver = PnPSolver()
    # 初始化目标选择类
    target_selector = TargetSelector()
    # 初始化机器人预测类
    robot = GuardRobot()

    t = time.time()  # 初始化时间
    time1 = time.time()
    cnt = 0
    last_vision_yaw = 0

    if save_video_time > 0:
        output_file = time.strftime("%Y%m%d_%H%M%S") + "_output.mp4"  # 导出文件名为时间
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器（MP4格式）
        fps = 30  # 帧率
        ret, orig_frame = camera.get_photo()
        frame_size = (orig_frame.shape[1], orig_frame.shape[0])  # 视频帧大小（宽度, 高度）
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    start_time = time.time()
    armor = None
    predict_armor = None
    predicted_armor_yaw = 0

    print("Start working...")
    while True:
        # 读取视频流的一帧
        ret, orig_frame = camera.get_photo()
        if camera_flip:
            orig_frame = cv2.flip(orig_frame, -1)
        if not ret:
            continue

        detected_point = []  # 初始化装甲板中心点结果列表
        robot.armor_plates_camera_positions.clear() # 清空上一帧数据

        if used_yolo:
            all_detect_armor, out_img = armor_de.detect_armor(orig_frame)
        else:
            ret, all_detect_armor, out_img = armor_de.get_armors_by_img(orig_frame)

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
                # 注意：pnp_solver.get_armor_target 返回 4 个值
                ret_pnp, armor, out_img, object_points_cam = pnp_solver.get_armor_target(detected_armor, out_img, vision.pitch, vision.yaw)
                if ret_pnp:  # 如果PnP解算成功, 将装甲板中心点添加到结果列表
                    detected_point.append(armor)

                    # 添加到GuardRobot的数据中 (需要检查数据完整性)
                    if object_points_cam is not None and len(object_points_cam) >= 4:
                        valid_armor = True
                        for corner in object_points_cam:
                            if len(corner) < 3:
                                valid_armor = False
                                break
                        if valid_armor:
                            robot.armor_plates_camera_positions.append(object_points_cam)

        cv2.putText(out_img,
                    f"receive yaw:{vision.yaw * 180 / math.pi:<9.3f} pitch:{vision.pitch * 180 / math.pi:<9.3f} ",
                    (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

        # 预测与瞄准逻辑
        predict_armor_pos = None
        predicted_armor_yaw = 0

        # 使用GuardRobot进行预测
        if len(robot.armor_plates_camera_positions) > 0:
             try:
                robot.use_robot_prediction()
                is_find = True
             except Exception as e:
                print(f"Prediction error: {e}")

        # 绘制机器人和装甲板 (移植自main_video.py)
        img_height, img_width = out_img.shape[:2]

        # 1. 绘制机器人中心
        if robot.center is not None:
             robot_center_pixel = camera2xy(robot.center)
             if 0 <= robot_center_pixel[0] < img_width and 0 <= robot_center_pixel[1] < img_height:
                 cv2.circle(out_img, (int(robot_center_pixel[0]), int(robot_center_pixel[1])), 12, (0, 0, 255), -1)
                 cv2.putText(out_img, "CENTER", (int(robot_center_pixel[0])+15, int(robot_center_pixel[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 2. 绘制检测到的装甲板
        for i, armor_corners in enumerate(robot.armor_plates_camera_positions):
            if i >= len(robot.armor_plates_camera_positions): break
            center_3d = np.mean(armor_corners, axis=0)
            center_pixel = camera2xy(center_3d)
            if 0 <= center_pixel[0] < img_width and 0 <= center_pixel[1] < img_height:
                cv2.circle(out_img, (int(center_pixel[0]), int(center_pixel[1])), 8, (255, 0, 0), -1)

        # 3. 绘制预测的装甲板并选择最佳攻击目标
        best_target_pos = None
        min_distance = float('inf')

        if hasattr(robot, 'armor_center_point') and robot.armor_center_point:
             armor_centers = robot.armor_center_point[0] if isinstance(robot.armor_center_point[0], list) else robot.armor_center_point
             for i, center_3d in enumerate(armor_centers):
                 if isinstance(center_3d, (list, np.ndarray)) and len(center_3d) >= 3:
                     center_3d = np.array(center_3d, dtype=np.float32)
                     center_pixel = camera2xy(center_3d)

                     # 绘制预测点
                     if 0 <= center_pixel[0] < img_width and 0 <= center_pixel[1] < img_height:
                         cv2.circle(out_img, (int(center_pixel[0]), int(center_pixel[1])), 10, (0, 255, 0), -1)
                         cv2.putText(out_img, f"Pred {i}", (int(center_pixel[0]), int(center_pixel[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                     # 选择策略: 优先选择视野中心 (或最近的)
                     # 这里简单选择距离相机Z轴最近的 (或者距离图像中心最近的)
                     # 距离图像中心距离:
                     dist_to_center = (center_pixel[0] - img_width/2)**2 + (center_pixel[1] - img_height/2)**2
                     if dist_to_center < min_distance:
                         min_distance = dist_to_center
                         best_target_pos = center_3d

        # 射击逻辑
        if best_target_pos is not None:
            # best_target_pos 是相机坐标系下的 3D 点
            # 转换为云台坐标系
            gimbal_pos = camera2gimbal(best_target_pos, vision.pitch)

            # 补偿 (ballistic_compensation 需要云台坐标)
            # 注意: main_cam 原逻辑中使用了 predict_armor.gimbal_pos

            # 计算弹道
            ax, ay, az = gimbal_pos
            change_angle = ballistic_compensation(gimbal_pos)

            cv2.putText(out_img,
                        f"target x:{ax:<9.3f} y:{ay:<9.3f} z:{az:<9.3f}",
                        (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

            angle_pitch = vision.pitch
            angle_yoz = -(change_angle - angle_pitch)

            if az > 0.1: # 距离判断
                angle_xoz = math.atan(ax / az) + vision.yaw
                while angle_xoz < -math.pi: angle_xoz += 2 * math.pi
                while angle_xoz > math.pi: angle_xoz -= 2 * math.pi

                # 简单发送逻辑，不再依赖 Tracker 的 state
                fire = 1 if abs(angle_xoz - vision.yaw) < 0.1 else 0 # 简单设定
                if angle_xoz > 0.1 or angle_yoz > 0.1:
                    vision.set_data(angle_xoz, angle_yoz, math.sqrt(az * az + ax * ax), 1, 0) # 这里的1, 0 是 fire_check 和 detection_status?
                else:
                    vision.set_data(angle_xoz, angle_yoz, math.sqrt(az * az + ax * ax), 1, 1)

                cv2.putText(out_img,
                            f"sending yaw:{angle_xoz * 180 / math.pi:<9.3f} pitch:{angle_yoz * 180 / math.pi:<9.3f}",
                            (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        else:
             vision.set_data(vision.yaw, 0, 0, 0, 0)

        cv2.putText(out_img, f"detected:{len(robot.armor_plates_camera_positions)}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)

        if save_video_time > 0:
            video_writer.write(out_img)

        # 显示图像和FPS计算
        if is_show_video:
            cv2.imshow("vision output", out_img)
            cv2.waitKey(1)
        cnt += 1
        if cnt == 20:
            fps = 20 / (time.time() - time1)
            time1 = time.time()
            cnt = 0
            print("fps", fps)

        if 0 < save_video_time < time.time() - start_time:
            video_writer.release()
            cv2.destroyAllWindows()
            camera.delete()
            print("video write to", output_file, "over")
            break


if __name__ == "__main__":
    t1 = threading.Thread(target=vision.start)
    t2 = threading.Thread(target=run)
    t1.start()
    t2.start()
    # run()