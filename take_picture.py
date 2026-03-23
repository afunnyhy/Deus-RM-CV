import argparse
import os
import sys
import onnxruntime as ort
import cv2
import torch
import numpy as np
import time
import math
from collections import deque
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

WINDOW_SECONDS = 8
DISPLAY_FPS = 20
buffers = {
    "ch1": deque(),
    "ch2": deque(),
    "ch3": deque(),
}
buffer_lock = threading.Lock()


# ========================
# 示波器线程
# ========================
def oscilloscope():
    plt.ion()
    fig, ax = plt.subplots()
    while True:
        now = time.time()
        t_min = now - WINDOW_SECONDS

        with buffer_lock:
            for buf in buffers.values():
                while buf and buf[0][0] < t_min:
                    buf.popleft()

            ch1 = list(buffers["ch1"])
            ch2 = list(buffers["ch2"])
            ch3 = list(buffers["ch3"])

        ax.clear()

        if ch1:
            t1, v1 = zip(*ch1)
            x1 = [t - now for t in t1]
            ax.plot(x1, v1, label="receive")

        if ch2:
            t2, v2 = zip(*ch2)
            x2 = [t - now for t in t2]
            ax.plot(x2, v2, label="send")

        if ch3:
            t3, v3 = zip(*ch3)
            x3 = [t - now for t in t3]
            ax.plot(x3, v3, label="old_diff")

        ax.set_xlim(-WINDOW_SECONDS, 0)
        ax.set_ylim(-25, 0)
        ax.set_title("Dual-Channel Oscilloscope")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)

        plt.pause(1.0 / DISPLAY_FPS)


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
    # #初始化预测类
    tra = Tracker()

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
    t0 = time.time()
    while True:
        # 读取视频流的一帧
        ret, orig_frame = camera.get_photo()
        if camera_flip:
            orig_frame = cv2.flip(orig_frame, -1)
        if not ret:
            continue

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
                ret_pnp, armor, out_img = pnp_solver.get_armor_target(detected_armor, out_img, vision.pitch, vision.yaw)
                if ret_pnp:  # 如果PnP解算成功, 将装甲板中心点添加到结果列表
                    detected_point.append(armor)

        if len(detected_point) > 0:
            # 选择最佳目标
            armor = target_selector.select_best_target(detected_point)
            if armor is None:
                continue
            # 标记显示识别到的装甲板
            if used_predict:
                found_pos2d = camera2xy(gimbal2camera(rotate_around_y(armor.gimbal_pos, -vision.yaw), vision.pitch))
            else:
                found_pos2d = camera2xy(gimbal2camera(armor.gimbal_pos, vision.pitch))
            cv2.circle(out_img, found_pos2d, 11, (0, 200, 200), 4)
            ax, ay, az = armor.gimbal_pos

            is_find = True
            # print(armor)
            t_n = time.time()

            # 初始化跟踪器

            if tra.state == TracState.LOST:
                tra.initial(armor)
                t = t_n
                continue
            # ///////////////////////////
            predicted_pos2d = camera2xy(gimbal2camera(armor.gimbal_pos, vision.pitch))
            cv2.circle(out_img, predicted_pos2d, 14, (174, 29, 128), 4)
            # ///////////////////
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
            last_vision_yaw = vision.yaw

        # 处理跟踪状态下的目标
        if tra.state == TracState.TRACKING or tra.state == TracState.TEMP_LOST:
            angle_pitch = vision.pitch
            if used_predict:
                re_transform_pos = rotate_around_y(predict_armor.gimbal_pos, -vision.yaw)
                predict_armor.gimbal_pos = re_transform_pos
            else:
                predict_armor = armor
            # 用运动云台坐标系计算弹道
            change_angle = ballistic_compensation(predict_armor.gimbal_pos)
            ax, ay, az = predict_armor.gimbal_pos

            cv2.putText(out_img,
                        f"predicted x:{ax:<9.3f} y:{ay:<9.3f} z:{az:<9.3f} yaw:{predicted_armor_yaw * 180.0 / math.pi:<9.3f}",
                        (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

            angle_yoz = - (change_angle - angle_pitch)
            angle_yoz = angle_pitch + angle_yoz

            # 补丁
            diff = - (change_angle - angle_pitch)
            angle_yoz = angle_yoz - diff + 0.25 * diff

            if az < 0.1:  # 距离过近
                continue
            angle_xoz = math.atan(ax / az) + vision.yaw
            while angle_xoz < -math.pi:
                angle_xoz += 2 * math.pi
            while angle_xoz > math.pi:
                angle_xoz -= 2 * math.pi
            if tra.state == TracState.TEMP_LOST:
                angle_xoz = angle_xoz - (vision.yaw - last_vision_yaw)
            if str(angle_xoz) == "nan":
                continue
            lock = 0
            if max(abs(angle_xoz - vision.yaw), abs(angle_yoz)) > 1.0 * math.pi / 180:
                vision.set_data(angle_xoz, angle_yoz, math.sqrt(az * az + ax * ax), 1, 0)
                lock = 0
            else:
                vision.set_data(angle_xoz, angle_yoz, math.sqrt(az * az + ax * ax), 1, 1)
                lock = 1
            now = time.time()
            dt = now - t0
            v1 = vision.pitch * 180 / math.pi
            v2 = angle_yoz * 180 / math.pi
            v3 = - (change_angle - angle_pitch) * 180 / math.pi
            with buffer_lock:
                buffers["ch1"].append((now, v1))
                buffers["ch2"].append((now, v2))
                buffers["ch3"].append((now, v3))
            # 标记显示预测后的装甲板
            # predicted_pos2d = camera2xy(gimbal2camera(armor.gimbal_pos, vision.pitch))
            # cv2.circle(out_img, predicted_pos2d, 14, (174, 29, 128), 4)

            # print(f"yaw旋转到{angle_xoz * 180 / math.pi}°,pitch旋转{angle_yoz * 180 / math.pi}°")
            # vision.send()
        else:
            vision.set_data(vision.yaw, 0, 0, 0, 0)
        # cv2.putText(out_img,
        #             f"received pitch:{(vision.pitch * 180 / math.pi) if vision.pitch is not None else 0:<9.3f} yaw:{(vision.yaw * 180 / math.pi) if vision.yaw is not None else 0:<9.3f} ",
        #             (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        if save_video_time > 0:
            video_writer.write(out_img)

        # 显示图像和FPS计算
        if is_show_video:
            # 缩放用于显示的图像（out_img 带有调试信息）
            small_img = cv2.resize(out_img, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
            cv2.imshow("vision output", small_img)

            # --- 新增：干净的拍照功能 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # 按下 's' 键拍照
                # 使用时间戳命名
                photo_time = time.strftime("%Y%m%d_%H%M%S")
                photo_name = f"raw_capture_{photo_time}.jpg"

                # 【关键点】：这里保存的是 orig_frame 而不是 out_img
                # 如果你在循环开始处对 orig_frame 做了 flip，这里保存的就是翻转后但无文字的图
                cv2.imwrite(photo_name, orig_frame)
                print(f"--- 原始照片（无参数）已保存: {photo_name} ---")

            elif key == ord('q'):
                break
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
    osc_thread = threading.Thread(target=oscilloscope)
    t1.start()
    t2.start()
    osc_thread.start()
    # run()
