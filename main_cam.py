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
# from pre_armor import Tracker  # 跟踪器类（已弃用，使用 KalmanFilter）
from detect_armor import ArmorDetector  # 模型推理类
from get_armor_points_cv import armor_getter  # 初始化装甲板检测类
from UART import VisionData_t  # 通信类
from camera_get_photo import InitCamera  # 相机类
from light_detector import LightDetector  # 导入灯条解算类
# from armor_chose import TargetSelector  # 导入目标选择类（按 main_video 逻辑不再使用）
from pnp_solver import PnPSolver  # 导入PnP解算类
from KalmanFilter import KalmanFilter as KF  # 新增：常速度卡尔曼滤波器

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
        # 初始化模型推断类（YOLO 强制）
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
    # 按 main_video 逻辑：不再使用目标选择器/Tracker
    # target_selector = TargetSelector()
    # tra = Tracker()

    # ========== KalmanFilter 初始化 ==========
    # 调整卡尔曼滤波器参数以更好地适应帧率变化
    # 降低初始协方差，增加过程噪声，使预测更依赖测量值
    kf = KF(init_cov=1e2, measure_noise=0.15, process_noise=0.8)
    # 使用名义 FPS（或后续动态 dt）初始化
    kf.init_kf(dt=1.0 / 30.0)
    kf_inited = False
    last_time = time.time()
    # =======================================

    # 添加 dt 平滑处理相关变量
    dt_history = []  # 存储最近几次的 dt 值用于平滑处理
    dt_history_maxlen = 2  # 进一步减少保留的 dt 值数量，提高响应速度

    # 添加帧率监控
    fps_history = []
    fps_history_maxlen = 10

    t = time.time()  # 历史保留
    time1 = time.time()
    cnt = 0
    last_vision_yaw = 0

    if save_video_time > 0:
        output_file = time.strftime("%Y%m%d_%H%M%S") + "_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        ret, orig_frame = camera.get_photo()
        frame_size = (orig_frame.shape[1], orig_frame.shape[0])
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    start_time = time.time()

    print("Start working...")
    while True:
        # 读取视频流的一帧
        ret, orig_frame = camera.get_photo()
        if camera_flip:
            orig_frame = cv2.flip(orig_frame, -1)
        if not ret:
            continue

        # 计算 dt（用于 KF）
        now = time.time()
        raw_dt = now - last_time
        last_time = now

        # 对 dt 进行限制和平滑处理
        clipped_dt = float(np.clip(raw_dt, 1e-3, 0.3))  # 稍微放宽上限到0.3秒

        # 更新 dt 历史记录
        dt_history.append(clipped_dt)
        if len(dt_history) > dt_history_maxlen:
            dt_history.pop(0)

        # 使用改进的方法计算 dt
        if len(dt_history) > 1:
            # 如果当前 dt 明显大于历史平均值，更相信当前值（目标可能开始快速移动）
            avg_dt = sum(dt_history) / len(dt_history)
            if clipped_dt > avg_dt * 1.3:  # 降低阈值使响应更快
                dt = clipped_dt
            else:
                dt = avg_dt * 0.7 + clipped_dt * 0.3  # 加权平均，增加当前值权重
        else:
            dt = clipped_dt

        # 根据帧率动态调整卡尔曼滤波器参数
        kf.adjust_for_frame_rate(dt)

        detected_point = []  # 不再使用旧逻辑
        # 1) 检测
        if used_yolo:
            all_detect_armor, out_img = armor_de.detect_armor(orig_frame)
        else:
            ret_cv, all_detect_armor, out_img = armor_de.get_armors_by_img(orig_frame)

        # 2) 灯条提点 + PnP，取面积最大的装甲板
        meas = None  # z = [x,y,z]
        candidates = []
        for detected_armor_box in all_detect_armor:
            if used_yolo:
                ret_detected, detected_armor, out_img = light_pos.extract_light_points(orig_frame, detected_armor_box,
                                                                                       out_img)
            else:
                ret_detected = True
                detected_armor = detected_armor_box
            if not ret_detected:
                continue
            ret_pnp, armor_candidate, out_img = pnp_solver.get_armor_target(detected_armor, out_img, vision.pitch,
                                                                            vision.yaw)
            if ret_pnp and armor_candidate and getattr(armor_candidate, 'gimbal_pos', None) is not None:
                candidates.append(armor_candidate)
        if candidates:
            best = max(candidates, key=lambda a: getattr(a, 'area', 0.0))
            meas = best.gimbal_pos

        # 3) 常速度 KF + 射击点（仅红圈）
        if meas is not None:
            gx, gy, gz = float(meas[0]), float(meas[1]), float(meas[2])
            if not kf_inited:
                kf.reset_state(x=gx, y=gy, z=gz, vx=0.0, vy=0.0, vz=0.0, init_cov=1e2)
                kf_inited = True
            else:
                # 先验
                _ = kf.predict_next(dt)
                # 融合当前量测（后验）
                kf.correct_by_sensor([gx, gy, gz])
                post_state, _P = kf.get_state()
                posterior_pos = post_state[:3].reshape(-1)
                posterior_vel = post_state[3:].reshape(-1)
                # 速度大小与方向（水平/俯仰角）
                speed_mag = float(np.linalg.norm(posterior_vel))
                horiz_angle_deg = math.degrees(math.atan2(posterior_vel[0], posterior_vel[2] + 1e-9))  # vx vs depth
                pitch_angle_deg = math.degrees(
                    math.atan2(posterior_vel[1], math.sqrt(posterior_vel[0] ** 2 + posterior_vel[2] ** 2) + 1e-9))
                # 子弹飞行时间（距离/初速度）+ 匀速直线前馈
                bullet_speed = defaults_bullet_speed if 'defaults_bullet_speed' in globals() else 25.0
                distance = float(np.linalg.norm(posterior_pos))
                bullet_time = distance / bullet_speed if bullet_speed > 1e-6 else 0.0

                # 改进的预测算法：根据帧率动态调整预测时间
                # 当帧率较低时，增加预测时间以补偿较大的时间间隔
                current_fps = 1.0 / max(dt, 1e-6)
                if current_fps < 20:  # 当帧率低于20FPS时
                    # 增加预测时间以补偿低帧率
                    prediction_factor = min(2.0, 20.0 / max(current_fps, 1e-6))
                    aim_pos = posterior_pos + posterior_vel * bullet_time * prediction_factor
                else:
                    aim_pos = posterior_pos + posterior_vel * bullet_time

                # 仅绘制射击点（红色圆圈）
                aim_proj = camera2xy(gimbal2camera(aim_pos, 0))
                cv2.circle(out_img, aim_proj, 11, (0, 0, 255), 2)
                # 速度箭头（基于当前位置，延伸 0.05s 的运动预测）
                base_proj = camera2xy(gimbal2camera(posterior_pos, 0))
                arrow_tip_pos = posterior_pos + posterior_vel * 0.05
                arrow_tip_proj = camera2xy(gimbal2camera(arrow_tip_pos, 0))
                cv2.arrowedLine(out_img, base_proj, arrow_tip_proj, (255, 255, 0), 2, tipLength=0.3)
                # 叠加文字信息：速度、方向、当前位置与射击点
                cv2.putText(out_img,
                            f"SPD:{speed_mag:.2f}m/s YawV:{horiz_angle_deg:.1f}deg PitchV:{pitch_angle_deg:.1f}deg",
                            (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                cv2.putText(out_img, f"TGT x:{posterior_pos[0]:.2f} y:{posterior_pos[1]:.2f} z:{posterior_pos[2]:.2f}",
                            (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
                cv2.putText(out_img, f"AIM x:{aim_pos[0]:.2f} y:{aim_pos[1]:.2f} z:{aim_pos[2]:.2f}", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                # 控制台实时打印（可按需限频）
                print(
                    f"speed={speed_mag:.2f} m/s, yawV={horiz_angle_deg:.1f} deg, pitchV={pitch_angle_deg:.1f} deg, pos=({posterior_pos[0]:.2f},{posterior_pos[1]:.2f},{posterior_pos[2]:.2f}), aim=({aim_pos[0]:.2f},{aim_pos[1]:.2f},{aim_pos[2]:.2f})")
        # 无量测：按题设通常不会发生；此处不显示提示

        # 写视频（可选）
        if save_video_time > 0:
            video_writer.write(out_img)

        # 显示图像
        if is_show_video:
            cv2.imshow("vision output", out_img)
            cv2.waitKey(1)

        # FPS 统计
        cnt += 1
        if cnt == 20:
            current_fps = 20 / (time.time() - time1)
            fps_history.append(current_fps)
            if len(fps_history) > fps_history_maxlen:
                fps_history.pop(0)

            time1 = time.time()
            cnt = 0
            print("fps", current_fps)

            # 如果帧率过低，可以考虑调整一些参数
            if len(fps_history) >= 5:
                avg_fps = sum(fps_history) / len(fps_history)
                if avg_fps < 10:  # 降低阈值到10FPS
                    print(f"Warning: Low average FPS ({avg_fps:.1f}), prediction accuracy may be affected")

        # 录制时长到达自动退出
        if 0 < save_video_time < time.time() - start_time:
            if save_video_time > 0:
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
