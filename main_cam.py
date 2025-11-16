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

# CUDA 环境与 main_video 对齐：自动探测 GPU，可在无 GPU 环境下回退 CPU
CUDA = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
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

# 如需 3D 可视化，可复用 main_video 的方式；此处保持原逻辑
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
    """相机在线推理主流程：与 main_video 统一思路，但输入来自实时相机。"""
    # 1) 初始化相机
    print("Camera type:", cameraType, "    ID:", cameraID)
    camera = InitCamera(cameraType)
    print(cameraID, "init success.")

    # 2) 初始化检测器（与 main_video 统一：优先 YOLO）
    if used_yolo:
        print("model:", model_path + model_name, "   use_cuda:", CUDA)
        armor_de = ArmorDetector(model_path, model_name, CUDA, friend_color)
        print("armor detector init success.")
        print("Troop type:", my_TroopType, "   Friend color:", friend_color)
        print("Is show video:", is_show_video, "   Save video times:", save_video_time)
    else:
        armor_de = armor_getter(friend_color)

    light_pos = LightDetector()
    pnp_solver = PnPSolver()

    # 3) 四个角点 3D KalmanFilter（平移常速度），与 main_video 的角点 KF 思路一致
    corner_kfs = [None] * 4
    corner_kf_inited = [False] * 4
    corner_kf_init_cov = 1e2
    corner_kf_measure_noise = 0.1
    corner_kf_process_noise = 0.5

    # 录制视频（在线情况下可选）
    if save_video_time > 0:
        output_file = time.strftime("%Y%m%d_%H%M%S") + "_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        ret, orig_frame = camera.get_photo()
        if not ret:
            print("Error: camera get first frame failed")
            return
        frame_size = (orig_frame.shape[1], orig_frame.shape[0])
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
    else:
        video_writer = None

    last_time = time.time()
    start_time = time.time()

    print("Start working...")
    while True:
        # 读取一帧
        ret, orig_frame = camera.get_photo()
        if not ret:
            continue
        if camera_flip:
            orig_frame = cv2.flip(orig_frame, -1)

        out_img = orig_frame.copy()

        # dt 供 KF 使用（与 main_video 一致裁剪）
        now = time.time()
        dt = float(np.clip(now - last_time, 1e-3, 0.2))
        last_time = now

        # 1) 检测
        if used_yolo:
            all_detect_armor, out_img = armor_de.detect_armor(orig_frame)
        else:
            ret_cv, all_detect_armor, out_img = armor_de.get_armors_by_img(orig_frame)

        # 2) 灯条提点 + PnP，选面积最大装甲
        corner_meas_cam = None
        candidates = []
        for detected_armor_box in all_detect_armor:
            if used_yolo:
                ret_detected, detected_armor, out_img = light_pos.extract_light_points(orig_frame, detected_armor_box, out_img)
            else:
                ret_detected = True
                detected_armor = detected_armor_box
            if not ret_detected:
                continue

            # 中心点 PnP，使用云台当前姿态 vision.pitch/vision.yaw
            ret_pnp, armor_candidate, out_img = pnp_solver.get_armor_target(
                detected_armor, out_img, vision.pitch, vision.yaw
            )
            if not ret_pnp or armor_candidate is None:
                continue

            # 角点 PnP：获得 4 角点在相机坐标系下的 3D
            ret_pnp2, rvec2, tvec2, obj_pts_cam = pnp_solver.solve_pnp(detected_armor)
            if not ret_pnp2 or obj_pts_cam is None:
                continue

            candidates.append((armor_candidate, obj_pts_cam))

        if candidates:
            best_armor, best_obj_pts_cam = max(
                candidates,
                key=lambda item: getattr(item[0], 'area', 0.0)
            )
            corner_meas_cam = best_obj_pts_cam

        # 3) 角点 KF + 原始/滤波后矩形 + 射击预测（完全沿用 main_video 流程）
        if corner_meas_cam is not None:
            # 顺序约定：LightDetector 输出为 [top_left, bottom_left, top_right, bottom_right]
            h, w = out_img.shape[:2]
            raw_pixels = []
            for p in corner_meas_cam:
                u, v = camera2xy(p)
                u = int(max(0, min(w - 1, u)))
                v = int(max(0, min(h - 1, v)))
                raw_pixels.append((u, v))

            # 原始矩形（蓝色）
            tl, bl, tr, br = raw_pixels
            raw_rect = np.array([tl, bl, br, tr], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out_img, [raw_rect], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.line(out_img, tl, br, (255, 0, 0), 1)
            cv2.line(out_img, bl, tr, (255, 0, 0), 1)

            # 角点量测：相机 -> 云台
            corner_meas_gimbal = [camera2gimbal(p, 0) for p in corner_meas_cam]

            filtered_gimbal = []
            filtered_vel = []
            filtered_pixels = []

            for idx in range(4):
                px, py, pz = map(float, corner_meas_gimbal[idx])

                if not corner_kf_inited[idx]:
                    kf_point = KF(
                        init_cov=corner_kf_init_cov,
                        measure_noise=corner_kf_measure_noise,
                        process_noise=corner_kf_process_noise,
                        x=px, y=py, z=pz,
                        vx=0.0, vy=0.0, vz=0.0,
                    )
                    # 在线场景使用一个名义 FPS 初始化
                    kf_point.init_kf(dt=1.0 / 30.0)
                    corner_kfs[idx] = kf_point
                    corner_kf_inited[idx] = True

                kf_point = corner_kfs[idx]
                kf_point.build_F_Q(dt)
                kf_point.predict_next(dt)
                kf_point.correct_by_sensor([px, py, pz])

                state_post, _P = kf_point.get_state()
                pos_post = state_post[:3].reshape(-1)
                vel_post = state_post[3:].reshape(-1)
                filtered_gimbal.append(pos_post)
                filtered_vel.append(vel_post)

                u_f, v_f = camera2xy(gimbal2camera(pos_post, 0))
                u_f = int(max(0, min(w - 1, u_f)))
                v_f = int(max(0, min(h - 1, v_f)))
                filtered_pixels.append((u_f, v_f))

            # 滤波后矩形（绿色）
            tl_f, bl_f, tr_f, br_f = filtered_pixels
            filt_rect = np.array([tl_f, bl_f, br_f, tr_f], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out_img, [filt_rect], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.line(out_img, tl_f, br_f, (0, 255, 0), 1)
            cv2.line(out_img, bl_f, tr_f, (0, 255, 0), 1)

            # 计算四边形的边长与角度（与 main_video 保持一致显示逻辑）
            points = [tl_f, bl_f, br_f, tr_f]
            edges = []
            for i in range(4):
                p1 = np.array(points[i])
                p2 = np.array(points[(i + 1) % 4])
                length = np.linalg.norm(p1 - p2)
                edges.append(length)

            angles = []
            for i in range(4):
                prev_point = np.array(points[(i - 1) % 4])
                curr_point = np.array(points[i])
                next_point = np.array(points[(i + 1) % 4])
                vec1 = prev_point - curr_point
                vec2 = next_point - curr_point
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)

            for i in range(4):
                p1 = np.array(points[i])
                p2 = np.array(points[(i + 1) % 4])
                midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.putText(out_img, f"{edges[i]:.1f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            for i in range(4):
                point = points[i]
                text_pos = (point[0] + 5, point[1] + 5)
                cv2.putText(out_img, f"{angles[i]:.1f}°", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # 由角点 KF 求中心位置与速度，进行射击预测
            center_pos = np.mean(np.vstack(filtered_gimbal), axis=0)
            center_vel = np.mean(np.vstack(filtered_vel), axis=0)

            bullet_speed = defaults_bullet_speed if 'defaults_bullet_speed' in globals() else 25.0
            distance = float(np.linalg.norm(center_pos))
            bullet_time = distance / bullet_speed if bullet_speed > 1e-6 else 0.0
            aim_pos = center_pos + center_vel * bullet_time

            aim_proj = camera2xy(gimbal2camera(aim_pos, 0))
            cv2.circle(out_img, aim_proj, 11, (0, 0, 255), 2)

        # 写出与显示（在线场景写文件可选）
        if video_writer is not None:
            video_writer.write(out_img)

        if is_show_video:
            cv2.imshow("vision output", out_img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # 录制时长到达自动退出
        if 0 < save_video_time < time.time() - start_time:
            break

    # 清理资源
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    camera.delete()


if __name__ == "__main__":
    t1 = threading.Thread(target=vision.start)
    t2 = threading.Thread(target=run)
    t1.start()
    t2.start()
