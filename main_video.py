"""
main_video.py — 离线视频推理与可视化入口

功能概览
- 从本地视频文件读取图像帧；
- 使用 YOLO（或传统 CV）检测装甲板大致位置；
- 灯条定位与四角点提取（LightDetector）；
- PnP 求解装甲板相对相机/云台的 3D 位姿（PnPSolver）；
- 目标选择（TargetSelector）与 EKF 跟踪预测（Tracker）；
- 弹道俯仰角补偿与可视化绘制；
- 输出实时可视化窗口，以及将结果写入新的视频文件 *_output.mp4。

输入/输出
- 输入：run(video_path: str) 指定视频路径，例如 ./test_data/0325blue.mp4；
- 输出：在项目根目录生成同名 *_output.mp4 视频；命令行打印 FPS；可选显示窗口（is_show_video）。

坐标系与单位
- 相机系：右手系，x 向右，y 向下，z 朝前；
- 云台系���项目内使用 x 向右，y 向上，z 指向目标；
- 角度：内部多为弧度，显示或日志中一般转成角度（°）。

依赖与约定
- 关键参数在 setting.py 中集中配置（内参、畸变、平移向量、是否使用 YOLO 等）；
- 如果没有 GPU，自动回退到 CPU 推理（避免 Ultralytics 对 device=0 报错）。

术语速览（给零基础读者）
- 帧（frame）：视频由很多静止图片组成，每一张就是一帧。
- 像素坐标（pixel, u/v）：图像左上角为 (0,0)，向右是 u 轴，向下是 v 轴。
- 边界框（bounding box, bbox）：一个矩形，圈出目标的大致范围，用左上角与右下角两个点表示。
- 相机内参（intrinsics, K）：描述相机“焦距和主点”的 3x3 矩阵，用来把 3D 点投影成像素点。
- PnP（Perspective-n-Point）：已知相机内参和物体上一些已知真实尺寸的 3D 点与它们在图片中的 2D 位置，
  反过来求相机与物体之间的位置和方向（位姿）。
- yaw / pitch：
  - yaw：水平转动的角度（左右转头）；
  - pitch：竖直转动的角度（抬头/低头）。
- EKF（扩展卡尔曼滤波）：在“有噪声”的观测下，对目标的状态（位置/速度/角度）做“预测+校正”，
  用于平滑与在短暂丢失时继续给出较合理的估计。
- 弹道补偿（ballistic compensation）：考虑子弹飞行时重力让它下坠，为了命中，需要把枪口稍微抬高一个角度。

"""
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
# from pre_armor import Tracker  # 旧跟踪器：已弃用
from detect_armor import ArmorDetector  # 强制使用 YOLO 检测
# from get_armor_points_cv import armor_getter  # 经典CV流程已停用
from light_detector import LightDetector
# from armor_chose import TargetSelector  # 目标选择：本测试不需要
from pnp_solver import PnPSolver
from KalmanFilter import KalmanFilter as KF  # 常速度卡尔曼滤波（仅保留这一种跟踪）

# CUDA 环境
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

# 可选 3D 显示（本测试不使用）
# if is_show_3d:
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
#     ax.set_xlim(-3, 3); ax.set_ylim(-0.5, 3); ax.set_zlim(0, 2)
#     plt.ion(); plt.show()

# def update_3d_fig(pre_amror):
#     pass  # 本测试不需要 3D 可视化


def write1(x, y, z):
    with open('data.txt', 'a') as file:
        file.write(f"{x} {y} {z}\n")


def run(video_path):
    """离线视频主流程：检测 -> 角点提取 -> PnP -> 角点 3D KF -> 中心点用于射击预测。"""
    # 颜色推断（保持原逻辑）
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

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    ret = cap.isOpened()
    if not ret:
        print("Error: Unable to open video file:", video_path)
        return

    # 初始化视频写出器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 240:
        fps = 60
    ret, orig_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        cap.release()
        return
    frame_size = (orig_frame.shape[1], orig_frame.shape[0])
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    # 检测器（强制 YOLO）
    armor_de = ArmorDetector(model_path, model_name, CUDA, test_color, ".pt")
    light_pos = LightDetector()
    pnp_solver = PnPSolver()

    # ========== 四个角点 3D KalmanFilter（仅平移常速度） ==========
    corner_kfs = [None] * 4
    corner_kf_inited = [False] * 4
    corner_kf_init_cov = 1e3
    corner_kf_measure_noise = 0.05
    corner_kf_process_noise = 0.2

    last_time = time.time()

    print("Start processing...")
    while True:
        ret, orig_frame = cap.read()
        if not ret:
            video_writer.release()
            cap.release()
            cv2.destroyAllWindows()
            print("video write to", output_file, "over")
            break

        out_img = orig_frame.copy()

        # dt 供 KF 使用
        now = time.time()
        dt = float(np.clip(now - last_time, 1e-3, 0.2))
        last_time = now

        # 1) YOLO 检测
        all_detect_armor, out_img = armor_de.detect_armor(orig_frame)

        # 2) 灯条提点 + PnP，取面积最大的装甲作为量测
        corner_meas_cam = None  # 相机坐标系下 4 角点
        candidates = []
        for detected_armor_box in all_detect_armor:
            ret_detected, detected_armor, out_img = light_pos.extract_light_points(orig_frame, detected_armor_box, out_img)
            if not ret_detected:
                continue
            # 中心点 PnP 及云台坐标
            ret_pnp, armor_candidate, out_img = pnp_solver.get_armor_target(detected_armor, out_img, 0, 0)
            if not ret_pnp or armor_candidate is None:
                continue
            # 拿 4 个角点 3D（相机坐标系）
            ret_pnp2, rvec2, tvec2, obj_pts_cam = pnp_solver.solve_pnp(detected_armor)
            if not ret_pnp2 or obj_pts_cam is None:
                continue
            candidates.append((armor_candidate, obj_pts_cam))

        if candidates:
            # 选面积最大的装甲板
            best_armor, best_obj_pts_cam = max(
                candidates,
                key=lambda item: getattr(item[0], 'area', 0.0)
            )
            corner_meas_cam = best_obj_pts_cam

        # 3) 角点 3D KF + 原始/滤波后矩形绘制 + 由角点中心进行射击预测
        if corner_meas_cam is not None:
            # 顺序约定：LightDetector 输出为 [top_left, bottom_left, top_right, bottom_right]
            # PnPSolver.solve_pnp 使用 detected_armor.camera_pos 作为 2D 输入，故此顺序保持一致

            # 原始 3D 角点：相机 -> 像素
            h, w = out_img.shape[:2]
            raw_pixels = []
            for p in corner_meas_cam:
                u, v = camera2xy(p)
                u = int(max(0, min(w - 1, u)))
                v = int(max(0, min(h - 1, v)))
                raw_pixels.append((u, v))

            # 画原始矩形（蓝色）：top_left -> bottom_left -> bottom_right -> top_right
            tl, bl, tr, br = raw_pixels
            raw_rect = np.array([tl, bl, br, tr], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out_img, [raw_rect], isClosed=True, color=(255, 0, 0), thickness=2)
            # 原始矩形对角线
            cv2.line(out_img, tl, br, (255, 0, 0), 1)
            cv2.line(out_img, bl, tr, (255, 0, 0), 1)

            # 角点量测：相机 -> 云台
            corner_meas_gimbal = [camera2gimbal(p, 0) for p in corner_meas_cam]

            filtered_gimbal = []   # 每个角点滤波后的 3D 位置
            filtered_vel = []      # 每个角点滤波后的 3D 速度
            filtered_pixels = []   # 每个角点滤波后的像素坐标

            for idx in range(4):
                px, py, pz = map(float, corner_meas_gimbal[idx])

                if not corner_kf_inited[idx]:
                    kf_point = KF(
                        init_cov=corner_kf_init_cov,
                        measure_noise=corner_kf_measure_noise,
                        process_noise=corner_kf_process_noise,
                        x=px, y=py, z=pz,
                        vx=0.0, vy=0.0, vz=0.0
                    )
                    kf_point.init_kf(dt=1.0 / fps if fps > 1e-6 else 1e-2)
                    corner_kfs[idx] = kf_point
                    corner_kf_inited[idx] = True

                kf_point = corner_kfs[idx]
                # 使用当前 dt 更新 F/Q、预测和校正（仅平移量测）
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

            # 画滤波后矩形（绿色），仍然按 tl, bl, br, tr 顺序，避免“8字形”
            tl_f, bl_f, tr_f, br_f = filtered_pixels
            filt_rect = np.array([tl_f, bl_f, br_f, tr_f], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out_img, [filt_rect], isClosed=True, color=(0, 255, 0), thickness=2)
            # 滤波后矩形对角线
            cv2.line(out_img, tl_f, br_f, (0, 255, 0), 1)
            cv2.line(out_img, bl_f, tr_f, (0, 255, 0), 1)

            # 计算四边形的角度和边长并显示在图像上
            # 定义四个点的顺序: tl_f, bl_f, br_f, tr_f
            points = [tl_f, bl_f, br_f, tr_f]
            
            # 计算每条边的长度
            edges = []
            for i in range(4):
                p1 = np.array(points[i])
                p2 = np.array(points[(i + 1) % 4])
                length = np.linalg.norm(p1 - p2)
                edges.append(length)
            
            # 计算每个角的角度
            angles = []
            for i in range(4):
                prev_point = np.array(points[(i - 1) % 4])
                curr_point = np.array(points[i])
                next_point = np.array(points[(i + 1) % 4])
                
                # 计算两条边的向量
                vec1 = prev_point - curr_point
                vec2 = next_point - curr_point
                
                # 计算夹角
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                # 限制cos值在[-1, 1]范围内，防止数值误差
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
            
            # 将边长和角度信息显示在图像上
            # 显示边长（在每条边的中点附近）
            for i in range(4):
                p1 = np.array(points[i])
                p2 = np.array(points[(i + 1) % 4])
                midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.putText(out_img, f"{edges[i]:.1f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 显示角度（在每个顶点附近）
            for i in range(4):
                point = points[i]
                # 稍微偏移文本位置以避免与点重叠
                text_pos = (point[0] + 5, point[1] + 5)
                cv2.putText(out_img, f"{angles[i]:.1f}°", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # 由 4 个角点状态求中心位置和速度（即中心是角点 KF 的结果，不是单独 KF）
            center_pos = np.mean(np.vstack(filtered_gimbal), axis=0)
            center_vel = np.mean(np.vstack(filtered_vel), axis=0)

            # 用中心位置和速度做射击预测
            bullet_speed = defaults_bullet_speed if 'defaults_bullet_speed' in globals() else 25.0
            distance = float(np.linalg.norm(center_pos))
            bullet_time = distance / bullet_speed if bullet_speed > 1e-6 else 0.0
            aim_pos = center_pos + center_vel * bullet_time

            aim_proj = camera2xy(gimbal2camera(aim_pos, 0))
            cv2.circle(out_img, aim_proj, 11, (0, 0, 255), 2)

        # 写出与显示
        video_writer.write(out_img)
        if is_show_video:
            cv2.imshow("vision output", out_img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # 结束清理
    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # run(video_path=r"./test_data/blue10.25.mp4")
    # 其他可选视频：
    # run(video_path=r"./test_data/small_blue.avi")
    # run(video_path=r"./test_data/small_red.avi")
    # run(video_path=r"./test_data/big_red.avi")
    # run(video_path=r"./test_data/big_blue.avi")
    # run(video_path="./test_data/0323blue1.mp4")
    # run(video_path="./test_data/0323blue2.mp4")
    run(video_path=r"./test_data/0325blue.mp4")
