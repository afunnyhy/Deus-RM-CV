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
- 云台系：项目内使用 x 向右，y 向上，z 指向目标；
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
    """离线视频主流程（仅保留：检测 -> PnP -> 常速度 KF 跟踪）。"""
    # 颜色推断（保持原逻辑，不影响 KF）
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
        fps = 30
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
    # target_selector = TargetSelector()  # 已移除

    # ========== KalmanFilter 初始化 ==========
    kf = KF(init_cov=1e3, measure_noise=0.05, process_noise=0.2)
    kf.init_kf(dt=1.0 / fps)
    kf_inited = False  # 首帧量测启动 KF
    last_time = time.time()
    # =======================================

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

        # 计算 dt（用于 KF）
        now = time.time()
        dt = float(np.clip(now - last_time, 1e-3, 0.2))
        last_time = now

        # 1) 检测（仅 YOLO 路径）
        all_detect_armor, out_img = armor_de.detect_armor(orig_frame)

        # 2) 灯条提点 + PnP，取面积最大的装甲作为量测（本测试假设无遮挡且不会丢失）
        meas = None  # 量测 z = [x,y,z]
        candidates = []  # 收集成功解算的 ArmorTargetPoint
        for detected_armor_box in all_detect_armor:
            ret_detected, detected_armor, out_img = light_pos.extract_light_points(orig_frame, detected_armor_box, out_img)
            if not ret_detected:
                continue
            ret_pnp, armor_candidate, out_img = pnp_solver.get_armor_target(detected_armor, out_img, 0, 0)
            if ret_pnp and armor_candidate and hasattr(armor_candidate, 'gimbal_pos') and armor_candidate.gimbal_pos is not None:
                candidates.append(armor_candidate)
        if candidates:
            # 选择面积最大的装甲板
            best = max(candidates, key=lambda a: getattr(a, 'area', 0.0))
            meas = best.gimbal_pos

        # 3) 常速度 KF：本测试假设永不丢失量测
        if meas is not None:
            gx, gy, gz = float(meas[0]), float(meas[1]), float(meas[2])
            if not kf_inited:
                kf.reset_state(x=gx, y=gy, z=gz, vx=0.0, vy=0.0, vz=0.0, init_cov=1e3)
                kf_inited = True
            else:
                # ========== 分离先验与后验 ==========
                # 先验预测（未融合当前量测），不改变上一帧后验之外再进行的状态提前一步
                prior_state = kf.predict_next(dt)          # statePre
                prior_pos = prior_state[:3].reshape(-1)
                # 校正（融合当前量测）
                kf.correct_by_sensor([gx, gy, gz])
                post_state, _P = kf.get_state()
                posterior_pos = post_state[:3].reshape(-1)
                posterior_vel = post_state[3:].reshape(-1)
                # 计算射击提前量：简单采用“子弹飞行时间=当前距离/初速度” + 匀速直线假设
                bullet_speed = defaults_bullet_speed if 'defaults_bullet_speed' in globals() else 25.0
                distance = float(np.linalg.norm(posterior_pos))
                bullet_time = distance / bullet_speed if bullet_speed > 1e-6 else 0.0
                aim_pos = posterior_pos + posterior_vel * bullet_time
                # ========== 可视化 ==========
                # 投影函数：云台坐标 -> 相机坐标 -> 像素
                prior_proj = camera2xy(gimbal2camera(prior_pos, 0))
                post_proj = camera2xy(gimbal2camera(posterior_pos, 0))
                aim_proj = camera2xy(gimbal2camera(aim_pos, 0))
                # 先验：黄色（暂不显示）
                # cv2.circle(out_img, prior_proj, 10, (0, 255, 255), 2)
                # cv2.putText(out_img, f"prior x:{prior_pos[0]:.2f} y:{prior_pos[1]:.2f} z:{prior_pos[2]:.2f}",
                #             (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                # 后验：绿色（暂不显示）
                # cv2.circle(out_img, post_proj, 10, (0, 255, 0), 2)
                # cv2.putText(out_img, f"post  x:{posterior_pos[0]:.2f} y:{posterior_pos[1]:.2f} z:{posterior_pos[2]:.2f}",
                #             (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
                # 射击点：红色（仅保留圈出位置）
                cv2.circle(out_img, aim_proj, 11, (0, 0, 255), 2)
                # cv2.putText(out_img, f"aim   x:{aim_pos[0]:.2f} y:{aim_pos[1]:.2f} z:{aim_pos[2]:.2f}",
                #             (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            # 初次只显示后验（与量测一致）（暂不显示）
            if not kf_inited:
                proj = camera2xy(gimbal2camera([gx, gy, gz], 0))
                # cv2.circle(out_img, proj, 10, (0, 255, 0), 2)
                # cv2.putText(out_img, f"init x:{gx:.2f} y:{gy:.2f} z:{gz:.2f}", (50,60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,0), 2)
        else:
            # 理论上不会发生（题设保证无遮挡且不会丢失）（暂不显示文本）
            # cv2.putText(out_img, "No measurement!", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            pass

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
    # run(video_path=r"./test_data/0325blue.mp4")
    # 其他可选视频：
    # run(video_path=r"./test_data/small_blue.avi")
    # run(video_path=r"./test_data/small_red.avi")
    # run(video_path=r"./test_data/big_red.avi")
    # run(video_path=r"./test_data/big_blue.avi")
    # run(video_path="./test_data/0323blue1.mp4")
    # run(video_path="./test_data/0323blue2.mp4")
    run(video_path=r"./test_data/0325blue.mp4")
