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
- 相机系：右手系，x 向右，y 向下，z 朝���；
- 云台系���项目内使用 x 向右，y 向上，z 指向目标；
- ���度：内部多为弧度，显示或日志中一般转成角度（°）。

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
- EKF（���展卡尔曼滤波）：在“有噪声”的观测下，对目标的状态（位置/速度/角度）做“���测+校正”，
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
# from KalmanFilter import KalmanFilter as KF  # 常速度卡尔曼滤波（先整体注释掉以简化流程）
from guardRobot import GuardRobot
from motion_state_detector import MotionStateDetector  # 运动状态检测器
from rotation_velocity_estimator import RotationVelocityEstimator  # 旋转角速度估计器
from KalmanFilter import KalmanFilter as KF  # 卡尔曼滤波器

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
    """离线视频主流程：检测 -> 角点提取 -> PnP -> 可视化（暂时关闭卡尔曼滤波）。"""
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
    motion_detector = MotionStateDetector()  # 创建运动状态检测器实例
    rotation_estimator = RotationVelocityEstimator()  # 创建旋转角速度估计器实例

    # ========== 多装甲板 3D KalmanFilter 管理 ==========
    # key: armor_id  ->  value: {"kfs": [KF*4], "inited": [bool*4], "miss_cnt": int, "center_x": float,
    #                             "color": Color, "troop_type": TroopType,
    #                             "smooth_pixels": list[tuple[int,int]] | None}
    armor_kf_dict = {}
    corner_kf_init_cov = 1e3
    # 测量噪声稍大一些，让KF对单帧抖动更不敏感
    corner_kf_measure_noise = 0.15
    corner_kf_process_noise = 0.2
    # 连续丢失多少帧才真正认为装甲板消失
    max_miss_frames = 8
    # 像素坐标的一阶低通滤波系数（0~1，越小越平滑）
    pixel_smooth_alpha = 0.4

    # 小车中心点的3D平滑系数（0~1，越小越平滑）
    center_smooth_alpha = 0.3
    center_cam_prev = None

    last_time = time.time()
    
    robot_id = 1  # 假设我们跟踪的机器人ID为1
    frame_count = 0  # 帧计数器

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
        
        # 记录装甲板法向量用于角速度计算
        visible_armor_ids = []  # 当前可见的装甲板ID列表
        for i, detected_armor_box in enumerate(all_detect_armor):
            # 提取灯条四角点
            ret_detected, detected_armor, out_img = light_pos.extract_light_points(orig_frame, detected_armor_box, out_img)
            if not ret_detected:
                continue

            # 中心点 PnP 及云台坐标
            ret_pnp, armor_candidate, out_img = pnp_solver.get_armor_target(detected_armor, out_img, 0, 0)
            if not ret_pnp or armor_candidate is None:
                continue

            # 拿 4 个角点 3D（相机坐标系），并将其写回 ArmorPlate，供 GuardRobot 使用
            ret_pnp2, rvec, tvec, obj_pts_cam = pnp_solver.solve_pnp(detected_armor)
            if not ret_pnp2 or obj_pts_cam is None:
                continue
                
            # 计算法向量并更新到旋转估计器
            armor_id = i  # 使用索引作为装甲板ID
            visible_armor_ids.append(armor_id)
            
            # 从3D点计算法向量
            if len(obj_pts_cam) >= 3:
                p1 = np.array(obj_pts_cam[0])
                p2 = np.array(obj_pts_cam[1])
                p3 = np.array(obj_pts_cam[2])
                
                # 计算两个边向量
                v1 = p2 - p1
                v2 = p3 - p1
                
                # 计算法向量
                normal_vector = np.cross(v1, v2)
                if np.linalg.norm(normal_vector) > 1e-6:
                    rotation_estimator.update_armor_normal(armor_id, now, normal_vector)
        
        # 更新运动状态检测器
        armor_count = len(all_detect_armor)
        motion_detector.update(robot_id, armor_count, now)
        motion_state = motion_detector.get_motion_state(robot_id)
        
        # 如果处于旋转状态，计算角速度
        angular_velocity_info = None
        if motion_state == MotionStateDetector.ROTATION and visible_armor_ids:
            angular_velocity_info = rotation_estimator.estimate_robot_angular_velocity(visible_armor_ids)

        # 在图像上显示运动状态
        cv2.putText(out_img, f"Motion State: {motion_state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(out_img, f"Armor Count: {armor_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        # 如果有角速度信息，显示在图像上
        if angular_velocity_info is not None:
            angular_velocity, rotation_axis = angular_velocity_info
            cv2.putText(out_img, f"Angular Velocity: {angular_velocity:.2f} rad/s", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(out_img, f"Rotation Axis: [{rotation_axis[0]:.2f}, {rotation_axis[1]:.2f}, {rotation_axis[2]:.2f}]", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 记录本帧检测到的装甲板中心x，后面用于和已有KF做简单位置匹配
        h, w = out_img.shape[:2]

        def get_center_x(armor_box):
            # armor_box.camera_pos 是 xyxy 或四点，当前这里按xyxy处理
            pts = np.array(armor_box.camera_pos).reshape(-1, 2)
            return float(np.mean(pts[:, 0]))

        # 本帧中已成功完成 PnP 的装甲板（用于 GuardRobot 计算小车中心）
        guardrobot_candidates = []  # 存储 (detected_armor, area)

        # 2) 对每个检测到的装甲板：灯条提点 + PnP
        for detected_armor_box in all_detect_armor:
            # 提取灯条四角点
            ret_detected, detected_armor, out_img = light_pos.extract_light_points(orig_frame, detected_armor_box, out_img)
            if not ret_detected:
                continue

            # 中心点 PnP 及云台坐标
            ret_pnp, armor_candidate, out_img = pnp_solver.get_armor_target(detected_armor, out_img, 0, 0)
            if not ret_pnp or armor_candidate is None:
                continue

            # 拿 4 个角点 3D（相机坐标系），并将其写回 ArmorPlate，供 GuardRobot 使用
            ret_pnp2, rvec2, tvec2, obj_pts_cam = pnp_solver.solve_pnp(detected_armor)
            if not ret_pnp2 or obj_pts_cam is None:
                continue

            # 记录这块装甲板可以作为 GuardRobot 候选（直接用3D四角点）
            detected_armor.camera_pos = np.asarray(obj_pts_cam, dtype=np.float32)
            armor_area = getattr(detected_armor_box, "area", 0.0)
            guardrobot_candidates.append((detected_armor, armor_area))

            # 如果处于平移状态，启用卡尔曼滤波器
            if motion_state == MotionStateDetector.TRANSLATION:
                # 为每个装甲板创建唯一ID（基于装甲板中心位置）
                armor_center_x = get_center_x(detected_armor_box)
                
                # 简单的装甲板匹配逻辑（基于中心x坐标）
                matched_armor_id = None
                for armor_id, state in armor_kf_dict.items():
                    if abs(state["center_x"] - armor_center_x) < 50:  # 阈值可根据需要调整
                        matched_armor_id = armor_id
                        break
                
                # 如果没有匹配的装甲板，则创建新的ID
                if matched_armor_id is None:
                    matched_armor_id = len(armor_kf_dict)
                    armor_kf_dict[matched_armor_id] = {
                        "kfs": [None] * 4,
                        "inited": [False] * 4,
                        "miss_cnt": 0,
                        "center_x": armor_center_x,
                        "color": test_color,
                        "troop_type": None,
                        "smooth_pixels": None
                    }
                
                # 更新装甲板状态
                armor_kf_dict[matched_armor_id]["miss_cnt"] = 0
                armor_kf_dict[matched_armor_id]["center_x"] = armor_center_x
                
                # 为4个角点应用卡尔曼滤波
                kfs = armor_kf_dict[matched_armor_id]["kfs"]
                inited = armor_kf_dict[matched_armor_id]["inited"]
                
                filtered_pixels = []
                for idx, p in enumerate(obj_pts_cam):
                    px, py, pz = map(float, p)
                    
                    if not inited[idx]:
                        kf_point = KF(
                            init_cov=corner_kf_init_cov,
                            measure_noise=corner_kf_measure_noise,
                            process_noise=corner_kf_process_noise,
                            x=px, y=py, z=pz,
                            vx=0.0, vy=0.0, vz=0.0,
                        )
                        kf_point.init_kf(dt=dt)
                        kfs[idx] = kf_point
                        inited[idx] = True
                    
                    kf_point = kfs[idx]
                    kf_point.predict_next(dt)
                    kf_point.correct_by_sensor([px, py, pz])
                    
                    state_post, _P = kf_point.get_state()
                    pos_post = state_post[:3].reshape(-1)
                    
                    u_f, v_f = camera2xy(pos_post)
                    u_f = int(max(0, min(w - 1, u_f)))
                    v_f = int(max(0, min(h - 1, v_f)))
                    filtered_pixels.append((u_f, v_f))
                
                # 使用滤波后的角点绘制装甲板
                if len(filtered_pixels) == 4:
                    tl_f, bl_f, tr_f, br_f = filtered_pixels
                    filt_rect = np.array([tl_f, bl_f, tr_f, br_f], dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(out_img, [filt_rect], isClosed=True, color=(0, 255, 0), thickness=2)
            else:
                # 非平移状态直接使用PnP结果绘制
                raw_pixels = []
                for p in obj_pts_cam:
                    u, v = camera2xy(p)
                    u = int(max(0, min(w - 1, u)))
                    v = int(max(0, min(h - 1, v)))
                    raw_pixels.append((u, v))
                if len(raw_pixels) != 4:
                    continue
                tl_f, bl_f, tr_f, br_f = raw_pixels
                filt_rect = np.array([tl_f, bl_f, tr_f, br_f], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(out_img, [filt_rect], isClosed=True, color=(0, 255, 0), thickness=2)

        # ====== 装甲板消失：连续丢失若干帧后删除其对应的运动模型（KF） ======
        for armor_id, state in list(armor_kf_dict.items()):
            # 如果这一帧没有被匹配更新，则视为丢失一帧
            if state.get("miss_cnt", 0) is not None:
                state["miss_cnt"] = state.get("miss_cnt", 0) + 1
            
            if state["miss_cnt"] > max_miss_frames:
                del armor_kf_dict[armor_id]

        # ====== 如果本帧至少有两块通过KF链路且有3D角点的装甲板，则用法向量直线最近点中点作为小车中心 ======
        if len(guardrobot_candidates) >= 2:
            try:
                # 按面积从大到小排序，取前两块
                guardrobot_candidates.sort(key=lambda x: x[1], reverse=True)
                top_two_armors = [guardrobot_candidates[0][0], guardrobot_candidates[1][0]]

                # 计算两块装甲板各自的3D中心（相机坐标系），仅用于调试输出
                armor_centers = []
                for armor in top_two_armors:
                    pts = np.asarray(armor.camera_pos, dtype=float).reshape(-1, 3)
                    center_i = pts.mean(axis=0)
                    armor_centers.append(center_i)
                c1, c2 = armor_centers

                print(
                    f"[Center-Normal] using two armors: areas=({guardrobot_candidates[0][1]:.3f}, {guardrobot_candidates[1][1]:.3f}), "
                    f"centers=({c1[0]:.3f},{c1[1]:.3f},{c1[2]:.3f}) & ({c2[0]:.3f},{c2[1]:.3f},{c2[2]:.3f})"
                )

                # 使用 GuardRobot 计算两法向量在 xz 平面投影直线的交点作为二维小车中心
                robot = GuardRobot(top_two_armors)
                center_xz = robot.get_center_from_normals()  # 相机坐标系二维点 [x, z]

                # 为每块装甲板构造各自对应的 3D 中心点：[center_x, armor_y, center_z]
                per_armor_centers = []
                for armor_center in armor_centers:
                    y_i = float(armor_center[1])
                    center_i_3d = np.array([
                        float(center_xz[0]),
                        y_i,
                        float(center_xz[1])
                    ], dtype=float)
                    per_armor_centers.append(center_i_3d)

                # 输出每块装甲板对应的车中心（两个中心都体现出来）
                print("[Center-Normal] per-armor centers:")
                for idx_c, c_cam in enumerate(per_armor_centers):
                    print(
                        f"  armor #{idx_c} center_cam=({c_cam[0]:.3f}, {c_cam[1]:.3f}, {c_cam[2]:.3f})"
                    )

                # 使用面积更大的第一块装甲板对应的中心点作为整体车中心的原始值，用于时间平滑
                center_cam_raw = per_armor_centers[0]

                # 基于当前两块装甲推算对面两块装甲，并追加到 robot.armor_plate
                robot.calculate_another_armor_by_center()

                # 运行时调试：在 main_video 中打印由 GuardRobot 预测得到的对面装甲板的四个点坐标和面积
                inferred_armors = []
                if len(robot.armor_plate) >= 4:
                    inferred_armors = robot.armor_plate[2:4]
                    for idx_inf, inf_armor in enumerate(inferred_armors):
                        try:
                            area_inf = float(getattr(inf_armor, "area", 0.0))
                            print(f"[PredArmor-main] new armor #{idx_inf}: area={area_inf:.3f}")
                        except Exception:
                            print(f"[PredArmor-main] new armor #{idx_inf}: area=<unknown>")
                        pts3d_inf = np.asarray(inf_armor.camera_pos, dtype=float).reshape(-1, 3)
                        # 打印四个3D角点
                        for j, pt in enumerate(pts3d_inf):
                            x, y, z = pt
                            print(f"[PredArmor-main] new armor #{idx_inf} pt{j}: ({x:.3f}, {y:.3f}, {z:.3f})")
                        # 额外打印对面装甲板的3D中心点
                        center_inf = pts3d_inf.mean(axis=0)
                        print(
                            f"[PredArmor-main] new armor #{idx_inf} center: ("
                            f"{center_inf[0]:.3f}, {center_inf[1]:.3f}, {center_inf[2]:.3f})"
                        )

                print(
                    f"[Center-Normal] center_cam_raw=({center_cam_raw[0]:.3f}, {center_cam_raw[1]:.3f}, {center_cam_raw[2]:.3f})"
                )

                # 3D 中心点平滑：一阶低通滤波，仅对整体车中心进行
                if center_cam_prev is None:
                    center_cam_smooth = center_cam_raw
                else:
                    center_cam_smooth = (
                        center_smooth_alpha * center_cam_raw +
                        (1.0 - center_smooth_alpha) * center_cam_prev
                    )
                center_cam_prev = center_cam_smooth

                # 简单保护：如果z<=0，说明中心点在相机后方或数值异常，本帧不画中心
                if center_cam_smooth[2] <= 0:
                    print("[Center-Normal] z<=0, skip drawing this frame, center_cam=", center_cam_smooth)
                else:
                    # 将两个 3D 中心点和整体平滑中心投影到像素平面，并在图像上分别绘制
                    centers_to_draw = list(per_armor_centers)
                    centers_to_draw.append(center_cam_smooth)

                    for idx_draw, c_cam_draw in enumerate(centers_to_draw):
                        u_c, v_c = camera2xy(c_cam_draw)
                        u_c = int(max(0, min(w - 1, u_c)))
                        v_c = int(max(0, min(h - 1, v_c)))

                        if idx_draw < 2:
                            # 两块装甲板各自对应的中心：使用黄色小十字
                            color_outer = (0, 0, 0)
                            color_inner = (255, 255, 0)
                            radius_outer = 14
                            radius_inner = 10
                            line_len = 16
                            label = f"CENTER#{idx_draw}"
                        else:
                            # 平滑后的整体车中心：使用青色大十字
                            color_outer = (0, 0, 0)
                            color_inner = (0, 255, 255)
                            radius_outer = 18
                            radius_inner = 16
                            line_len = 20
                            label = "CAR CENTER"

                        cv2.circle(out_img, (u_c, v_c), radius_outer, color_outer, 4)
                        cv2.line(out_img, (u_c - (radius_outer+4), v_c), (u_c + (radius_outer+4), v_c), color_outer, 3)
                        cv2.line(out_img, (u_c, v_c - (radius_outer+4)), (u_c, v_c + (radius_outer+4)), color_outer, 3)

                        cv2.circle(out_img, (u_c, v_c), radius_inner, color_inner, 2)
                        cv2.circle(out_img, (u_c, v_c), 4, color_inner, -1)
                        cv2.line(out_img, (u_c - line_len, v_c), (u_c + line_len, v_c), color_inner, 2)
                        cv2.line(out_img, (u_c, v_c - line_len), (u_c, v_c + line_len), color_inner, 2)

                        cv2.putText(out_img, label, (u_c + 10, v_c - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                        cv2.putText(out_img, label, (u_c + 10, v_c - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_inner, 2)

                # ====== 画出由 GuardRobot 推算得到的对面装甲板（两块） ======
                # 这里使用上面已经获取的 inferred_armors，确保只画一次
                if inferred_armors:
                    for idx_inf, inf_armor in enumerate(inferred_armors):
                        pts3d = np.asarray(inf_armor.camera_pos, dtype=float).reshape(-1, 3)
                        if pts3d.shape[0] != 4:
                            continue

                        pts2d = [camera2xy(p) for p in pts3d]
                        pts2d = [
                            (int(max(0, min(w - 1, u))), int(max(0, min(h - 1, v))))
                            for (u, v) in pts2d
                        ]

                        tl_i, bl_i, tr_i, br_i = pts2d

                        # 用亮黄色画矩形表示对面装甲板，增加透明度和填充效果提升3D感
                        # 创建装甲板区域的半透明填充效果
                        overlay = out_img.copy()
                        pts_array = np.array([tl_i, tr_i, br_i, bl_i], dtype=np.int32)
                        cv2.fillPoly(overlay, [pts_array], color=(0, 128, 255))  # 半透明填充
                        cv2.addWeighted(overlay, 0.3, out_img, 0.7, 0, out_img)  # 混合图像
                        
                        # 绘制装甲板边界，增强3D效果
                        cv2.line(out_img, tl_i, tr_i, (0, 255, 255), 2)  # 上边缘
                        cv2.line(out_img, tr_i, br_i, (0, 200, 255), 2)  # 右边缘
                        cv2.line(out_img, br_i, bl_i, (0, 150, 255), 2)  # 下边缘
                        cv2.line(out_img, bl_i, tl_i, (0, 100, 255), 2)  # 左边缘
                        
                        # 绘制对角线
                        cv2.line(out_img, tl_i, br_i, (0, 255, 255), 1)  # 主对角线
                        cv2.line(out_img, bl_i, tr_i, (0, 255, 255), 1)  # 副对角线

                        # 在中心处标注 OPP#idx
                        center_inf = pts3d.mean(axis=0)
                        u_o, v_o = camera2xy(center_inf)
                        u_o = int(max(0, min(w - 1, u_o)))
                        v_o = int(max(0, min(h - 1, v_o)))
                        cv2.putText(out_img, f"OPP#{idx_inf}", (u_o + 5, v_o - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            except Exception as e:
                print("Center-Normal compute error:", repr(e))
        else:
            # 调试：没有足够装甲板用于计算中心
            if len(guardrobot_candidates) > 0:
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
    # run(video_path=r"./test_data/blue10.25.mp4")
    # 其他可选视频：
    # run(video_path=r"./test_data/small_blue.avi")
    # run(video_path=r"./test_data/small_red.avi")
    # run(video_path=r"./test_data/big_red.avi")
    # run(video_path=r"./test_data/big_blue.avi")
    # run(video_path="./test_data/0323blue1.mp4")
    # run(video_path="./test_data/0323blue2.mp4")
    run(video_path=r"./test_data/0325blue.mp4")
    # run(video_path=r"C:\Users\sjj\Desktop\Deus-RM-CV\test_data\b3bf7e0c4e52cb8e0b8ec66dc7d7e055.mp4")
