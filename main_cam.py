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
from motion_state_detector import MotionStateDetector  # 运动状态检测器
from rotation_velocity_estimator import RotationVelocityEstimator  # 旋转角速度估计器
from guardRobot import GuardRobot  # 添加GuardRobot导入
from KalmanFilter import KalmanFilter as KF  # 卡尔曼滤波器

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
    """相机在线推理主流程：移植 main_video 逻辑，包含绘图与控制台输出。"""
    # 1) 初始化相机
    print("Camera type:", cameraType, "    ID:", cameraID)
    camera = InitCamera(cameraType)
    print(cameraID, "init success.")

    # 2) 初始化检测器
    if used_yolo:
        print("model:", model_path + model_name, "   use_cuda:", CUDA)
        armor_de = ArmorDetector(model_path, model_name, CUDA, friend_color)
        print("armor detector init success.")
        print("Troop type:", my_TroopType, "   Friend color:", friend_color)
    else:
        armor_de = armor_getter(friend_color)

    light_pos = LightDetector()
    pnp_solver = PnPSolver()

    # 初始化 GuardRobot
    robot = GuardRobot()

    # 视频录制变量初始化
    video_writer = None

    print("Start working...")

    cnt = 0
    time1 = time.time()

    while True:
        # 读取一帧
        ret, orig_frame = camera.get_photo()
        if not ret:
            # print("Detected not ret, continue")
            continue
        if camera_flip:
            orig_frame = cv2.flip(orig_frame, -1)

        out_img = orig_frame.copy()
        img_height, img_width = out_img.shape[:2]

        # 清空当前帧装甲板数据
        robot.armor_plates_camera_positions.clear()

        # 1) 检测
        if used_yolo:
            all_detect_armor, out_img = armor_de.detect_armor(orig_frame)
        else:
            ret_cv, all_detect_armor, out_img = armor_de.get_armors_by_img(orig_frame)
            
        for detected_armor_box in all_detect_armor:
            if used_yolo:
                ret_detected, detected_armor, out_img = light_pos.extract_light_points(orig_frame, detected_armor_box, out_img)
            else:
                ret_detected = True
                detected_armor = detected_armor_box

            if ret_detected:
                # 注意：main_cam 使用实时云台角度 vision.pitch, vision.yaw
                try:
                    current_pitch = vision.pitch
                    current_yaw = vision.yaw
                except:
                    current_pitch = 0
                    current_yaw = 0

                ret_pnp, armor, out_img, object_points_cam = pnp_solver.get_armor_target(
                    detected_armor, out_img, current_pitch, current_yaw
                )

                if ret_pnp:
                    # 检查角点数据是否完整
                    if object_points_cam is not None and len(object_points_cam) >= 4:
                        valid_armor = True
                        for corner in object_points_cam:
                            if len(corner) < 3:
                                valid_armor = False
                                break

                        if valid_armor:
                            robot.armor_plates_camera_positions.append(object_points_cam)

        # 只有在有装甲板数据时才进行预测
        if len(robot.armor_plates_camera_positions) > 0:
            detected_num = len(robot.armor_plates_camera_positions)
            try:
                robot.use_robot_prediction()

                # 获取并打印信息
                if robot.center is not None:
                    # ==================== 1. 绘制机器人中心点 ====================
                    robot_center_pixel = camera2xy(robot.center)
                    # (traj_img drawing removed as it's not applicable here easily)

                    if (0 <= robot_center_pixel[0] < img_width and
                            0 <= robot_center_pixel[1] < img_height):
                        # 绘制机器人中心点（红色大圆点）
                        cv2.circle(out_img, (int(robot_center_pixel[0]), int(robot_center_pixel[1])),
                                   12, (0, 0, 255), -1)
                        cv2.circle(out_img, (int(robot_center_pixel[0]), int(robot_center_pixel[1])),
                                   12, (255, 255, 255), 2)

                        # 添加机器人中心点信息文本
                        cv2.putText(out_img, "ROBOT CENTER",
                                    (int(robot_center_pixel[0]) + 15, int(robot_center_pixel[1]) - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(out_img, f"X:{robot.center[0]:.2f} Z:{robot.center[2]:.2f}",
                                    (int(robot_center_pixel[0]) + 15, int(robot_center_pixel[1]) + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # ==================== 2. 绘制所有检测到的装甲板 ====================
                    for i, armor_corners in enumerate(robot.armor_plates_camera_positions):
                        # 限制绘制装甲板数量，但确保包含所有预测的装甲板
                        if i >= len(robot.armor_plates_camera_positions):
                            break

                        if armor_corners and len(armor_corners) >= 4:
                            # 绘制装甲板的四个角点
                            armor_points_pixel = []
                            for corner in armor_corners:
                                corner_pixel = camera2xy(corner)
                                armor_points_pixel.append(corner_pixel)

                            # 区分检测到的装甲板和预测的装甲板
                            if i < len(robot.armor_plates_camera_positions) - (len(robot.armor_plates_camera_positions) - len(robot.armor_plates_camera_positions)):
                                # 检测到的装甲板（蓝色实线）
                                color = (255, 0, 0)
                                line_type = cv2.LINE_AA
                                line_thickness = 3
                                label_prefix = "Detected"
                            else:
                                # 预测的装甲板（绿色虚线）
                                color = (0, 255, 0)
                                line_type = cv2.LINE_AA
                                line_thickness = 2
                                label_prefix = "Predicted"

                            # 绘制装甲板边框
                            for j in range(4):
                                pt1 = armor_points_pixel[j]
                                pt2 = armor_points_pixel[(j + 1) % 4]
                                cv2.line(out_img,
                                         (int(pt1[0]), int(pt1[1])),
                                         (int(pt2[0]), int(pt2[1])),
                                         color, line_thickness, line_type)

                            # 计算装甲板中心点并绘制
                            center_3d = np.mean(armor_corners, axis=0)
                            center_pixel = camera2xy(center_3d)
                            cv2.circle(out_img, (int(center_pixel[0]), int(center_pixel[1])),
                                       8, color, -1)
                            cv2.putText(out_img, f"{label_prefix} Armor {i + 1}",
                                        (int(center_pixel[0]) + 10, int(center_pixel[1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # ==================== 3. 绘制装甲板中心点和连线 ====================
                    if hasattr(robot, 'armor_center_point') and robot.armor_center_point:
                        # 直接使用armor_center_point，不需要处理嵌套结构
                        armor_centers = robot.armor_center_point

                        # 绘制每个装甲板中心点
                        for i, center_3d in enumerate(armor_centers):
                            if isinstance(center_3d, (list, np.ndarray)) and len(center_3d) >= 3:
                                center_3d = np.array(center_3d, dtype=np.float32)

                                # 将装甲板中心点转换为像素坐标
                                center_pixel = camera2xy(center_3d)

                                # 检查点是否在图像范围内
                                if (0 <= center_pixel[0] < img_width and
                                        0 <= center_pixel[1] < img_height):

                                    # 区分检测到的装甲板中心点和预测的装甲板中心点
                                    if i < len(robot.armor_plates_camera_positions):
                                        # 检测到的装甲板中心点（蓝色）
                                        color = (255, 0, 0)
                                        label = f"Detected Center {i + 1}"
                                    else:
                                        # 预测的装甲板中心点（绿色）
                                        color = (0, 255, 0)
                                        label = f"Predicted Center {i + 1}"

                                    # 绘制装甲板中心点
                                    cv2.circle(out_img,
                                               (int(center_pixel[0]), int(center_pixel[1])),
                                               10, color, -1)
                                    cv2.circle(out_img,
                                               (int(center_pixel[0]), int(center_pixel[1])),
                                               10, (255, 255, 255), 1)

                                    # 添加文本标签
                                    cv2.putText(out_img, label,
                                                (int(center_pixel[0]) + 15,
                                                 int(center_pixel[1]) - 15),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                    # 绘制从机器人中心到装甲板中心的连线
                                    if (0 <= robot_center_pixel[0] < img_width and
                                            0 <= robot_center_pixel[1] < img_height):
                                        # 使用不同颜色区分检测和预测的装甲板连线
                                        line_color = (0, 255, 255) if i >= len(robot.armor_plates_camera_positions) else (255, 255, 0)
                                        cv2.line(out_img,
                                                 (int(robot_center_pixel[0]), int(robot_center_pixel[1])),
                                                 (int(center_pixel[0]), int(center_pixel[1])),
                                                 line_color, 2, cv2.LINE_AA)
                                else:
                                    # 如果装甲板中心点不在图像内，在图像边缘绘制一个指示箭头
                                    # 计算方向向量
                                    dir_x = center_pixel[0] - img_width / 2
                                    dir_y = center_pixel[1] - img_height / 2

                                    # 归一化
                                    norm = np.sqrt(dir_x ** 2 + dir_y ** 2)
                                    if norm > 0:
                                        dir_x /= norm
                                        dir_y /= norm

                                    # 在图像边缘绘制箭头
                                    if center_pixel[0] < 0:
                                        edge_x = 10
                                        edge_y = int(img_height / 2 + dir_y * img_height / 3)
                                    elif center_pixel[0] >= img_width:
                                        edge_x = img_width - 10
                                        edge_y = int(img_height / 2 + dir_y * img_height / 3)
                                    elif center_pixel[1] < 0:
                                        edge_x = int(img_width / 2 + dir_x * img_width / 3)
                                        edge_y = 10
                                    else:  # center_pixel[1] >= img_height
                                        edge_x = int(img_width / 2 + dir_x * img_width / 3)
                                        edge_y = img_height - 10

                                    # 绘制指示箭头
                                    arrow_color = (0, 255, 0) if i >= len(robot.armor_plates_camera_positions) else (255, 0, 0)
                                    arrow_label = "Detected" if i < len(robot.armor_plates_camera_positions) else "Predicted"
                                    cv2.arrowedLine(out_img,
                                                    (edge_x - int(30 * dir_x), edge_y - int(30 * dir_y)),
                                                    (edge_x, edge_y),
                                                    arrow_color, 2, tipLength=0.3)
                                    cv2.putText(out_img, f"{arrow_label} {i + 1}",
                                                (edge_x + 10, edge_y),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, arrow_color, 2)

                    # ==================== 4. 绘制所有预测的装甲板中心点和边框 (Predicted) ====================
                    if hasattr(robot, 'armor_center_point') and robot.armor_center_point:
                        # 确保armor_center_point是列表的列表格式
                        if isinstance(robot.armor_center_point[0], list):
                            armor_centers = robot.armor_center_point[0]
                        else:
                            armor_centers = robot.armor_center_point

                        # 绘制每个预测的装甲板
                        for i, center_3d in enumerate(armor_centers):
                            if isinstance(center_3d, (list, np.ndarray)) and len(center_3d) >= 3:
                                center_3d = np.array(center_3d, dtype=np.float32)

                                # 将预测的装甲板中心点转换为像素坐标
                                predicted_center_pixel = camera2xy(center_3d)

                                # 检查点是否在图像范围内
                                if (0 <= predicted_center_pixel[0] < img_width and
                                        0 <= predicted_center_pixel[1] < img_height):

                                    # 绘制预测的装甲板中心点（绿色大圆点）
                                    cv2.circle(out_img,
                                               (int(predicted_center_pixel[0]), int(predicted_center_pixel[1])),
                                               10, (0, 255, 0), -1)
                                    cv2.circle(out_img,
                                               (int(predicted_center_pixel[0]), int(predicted_center_pixel[1])),
                                               10, (255, 255, 255), 1)

                                    # 添加文本标签
                                    label = f"Pred Armor {i + 1}"
                                    cv2.putText(out_img, label,
                                                (int(predicted_center_pixel[0]) + 15,
                                                 int(predicted_center_pixel[1]) - 15),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                                    # 绘制从机器人中心到装甲板中心的连线（黄色虚线）
                                    if (0 <= robot_center_pixel[0] < img_width and
                                            0 <= robot_center_pixel[1] < img_height):
                                        cv2.line(out_img,
                                                 (int(robot_center_pixel[0]), int(robot_center_pixel[1])),
                                                 (int(predicted_center_pixel[0]), int(predicted_center_pixel[1])),
                                                 (0, 255, 255), 2, cv2.LINE_AA)

                                    # 绘制预测装甲板的边框（绿色虚线）
                                    # 假设装甲板尺寸为0.26m x 0.13m
                                    armor_width = 0.26
                                    armor_height = 0.13

                                    # 获取装甲板方向（从机器人中心到装甲板中心的方向）
                                    direction = center_3d - robot.center
                                    direction_norm = np.linalg.norm(direction)
                                    if direction_norm > 0:
                                        direction = direction / direction_norm

                                    # 计算装甲板的四个角点
                                    # 垂直于方向向量的向量
                                    perpendicular = np.array([-direction[2], 0, direction[0]])
                                    perpendicular_norm = np.linalg.norm(perpendicular)
                                    if perpendicular_norm > 0:
                                        perpendicular = perpendicular / perpendicular_norm

                                    # 计算四个角点的3D坐标
                                    half_width = armor_width / 2
                                    half_height = armor_height / 2

                                    corner1 = center_3d + direction * half_height + perpendicular * half_width
                                    corner2 = center_3d + direction * half_height - perpendicular * half_width
                                    corner3 = center_3d - direction * half_height - perpendicular * half_width
                                    corner4 = center_3d - direction * half_height + perpendicular * half_width

                                    # 转换为像素坐标
                                    corner1_pixel = camera2xy(corner1)
                                    corner2_pixel = camera2xy(corner2)
                                    corner3_pixel = camera2xy(corner3)
                                    corner4_pixel = camera2xy(corner4)

                                    # 绘制装甲板边框（绿色虚线）
                                    corners_pixel = [corner1_pixel, corner2_pixel, corner3_pixel, corner4_pixel]
                                    for j in range(4):
                                        pt1 = corners_pixel[j]
                                        pt2 = corners_pixel[(j + 1) % 4]
                                        cv2.line(out_img,
                                                 (int(pt1[0]), int(pt1[1])),
                                                 (int(pt2[0]), int(pt2[1])),
                                                 (0, 255, 0), 2, cv2.LINE_AA)

                                else:
                                    # 如果预测的装甲板中心点不在图像内，在图像边缘绘制一个指示箭头
                                    # 计算方向向量
                                    dir_x = predicted_center_pixel[0] - img_width / 2
                                    dir_y = predicted_center_pixel[1] - img_height / 2

                                    # 归一化
                                    norm = np.sqrt(dir_x ** 2 + dir_y ** 2)
                                    if norm > 0:
                                        dir_x /= norm
                                        dir_y /= norm

                                    # 在图像边缘绘制箭头
                                    if predicted_center_pixel[0] < 0:
                                        edge_x = 10
                                        edge_y = int(img_height / 2 + dir_y * img_height / 3)
                                    elif predicted_center_pixel[0] >= img_width:
                                        edge_x = img_width - 10
                                        edge_y = int(img_height / 2 + dir_y * img_height / 3)
                                    elif predicted_center_pixel[1] < 0:
                                        edge_x = int(img_width / 2 + dir_x * img_width / 3)
                                        edge_y = 10
                                    else:  # predicted_center_pixel[1] >= img_height
                                        edge_x = int(img_width / 2 + dir_x * img_width / 3)
                                        edge_y = img_height - 10

                                    # 绘制指示箭头
                                    cv2.arrowedLine(out_img,
                                                    (edge_x - int(30 * dir_x), edge_y - int(30 * dir_y)),
                                                    (edge_x, edge_y),
                                                    (0, 255, 0), 2, tipLength=0.3)
                                    cv2.putText(out_img, f"Pred {i + 1}",
                                                (edge_x + 10, edge_y),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # ==================== 5. 绘制机器人轮廓 (Polygon) ====================
                    if hasattr(robot, 'armor_center_point') and robot.armor_center_point:
                        armor_centers = robot.armor_center_point[0] if isinstance(robot.armor_center_point[0],
                                                                                  list) else robot.armor_center_point

                        if len(armor_centers) >= 4:
                            # 将装甲板中心点按顺时针顺序连接
                            points_pixel = []
                            for center_3d in armor_centers[:4]:  # 取前4个点
                                if isinstance(center_3d, (list, np.ndarray)) and len(center_3d) >= 3:
                                    point_pixel = camera2xy(np.array(center_3d, dtype=np.float32))
                                    points_pixel.append((int(point_pixel[0]), int(point_pixel[1])))

                            # 绘制机器人轮廓（多边形）- 只绘制在图像内的点
                            valid_points = []
                            for pt in points_pixel:
                                if 0 <= pt[0] < img_width and 0 <= pt[1] < img_height:
                                    valid_points.append(pt)

                            if len(valid_points) >= 3:
                                # 绘制轮廓线（青色）
                                for i in range(len(valid_points)):
                                    pt1 = valid_points[i]
                                    pt2 = valid_points[(i + 1) % len(valid_points)]
                                    cv2.line(out_img, pt1, pt2, (255, 255, 0), 3, cv2.LINE_AA)

                                # 填充机器人内部（半透明青色）
                                if len(valid_points) >= 3:
                                    overlay = out_img.copy()
                                    pts = np.array(valid_points, dtype=np.int32)
                                    cv2.fillPoly(overlay, [pts], (255, 255, 0))
                                    cv2.addWeighted(overlay, 0.2, out_img, 0.8, 0, out_img)

                    # ==================== [新增] 绘制预测的所有射击点 (Future Shots) ====================
                    if hasattr(robot, 'predicted_shoot_points') and robot.predicted_shoot_points:
                        for idx, p_shoot in enumerate(robot.predicted_shoot_points):
                            p_shoot_arr = np.array(p_shoot, dtype=np.float32)
                            p_shoot_pixel = camera2xy(p_shoot_arr)

                            if (0 <= p_shoot_pixel[0] < img_width and 0 <= p_shoot_pixel[1] < img_height):
                                # 绘制红色十字准星
                                px, py = int(p_shoot_pixel[0]), int(p_shoot_pixel[1])
                                cv2.drawMarker(out_img, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                                cv2.putText(out_img, f"Shot {idx+1}", (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (0, 0, 255), 1)

                    # ==================== 6. 显示状态信息 ====================
                    detected_count = len(robot.armor_plates_camera_positions)

                    total_centers_count = 0
                    if hasattr(robot, 'armor_center_point') and robot.armor_center_point:
                        temp_centers = robot.armor_center_point
                        if isinstance(temp_centers, list) and len(temp_centers) > 0 and isinstance(
                                temp_centers[0], list):
                            total_centers_count = len(temp_centers[0])
                        elif isinstance(temp_centers, list):
                            total_centers_count = len(temp_centers)

                    predicted_count = max(0, total_centers_count - detected_count)

                    status_y = 30
                    # 1. 显示机器人中心坐标
                    cv2.putText(out_img,
                                f"Robot pos: X={robot.center[0]:.2f}, Z={robot.center[2]:.2f}",
                                (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # [新增] 显示速度和角速度
                    mgr = getattr(robot.test_robot_center, 'spin_manager', None)
                    omega_val = mgr.omega if mgr else 0.0
                    vel_str = f"Vel: ({robot.velocity[0]:.1f}, {robot.velocity[2]:.1f}) m/s"
                    omega_str = f"Omega: {omega_val:.2f} rad/s"

                    cv2.putText(out_img, vel_str, (10, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                    cv2.putText(out_img, omega_str, (10, status_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                    # 2. 显示装甲板数量
                    cv2.putText(out_img,
                                f"Armors: Detected={detected_count}, Pred={predicted_count}",
                                (10, status_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 3. [新增] 显示双板夹角 (当检测到 >= 2 个装甲板时)
                    next_y = status_y + 60
                    if detected_count >= 2:
                        # 获取上一轮添加到 GuardRobot 中的 angle_between_plates 属性
                        # 使用 getattr 防止旧版本类定义报错
                        angle_val = getattr(robot, 'angle_between_plates', 0.0)

                        # 绘制角度信息 (使用青色/黄色高亮)
                        cv2.putText(out_img, f"Dual Angle: {angle_val:.2f} deg",
                                    (10, next_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        next_y += 30  # 下移一行，为图例腾出空间

                    # 添加图例
                    legend_y = next_y + 10
                    cv2.putText(out_img, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)

                    cv2.circle(out_img, (100, legend_y - 5), 6, (0, 0, 255), -1)
                    cv2.putText(out_img, ": Robot Center", (110, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255), 1)

                    cv2.circle(out_img, (230, legend_y - 5), 5, (255, 0, 0), -1)
                    cv2.putText(out_img, ": Detected Armor", (240, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255), 1)

                    cv2.circle(out_img, (380, legend_y - 5), 5, (0, 255, 0), -1)
                    cv2.putText(out_img, ": Predicted Armor", (390, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255), 1)

                    # ==================== Console Output ====================
                    # 1. 打印机器人中心坐标
                    print(f"Robot Center: X={robot.center[0]:.4f}m, Z={robot.center[2]:.4f}m")

                    # 2. 打印装甲板数量
                    print(f"Armors: Detected={detected_count}, Predicted={predicted_count}")

                    # 3. 打印双板夹角
                    if detected_count >= 2:
                        angle_val = getattr(robot, 'angle_between_plates', 0.0)
                        print(f"Dual Angle: {angle_val:.2f} deg")

                    # 分隔符
                    print("-" * 30)

            except Exception as e:
                print(f"Prediction Error: {e}")
                # import traceback
                # traceback.print_exc()

        cnt += 1
        if cnt == 20:
            fps = 20 / (time.time() - time1)
            time1 = time.time()
            cnt = 0
            print(f"FPS: {fps:.2f}")

        # 如果需要显示视频流
        if is_show_video:
            cv2.imshow("camera_output", out_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
