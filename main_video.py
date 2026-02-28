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
from guardRobot import TestRobotCenter
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


# def camera2xy(point_3d):
#     """
#     将3D相机坐标系中的点转换为2D图像像素坐标
#     point_3d: [x, y, z] 在相机坐标系中（z为深度）
#     返回: [u, v] 像素坐标
#     """
#     # 这里的相机内参需要根据你的相机标定结果调整
#     fx = 1600  # 焦距x（像素）
#     fy = 1600  # 焦距y（像素）
#     cx = 640  # 主点x（像素）
#     cy = 360  # 主点y（像素）
#
#     x, y, z = point_3d[0], point_3d[1], point_3d[2]
#
#     # 避免除零
#     if z <= 0:
#         z = 0.001
#
#     # 透视投影
#     u = (fx * x / z) + cx
#     v = (fy * y / z) + cy
#
#     return np.array([u, v])


def run(video_path):
    """默认通信发送的pitch和yaw角度为0"""
    # 测试敌方颜色
    test_color = Color.RED
    # if video_path.find("red") != -1:
    #     test_color = Color.RED
    # elif video_path.find("blue") != -1:
    #     test_color = Color.BLUE
    # if test_color == Color.RED:
    #     test_color = Color.BLUE
    # else:
    #     test_color = Color.RED
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
    # 初始化机器人中心类
    robot_center = TestRobotCenter()
    robot = GuardRobot()

    # 初始化轨迹显示相关
    traj_output_file = video_path[:-4] + "_trajectory.mp4"
    traj_width, traj_height = 800, 800
    traj_video_writer = cv2.VideoWriter(traj_output_file, fourcc, fps, (traj_width, traj_height))
    traj_img = np.zeros((traj_height, traj_width, 3), dtype=np.uint8) + 255  # 白底

    # 初始化装甲板中心轨迹显示相关
    armor_traj_output_file = video_path[:-4] + "_armor_center_trajectory.mp4"
    armor_traj_video_writer = cv2.VideoWriter(armor_traj_output_file, fourcc, fps, (traj_width, traj_height))
    armor_traj_img = np.zeros((traj_height, traj_width, 3), dtype=np.uint8) + 255  # 白底

    traj_scale = 80  # 像素/米
    traj_origin_x = traj_width // 2
    traj_origin_z = traj_height - 50

    # 绘制坐标轴功能函数
    def draw_traj_axes(img):
        cv2.line(img, (traj_origin_x, 0), (traj_origin_x, traj_height), (200, 200, 200), 2)
        cv2.line(img, (0, traj_origin_z), (traj_width, traj_origin_z), (200, 200, 200), 2)

        # 绘制X轴刻度 (每0.1m)
        x_range_m = traj_width / traj_scale / 2 + 1
        for i in range(1, int(x_range_m * 10)):
            val = i * 0.1
            # 正半轴
            px = int(traj_origin_x + val * traj_scale)
            if px < traj_width:
                cv2.line(img, (px, traj_origin_z), (px, traj_origin_z - 3), (200, 200, 200), 1)
                if i % 5 == 0:  # 每0.5m标记数值
                    cv2.line(img, (px, traj_origin_z), (px, traj_origin_z - 6), (150, 150, 150), 2)
                    cv2.putText(img, f"{val:.1f}", (px - 10, traj_origin_z + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            # 负半轴
            mx = int(traj_origin_x - val * traj_scale)
            if mx > 0:
                cv2.line(img, (mx, traj_origin_z), (mx, traj_origin_z - 3), (200, 200, 200), 1)
                if i % 5 == 0:  # 每0.5m标记数值
                    cv2.line(img, (mx, traj_origin_z), (mx, traj_origin_z - 6), (150, 150, 150), 2)
                    cv2.putText(img, f"-{val:.1f}", (mx - 15, traj_origin_z + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # 绘制Z轴刻度 (每0.1m) - 注意Z轴在图像上是向上的（y轴负方向）
        z_range_m = traj_height / traj_scale
        for i in range(1, int(z_range_m * 10)):
            val = i * 0.1
            py = int(traj_origin_z - val * traj_scale)
            if py > 0:
                cv2.line(img, (traj_origin_x, py), (traj_origin_x + 3, py), (200, 200, 200), 1)
                if i % 5 == 0:  # 每0.5m标记数值
                    cv2.line(img, (traj_origin_x, py), (traj_origin_x + 6, py), (150, 150, 150), 2)
                    cv2.putText(img, f"{val:.1f}", (traj_origin_x + 10, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        cv2.putText(img, "Z (m)", (traj_origin_x + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
        cv2.putText(img, "X (m)", (traj_width - 60, traj_origin_z - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

        # 添加方向标注
        cv2.putText(img, "Forward", (traj_origin_x - 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(img, "Right", (traj_width - 60, traj_origin_z + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(img, "Left", (20, traj_origin_z + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(img, "Camera Origin", (traj_origin_x - 50, traj_origin_z + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 绘制坐标轴
    draw_traj_axes(traj_img)
    draw_traj_axes(armor_traj_img)
    # 给装甲板轨迹视频添加标题
    cv2.putText(armor_traj_img, "Armor Center Trajectory", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    print("Start processing...")
    while True:
        # 读取视频流的一帧
        ret, orig_frame = cap.read()
        # orig_frame = cv2.flip(orig_frame, -1)
        if not ret:
            video_writer.release()
            traj_video_writer.release()
            armor_traj_video_writer.release()
            cap.release()
            cv2.destroyAllWindows()
            print("video write to", output_file, "over")
            break
        detected_point = []  # 初始化装甲板中心点结果列表
        detected_armor_all_point = []  # 存储装甲板所有点的3D坐标

        # 在每帧开始时清空当前帧的装甲板数据
        robot.armor_plates_camera_positions.clear()

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
                ret_pnp, armor, out_img, object_points_cam = pnp_solver.get_armor_target(detected_armor, out_img, 0, 0)
                if ret_pnp:  # 如果PnP解算成功, 将装甲板中心点添加到结果列表
                    # 检查角点数据是否完整（需要4个角点）
                    if object_points_cam is not None and len(object_points_cam) >= 4:
                        # 确保每个角点都有3个坐标值
                        valid_armor = True
                        for corner in object_points_cam:
                            if len(corner) < 3:
                                valid_armor = False
                                break

                        if valid_armor:
                            robot.armor_plates_camera_positions.append(object_points_cam)
                            # print(f"添加装甲板数据，当前装甲板数量: {len(robot.armor_plates_camera_positions)}")

        # 只有在有装甲板数据时才进行预测
        if len(robot.armor_plates_camera_positions) > 0:
            detected_num = len(robot.armor_plates_camera_positions)
            try:
                robot.use_robot_prediction()
                # 获取预测结果
                if robot.center is not None:
                    # print(f"机器人中心坐标: {robot.center}")

                    # ==================== 1. 绘制机器人中心点 ====================
                    robot_center_pixel = camera2xy(robot.center)

                    # 绘制轨迹到平面图（只在有两块或更多装甲板检测到时绘制）
                    if detected_num >= 2:
                        tx = int(traj_origin_x + robot.center[0] * traj_scale)
                        ty = int(traj_origin_z - robot.center[2] * traj_scale)
                        if 0 <= tx < traj_width and 0 <= ty < traj_height:
                            cv2.circle(traj_img, (tx, ty), 2, (0, 0, 255), -1)

                    # 绘制正对装甲板中心轨迹到平面图
                    if len(robot.armor_plates_camera_positions) > 0:
                        # 找到Z值最小（离相机最近）的装甲板中心作为正对装甲板
                        min_z = float('inf')
                        closest_armor_center = None

                        for armor_pts in robot.armor_plates_camera_positions:
                            # 计算装甲板中心点 (4个角点的平均值)
                            pts = np.array(armor_pts, dtype=float)
                            center = np.mean(pts, axis=0)

                            if center[2] < min_z:
                                min_z = center[2]
                                closest_armor_center = center

                        if closest_armor_center is not None:
                            ax = int(traj_origin_x + closest_armor_center[0] * traj_scale)
                            ay = int(traj_origin_z - closest_armor_center[2] * traj_scale)
                            if 0 <= ax < traj_width and 0 <= ay < traj_height:
                                # 绘制蓝色点表示装甲板中心轨迹
                                cv2.circle(armor_traj_img, (ax, ay), 2, (255, 0, 0), -1)

                    img_height, img_width = out_img.shape[:2]

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

                        # print("成功绘制机器人中心点")

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

                    # ==================== 3. 绘制所有预测的装甲板中心点和边框 ====================
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

                    # ==================== 4. 绘制机器人轮廓 ====================
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

                    # ==================== 5. 在图像顶部显示状态信息 ====================
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
                                f"Robot Center: X={robot.center[0]:.2f}m, Z={robot.center[2]:.2f}m",
                                (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # 2. 显示装甲板数量
                    cv2.putText(out_img,
                                f"Armors: Detected={detected_count}, Predicted={predicted_count}",
                                (10, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

                    # 添加图例 (位置根据上方文本动态调整)
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

            except Exception as e:
                # print(f"机器人预测出错: {e}")
                import traceback
                traceback.print_exc()
        # else:
        #     print("当前帧未检测到有效装甲板")

        # ... existing code ...
        frame_size = (out_img.shape[1], out_img.shape[0])

        # 继续处理其他代码...

        # predicted_armor_yaw = math.atan2(predict_armor[1], predict_armor[0])
        # cv2.circle(out_img, (int(predict_armor[0]), int(predict_armor[1])), 5, (0, 0, 255), -1)
        # cv2.putText(out_img, f"yaw:{predicted_armor_yaw:.2f}", (int(predict_armor[0]), int(predict_armor[1])),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        video_writer.write(out_img)
        traj_video_writer.write(traj_img)
        armor_traj_video_writer.write(armor_traj_img)
        if is_show_video:
            cv2.imshow("vision output", out_img)
            cv2.waitKey(1)

        cnt += 1
        if cnt == 20:
            fps = 20 / (time.time() - time1)
            time1 = time.time()
            cnt = 0
            print("fps", fps)

        # 注意：不要清空robot_center.robot_armor_coordinate，因为GuardRobot需要历史数据
        # robot_center.robot_armor_coordinate.clear()


if __name__ == "__main__":
    # 根据视频文件名自动选择颜色，文件名中包含"red"或"blue"
    # run(video_path=r"./test_data/0325blue.mp4")
    # run(video_path=r"./test_data/small_blue.avi")
    # run(video_path=r"./test_data/small_red.avi")
    # run(video_path=r"./test_data/big_red.avi")
    # run(video_path=r"./test_data/big_blue.avi")
    # run(video_path="./test_data/0323blue1.mp4")
    # run(video_path="./test_data/0323blue2.mp4")
    run(video_path=r"C:\Users\sjj\Desktop\新建文件夹\Deus-RM-CV\test_data\20251217_163948_captured.mp4")