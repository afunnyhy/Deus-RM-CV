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
                robot.use_robot_prediction(vision.pitch, vision.yaw, vision_manager=vision)
                is_find = True
             except Exception as e:
                print(f"Prediction error: {e}")

        # 绘制机器人和装甲板 (移植自main_video.py的完整可视化逻辑)
        img_height, img_width = out_img.shape[:2]

        robot_center_pixel = (-100, -100)
        if robot.center is not None:
             robot_center_pixel = camera2xy(robot.center)

        # ==================== 1. 绘制机器人中心点 ====================
        if (0 <= robot_center_pixel[0] < img_width and 0 <= robot_center_pixel[1] < img_height):
            # 绘制机器人中心点（红色大圆点）
            cv2.circle(out_img, (int(robot_center_pixel[0]), int(robot_center_pixel[1])), 12, (0, 0, 255), -1)
            cv2.circle(out_img, (int(robot_center_pixel[0]), int(robot_center_pixel[1])), 12, (255, 255, 255), 2)
            # 添加机器人中心点信息文本
            cv2.putText(out_img, "ROBOT CENTER", (int(robot_center_pixel[0]) + 15, int(robot_center_pixel[1]) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ==================== 2. 绘制所有检测到的装甲板 ====================
        for i, armor_corners in enumerate(robot.armor_plates_camera_positions):
            if i >= len(robot.armor_plates_camera_positions): break
            if armor_corners and len(armor_corners) >= 4:
                # 绘制装甲板的四个角点
                armor_points_pixel = []
                for corner in armor_corners:
                    corner_pixel = camera2xy(corner)
                    armor_points_pixel.append(corner_pixel)

                color = (255, 0, 0) # Detected: Blue
                label_prefix = "Detected"

                # 绘制装甲板边框
                for j in range(4):
                    pt1 = armor_points_pixel[j]
                    pt2 = armor_points_pixel[(j + 1) % 4]
                    cv2.line(out_img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 3, cv2.LINE_AA)

                # 计算装甲板中心点并绘制
                center_3d = np.mean(armor_corners, axis=0)
                center_pixel = camera2xy(center_3d)
                cv2.circle(out_img, (int(center_pixel[0]), int(center_pixel[1])), 8, color, -1)
                cv2.putText(out_img, f"{label_prefix} Armor {i + 1}", (int(center_pixel[0]) + 10, int(center_pixel[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ==================== 3. 绘制所有预测的装甲板中心点和边框 ====================
        best_target_pos = None
        min_distance = float('inf')

        if hasattr(robot, 'armor_center_point') and robot.armor_center_point:
             # Handle nested list structure if necessary
             armor_centers = robot.armor_center_point[0] if (len(robot.armor_center_point) > 0 and isinstance(robot.armor_center_point[0], list)) else robot.armor_center_point

             for i, center_3d in enumerate(armor_centers):
                 if isinstance(center_3d, (list, np.ndarray)) and len(center_3d) >= 3:
                     center_3d = np.array(center_3d, dtype=np.float32)
                     center_pixel = camera2xy(center_3d) # Predicted: Green

                     # Check if detected or predicted based on index
                     is_predicted = i >= len(robot.armor_plates_camera_positions)

                     if is_predicted:
                         color = (0, 255, 0)
                         label = f"Pred Armor {i}"

                         if (0 <= center_pixel[0] < img_width and 0 <= center_pixel[1] < img_height):
                             # 绘制预测点
                             cv2.circle(out_img, (int(center_pixel[0]), int(center_pixel[1])), 10, (0, 255, 0), -1)
                             cv2.circle(out_img, (int(center_pixel[0]), int(center_pixel[1])), 10, (255, 255, 255), 1)
                             cv2.putText(out_img, label, (int(center_pixel[0]) + 15, int(center_pixel[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                             # 绘制连线
                             if 0 <= robot_center_pixel[0] < img_width:
                                 cv2.line(out_img, (int(robot_center_pixel[0]), int(robot_center_pixel[1])), (int(center_pixel[0]), int(center_pixel[1])), (0, 255, 255), 2, cv2.LINE_AA)

                             # 绘制预测装甲板边框 (虚线效果简化为实线，因为cv2.line不支持虚线参数直接调用)
                             # 方向计算
                             direction = center_3d - robot.center
                             direction_norm = np.linalg.norm(direction)
                             if direction_norm > 0: direction /= direction_norm

                             perpendicular = np.array([-direction[2], 0, direction[0]])
                             perpendicular_norm = np.linalg.norm(perpendicular)
                             if perpendicular_norm > 0: perpendicular /= perpendicular_norm

                             armor_width, armor_height = 0.26, 0.13
                             half_w, half_h = armor_width/2, armor_height/2

                             c1 = center_3d + direction*half_h + perpendicular*half_w
                             c2 = center_3d + direction*half_h - perpendicular*half_w
                             c3 = center_3d - direction*half_h - perpendicular*half_w
                             c4 = center_3d - direction*half_h + perpendicular*half_w

                             corners_px = [camera2xy(c) for c in [c1, c2, c3, c4]]
                             for j in range(4):
                                 cv2.line(out_img, (int(corners_px[j][0]), int(corners_px[j][1])), (int(corners_px[(j+1)%4][0]), int(corners_px[(j+1)%4][1])), (0, 255, 0), 2, cv2.LINE_AA)

                             # 选择最佳目标 logic (retained from main_cam)
                             dist_to_center = (center_pixel[0] - img_width/2)**2 + (center_pixel[1] - img_height/2)**2
                             if dist_to_center < min_distance:
                                 min_distance = dist_to_center
                                 best_target_pos = center_3d

                         else:
                             # 绘制屏幕外箭头指示
                             dir_x = center_pixel[0] - img_width/2
                             dir_y = center_pixel[1] - img_height/2
                             norm = np.sqrt(dir_x**2 + dir_y**2)
                             if norm > 0:
                                 dir_x /= norm
                                 dir_y /= norm

                             edge_x, edge_y = 0, 0
                             if center_pixel[0] < 0: edge_x = 10; edge_y = int(img_height/2 + dir_y * img_height/3)
                             elif center_pixel[0] >= img_width: edge_x = img_width-10; edge_y = int(img_height/2 + dir_y * img_height/3)
                             elif center_pixel[1] < 0: edge_x = int(img_width/2 + dir_x * img_width/3); edge_y = 10
                             else: edge_x = int(img_width/2 + dir_x * img_width/3); edge_y = img_height-10

                             cv2.arrowedLine(out_img, (edge_x - int(30*dir_x), edge_y - int(30*dir_y)), (edge_x, edge_y), (0, 255, 0), 2, tipLength=0.3)
                             cv2.putText(out_img, f"Pred {i}", (edge_x+10, edge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ==================== 4. 绘制机器人轮廓 (多边形) ====================
        if hasattr(robot, 'armor_center_point') and robot.armor_center_point:
             armor_centers = robot.armor_center_point[0] if (len(robot.armor_center_point) > 0 and isinstance(robot.armor_center_point[0], list)) else robot.armor_center_point
             if len(armor_centers) >= 3:
                 points_pixel = [camera2xy(np.array(c, dtype=np.float32)) for c in armor_centers[:4] if len(c)>=3]
                 valid_points = [(int(p[0]), int(p[1])) for p in points_pixel if 0 <= p[0] < img_width and 0 <= p[1] < img_height]

                 if len(valid_points) >= 3:
                     # 绘制填充轮廓
                     overlay = out_img.copy()
                     cv2.fillPoly(overlay, [np.array(valid_points, dtype=np.int32)], (255, 255, 0))
                     cv2.addWeighted(overlay, 0.2, out_img, 0.8, 0, out_img)
                     # 绘制轮廓线
                     for k in range(len(valid_points)):
                         cv2.line(out_img, valid_points[k], valid_points[(k+1)%len(valid_points)], (255, 255, 0), 2, cv2.LINE_AA)

        # ==================== 5. 显示状态信息 ====================
        detected_count = len(robot.armor_plates_camera_positions)
        predicted_count = max(0, (len(armor_centers) if 'armor_centers' in locals() else 0) - detected_count)

        cv2.putText(out_img, f"Robot Center: X={robot.center[0]:.2f}m, Z={robot.center[2]:.2f}m" if robot.center is not None else "Robot Center: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(out_img, f"Armors: Detected={detected_count}, Predicted={predicted_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if detected_count >= 2:
            angle_val = getattr(robot, 'angle_between_plates', 0.0)
            cv2.putText(out_img, f"Dual Angle: {angle_val:.2f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 射击逻辑
        if best_target_pos is not None:
            # best_target_pos 是相机坐标系下的 3D 点

            # 使用 GuardRobot 封装的函数计算旋转角度
            target_yaw, target_pitch = robot.get_rotation_angle(best_target_pos, vision.pitch, vision.yaw)

            # 计算云台坐标用于距离和调试显示
            gimbal_pos = camera2gimbal(best_target_pos, vision.pitch)
            ax, ay, az = gimbal_pos

            cv2.putText(out_img,
                        f"target x:{ax:<9.3f} y:{ay:<9.3f} z:{az:<9.3f}",
                        (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

            angle_pitch = vision.pitch
            # 注意：这里的 signs 可能需要根据实际云台协议调整
            # get_rotation_angle 返回的是目标绝对角度
            # main_cam 原逻辑：angle_yoz = -(change_angle - angle_pitch)
            angle_yoz = -(target_pitch - angle_pitch)
            angle_xoz = target_yaw

            print(f"Target Point: {best_target_pos} | Target Rotation -> Yaw: {angle_xoz * 180 / math.pi:.2f} deg, Pitch: {angle_yoz * 180 / math.pi:.2f} deg")

            if az > 0.1: # 距离判断

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