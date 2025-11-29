import os
import sys
import time

# import onnxruntime as ort
import torch

from KalmanFilter import KalmanFilter as KF  # 卡尔曼滤波器
from all_function import *
# from pre_armor import Tracker  # 旧跟踪器：已弃用
from detect_armor import ArmorDetector  # 强制使用 YOLO 检测
# from KalmanFilter import KalmanFilter as KF  # 常速度卡尔曼滤波（先整体注释掉以简化流程）
from guardRobot import GuardRobot
# from get_armor_points_cv import armor_getter  # 经典CV流程已停用
from light_detector import LightDetector
from motion_state_detector import MotionStateDetector  # 运动状态检测器
# from armor_chose import TargetSelector  # 目标选择：本测试不需要
from pnp_solver import PnPSolver
from rotation_velocity_estimator import RotationVelocityEstimator  # 旋转角速度估计器
# import serial
# import UART
from setting import *

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
    # 卡尔曼滤波器参数
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

    # 创建 GuardRobot 实例
    robot = None

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

        # 创建或更新 GuardRobot 实例
        if guardrobot_candidates:
            # 提取装甲板对象
            armor_plates = [candidate[0] for candidate in guardrobot_candidates]
            
            # 如果还没有创建 robot 实例，则创建一个
            if robot is None:
                robot = GuardRobot(armor_plates)
            else:
                # 动态更新 robot 实例中的装甲板
                robot.update_armor_plates(armor_plates)
            
            # 如果检测到两个或更多装甲板，记录初始半径
            if len(armor_plates) >= 2 and len(robot.recorded_radii) < 2:
                try:
                    robot.record_initial_radii()
                    print(f"[Prediction] Recorded initial radii: {robot.recorded_radii}")
                    print(f"[Prediction] Height to radius mapping: {robot.height_to_radius}")
                except Exception as e:
                    print(f"[Prediction] Failed to record initial radii: {e}")
            
            # 根据装甲板数量选择合适的中心点计算方法
            if len(armor_plates) >= 2:
                # 使用两个装甲板计算中心点
                try:
                    center_xz = robot.get_center_from_normals()
                    robot.center_point = center_xz
                    print(f"[Center] Updated center point from 2+ armors: x={center_xz[0]:.3f}, z={center_xz[1]:.3f}")
                except Exception as e:
                    print(f"[Center] Failed to compute center from normals: {e}")
            elif len(armor_plates) == 1 and len(robot.recorded_radii) >= 2:
                # 使用单个装甲板和记录的半径预测中心点
                try:
                    center_xz = robot.predict_center_from_single_armor(0)
                    print(f"[Center] Predicted center point from 1 armor: x={center_xz[0]:.3f}, z={center_xz[1]:.3f}")
                except Exception as e:
                    print(f"[Center] Failed to predict center from single armor: {e}")
            
            # 如果只检测到一个装甲板且已记录初始半径，则进行预测
            predicted_armors = []
            if len(armor_plates) == 1 and len(robot.recorded_radii) >= 2:
                try:
                    # 使用预测功能获取其他装甲板位置
                    predicted_armors = robot.predict_other_armors(0)  # 0表示第一个（也是唯一一个）可见装甲板
                    print(f"[Prediction] Predicted {len(predicted_armors)} armors based on 1 visible armor")
                except Exception as e:
                    print(f"[Prediction] Failed to predict other armors: {e}")
            
            # 直接绘制装甲板，不使用卡尔曼滤波
            # 绘制实际检测到的装甲板
            for armor_plate in armor_plates:
                raw_pixels = []
                for p in armor_plate.camera_pos:
                    u, v = camera2xy(p)
                    u = int(max(0, min(w - 1, u)))
                    v = int(max(0, min(h - 1, v)))
                    raw_pixels.append((u, v))
                if len(raw_pixels) != 4:
                    continue
                tl_f, bl_f, tr_f, br_f = raw_pixels
                filt_rect = np.array([tl_f, bl_f, tr_f, br_f], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(out_img, [filt_rect], isClosed=True, color=(0, 255, 0), thickness=2)
                
            # 绘制预测的装甲板（如果有）
            if predicted_armors:
                for i, pred_armor in enumerate(predicted_armors):
                    raw_pixels = []
                    for p in pred_armor.camera_pos:
                        u, v = camera2xy(p)
                        u = int(max(0, min(w - 1, u)))
                        v = int(max(0, min(h - 1, v)))
                        raw_pixels.append((u, v))
                    if len(raw_pixels) == 4:
                        tl_f, bl_f, tr_f, br_f = raw_pixels
                        # 用不同颜色绘制预测的装甲板
                        filt_rect = np.array([tl_f, bl_f, tr_f, br_f], dtype=np.int32).reshape(-1, 1, 2)
                        cv2.polylines(out_img, [filt_rect], isClosed=True, color=(255, 0, 255), thickness=2)  # 紫色表示预测
                        
                        # 添加标签
                        center_u = int(np.mean([tl_f[0], bl_f[0], tr_f[0], br_f[0]]))
                        center_v = int(np.mean([tl_f[1], bl_f[1], tr_f[1], br_f[1]]))
                        cv2.putText(out_img, f"PREDICTED#{i}", (center_u, center_v), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # 绘制小车中心点（如果已计算）
            if robot and robot.center_point is not None:
                center_x, center_z = robot.center_point[0], robot.center_point[1]
                # 为了在图像上显示，我们需要一个y坐标，这里使用第一个装甲板的y坐标
                if len(robot.armor_plate_center) > 0:
                    center_y = robot.armor_plate_center[0][1]
                    center_point_3d = np.array([center_x, center_y, center_z])
                    u, v = camera2xy(center_point_3d)
                    u = int(max(0, min(w - 1, u)))
                    v = int(max(0, min(h - 1, v)))
                    
                    # 绘制中心点
                    cv2.circle(out_img, (u, v), 8, (0, 0, 255), -1)  # 红色实心圆
                    cv2.putText(out_img, "CENTER", (u + 10, v - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # 打印中心点坐标
                    print(f"[Center] Car center point: x={center_x:.3f}, y={center_y:.3f}, z={center_z:.3f}")

        else:
            # 没有检测到装甲板
            if robot is not None:
                # 清除装甲板信息
                robot.update_armor_plates([])
            pass

        # ====== 如果本帧至少有两块通过KF链路且有3D角点的装甲板，则用法向量直线最近点中点作为小车中心 ======
        if len(guardrobot_candidates) >= 2 and robot is not None:
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
                # 更新 robot 实例中的装甲板
                robot.armor_plate = top_two_armors
                center_xz = robot.get_center_from_normals()  # 相机坐标系二维点 [x, z]
                
                # 注意：此处不再更新 robot.center_point，因为已经在上面的代码中更新过了

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

                # 基于当前两块推算对面两块装甲，并追加到 robot.armor_plate
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
                            pass
                        else:
                            # 平滑后的整体车中心：使用青色大十字
                            pass

                        # cv2.circle(out_img, (u_c, v_c), radius_outer, color_outer, 4)
                        # cv2.line(out_img, (u_c - (radius_outer+4), v_c), (u_c + (radius_outer+4), v_c), color_outer, 3)
                        # cv2.line(out_img, (u_c, v_c - (radius_outer+4)), (u_c, v_c + (radius_outer+4)), color_outer, 3)

                        # cv2.circle(out_img, (u_c, v_c), radius_inner, color_inner, 2)
                        # cv2.circle(out_img, (u_c, v_c), 4, color_inner, -1)
                        # cv2.line(out_img, (u_c - line_len, v_c), (u_c + line_len, v_c), color_inner, 2)
                        # cv2.line(out_img, (u_c, v_c - line_len), (u_c, v_c + line_len), color_inner, 2)

                        # cv2.putText(out_img, label, (u_c + 10, v_c - 10),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                        # cv2.putText(out_img, label, (u_c + 10, v_c - 10),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_inner, 2)
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
                        pts_array = np.array([tl_i, bl_i, br_i, tr_i], dtype=np.int32)  # 正确的点顺序
                        cv2.fillPoly(overlay, [pts_array], color=(0, 128, 255))  # 半透明填充
                        cv2.addWeighted(overlay, 0.3, out_img, 0.7, 0, out_img)  # 混合图像
                        
                        # 绘制装甲板边界，增强3D效果
                        cv2.line(out_img, tl_i, bl_i, (0, 100, 255), 2)  # 左边缘
                        cv2.line(out_img, bl_i, br_i, (0, 150, 255), 2)  # 下边缘
                        cv2.line(out_img, br_i, tr_i, (0, 200, 255), 2)  # 右边缘
                        cv2.line(out_img, tr_i, tl_i, (0, 255, 255), 2)  # 上边缘
                        
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
    run(video_path=r"C:\Users\sjj\Desktop\Deus-RM-CV\test_data\0325blue.mp4")