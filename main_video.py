import os
import sys
import time
import math

# import onnxruntime as ort
import torch

from KalmanFilter import KalmanFilter as KF  # 卡尔曼滤波器（9维状态：位置+半径+角度+角速度+速度）
from all_function import *
# from pre_armor import Tracker  # 旧跟踪器：已弃用
from detect_armor import ArmorDetector  # 强制使用 YOLO 检测
# from KalmanFilter import KalmanFilter as KF  # 常速度卡尔曼滤波（先整体注释掉以简化流程）
from guardRobot import GuardRobot
from rotation_model import GetAngularVelocity
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


def get_armor_angle(armor_center, robot_center):
    """计算装甲板相对于机器人中心的角度"""
    if robot_center is None or len(robot_center) < 2:
        return 0.0
    
    # armor_center 应该是 [x, y, z] 格式
    # robot_center 应该是 [x, z] 格式 (xz平面坐标)
    dx = armor_center[0] - robot_center[0]
    dz = armor_center[2] - robot_center[1]  # 注意这里 robot_center[1] 是 z 坐标
    angle = math.atan2(dz, dx)
    return angle


def assign_tracker_by_angle(angle):
    """根据角度分配追踪器ID"""
    # 将角度标准化到 [0, 2π]
    normalized_angle = (angle + 2 * math.pi) % (2 * math.pi)
    # 将圆周分为4个象限
    if 0 <= normalized_angle < math.pi/2:
        return 0  # 前方
    elif math.pi/2 <= normalized_angle < math.pi:
        return 1  # 左侧
    elif math.pi <= normalized_angle < 3*math.pi/2:
        return 2  # 后方
    else:
        return 3  # 右侧


def run(video_path):
    """离线视频主流程：检测 -> 角点提取 -> PnP -> 可视化 + 稳定的多装甲板卡尔曼跟踪。"""
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

    # 检测器 / 求解器
    armor_de = ArmorDetector(model_path, model_name, CUDA, test_color, ".pt")
    light_pos = LightDetector()
    pnp_solver = PnPSolver()
    motion_detector = MotionStateDetector()
    rotation_estimator = RotationVelocityEstimator()

    # ================== 多目标卡尔曼跟踪配置 ==================
    # KF 参数调整：测量噪声大一点、过程噪声小一点，让轨迹更稳
    corner_kf_init_cov = 1e3
    corner_kf_measure_noise = 0.3
    corner_kf_process_noise = 0.05
    # track 丢失多少帧后删除
    max_miss_frames = 10
    # 新建 track 的最大允许位置距离（m，3D 坐标）和角度差（rad）
    # 原来比较宽松，身份经常互换，这里调严一些，避免轨迹互相抢观测
    max_match_dist = 0.25  # 原 0.6，位置要求更近
    max_match_angle = math.radians(15.0)  # 原 35 度，只允许相对法向夹角在 15 度内

    # track 结构：
    # {
    #   'id': int,
    #   'kfs': [KF*4],
    #   'inited': [bool*4],
    #   'miss_cnt': int,
    #   'last_center3d': np.array([x,y,z]),
    #   'last_angle': float,
    #   'color': Color,
    #   'troop_type': TroopType
    # }
    next_track_id = 0
    armor_tracks = []

    def get_center_3d_from_pts(pts3d):
        pts = np.asarray(pts3d, dtype=float).reshape(-1, 3)
        return pts.mean(axis=0)

    def angle_between(v1, v2):
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            return 0.0
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosang = float(np.clip(cosang, -1.0, 1.0))
        return math.acos(cosang)

    center_cam_prev = None
    center_smooth_alpha = 0.3

    last_time = time.time()
    robot_id = 1
    frame_count = 0

    recorded_radii = []
    height_to_radius = {}
    armor_dimensions = {}
    center_point = None

    print("Start processing with stable multi-armor tracking...")

    # 在视频循环前初始化角速度估计器
    angular_velocity_estimator = GetAngularVelocity()

    while True:
        ret, orig_frame = cap.read()
        if not ret:
            video_writer.release()
            cap.release()
            cv2.destroyAllWindows()
            print("video write to", output_file, "over")
            break

        out_img = orig_frame.copy()
        h, w = out_img.shape[:2]

        now = time.time()
        dt = float(np.clip(now - last_time, 1e-3, 0.2))
        last_time = now

        # 1) YOLO 检测
        all_detect_armor, out_img = armor_de.detect_armor(orig_frame)

        # 记录可见 armor id 用于角速度估计
        visible_armor_ids = []
        for i, detected_armor_box in enumerate(all_detect_armor):
            ret_detected, detected_armor, out_img = light_pos.extract_light_points(orig_frame, detected_armor_box, out_img)
            if not ret_detected:
                continue
            ret_pnp2, rvec, tvec, obj_pts_cam = pnp_solver.solve_pnp(detected_armor)
            if not ret_pnp2 or obj_pts_cam is None:
                continue
            armor_id = i
            visible_armor_ids.append(armor_id)
            if len(obj_pts_cam) >= 3:
                p1 = np.array(obj_pts_cam[0])
                p2 = np.array(obj_pts_cam[1])
                p3 = np.array(obj_pts_cam[2])
                v1 = p2 - p1
                v2 = p3 - p1
                normal_vector = np.cross(v1, v2)
                if np.linalg.norm(normal_vector) > 1e-6:
                    rotation_estimator.update_armor_normal(armor_id, now, normal_vector)

        armor_count = len(all_detect_armor)
        motion_detector.update(robot_id, armor_count, now)
        motion_state = MotionStateDetector.TRANSLATION

        angular_velocity_info = None
        if visible_armor_ids:
            angular_velocity_info = rotation_estimator.estimate_robot_angular_velocity(visible_armor_ids)

        cv2.putText(out_img, f"Motion State: {motion_state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(out_img, f"Armor Count: {armor_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if angular_velocity_info is not None:
            angular_velocity, rotation_axis = angular_velocity_info
            cv2.putText(out_img, f"Angular Velocity: {angular_velocity:.2f} rad/s", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(out_img, f"Rotation Axis: [{rotation_axis[0]:.2f}, {rotation_axis[1]:.2f}, {rotation_axis[2]:.2f}]", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 本帧通过 PnP 得到的 armor_plate 列表，用于 GuardRobot 和跟踪
        guardrobot_candidates = []  # (ArmorPlate, area)

        # 2) 对每一块检测到的装甲板做精确角点 + PnP
        for detected_armor_box in all_detect_armor:
            ret_detected, detected_armor, out_img = light_pos.extract_light_points(orig_frame, detected_armor_box, out_img)
            if not ret_detected:
                continue
            ret_pnp, armor_candidate, out_img = pnp_solver.get_armor_target(detected_armor, out_img, 0, 0)
            if not ret_pnp or armor_candidate is None:
                continue
            ret_pnp2, rvec2, tvec2, obj_pts_cam = pnp_solver.solve_pnp(detected_armor)
            if not ret_pnp2 or obj_pts_cam is None:
                continue
            # 直接用四个 3D 角点
            detected_armor.camera_pos = np.asarray(obj_pts_cam, dtype=np.float32)
            armor_area = getattr(detected_armor_box, "area", 0.0)
            guardrobot_candidates.append((detected_armor, armor_area))
            print(detected_armor.camera_pos)

        # =============== 多目标数据关联：为每块观测匹配最近的轨迹 ===============
        # 先默认所有轨迹 miss_cnt+1
        for trk in armor_tracks:
            trk['miss_cnt'] += 1
            # 本帧是否被观测更新的标记，先清零
            trk['updated'] = False

        # 准备当前帧观测的中心和朝向（用于匹配）
        obs_centers = []
        obs_normals = []
        obs_plates = []
        for armor_plate, area in guardrobot_candidates:
            pts3d = np.asarray(armor_plate.camera_pos, dtype=float).reshape(-1, 3)
            if pts3d.shape[0] != 4:
                continue
            center3d = get_center_3d_from_pts(pts3d)
            obs_centers.append(center3d)
            obs_plates.append((armor_plate, area))
            # 用前三个点近似一个法向量
            nrm = np.cross(pts3d[1] - pts3d[0], pts3d[2] - pts3d[0])
            obs_normals.append(nrm)

        # 对每个观测尝试匹配已有轨迹
        used_track_indices = set()
        for obs_idx, center3d in enumerate(obs_centers):
            best_tid = -1
            best_cost = 1e9
            for ti, trk in enumerate(armor_tracks):
                if trk['miss_cnt'] > max_miss_frames:
                    continue
                if ti in used_track_indices:
                    continue
                # 距离约束
                dist = np.linalg.norm(center3d - trk['last_center3d'])
                if dist > max_match_dist:
                    continue
                cost = dist
                # 朝向约束（尽量同一面）
                if np.linalg.norm(obs_normals[obs_idx]) > 1e-6 and trk['last_angle'] is not None:
                    # 这里只用角度差的粗略信息：obs_normals与上一帧normal夹角
                    ang = angle_between(obs_normals[obs_idx], trk['last_angle'])  # last_angle 暂时存 normal 向量
                    if ang > max_match_angle:
                        continue
                    cost += 0.2 * ang
                if cost < best_cost:
                    best_cost = cost
                    best_tid = ti

            armor_plate, area = obs_plates[obs_idx]
            pts3d = np.asarray(armor_plate.camera_pos, dtype=float).reshape(-1, 3)

            if best_tid >= 0:
                # 匹配到已有轨迹：用观测更新 KF
                trk = armor_tracks[best_tid]
                used_track_indices.add(best_tid)
                trk['miss_cnt'] = 0
                trk['updated'] = True
                trk['last_center3d'] = center3d
                trk['last_angle'] = obs_normals[obs_idx]
                trk['color'] = armor_plate.color
                trk['troop_type'] = armor_plate.troop_type

                # === 利用 GuardRobot/旋转模型计算 r, theta, omega, vx, vy, vz ===
                # 1) 基于中心点和车中心估计 r, theta
                r = 0.0
                theta = 0.0
                vx = 0.0
                vy = 0.0
                vz = 0.0
                omega_scalar = 0.0
                try:
                    # center_point 在前面对 GuardRobot 计算中心时已经维护
                    if center_point is not None:
                        cx, cz = float(center_point[0]), float(center_point[1])
                        dx = float(center3d[0]) - cx
                        dz = float(center3d[2]) - cz
                        r = float((dx ** 2 + dz ** 2) ** 0.5)
                        theta = float(np.arctan2(dz, dx))
                except Exception:
                    pass

                # 2) 利用 PnP 得到的旋转向量 rvec 估计角速度 omega
                try:
                    if hasattr(armor_plate, 'rvec') and armor_plate.rvec is not None:
                        omega_vec, _mag, yaw_rate = angular_velocity_estimator.update_with_rvec(armor_plate.rvec)
                        omega_scalar = float(yaw_rate)
                except Exception:
                    pass

                # 3) 简单帧间差分估计线速度（基于 last_center3d）
                if 'prev_center3d' in trk and trk['prev_center3d'] is not None and dt > 1e-6:
                    v3 = (center3d - trk['prev_center3d']) / float(dt)
                    vx, vy, vz = map(float, v3)
                trk['prev_center3d'] = center3d.copy()

                for ci in range(4):
                    px, py, pz = map(float, pts3d[ci])
                    if not trk['inited'][ci]:
                        # 9维状态KF：初始化时把所有 9 维参数都传入
                        kf_point = KF(
                            state_dim=9,
                            init_cov=corner_kf_init_cov,
                            measure_noise=corner_kf_measure_noise,
                            process_noise=corner_kf_process_noise,
                            x=px, y=py, z=pz,
                            r=r, theta=theta, omega=omega_scalar,
                            vx=vx, vy=vy, vz=vz,
                        )
                        kf_point.init_kf(dt=dt)
                        trk['kfs'][ci] = kf_point
                        trk['inited'][ci] = True
                    kf_point = trk['kfs'][ci]
                    kf_point.predict_next(dt)
                    kf_point.correct_by_sensor([px, py, pz])
            else:
                # 没有匹配到轨迹：新建一个 track
                kfs = [None] * 4
                inited = [False] * 4

                # 尝试为新轨迹也估计一次 r/theta 等（用当前中心）
                r = 0.0
                theta = 0.0
                vx = 0.0
                vy = 0.0
                vz = 0.0
                omega_scalar = 0.0
                try:
                    if center_point is not None:
                        cx, cz = float(center_point[0]), float(center_point[1])
                        dx = float(center3d[0]) - cx
                        dz = float(center3d[2]) - cz
                        r = float((dx ** 2 + dz ** 2) ** 0.5)
                        theta = float(np.arctan2(dz, dx))
                except Exception:
                    pass
                try:
                    if hasattr(armor_plate, 'rvec') and armor_plate.rvec is not None:
                        omega_vec, _mag, yaw_rate = angular_velocity_estimator.update_with_rvec(armor_plate.rvec)
                        omega_scalar = float(yaw_rate)
                except Exception:
                    pass

                for ci in range(4):
                    px, py, pz = map(float, pts3d[ci])
                    kf_point = KF(
                        state_dim=9,
                        init_cov=corner_kf_init_cov,
                        measure_noise=corner_kf_measure_noise,
                        process_noise=corner_kf_process_noise,
                        x=px, y=py, z=pz,
                        r=r, theta=theta, omega=omega_scalar,
                        vx=vx, vy=vy, vz=vz,
                    )
                    kf_point.init_kf(dt=dt)
                    kfs[ci] = kf_point
                    inited[ci] = True
                new_trk = {
                    'id': next_track_id,
                    'kfs': kfs,
                    'inited': inited,
                    'miss_cnt': 0,
                    'last_center3d': center3d,
                    'prev_center3d': center3d.copy(),
                    'last_angle': obs_normals[obs_idx],
                    'color': armor_plate.color,
                    'troop_type': armor_plate.troop_type,
                    'updated': True,
                }
                armor_tracks.append(new_trk)
                next_track_id += 1

        # 对于本帧没有被观测到的轨迹，只做 predict，不做 correct
        for ti, trk in enumerate(armor_tracks):
            if ti in used_track_indices:
                continue
            for ci in range(4):
                if not trk['inited'][ci] or trk['kfs'][ci] is None:
                    continue
                trk['kfs'][ci].predict_next(dt)

        # 删除长期丢失的 track
        armor_tracks = [t for t in armor_tracks if t['miss_cnt'] <= max_miss_frames]

        # ========= 将 guardrobot_candidates 转给 GuardRobot，用于中心点和对面装甲推算 =========
        robot = None
        if guardrobot_candidates:
            armor_plates_for_robot = [c[0] for c in guardrobot_candidates]
            robot = GuardRobot(armor_plates_for_robot)
            robot.recorded_radii = recorded_radii
            robot.height_to_radius = height_to_radius
            robot.armor_dimensions = armor_dimensions
            robot.center_point = center_point

            if len(armor_plates_for_robot) >= 2 and len(recorded_radii) < 2:
                try:
                    robot.record_initial_radii()
                    recorded_radii = robot.recorded_radii
                    height_to_radius = robot.height_to_radius
                    armor_dimensions = robot.armor_dimensions
                    print(f"[Prediction] Recorded initial radii: {recorded_radii}")
                except Exception as e:
                    print(f"[Prediction] Failed to record initial radii: {e}")

            if len(armor_plates_for_robot) >= 2:
                try:
                    center_xz = robot.get_center_from_normals()
                    center_point = center_xz
                    # 打印每帧小车的半径
                    if len(recorded_radii) >= 2:
                        print(f"Frame {frame_count}: Car radius = {recorded_radii}")
                except Exception as e:
                    print(f"[Center] Failed to compute center from normals: {e}")
            elif len(armor_plates_for_robot) == 1 and len(recorded_radii) >= 2:
                try:
                    robot.center_point = center_point
                    center_xz = robot.predict_center_from_single_armor(0)
                    center_point = center_xz
                    # 打印每帧小车的半径
                    if len(recorded_radii) >= 2:
                        print(f"Frame {frame_count}: Car radius = {recorded_radii}")
                except Exception as e:
                    print(f"[Center] Failed to predict center from single armor: {e}")

        # 生成基于 GuardRobot 的预测装甲板（半径法）
        predicted_armors = []
        if robot is not None and len(robot.armor_plate) == 1 and len(recorded_radii) >= 2:
            try:
                robot.center_point = center_point
                predicted_armors = robot.predict_other_armors(0)
            except Exception as e:
                print(f"[Prediction] Failed to predict other armors: {e}")

        # ========= 从 KF 轨迹构造预测装甲，用于可视化 =========
        kf_predicted_armors = []
        for trk in armor_tracks:
            # 只绘制当前帧真正出现（有观测更新）的装甲板
            if not trk.get('updated', False):
                continue
            pts_pred = []
            for ci in range(4):
                if not trk['inited'][ci] or trk['kfs'][ci] is None:
                    continue
                state_post, _ = trk['kfs'][ci].get_state()
                # 9维状态下前3维仍然是(x,y,z)，直接取用
                pos_post = state_post[:3].reshape(-1)
                pts_pred.append(pos_post)
            if len(pts_pred) == 4:
                pred_plate = ArmorPlate(
                    points=np.array(pts_pred, dtype=float),
                    color=trk['color'],
                    troop_type=trk['troop_type'],
                    area=0,
                    confident=0.5,
                )
                pred_plate.armor_id = trk['id']
                kf_predicted_armors.append(pred_plate)

        # =============== 绘制部分：检测 / 半径预测 / KF 预测 / 中心点 ===============
        # 绘制真实检测到的装甲板（绿色）
        for armor_plate, area in guardrobot_candidates:
            pts3d = np.asarray(armor_plate.camera_pos, dtype=float).reshape(-1, 3)
            if pts3d.shape[0] != 4:
                continue
            pts2d = [camera2xy(p) for p in pts3d]
            pts2d = [
                (int(max(0, min(w - 1, u))), int(max(0, min(h - 1, v))))
                for (u, v) in pts2d
            ]
            tl_i, bl_i, tr_i, br_i = pts2d
            overlay = out_img.copy()
            pts_array = np.array([tl_i, bl_i, br_i, tr_i], dtype=np.int32)
            cv2.fillPoly(overlay, [pts_array], color=(0, 128, 0))
            cv2.addWeighted(overlay, 0.3, out_img, 0.7, 0, out_img)
            cv2.line(out_img, tl_i, bl_i, (0, 100, 0), 2)
            cv2.line(out_img, bl_i, br_i, (0, 150, 0), 2)
            cv2.line(out_img, br_i, tr_i, (0, 200, 0), 2)
            cv2.line(out_img, tr_i, tl_i, (0, 255, 0), 2)
            cv2.line(out_img, tl_i, br_i, (0, 255, 255), 1)
            cv2.line(out_img, bl_i, tr_i, (0, 255, 255), 1)
            center_detected = pts3d.mean(axis=0)
            u_o, v_o = camera2xy(center_detected)
            u_o = int(max(0, min(w - 1, u_o)))
            v_o = int(max(0, min(h - 1, v_o)))
            cv2.putText(out_img, "DETECTED", (u_o + 5, v_o - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 绘制半径法预测装甲（黄色）
        for i, pred_armor in enumerate(predicted_armors):
            pts3d = np.asarray(pred_armor.camera_pos, dtype=float).reshape(-1, 3)
            if pts3d.shape[0] != 4:
                continue
            pts2d = [camera2xy(p) for p in pts3d]
            pts2d = [
                (int(max(0, min(w - 1, u))), int(max(0, min(h - 1, v))))
                for (u, v) in pts2d
            ]
            tl_i, bl_i, tr_i, br_i = pts2d
            overlay = out_img.copy()
            pts_array = np.array([tl_i, bl_i, br_i, tr_i], dtype=np.int32)
            cv2.fillPoly(overlay, [pts_array], color=(0, 128, 255))
            cv2.addWeighted(overlay, 0.3, out_img, 0.7, 0, out_img)
            cv2.line(out_img, tl_i, bl_i, (0, 100, 255), 2)
            cv2.line(out_img, bl_i, br_i, (0, 150, 255), 2)
            cv2.line(out_img, br_i, tr_i, (0, 200, 255), 2)
            cv2.line(out_img, tr_i, tl_i, (0, 255, 255), 2)
            cv2.line(out_img, tl_i, br_i, (0, 255, 255), 1)
            cv2.line(out_img, bl_i, tr_i, (0, 255, 255), 1)
            center_pred = pts3d.mean(axis=0)
            u_o, v_o = camera2xy(center_pred)
            u_o = int(max(0, min(w - 1, u_o)))
            v_o = int(max(0, min(h - 1, v_o)))
            cv2.putText(out_img, f"PRED#{i}", (u_o + 5, v_o - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 绘制 KF 轨迹预测的装甲（橙色），并打上稳定的 track_id
        for pred_armor in kf_predicted_armors:
            pts3d = np.asarray(pred_armor.camera_pos, dtype=float).reshape(-1, 3)
            if pts3d.shape[0] != 4:
                continue
            pts2d = [camera2xy(p) for p in pts3d]
            pts2d = [
                (int(max(0, min(w - 1, u))), int(max(0, min(h - 1, v))))
                for (u, v) in pts2d
            ]
            tl_i, bl_i, tr_i, br_i = pts2d
            overlay = out_img.copy()
            pts_array = np.array([tl_i, bl_i, br_i, tr_i], dtype=np.int32)
            cv2.fillPoly(overlay, [pts_array], color=(255, 128, 0))
            cv2.addWeighted(overlay, 0.3, out_img, 0.7, 0, out_img)
            cv2.line(out_img, tl_i, bl_i, (255, 100, 0), 2)
            cv2.line(out_img, bl_i, br_i, (255, 150, 0), 2)
            cv2.line(out_img, br_i, tr_i, (255, 200, 0), 2)
            cv2.line(out_img, tr_i, tl_i, (255, 255, 0), 2)
            cv2.line(out_img, tl_i, br_i, (255, 255, 0), 1)
            cv2.line(out_img, bl_i, tr_i, (255, 255, 0), 1)
            center_pred = pts3d.mean(axis=0)
            u_o, v_o = camera2xy(center_pred)
            u_o = int(max(0, min(w - 1, u_o)))
            v_o = int(max(0, min(h - 1, v_o)))
            label = f"KF#{getattr(pred_armor, 'armor_id', -1)}"
            cv2.putText(out_img, label, (u_o + 5, v_o - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 小车中心点绘制（沿用原来的 GuardRobot 结果，做 3D 平滑）
        if center_point is not None and guardrobot_candidates:
            first_armor = guardrobot_candidates[0][0]
            pts = np.asarray(first_armor.camera_pos, dtype=float).reshape(-1, 3)
            center_y = float(pts[0][1])
            center_cam_raw = np.array([center_point[0], center_y, center_point[1]], dtype=float)
            if center_cam_prev is None:
                center_cam_smooth = center_cam_raw
            else:
                center_cam_smooth = (
                    center_smooth_alpha * center_cam_raw +
                    (1.0 - center_smooth_alpha) * center_cam_prev
                )
            center_cam_prev = center_cam_smooth
            if center_cam_smooth[2] > 0:
                u, v = camera2xy(center_cam_smooth)
                u = int(max(0, min(w - 1, u)))
                v = int(max(0, min(h - 1, v)))
                cv2.circle(out_img, (u, v), 8, (0, 0, 255), -1)
                cv2.putText(out_img, "CENTER", (u + 10, v - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        frame_count += 1

        video_writer.write(out_img)
        if is_show_video:
            cv2.imshow("vision output", out_img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run(r"test_data/0323blue1.mp4")