from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time
import math
from collections import deque

# 假设这些类定义在 all_type 模块中
from all_type import ArmorPlate, Color, TroopType


# ==========================================
# 核心类 1: 半径与高度管理器 (状态观测器)
# ==========================================
class SpinRadiusManager:
    """
    管理机器人的结构参数：半径(Radius) 和 高度偏移(Y-Offset)。
    处理长方体底盘（如平衡步兵）前后、左右装甲板尺寸和安装高度不一致的问题。
    """

    def __init__(self):
        # --- 结构参数缓存 ---
        # 默认假设：长半径0.25，短半径0.20，高度无差异(0.0)
        self.long_radius = 0.25
        self.short_radius = 0.20

        # 高度偏移：指装甲板中心相对于机器人几何中心在Y轴上的距离
        # Offset = Armor_Y - Robot_Center_Y
        self.long_y_offset = 0.0
        self.short_y_offset = 0.0

        # --- 状态机 ---
        # 0 = Long (前/后), 1 = Short (左/右)
        self.current_state = 0

        # --- 积分与预测 ---
        self.accumulated_angle = 0.0
        self.last_update_time = time.time()
        self.last_armor_yaw = 0.0

        self.omega = 0.0  # 角速度
        self.omega_alpha = 0.2

        self.is_initialized = False

    def update_dual_plate(self, r1, r2, y1, y2, c1_x, c2_x, current_yaw, center_y):
        """
        [双板模式 - 绝对校准]
        参数:
          r1, r2: 两块板的物理半径
          y1, y2: 两块板的Y坐标 (相机坐标系)
          c1_x, c2_x: X坐标 (用于左右判断)
          current_yaw: 偏航角
          center_y: 计算出的机器人中心Y坐标 (基准)
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # 1. 识别长短板 (基于半径大小)
        # 假设 r1 对应 idx1, r2 对应 idx2
        if r1 > r2:
            l_r, s_r = r1, r2
            l_y, s_y = y1, y2
        else:
            l_r, s_r = r2, r1
            l_y, s_y = y2, y1

        # 2. 更新结构参数 (平滑滤波)
        alpha = 0.1 if self.is_initialized else 1.0
        self.long_radius = self.long_radius * (1 - alpha) + l_r * alpha
        self.short_radius = self.short_radius * (1 - alpha) + s_r * alpha

        # 更新高度偏移: Offset = Plate_Y - Center_Y
        curr_l_offset = l_y - center_y
        curr_s_offset = s_y - center_y

        self.long_y_offset = self.long_y_offset * (1 - alpha) + curr_l_offset * alpha
        self.short_y_offset = self.short_y_offset * (1 - alpha) + curr_s_offset * alpha

        self.is_initialized = True

        # 3. 计算角速度
        diff_yaw = current_yaw - self.last_armor_yaw
        while diff_yaw > np.pi: diff_yaw -= 2 * np.pi
        while diff_yaw < -np.pi: diff_yaw += 2 * np.pi

        if dt > 0.001:
            raw_omega = diff_yaw / dt
            if abs(raw_omega) < 15.0:
                self.omega = self.omega * (1 - self.omega_alpha) + raw_omega * self.omega_alpha

        self.last_armor_yaw = current_yaw

        # 4. 旋转方向锁定逻辑 (Determining Next Survivor)
        # 根据 X 坐标判断左右: c1_x < c2_x 意味着 1在左, 2在右
        if c1_x < c2_x:
            # left=1, right=2
            r_right = r2
            r_left = r1
        else:
            # left=2, right=1
            r_right = r1
            r_left = r2

        target_radius = None
        ROTATION_THRESHOLD = 0.1

        if self.omega > ROTATION_THRESHOLD:
            # CCW (逆时针, 向左转) -> 保留右边的板
            target_radius = r_right
        elif self.omega < -ROTATION_THRESHOLD:
            # CW (顺时针, 向右转) -> 保留左边的板
            target_radius = r_left

        # 5. 更新状态机
        if target_radius is not None:
            # 判断保留下来的板是长还是短
            dist_long = abs(target_radius - self.long_radius)
            dist_short = abs(target_radius - self.short_radius)

            if dist_long < dist_short:
                self.current_state = 0  # Long
            else:
                self.current_state = 1  # Short

            # 重置积分
            self.accumulated_angle = 0.0

    def predict_single_plate(self, current_yaw):
        """
        [单板模式 - 预测]
        返回: (predicted_radius, predicted_y_offset)
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # 1. 更新角速度
        diff_yaw = current_yaw - self.last_armor_yaw
        while diff_yaw > np.pi: diff_yaw -= 2 * np.pi
        while diff_yaw < -np.pi: diff_yaw += 2 * np.pi

        if dt > 0.001:
            raw_omega = diff_yaw / dt
            if abs(raw_omega) > 0.2 and abs(raw_omega) < 15.0:
                self.omega = self.omega * (1 - self.omega_alpha) + raw_omega * self.omega_alpha

        self.last_armor_yaw = current_yaw

        # 2. 积分与切换
        self.accumulated_angle += self.omega * dt
        pi_half = np.pi / 2

        if abs(self.accumulated_angle) >= pi_half:
            switches = int(abs(self.accumulated_angle) / pi_half)
            if switches % 2 != 0:
                self.current_state = 1 - self.current_state  # Toggle

            sign = 1 if self.accumulated_angle > 0 else -1
            self.accumulated_angle -= sign * switches * pi_half

        # 3. 返回对应的参数
        if self.current_state == 0:  # Long
            return self.long_radius, self.long_y_offset
        else:  # Short
            return self.short_radius, self.short_y_offset


# ==========================================
# 核心类 2: 几何解算
# ==========================================
class TestRobotCenter:
    spin_manager = SpinRadiusManager()

    def __init__(self, robot_armor_coordinate=None):
        if robot_armor_coordinate is None:
            robot_armor_coordinate = []
        self.robot_armor_coordinate = robot_armor_coordinate
        self.armor_center_point = []

    def get_armor_yaw(self, normal_vec):
        return math.atan2(normal_vec[0], normal_vec[2])

    def get_robot_center_by_two_armor(self, idx1=0, idx2=1):
        """双板解算"""
        self.armor_center_point.clear()
        c1, n1, _ = self.get_armor_normal_vector(self.robot_armor_coordinate[idx1])
        c2, n2, _ = self.get_armor_normal_vector(self.robot_armor_coordinate[idx2])

        if c1 is None or c2 is None: return None

        # XZ平面求交点
        p1_2d = np.array([c1[0], c1[2]], dtype=float)
        d1_2d = np.array([n1[0], n1[2]], dtype=float)
        p2_2d = np.array([c2[0], c2[2]], dtype=float)
        d2_2d = np.array([n2[0], n2[2]], dtype=float)

        A = np.array([[d1_2d[0], -d2_2d[0]], [d1_2d[1], -d2_2d[1]]], dtype=float)
        b = p2_2d - p1_2d

        if abs(np.linalg.det(A)) < 0.1: return None
        try:
            t, s = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        center_xz = p1_2d + t * d1_2d

        # 计算物理半径
        r1 = np.linalg.norm(center_xz - p1_2d)
        r2 = np.linalg.norm(center_xz - p2_2d)

        # 计算平均高度作为 Robot Center Y (基准)
        # 注意：这里假设 Center Y 位于两板高度的中间。
        # 如果机器人重心偏向某一方，这个基准可能会上下浮动，但相对 offset 是准的。
        center_y = (c1[1] + c2[1]) / 2.0

        # 更新管理器 (传入高度信息)
        yaw = self.get_armor_yaw(n1)
        TestRobotCenter.spin_manager.update_dual_plate(
            r1, r2, c1[1], c2[1], c1[0], c2[0], yaw, center_y
        )

        return np.array([center_xz[0], center_y, center_xz[1]], dtype=float)

    def get_robot_center_by_one_armor(self, idx=0):
        """单板解算"""
        self.armor_center_point.clear()
        center_armor, normal_unit, _ = self.get_armor_normal_vector(self.robot_armor_coordinate[idx])

        if center_armor is None or normal_unit is None: return None

        # 1. 获取预测参数
        curr_yaw = self.get_armor_yaw(normal_unit)
        pred_r, pred_offset = TestRobotCenter.spin_manager.predict_single_plate(curr_yaw)

        # 2. XZ平面反推
        normal_xz = np.array([normal_unit[0], normal_unit[2]], dtype=float)
        norm_xz = np.linalg.norm(normal_xz)

        if norm_xz > 1e-4:
            normal_xz /= norm_xz
            center_xz = np.array([center_armor[0], center_armor[2]]) - normal_xz * pred_r

            # 3. Y轴高度修正 (核心修改)
            # Armor_Y = Center_Y + Offset  =>  Center_Y = Armor_Y - Offset
            center_y = center_armor[1] - pred_offset

            return np.array([center_xz[0], center_y, center_xz[1]])
        else:
            return None

    def get_armor_normal_vector(self, four_armor_points):
        p1 = np.array(four_armor_points[0], dtype=float)
        p2 = np.array(four_armor_points[1], dtype=float)
        p3 = np.array(four_armor_points[2], dtype=float)
        p4 = np.array(four_armor_points[3], dtype=float)
        center = (p1 + p2 + p3 + p4) / 4.0
        self.armor_center_point.append(center.tolist())
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        return (center, n / norm, None) if norm > 1e-6 else (None, None, None)


# ==========================================
# 核心类 3: 机器人封装
# ==========================================
class GuardRobot:
    def __init__(self, armor_plates=None, color: Color = None, troop_type: TroopType = None):
        if armor_plates is None: armor_plates = []
        if armor_plates and hasattr(armor_plates[0], 'camera_pos'):
            self.armor_plates_camera_positions = [ap.camera_pos for ap in armor_plates]
        else:
            self.armor_plates_camera_positions = armor_plates

        self.color = color
        self.troop_type = troop_type
        self.test_robot_center = TestRobotCenter(self.armor_plates_camera_positions)
        self.center = None
        self.armor_center_point = []

        # Z轴滤波
        self.z_filter_val = None
        self.Z_ALPHA = 0.1

    def cal_armor(self):
        return len(self.armor_plates_camera_positions) > 0 and len(self.armor_plates_camera_positions[0]) >= 4

    def find_robot_center(self):
        self.armor_center_point.clear()

        if len(self.armor_plates_camera_positions) >= 2:
            raw_center = self.test_robot_center.get_robot_center_by_two_armor()
        elif len(self.armor_plates_camera_positions) == 1:
            raw_center = self.test_robot_center.get_robot_center_by_one_armor(0)
        else:
            raw_center = None

        if raw_center is not None:
            # Z轴滤波逻辑
            if self.z_filter_val is None:
                self.z_filter_val = raw_center[2]

            if abs(raw_center[2] - self.z_filter_val) > 1.0:
                raw_center[2] = self.z_filter_val
            else:
                self.z_filter_val = self.z_filter_val * (1 - self.Z_ALPHA) + raw_center[2] * self.Z_ALPHA
                raw_center[2] = self.z_filter_val
            self.center = raw_center
        else:
            self.center = None

        self.armor_center_point = self.test_robot_center.armor_center_point
        return self.center

    def get_another_armor_plate_center_by_center(self):
        """补全虚拟装甲板 (包含高度修正)"""
        if self.center is None: return
        mgr = TestRobotCenter.spin_manager

        if len(self.armor_plates_camera_positions) == 1 and len(self.armor_center_point) >= 1:
            known = np.array(self.armor_center_point[0])  # 当前可视板的中心

            # 1. 对面装甲板 (中心对称，高度一致)
            opp = 2 * self.center - known
            # 高度直接使用当前板高度，或者 CenterY + CurrentOffset
            opp[1] = known[1]
            self.armor_center_point.append(opp.tolist())

            # 2. 侧面装甲板 (需要切换高度)
            # 获取当前板的状态 (通过对比半径)
            curr_r = np.linalg.norm([known[0] - self.center[0], known[2] - self.center[2]])

            # 判定侧板参数
            if abs(curr_r - mgr.long_radius) < abs(curr_r - mgr.short_radius):
                # 当前是 Long -> 侧板是 Short
                side_r = mgr.short_radius
                side_offset = mgr.short_y_offset
            else:
                # 当前是 Short -> 侧板是 Long
                side_r = mgr.long_radius
                side_offset = mgr.long_y_offset

            # 计算侧板 XZ 坐标
            vec = known - self.center
            vec_xz = np.array([vec[0], vec[2]])
            norm = np.linalg.norm(vec_xz)
            if norm < 1e-4: return

            perp = np.array([-vec_xz[1], vec_xz[0]]) / norm

            s1_xz = np.array([self.center[0], self.center[2]]) + perp * side_r
            s2_xz = np.array([self.center[0], self.center[2]]) - perp * side_r

            # 计算侧板高度 Y = Center_Y + Side_Offset
            side_y = self.center[1] + side_offset

            self.armor_center_point.append([s1_xz[0], side_y, s1_xz[1]])
            self.armor_center_point.append([s2_xz[0], side_y, s2_xz[1]])

    def use_robot_prediction(self):
        if not self.cal_armor(): return
        self.find_robot_center()
        if self.center is not None:
            self.get_another_armor_plate_center_by_center()