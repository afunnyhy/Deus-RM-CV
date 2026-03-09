from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time
import math
from collections import deque
from all_function import camera2gimbal, ballistic_compensation

from sympy.abc import theta

# 假设这些类定义在 all_type 模块中
from all_type import ArmorPlate, Color, TroopType
from setting import PREDICTION_TIME_THRESHOLD

# =========================================================================
# [关键修改] 从外部文件导入 6D 卡尔曼滤波类
# 请确保 KalmanFilter.py 文件在同级目录下，且类名为 KalmanFilter6D
# =========================================================================
try:
    from KalmanFilter import KalmanFilter6D
except ImportError:
    print("Warning: KalmanFilter6D not found. KF feature will be disabled.")


    # 定义一个空类防止报错（如果文件缺失）
    class KalmanFilter6D:
        def __init__(self, **kwargs): pass

        def filter_once(self, z, dt): return z


# ==========================================
# 核心类 1: 半径与高度管理器 (保持原有逻辑)
# ==========================================
class SpinRadiusManager:
    """
    管理机器人的结构参数：半径(Radius) 和 高度偏移(Y-Offset)。
    处理长方体底盘（如平衡步兵）前后、左右装甲板尺寸和安装高度不一致的问题。
    """

    def __init__(self):
        # 初始设定的长半径（前后？）和短半径（左右？）
        # [修改] 按照要求将半径固定 (这里示例改为用户可能的真实值，如0.28/0.22，用户可自行微调)
        self.long_radius = 0.28
        self.short_radius = 0.22
        # 初始设定的高度偏移
        self.long_y_offset = 0.0
        self.short_y_offset = 0.0

        # --- 状态机 ---
        self.current_state = 0  # 0 = 正在面对长板面, 1 = 正在面对短板面 (或者对应状态索引)

        # --- 积分与预测 ---
        self.accumulated_angle = 0.0 # 累积旋转角度，用于状态切换预测
        # self.last_update_time = time.time() # [修改] 不再内部记录时间
        self.last_armor_yaw = 0.0 # 上一帧的装甲板 Yaw 角

        self.omega = 0.0  # 估算的旋转角速度 (rad/s)
        self.omega_alpha = 0.95 # 角速度低通滤波系数

        # [新增] 是否启用角速度滤波控制开关，默认为 False (不滤波)
        self.enable_omega_filter = True

        self.is_initialized = True # [修改] 半径已写死，视为已初始化
        self.T=1.0
    def find_shoot_time(self):
        theta=self.omega*self.T
        shoot_angle=self.accumulated_angle-theta
        return shoot_angle

    # [修改] update_dual_plate 接收外部传入的 dt
    def update_dual_plate(self, r1, r2, y1, y2, c1_x, c2_x, current_yaw, center_y, dt):
        """
        [双板模式 - 绝对校准]
        当视觉能同时看到两个装甲板时调用此函数。
        """
        # [修改] 移除内部时间计算，使用传入的 dt

        # 1. 识别长短板：假设距离中心更远的是长半径面，近的是短半径面
        if r1 > r2:
            l_r, s_r = r1, r2
            l_y, s_y = y1, y2
        else:
            l_r, s_r = r2, r1
            l_y, s_y = y2, y1

        # 2. 更新结构参数 (平滑滤波)：使用指数移动平均平滑参数震荡
        alpha = 0.1 #if self.is_initialized else 1.0 # 初次直接赋值，后续平滑更新
        # [修改] 半径已写死，不再动态更新半径
        # self.long_radius = self.long_radius * (1 - alpha) + l_r * alpha
        # self.short_radius = self.short_radius * (1 - alpha) + s_r * alpha

        # 计算高度相对于中心的偏移量
        curr_l_offset = l_y - center_y
        curr_s_offset = s_y - center_y
        self.long_y_offset = self.long_y_offset * (1 - alpha) + curr_l_offset * alpha
        self.short_y_offset = self.short_y_offset * (1 - alpha) + curr_s_offset * alpha

        self.is_initialized = True

        # 3. 计算角速度：差分 Yaw 角
        diff_yaw = current_yaw - self.last_armor_yaw
        # 处理角度跳变 (-pi 到 pi 的跨越)
        while diff_yaw > np.pi: diff_yaw -= 2 * np.pi
        while diff_yaw < -np.pi: diff_yaw += 2 * np.pi

        # [修改] 增加异常跳变过滤 (切板保护)
        # 如果两帧之间角度变化超过 0.8 rad (约45度)，认为发生了切板或异常，不更新 omega
        if abs(diff_yaw) < 0.8:
            if dt > 0.001:
                raw_omega = diff_yaw / dt
                # 简单的异常值过滤，只更新合理的转速
                if abs(raw_omega) < 15.0:
                    if self.enable_omega_filter:
                        self.omega = raw_omega/abs(raw_omega)*(abs(self.omega) * (1 - self.omega_alpha) + abs(raw_omega) * self.omega_alpha)
                    else:
                        self.omega = raw_omega

        # [新增] 角速度阈值：当角速度小于 0.5 时直接置为 0
        if abs(self.omega) < 0.5:
            self.omega = 0.0

        self.last_armor_yaw = current_yaw

        # 4. 旋转方向锁定与状态机更新
        # 根据左右位置确定旋转方向，尝试判断当前哪个板是目标板，更新 current_state
        if c1_x < c2_x:
            r_left, r_right = r1, r2
        else:
            r_left, r_right = r2, r1

        target_radius = None
        ROTATION_THRESHOLD = 0.1 # 旋转判定阈值

        # 根据旋转方向判断哪一侧是即将面对的主板
        if self.omega > ROTATION_THRESHOLD:
            target_radius = r_right
        elif self.omega < -ROTATION_THRESHOLD:
            target_radius = r_left

        # 更新状态：当前是面对长板(0) 还是 短板(1)
        if target_radius is not None:
            dist_long = abs(target_radius - self.long_radius)
            dist_short = abs(target_radius - self.short_radius)
            self.current_state = 0 if dist_long < dist_short else 1
            self.accumulated_angle = 0.0 # 重置角度积分

    # [修改] predict_single_plate 接收外部传入的 dt
    def predict_single_plate(self, current_yaw, dt):
        """
        [单板模式 - 预测]
        当只看到一个装甲板时，无法直接得知这是长板侧还是短板侧。
        利用角速度积分推算当前转到了哪一面。
        """
        # [修改] 移除内部时间计算，使用传入的 dt

        # 计算 yaw 变化量
        diff_yaw = current_yaw - self.last_armor_yaw
        while diff_yaw > np.pi: diff_yaw -= 2 * np.pi
        while diff_yaw < -np.pi: diff_yaw += 2 * np.pi

        # 持续更新角速度估计
        if abs(diff_yaw) < 0.8:
            if dt > 0.001:
                raw_omega = diff_yaw / dt
                if 0.2 < abs(raw_omega) < 15.0:
                    if self.enable_omega_filter:
                        self.omega = self.omega * (1 - self.omega_alpha) + raw_omega * self.omega_alpha
                    else:
                        self.omega = raw_omega

        # [新增] 角速度阈值：当角速度小于 0.5 时直接置为 0
        if abs(self.omega) < 0.5:
            self.omega = 0.0

        self.last_armor_yaw = current_yaw

        # 积分角度，用于推测是否切换了板面1
        # 假设板面之间间隔 90度 (Pi/2)
        self.accumulated_angle += self.omega * dt
        pi_half = np.pi / 2

        # 每次旋转超过 90 度，认为切换到了相邻的板（长变短，或短变长）
        if abs(self.accumulated_angle) >= pi_half:
            switches = int(abs(self.accumulated_angle) / pi_half)
            if switches % 2 != 0:
                self.current_state = 1 - self.current_state # 状态翻转 0<->1
            # 减去已经结算的角度，保留余数
            sign = 1 if self.accumulated_angle > 0 else -1
            self.accumulated_angle -= sign * switches * pi_half

        # 返回当前推测状态对应的物理参数
        if self.current_state == 0:
            return self.long_radius, self.long_y_offset
        else:
            return self.short_radius, self.short_y_offset


# ==========================================
# 核心类 2: 几何解算 (保持原有逻辑)
# ==========================================
class TestRobotCenter:
    # 静态成员，所有实例共享同一个自旋管理器
    spin_manager = SpinRadiusManager()

    def __init__(self, robot_armor_coordinate=None):
        if robot_armor_coordinate is None:
            robot_armor_coordinate = []
        # 注意：这里的 robot_armor_coordinate 将会被 GuardRobot 传入经过 KF 滤波后的数据
        # 数据格式: [Plate1_Points, Plate2_Points, ...]，每个 Points 是 4个 [x,y,z]
        self.robot_armor_coordinate = robot_armor_coordinate
        self.armor_center_point = [] # 用于存储计算过程中提取的装甲板中心点
        self.last_dual_angle_deg = 0.0 # 记录双板之间的夹角，用于调试或验证

    def get_armor_yaw(self, normal_vec):
        """根据法向量计算 Yaw 角 (XZ平面投影)"""
        return math.atan2(normal_vec[0], normal_vec[2])

    def get_robot_center_by_two_armor(self, dt, idx1=0, idx2=1):
        """
        [双板解算策略]
        当能看到两个装甲板时，利用两个板的中心点和法向量，求解刚体中心。

        原理：
        两个装甲板的法向量理论上都指向（或背向）旋转中心。
        在XZ平面上，这是两条直线的交点问题。
        """
        self.armor_center_point.clear()
        # 获取两个板的中心和法向量
        c1, n1, _ = self.get_armor_normal_vector(self.robot_armor_coordinate[idx1])
        c2, n2, _ = self.get_armor_normal_vector(self.robot_armor_coordinate[idx2])

        if c1 is None or c2 is None: return None

        # 投影到 2D (XZ平面)
        p1_2d = np.array([c1[0], c1[2]], dtype=float)
        d1_2d = np.array([n1[0], n1[2]], dtype=float)
        p2_2d = np.array([c2[0], c2[2]], dtype=float)
        d2_2d = np.array([n2[0], n2[2]], dtype=float)

        # 构建线性方程组求解交点: P1 + t*N1 = P2 + s*N2
        # A * [t, -s]^T = P2 - P1
        A = np.array([[d1_2d[0], -d2_2d[0]], [d1_2d[1], -d2_2d[1]]], dtype=float)
        b = p2_2d - p1_2d

        # 矩阵行列式过小说明平行，无法求解交点
        if abs(np.linalg.det(A)) < 0.1: return None
        try:
            t, s = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        # 计算出中心点坐标
        center_xz = p1_2d + t * d1_2d

        # --- 计算板间夹角 (用于验证) ---
        vec1 = p1_2d - center_xz
        vec2 = p2_2d - center_xz
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 > 1e-4 and norm2 > 1e-4:
            cos_theta = np.dot(vec1, vec2) / (norm1 * norm2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            self.last_dual_angle_deg = np.degrees(np.arccos(cos_theta))
        else:
            self.last_dual_angle_deg = 0.0

        # 计算两个板各自的旋转半径
        r1 = np.linalg.norm(center_xz - p1_2d)
        r2 = np.linalg.norm(center_xz - p2_2d)
        # 高度取平均
        center_y = (c1[1] + c2[1]) / 2.0

        # 更新自旋管理器模型
        yaw = self.get_armor_yaw(n1)
        TestRobotCenter.spin_manager.update_dual_plate(
            r1, r2, c1[1], c2[1], c1[0], c2[0], yaw, center_y, dt
        )
        return np.array([center_xz[0], center_y, center_xz[1]], dtype=float)

    def get_robot_center_by_one_armor(self, dt, idx=0):
        """
        [单板解算策略]
        当只看到一个装甲板时，利用法向量方向，并在法向量方向上回退 "预测半径" 的距离来估算中心。
        预测半径由 SpinRadiusManager 提供（根据长短板状态机）。
        """
        self.armor_center_point.clear()
        self.last_dual_angle_deg = 0.0

        center_armor, normal_unit, _ = self.get_armor_normal_vector(self.robot_armor_coordinate[idx])
        if center_armor is None or normal_unit is None: return None

        # 1. 获取当前 Yaw，并询问管理器当前的半径和高度偏移
        curr_yaw = self.get_armor_yaw(normal_unit)
        pred_r, pred_offset = TestRobotCenter.spin_manager.predict_single_plate(curr_yaw, dt)

        # 2. XZ 平面上回退半径距离
        normal_xz = np.array([normal_unit[0], normal_unit[2]], dtype=float)
        norm_xz = np.linalg.norm(normal_xz)

        if norm_xz > 1e-4:
            normal_xz /= norm_xz # 归一化

            # [修改] 动态判定方向：计算两个可能的中心点，取Z轴更大的那个（即离相机更远的那个）
            # 这是基于机器人中心一定位于装甲板后方（深度更深）的物理约束

            # 候选1：减法
            cand1 = np.array([center_armor[0], center_armor[2]]) - normal_xz * pred_r
            # 候选2：加法
            cand2 = np.array([center_armor[0], center_armor[2]]) + normal_xz * pred_r

            # 比较 Z 值 (下标1)
            # cand1[1] 和 cand2[1] 分别对应 Z 坐标
            if cand1[1] > cand2[1]:
                center_xz = cand1
            else:
                center_xz = cand2

            # 高度修正
            center_y = center_armor[1] - pred_offset
            return np.array([center_xz[0], center_y, center_xz[1]])
        else:
            return None

    def get_armor_normal_vector(self, four_armor_points):
        """
        利用装甲板的四个角点计算中心点和法向量
        :param four_armor_points: 4个点，顺序通常为 左上, 左下, 右下, 右上 (或类似顺序，需保证交叉乘积方向正确)
        :return: (中心坐标, 单位法向量, None)
        """
        # 确保数据是 float 类型，避免整型溢出
        p1 = np.array(four_armor_points[0], dtype=float)
        p2 = np.array(four_armor_points[1], dtype=float)
        p3 = np.array(four_armor_points[2], dtype=float)
        p4 = np.array(four_armor_points[3], dtype=float)

        center = (p1 + p2 + p3 + p4) / 4.0
        self.armor_center_point.append(center.tolist())

        # 利用对角线或者边向量叉乘计算法向量
        # 假设 p1, p2, p3 分别是 左上, 左下, 右下
        # v1: p1 -> p2 (左边向下)
        # v2: p1 -> p3 (左边向对角) -- 这里代码写的是 v1=p2-p1, v2=p3-p1，需确认点的顺序才能确定法向
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2) # 叉乘得法向量

        # [修改] 强制统一法向量方向，解决法向量 Z 轴正负跳动导致的 Yaw 角 180 度突变问题
        # 这里强制 Z 分量为正 (指向远离相机方向，与之前的单板解算逻辑匹配)
        if n[2] < 0:
            n = -n

        norm = np.linalg.norm(n)
        # 归一化
        return (center, n / norm, None) if norm > 1e-6 else (None, None, None)


# ==========================================
# 核心类 3: 机器人封装 (整合 KF 逻辑)
# ==========================================
class GuardRobot:
    def __init__(self, armor_plates=None, color: Color = None, troop_type: TroopType = None, enable_kf: bool =False):
        """
        GuardRobot 构造函数
        :param armor_plates: 输入的装甲板列表 (ArmorPlate 对象或坐标列表)
        :param enable_kf: [控制开关] 是否启用卡尔曼滤波对角点进行平滑。启用后会对每个装甲板的每个角点进行独立滤波。
        """
        if armor_plates is None: armor_plates = []

        # 提取相机坐标：如果是 ArmorPlate 对象则取其 camera_pos 属性
        if armor_plates and hasattr(armor_plates[0], 'camera_pos'):
            self.armor_plates_camera_positions = [ap.camera_pos for ap in armor_plates]
        else:
            self.armor_plates_camera_positions = armor_plates

        self.color = color
        self.troop_type = troop_type

        # 初始化几何解算器
        self.test_robot_center = TestRobotCenter(self.armor_plates_camera_positions)

        self.center = None # 计算得到的机器人中心坐标
        self.armor_center_point = [] # 各个装甲板的中心点
        self.angle_between_plates = 0.0 # 双板夹角

        # Z轴低通滤波参数：用于平滑深度方向的抖动
        self.z_filter_val = None
        self.Z_ALPHA = 0.1

        # ==================================================
        # KF 相关初始化
        # ==================================================
        self.enable_kf = enable_kf  # 滤波总开关

        # 存储滤波器的字典: key=装甲板索引, value=长度为4的KF列表(对应4个角点)
        # 即使装甲板移动，只要索引对应关系不变，就能持续追踪
        self.kf_map: Dict[int, List[KalmanFilter6D]] = {}

        # 时间戳记录，用于计算 dt
        self.last_timestamp = time.time()

        # [新增] 预测角度与通信管理器
        self.prediction_time_threshold = PREDICTION_TIME_THRESHOLD
        self.vision_manager = None

    def _init_kf_for_plate(self, plate_idx: int, initial_points: List):
        """
        [内部方法] 为指定的装甲板初始化 KF 滤波器
        :param plate_idx: 装甲板索引
        :param initial_points: 初始观测点列表 (4个点)
        """
        # init_cov=1000: 初始协方差大，表示初始不确定性高，快速收敛到测量值
        # process_noise=5000: 过程噪声大，表示相信物体的运动是多变的，不过分依赖模型预测 (对抗滞后)
        # measure_noise=0.1: 测量噪声小，表示比较相信视觉测量的当前位置
        self.kf_map[plate_idx] = [
            KalmanFilter6D(measure_dim=3, init_cov=1000.0, measure_noise=0.1, process_noise=5000.0,
                           x=p[0], y=p[1], z=p[2])
            for p in initial_points
        ]

    def cal_armor(self):
        """检查是否有有效的装甲板数据"""
        return len(self.armor_plates_camera_positions) > 0 and len(self.armor_plates_camera_positions[0]) >= 4

    def _process_kf(self, raw_armors, dt):
        """
        [内部方法] 处理卡尔曼滤波的核心逻辑
        输入: 原始观测的装甲板坐标列表 (raw_armors)
        输出: 经过位置+速度平滑后的装甲板坐标列表

        逻辑:
        对看到的每个装甲板，为其 4 个角点分别维护一个 6维卡尔曼滤波器 (Location + Velocity)。
        这有助于平滑视觉检测的噪声，并提供短暂的运动预测。
        """
        filtered_armors = []

        for idx, points in enumerate(raw_armors):
            # 1. 检查该装甲板是否有对应的滤波器组，没有则初始化
            if idx not in self.kf_map:
                # 初始化：为该装甲板的 4 个角点分别创建 6D 卡尔曼滤波器
                self._init_kf_for_plate(idx, points)

            # 2. 对 4 个角点分别进行滤波
            filtered_points = []
            kf_list = self.kf_map[idx]

            for i in range(4):
                if i < len(points):
                    # 调用 filter_once 进行一步迭代 (预测 + 更正)
                    kf_result = kf_list[i].filter_once(points[i], dt=dt)

                    # ====================================================
                    # [修复] 兼容性处理：检查返回值是 元组 还是 数组
                    # ====================================================
                    if isinstance(kf_result, tuple):
                        # 如果返回的是 (X, P, K) 或 (X, P)，取第一个元素 X (状态向量)
                        state_vector = kf_result[0]
                    else:
                        # 如果直接返回的是 X
                        state_vector = kf_result

                    # 处理 numpy 形状，确保是 [x, y, z]
                    if hasattr(state_vector, 'flatten'):
                        # 展平并取前3个值 (x, y, z)，忽略速度等后续状态
                        xyz = state_vector.flatten()[:3]
                        filtered_points.append(xyz.tolist())
                    elif isinstance(state_vector, list):
                        filtered_points.append(state_vector[:3])
                    else:
                        # 兜底：假设已经是正确格式
                        filtered_points.append(state_vector)
                else:
                    # 异常保护：数据缺失还是用原始的
                    filtered_points.append(points[i])

            filtered_armors.append(filtered_points)

        return filtered_armors

    def update_velocity(self):
        """
        [新增] 更新机器人底盘线速度
        使用最近的历史中心点进行差分计算
        """
        if self.center is None: return

        current_time = self.last_timestamp
        self.center_history.append((self.center, current_time))

        if len(self.center_history) >= 2:
            # 取最新和最旧的数据计算平均速度 (简单低通)
            p_new, t_new = self.center_history[-1]
            p_old, t_old = self.center_history[0] # 使用队列头部数据，跨度更大，速度更平滑

            time_span = t_new - t_old
            if time_span > 0.001:
                raw_v = (p_new - p_old) / time_span

                # 简单的低通滤波
                alpha_v = 0.8
                self.velocity = self.velocity * (1 - alpha_v) + raw_v * alpha_v

                # [新增] 线速度阈值：3个方向的分量小于 0.05 m/s 时置为 0
                for i in range(3):
                    if abs(self.velocity[i]) < 0.05:
                        self.velocity[i] = 0.0
            else:
                self.velocity = np.array([0.0, 0.0, 0.0])

    def get_predicted_target_position(self, future_time_delta: float):
        """
        [新增] 获取未来时刻的预测打击坐标 (考虑平移 + 自旋)
        future_time_delta: 距离当前时刻的时间增量 (秒)
        """
        if self.center is None: return None

        mgr = TestRobotCenter.spin_manager

        # 1. 预测平移后的中心位置
        # P_pred = P_now + V * t
        pred_center = self.center + self.velocity * future_time_delta

        # 2. 预测自旋后的装甲板位置
        # 计算未来时刻的绝对 Yaw 角
        # 注意: accumulated_angle 是积分值，last_armor_yaw 是当前值
        # 我们使用 last_armor_yaw 作为基准推算
        total_yaw_pred = mgr.last_armor_yaw + mgr.omega * future_time_delta

        # 获取当前状态对应的物理半径 (假设短时间内状态不切换，或忽略切换带来的微小半径变化)
        # 这里传入 last_armor_yaw 主要是为了满足接口，实际上我们只关心状态对应的半径
        # 注意: dt 传 0 因为只是获取参数
        r, y_offset = mgr.predict_single_plate(mgr.last_armor_yaw, 1/30)

        # 3. 合成 3D 坐标
        # 假设装甲板在 XZ 平面上绕中心旋转
        # 坐标系: X右, Y下(高度), Z前
        # 旋转模型:
        #   x_offset = -r * sin(yaw)
        #   z_offset = -r * cos(yaw)
        #   (这个公式取决于之前的坐标系定义和法向量方向，这里沿用 get_predict_target 的推导)

        target_3d = np.array([
            pred_center[0] - r * math.sin(total_yaw_pred),
            pred_center[1] + y_offset, # 高度即使平移也加上 offset
            pred_center[2] - r * math.cos(total_yaw_pred)
        ])

        return target_3d

    def find_robot_center(self, dt):
        """
        计算机器人中心
        注意：此函数依赖 self.test_robot_center.robot_armor_coordinate
        该数据在调用本函数前已经被 use_robot_prediction 更新为 (滤波后 or 原始) 数据
        """
        self.armor_center_point.clear()

        # 根据可视装甲板数量选择解算策略
        if len(self.test_robot_center.robot_armor_coordinate) >= 2:
            raw_center = self.test_robot_center.get_robot_center_by_two_armor(dt)
        elif len(self.test_robot_center.robot_armor_coordinate) == 1:
            raw_center = self.test_robot_center.get_robot_center_by_one_armor(dt, 0)
        else:
            raw_center = None

        # 更新夹角信息
        self.angle_between_plates = self.test_robot_center.last_dual_angle_deg

        if raw_center is not None:
            # Z轴低通滤波：防止中心点前后跳动 (Depth 轴通常噪声最大)
            # Z_ALPHA 控制平滑程度，越小越平滑但滞后越大
            if self.z_filter_val is None:
                self.z_filter_val = raw_center[2]

            if abs(raw_center[2] - self.z_filter_val) > 1.0:
                # 突变过大直接跟随，防止跟丢快速移动物体
                raw_center[2] = self.z_filter_val
            else:
                # 正常范围内平滑
                self.z_filter_val = self.z_filter_val * (1 - self.Z_ALPHA) + raw_center[2] * self.Z_ALPHA
                raw_center[2] = self.z_filter_val
            self.center = raw_center

            # [新增] 更新速度估计
            self.update_velocity()
        else:
            self.center = None

        self.armor_center_point = self.test_robot_center.armor_center_point
        return self.center

    def get_another_armor_plate_center_by_center(self):
        """
        基于计算出的机器人中心，补全未观测到的虚拟装甲板
        用于辅助显示或进一步的逻辑判断
        """
        if self.center is None: return
        mgr = TestRobotCenter.spin_manager

        # 这里的 armor_center_point 来源于 find_robot_center 计算过程中记录的装甲板中心
        if len(self.armor_plates_camera_positions) == 1 and len(self.armor_center_point) >= 1:
            known = np.array(self.armor_center_point[0])

            # 1. 补全对面装甲板 (中心对称)
            opp = 2 * self.center - known
            opp[1] = known[1]  # 保持 Y 高度与观测板一致
            self.armor_center_point.append(opp.tolist())

            # 2. 补全侧面装甲板 (需判断长短板状态来决定半径和高度)
            # 计算当前观测板的旋转半径
            curr_r = np.linalg.norm([known[0] - self.center[0], known[2] - self.center[2]])

            # 判断当前板是 长板 还是 短板 (通过比较当前半径与模型参数)
            if abs(curr_r - mgr.long_radius) < abs(curr_r - mgr.short_radius):
                # 当前是长板 -> 侧板应为短板
                side_r = mgr.short_radius
                side_offset = mgr.short_y_offset
            else:
                # 当前是短板 -> 侧板应为长板
                side_r = mgr.long_radius
                side_offset = mgr.long_y_offset

            # 计算垂直方向向量 (XZ平面)
            vec = known - self.center
            vec_xz = np.array([vec[0], vec[2]])
            norm = np.linalg.norm(vec_xz)
            if norm < 1e-4: return

            # 垂直向量: (x, y) -> (-y, x)
            perp = np.array([-vec_xz[1], vec_xz[0]]) / norm

            # 侧板的 XZ 坐标 (+90度 和 -90度)
            s1_xz = np.array([self.center[0], self.center[2]]) + perp * side_r
            s2_xz = np.array([self.center[0], self.center[2]]) - perp * side_r

            # 侧板的高度 Y
            side_y = self.center[1] + side_offset

            self.armor_center_point.append([s1_xz[0], side_y, s1_xz[1]])
            self.armor_center_point.append([s2_xz[0], side_y, s2_xz[1]])

    def predict_shooting_times(self, current_pitch=0, current_yaw=0) -> List[float]:
        """
        [修改功能] 预测适合发射子弹的时间节点
        逻辑：
        1. 仅当画面中只有一块装甲板时触发。
        2. 以当前时刻和当前板的角度为基准。
        3. 预测未来旋转 90, 180, 270... 度 (90的倍数) 的时刻。
        4. 发射时间 = 预测到达时刻 - 延迟(self.T)。
        5. 生成未来 10 个预测点。
        """
        predicted_times = []
        self.predicted_shoot_points = [] # 清空上一帧预测点

        # 1. 触发条件：只出现一块装甲板
        if len(self.armor_plates_camera_positions) != 1:
            return []

        mgr = TestRobotCenter.spin_manager
        omega = mgr.omega

        # 必须有有效转速，否则无法预测周期
        if abs(omega) < 0.1:
            return []

        current_time = self.last_timestamp # 当前帧的绝对时间
        period_angle = np.pi / 2 # 90度

        # 2. 生成 10 个预测点 (90度, 180度 ... 900度)
        print(f"[Prediction] Base Time: {current_time - self.program_start_time:.4f}s | Omega: {omega:.3f} rad/s")
        print(f"             Velocity: ({self.velocity[0]:.2f}, {self.velocity[1]:.2f}, {self.velocity[2]:.2f}) m/s")

        for k in range(1, 11):
            # 需要转过的角度 (弧度)
            angle_to_rotate = k * period_angle

            # 需要的时间 (无论正反转，时间都是正的 distance/speed)
            time_to_rotate = angle_to_rotate / abs(omega)

            # 目标时刻 (装甲板到位时刻)
            target_arrival_time = current_time + time_to_rotate

            # 击打时刻 (提前量)
            shoot_time_abs = target_arrival_time - self.T
            predicted_times.append(shoot_time_abs)

            # 计算预测点坐标和旋转角度
            target_3d = self.get_predicted_target_position(time_to_rotate)
            yaw_rot, pitch_rot = 0, 0
            if target_3d is not None:
                yaw_rot, pitch_rot = self.get_rotation_angle(target_3d, current_pitch, current_yaw)
                p_str = f"({target_3d[0]:.3f}, {target_3d[1]:.3f}, {target_3d[2]:.3f})"
            else:
                p_str = "None"

            print(f"      - {k * 90} deg: Target={p_str} | Rot: Yaw={yaw_rot*180/math.pi:.2f}, Pitch={pitch_rot*180/math.pi:.2f} | Arrival={target_arrival_time - self.program_start_time:.3f}s")

        return predicted_times

    def get_rotation_angle(self, armor_pos, current_pitch, current_yaw):
        """
        计算需要旋转的角度
        :param armor_pos: 预测后的装甲板坐标 (相机坐标系)
        :param current_pitch: 当前 pitch (弧度)
        :param current_yaw: 当前 yaw (弧度)
        :return: (yaw, pitch) 需要旋转的角度 (绝对角度)
        """
        if armor_pos is None:
            return 0, 0

        # 转为云台坐标
        gimbal_pos = camera2gimbal(armor_pos, current_pitch)
        ax, ay, az = gimbal_pos

        # 计算 Pitch (使用弹道补偿)
        # ballistic_compensation 返回的是补偿后的目标 Pitch 角度
        target_pitch = ballistic_compensation(gimbal_pos)

        # 计算 Yaw
        # ax 是水平方向 (左为正)，az 是深度 (前为正)
        # math.atan2(y, x) -> math.atan2(ax, az) 返回的是相对于 Z 轴的偏角
        if az == 0:
            target_yaw = current_yaw
        else:
            relative_yaw = math.atan2(ax, az)
            target_yaw = current_yaw + relative_yaw

        # 归一化 Yaw 到 [-pi, pi]
        while target_yaw < -math.pi: target_yaw += 2 * math.pi
        while target_yaw > math.pi: target_yaw -= 2 * math.pi

        return target_yaw, target_pitch

    def predict_and_send_target(self, vision_manager, time_threshold, current_pitch, current_yaw):
        """
        [修改功能] 预测多个未来节点，从中筛选出满足时间阈值的最近点，并发送给电控
        :param vision_manager: 视觉通信管理器实例 (VisionData_t)
        :param time_threshold: 最小时间延迟阈值 (秒, 例如 1.0)
        :param current_pitch: 当前云台 Pitch
        :param current_yaw: 当前云台 Yaw
        :return: (yaw_deg, pitch_deg, target_3d) 选中点的参数
        """
        # 1. 触发条件：单板且有有效转速
        if len(self.armor_plates_camera_positions) != 1:
            return None, None, None

        mgr = TestRobotCenter.spin_manager
        omega = mgr.omega
        if abs(omega) < 0.1:
            return None, None, None

        # 2. 遍历预测点 (90度步进, 最多10个点)
        target_time_offset = None
        target_3d = None
        found_target = False

        period_angle = np.pi / 2 # 90度

        for k in range(1, 11):
            # 需要转过的角度 (弧度)
            angle_to_rotate = k * period_angle
            # 对应的未来时间增量
            time_to_rotate = angle_to_rotate / abs(omega)

            # [筛选]: 找到第一个大于时间阈值的点
            if time_to_rotate > time_threshold:
                target_time_offset = time_to_rotate
                # 获取该时刻的三维坐标
                target_3d = self.get_predicted_target_position(target_time_offset)
                found_target = True
                print(f"[Send] Selected Prediction: k={k} ({k*90} deg), Time={target_time_offset:.3f}s (> {time_threshold}s)")
                break

        if not found_target:
            # 如果都不满足，可能转速太慢，或策略选择放弃
            # 这里可以选择返回最远的第10个点，或者不做任何操作
            # 策略：不发送
            return None, None, None

        if target_3d is None:
            return None, None, None

        # 4. 计算需要的云台角度 (绝对角度)
        target_yaw_rad, target_pitch_rad = self.get_rotation_angle(target_3d, current_pitch, current_yaw)

        # 计算相对旋转量 (用于返回显示 "need rotate angle")
        diff_yaw_rad = target_yaw_rad - current_yaw
        # 归一化 diff_yaw 到 [-pi, pi]
        while diff_yaw_rad < -math.pi: diff_yaw_rad += 2 * math.pi
        while diff_yaw_rad > math.pi: diff_yaw_rad -= 2 * math.pi

        diff_pitch_rad = target_pitch_rad - current_pitch

        # 5. 计算距离
        distance = np.linalg.norm(target_3d)

        # 6. 发送数据
        # 参考 main_cam 系统逻辑:
        # Yaw 发送绝对值
        # Pitch 发送 -(target - current) = current - target (可能是符号约定)
        send_yaw = target_yaw_rad
        send_pitch = -(target_pitch_rad - current_pitch)

        if vision_manager:
            vision_manager.set_data(send_yaw, send_pitch, distance, 1, 1, 0)
            vision_manager.send()

        # 7. 返回结果供打印 (转换为角度制，返回需要旋转的角度增量)
        yaw_deg_diff = math.degrees(diff_yaw_rad)
        pitch_deg_diff = math.degrees(diff_pitch_rad)

        return yaw_deg_diff, pitch_deg_diff, target_3d

    def use_robot_prediction(self, current_pitch=0, current_yaw=0, vision_manager=None):
        """
        [主入口函数] 流程控制
        每帧图像处理时调用此函数。

        流程:
        1. 计算帧间隔 dt
        2. (可选) 对所有装甲板角点进行卡尔曼滤波 (KF)
        3. 调用几何解算计算机器人中心
        4. 生成虚拟装甲板用于展示
        5. (可选) 调用预测并发送给电控
        """
        if vision_manager:
            self.vision_manager = vision_manager

        if not self.cal_armor(): return

        # --- 1. 计算时间步长 dt ---
        current_time = time.time()
        dt = current_time - self.last_timestamp
        self.last_timestamp = current_time
        # 限制 dt 防止首帧或卡顿导致发散 (假设最低 10fps -> 0.1s)
        if dt > 0.1: dt = 0.015
        if dt < 0.001: dt = 0.001

        # --- 2. 准备解算数据 (KF 滤波分支) ---
        target_coordinates = self.armor_plates_camera_positions

        if self.enable_kf:
            # 如果开启 KF，则计算平滑后的坐标
            # [关键]: 这一步对应 "对所有的装甲板的所有点都执行kf"
            # _process_kf 会返回经过 6D 模型平滑后的点坐标 (x,y,z)
            target_coordinates = self._process_kf(self.armor_plates_camera_positions, dt)

        # 将准备好的数据 (Raw 或 Filtered) 注入给解算器
        self.test_robot_center.robot_armor_coordinate = target_coordinates

        # --- 3. 计算中心 ---
        self.find_robot_center(dt)

        # --- 4. 补全虚拟板 ---
        if self.center is not None:
            self.get_another_armor_plate_center_by_center()

            # [新增] 调用射击时间预测
            self.predict_shooting_times(current_pitch, current_yaw)

            # [新增] 调用预测并发送给电控
            if self.vision_manager:
                 self.predict_and_send_target(self.vision_manager, self.prediction_time_threshold, current_pitch, current_yaw)

            # 调试用的打印 (可选)
            # if len(target_coordinates) >= 2:
            #     print(f"Angle: {self.angle_between_plates:.1f}, KF: {self.enable_kf}")