from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time
import math
from collections import deque

# 假设这些类定义在 all_type 模块中
from all_type import ArmorPlate, Color, TroopType

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
        self.long_radius = 0.25
        self.short_radius = 0.20
        # 初始设定的高度偏移
        self.long_y_offset = 0.0
        self.short_y_offset = 0.0

        # --- 状态机 ---
        self.current_state = 0  # 0 = 正在面对长板面, 1 = 正在面对短板面 (或者对应状态索引)

        # --- 积分与预测 ---
        self.accumulated_angle = 0.0 # 累积旋转角度，用于状态切换预测
        self.last_update_time = time.time()
        self.last_armor_yaw = 0.0 # 上一帧的装甲板 Yaw 角

        self.omega = 0.0  # 估算的旋转角速度 (rad/s)
        self.omega_alpha = 0.2 # 角速度低通滤波系数
        self.is_initialized = False # 是否已完成初次双板校准

    def update_dual_plate(self, r1, r2, y1, y2, c1_x, c2_x, current_yaw, center_y):
        """
        [双板模式 - 绝对校准]
        当视觉能同时看到两个装甲板时调用此函数。
        利用两个装甲板的几何关系直接计算并更新机器人的物理参数（长短半径、高度差）。

        参数:
        r1, r2: 两个装甲板到解算中心的距离
        y1, y2: 两个装甲板的高度(Y坐标)
        c1_x, c2_x: 两个装甲板的 X 坐标 (用于判断左右关系)
        current_yaw: 当前装甲板朝向角
        center_y: 机器人中心高度
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # 1. 识别长短板：假设距离中心更远的是长半径面，近的是短半径面
        if r1 > r2:
            l_r, s_r = r1, r2
            l_y, s_y = y1, y2
        else:
            l_r, s_r = r2, r1
            l_y, s_y = y2, y1

        # 2. 更新结构参数 (平滑滤波)：使用指数移动平均平滑参数震荡
        alpha = 0.1 #if self.is_initialized else 1.0 # 初次直接赋值，后续平滑更新
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

        if dt > 0.001:
            raw_omega = diff_yaw / dt
            # 简单的异常值过滤，只更新合理的转速
            if abs(raw_omega) < 15.0:
                self.omega = self.omega * (1 - self.omega_alpha) + raw_omega * self.omega_alpha

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

    def predict_single_plate(self, current_yaw):
        """
        [单板模式 - 预测]
        当只看到一个装甲板时，无法直接得知这是长板侧还是短板侧。
        利用角速度积分推算当前转到了哪一面。

        参数:
        current_yaw: 当前观测到的装甲板 Yaw 角

        返回:
        (预测半径, 预测高度偏移)
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time

        # 计算 yaw 变化量
        diff_yaw = current_yaw - self.last_armor_yaw
        while diff_yaw > np.pi: diff_yaw -= 2 * np.pi
        while diff_yaw < -np.pi: diff_yaw += 2 * np.pi

        # 持续更新角速度估计
        if dt > 0.001:
            raw_omega = diff_yaw / dt
            if abs(raw_omega) > 0.2 and abs(raw_omega) < 15.0:
                self.omega = self.omega * (1 - self.omega_alpha) + raw_omega * self.omega_alpha

        self.last_armor_yaw = current_yaw

        # 积分角度，用于推测是否切换了板面
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

    def get_robot_center_by_two_armor(self, idx1=0, idx2=1):
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
            r1, r2, c1[1], c2[1], c1[0], c2[0], yaw, center_y
        )
        return np.array([center_xz[0], center_y, center_xz[1]], dtype=float)

    def get_robot_center_by_one_armor(self, idx=0):
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
        pred_r, pred_offset = TestRobotCenter.spin_manager.predict_single_plate(curr_yaw)

        # 2. XZ 平面上回退半径距离
        normal_xz = np.array([normal_unit[0], normal_unit[2]], dtype=float)
        norm_xz = np.linalg.norm(normal_xz)

        if norm_xz > 1e-4:
            normal_xz /= norm_xz # 归一化
            # 中心 = 装甲板中心 - 法向量 * 半径
            center_xz = np.array([center_armor[0], center_armor[2]]) - normal_xz * pred_r
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
                # init_cov=1000: 初始协方差大，表示初始不确定性高，快速收敛到测量值
                # process_noise=5000: 过程噪声大，表示相信物体的运动是多变的，不过分依赖模型预测 (对抗滞后)
                # measure_noise=0.1: 测量噪声小，表示比较相信视觉测量的当前位置
                self.kf_map[idx] = [
                    KalmanFilter6D(measure_dim=3, init_cov=1000.0, measure_noise=0.1, process_noise=5000.0,
                                   x=p[0], y=p[1], z=p[2])
                    for p in points
                ]

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

    def find_robot_center(self):
        """
        计算机器人中心
        注意：此函数依赖 self.test_robot_center.robot_armor_coordinate
        该数据在调用本函数前已经被 use_robot_prediction 更新为 (滤波后 or 原始) 数据
        """
        self.armor_center_point.clear()

        # 根据可视装甲板数量选择解算策略
        if len(self.test_robot_center.robot_armor_coordinate) >= 2:
            raw_center = self.test_robot_center.get_robot_center_by_two_armor()
        elif len(self.test_robot_center.robot_armor_coordinate) == 1:
            raw_center = self.test_robot_center.get_robot_center_by_one_armor(0)
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

    def use_robot_prediction(self):
        """
        [主入口函数] 流程控制
        每帧图像处理时调用此函数。

        流程:
        1. 计算帧间隔 dt
        2. (可选) 对所有装甲板角点进行卡尔曼滤波 (KF)
        3. 调用几何解算计算机器人中心
        4. 生成虚拟装甲板用于展示
        """
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
        self.find_robot_center()

        # --- 4. 补全虚拟板 ---
        if self.center is not None:
            self.get_another_armor_plate_center_by_center()

            # 调试用的打印 (可选)
            # if len(target_coordinates) >= 2:
            #     print(f"Angle: {self.angle_between_plates:.1f}, KF: {self.enable_kf}")