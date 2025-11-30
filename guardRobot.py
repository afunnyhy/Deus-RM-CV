from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from all_type import ArmorPlate, Color, TroopType
from KalmanFilter import KalmanFilter as KF


class GuardRobot:
    def __init__(self, all_armor_plate: List):
        """根据若干装甲板估计小车相关几何信息（相机坐标系）。

        参数
        ------
        all_armor_plate : List[ArmorPlate]
            当前帧中可用的装甲板列表，每个 ArmorPlate 中应包含 4 个在相机坐标系下的 3D 角点 camera_pos。

        约定
        ------
        - 至少需要 2 块装甲板，几何上才能稳定估计出"小车中心"等信息；
        - 目前 get_center_from_normals 和 calculate_another_armor_by_center 只使用前两块装甲板；
        - 坐标系为相机坐标系：x 向右, y 向下, z 朝前（与工程中 PnP 得到的一致）。
        """
        if len(all_armor_plate) < 1:
            raise ValueError("all_armor_plate 至少需要 1 块装甲板")
        self.armor_plate = all_armor_plate

        # 代表整车中心的 3D 点（相机坐标系）
        self.center_point = None
        # 两个装甲板法平面的交线上的一点以及方向（目前保留, 用于旧接口）
        self.line_point = None
        self.line_dir = None
        # 每块装甲板所在平面的方程参数 [a,b,c,d]（ax+by+cz+d=0）
        self.normal_plane = []
        # 每块装甲板中心（4 个角点的算术平均）
        self.armor_plate_center = []
        # 每块装甲板的法向量（通过 3D 角点叉乘得到）
        self.normal_vector = []
        
        # 装甲板高度到旋转半径的映射
        self.height_to_radius = {}
        
        # 存储首次观测到的两个装甲板的半径，用于后续预测
        self.recorded_radii = []
        
        # 记录不同装甲板的尺寸信息：{height: (armor_type, length, width, top_z_diff, bottom_z_diff)}
        self.armor_dimensions = {}
        
        # 装甲板追踪器字典
        self.armor_trackers: Dict[int, Dict[str, Any]] = {}
        
        # 卡尔曼滤波器参数
        self.corner_kf_init_cov = 1e3
        self.corner_kf_measure_noise = 0.15
        self.corner_kf_process_noise = 0.2
        self.max_miss_frames = 8
        
        # 装甲板ID映射，用于追踪装甲板
        self.armor_id_mapping = {}  # 用于跟踪装甲板的身份
        
        # 帧计数器
        self.frame_count = 0
        
        # === SENSITIVITY IMPROVEMENT SUGGESTION ===
        # 添加配置参数以改善半径计算和选择的敏感性
        # 1. 高度匹配容差：用于确定两个装甲板是否属于同一高度组
        self.height_tolerance = 0.05  # 可根据实际情况调整
        # 2. 半径变化率限制：防止由于噪声导致的半径剧烈变化
        self.max_radius_change_rate = 0.01  # 最大10%的变化率
        # 3. 平滑因子：用于平滑半径计算的历史权重
        self.radius_smoothing_factor = 0.2
        # 范围0-1，越大越依赖历史值

    def update_armor_plates(self, new_armor_plates: List):
        """
        动态更新装甲板信息
        
        参数:
        new_armor_plates: 新的装甲板列表
        """
        self.frame_count += 1
        self.armor_plate = new_armor_plates
        
        # 清除旧的计算结果
        self.armor_plate_center.clear()
        self.normal_vector.clear()
        self.normal_plane.clear()
        
        # 重新计算装甲板中心和法向量
        if len(new_armor_plates) > 0:
            self.find_armor_center()
            self.find_armor_normal_vector()
            self.find_armor_normal_plane()

    def match_armor_plates(self, new_armor_plates: List):
        """
        匹配新旧装甲板，维护装甲板身份
        
        参数:
        new_armor_plates: 新检测到的装甲板列表
        
        返回:
        matched_pairs: 匹配对列表 [(old_index, new_index), ...]
        unmatched_new: 未匹配的新装甲板索引列表
        unmatched_old: 未匹配的旧装甲板索引列表
        """
        if not self.armor_plate:
            # 如果之前没有装甲板，所有新装甲板都是未匹配的
            return [], list(range(len(new_armor_plates))), []
        
        matched_pairs = []
        unmatched_new = list(range(len(new_armor_plates)))
        unmatched_old = list(range(len(self.armor_plate)))
        
        # 简单的基于中心点x坐标的匹配
        for new_idx, new_armor in enumerate(new_armor_plates):
            new_center_x = self.get_center_x(new_armor)
            
            best_match_idx = None
            best_distance = float('inf')
            
            for old_idx in unmatched_old[:]:  # 使用副本避免修改循环中的列表
                old_armor = self.armor_plate[old_idx]
                old_center_x = self.get_center_x(old_armor)
                
                distance = abs(new_center_x - old_center_x)
                if distance < best_distance and distance < 50:  # 阈值可根据需要调整
                    best_distance = distance
                    best_match_idx = old_idx
            
            if best_match_idx is not None:
                matched_pairs.append((best_match_idx, new_idx))
                unmatched_old.remove(best_match_idx)
                unmatched_new.remove(new_idx)
        
        return matched_pairs, unmatched_new, unmatched_old

    # === SENSITIVITY IMPROVEMENT SUGGESTION ===
    # ISSUE: 匹配过程仅基于x坐标，忽略了y和z坐标信息，且阈值固定
    # POTENTIAL PROBLEMS:
    # 1. 当装甲板在x方向重叠时，容易产生错误匹配
    # 2. 固定阈值不能适应不同距离下的匹配需求
    # SOLUTION SUGGESTIONS:
    # 1. 使用三维空间距离而非单一维度距离
    # 2. 实现自适应阈值，根据距离动态调整匹配阈值
    # 3. 引入外观特征匹配（如颜色、纹理）提高匹配准确性

    def find_armor_normal_vector(self):
        """根据装甲板的 4 个 3D 角点求平面法向量。

        约定 camera_pos 点序：
        - [0] top_left
        - [1] bottom_left
        - [2] top_right
        - [3] bottom_right

        法向量计算
        ------
        使用 top_left->top_right 与 top_left->bottom_left 的叉乘：
            n = (top_right - top_left) × (bottom_left - top_left)
        然后单位化。
        """
        self.normal_vector.clear()
        for armor in self.armor_plate:
            top_left = np.asarray(armor.camera_pos[0], dtype=np.float32)
            top_right = np.asarray(armor.camera_pos[2], dtype=np.float32)
            bottom_left = np.asarray(armor.camera_pos[1], dtype=np.float32)
            # bottom_right = np.asarray(armor.camera_pos[3], dtype=np.float32)

            # 由两个非共线边得到平面法向量
            n = np.cross(top_right - top_left, bottom_left - top_left)
            if np.linalg.norm(n) <= 1e-5:
                # 如果三点几乎共线, 无法定义一个稳定的法向量
                raise ValueError("法向量计算异常，可能是点共线")
            n = n / np.linalg.norm(n)
            self.normal_vector.append(n)

    def find_armor_center(self):
        """根据 4 个 3D 角点求装甲板中心点（简单平均）。

        注意
        ------
        - 这里的中心只是几何上的重心, 不带任何物理含义；
        - 该中心常被用作"装甲板中心 -> 车中心"的连线起点。
        """
        self.armor_plate_center.clear()
        for armor in self.armor_plate:
            top_left = np.asarray(armor.camera_pos[0], dtype=np.float32)
            top_right = np.asarray(armor.camera_pos[2], dtype=np.float32)
            bottom_left = np.asarray(armor.camera_pos[1], dtype=np.float32)
            bottom_right = np.asarray(armor.camera_pos[3], dtype=np.float32)
            center = (top_left + top_right + bottom_left + bottom_right) / 4.0
            self.armor_plate_center.append(center)

    def find_armor_normal_plane(self):
        """对每块装甲板, 根据其中心点和法向量构造平面方程 ax+by+cz+d=0。"""
        self.normal_plane.clear()

        def plane_from_point_normal(point, normal, normalize=True):
            point = np.asarray(point, dtype=float).reshape(3)
            normal = np.asarray(normal, dtype=float).reshape(3)
            if normalize:
                n_norm = np.linalg.norm(normal)
                if n_norm < 1e-12:
                    raise ValueError("法向量长度为 0，无法确定平面")
                normal = normal / n_norm
            a, b, c = normal
            # 令点在平面上满足: a*x0 + b*y0 + c*z0 + d = 0 -> d = -n·p
            d = -np.dot(normal, point)
            return [a, b, c, d]

        for center, n in zip(self.armor_plate_center, self.normal_vector):
            self.normal_plane.append(plane_from_point_normal(center, n))

    def intersect_two_planes(self, p1, p2, eps=1e-8):
        """求两平面交线的一点和方向。

        参数
        ------
        p1, p2 : [a,b,c,d]
            两个平面方程 ax+by+cz+d=0 的参数。

        返回
        ------
        (has_line, point, direction)
            has_line : bool, 是否存在唯一交线；
            point : ndarray(3,), 交线上的一点；
            direction : ndarray(3,), 交线方向单位向量。
        """
        a1, b1, c1, d1 = p1
        a2, b2, c2, d2 = p2

        n1 = np.array([a1, b1, c1], dtype=float)
        n2 = np.array([a2, b2, c2], dtype=float)

        # 交线方向 = 两法线叉乘
        direction = np.cross(n1, n2)
        if np.linalg.norm(direction) < eps:
            # 法向量平行 -> 平面平行或重合, 没有唯一交线
            return False, None, None

        direction = direction / np.linalg.norm(direction)

        # 在三维中, 任意两平面交线可以通过给某个坐标分量固定值, 解 2x2 线性方程得到一个交点。
        def solve_with_fixed(var_idx, var_value):
            # var_idx: 固定的坐标索引 (0:x,1:y,2:z)
            unknown_idx = [i for i in range(3) if i != var_idx]

            # 平面1: n1·[x,y,z] + d1 = 0
            coeffs1 = [n1[unknown_idx[0]], n1[unknown_idx[1]]]
            rhs1 = -d1 - n1[var_idx] * var_value

            # 平面2: n2·[x,y,z] + d2 = 0
            coeffs2 = [n2[unknown_idx[0]], n2[unknown_idx[1]]]
            rhs2 = -d2 - n2[var_idx] * var_value

            A = np.array([coeffs1, coeffs2], dtype=float)
            b_vec = np.array([rhs1, rhs2], dtype=float)

            if abs(np.linalg.det(A)) < eps:
                # 该固定方式下矩阵不可逆, 换一个坐标固定
                return None

            sol = np.linalg.solve(A, b_vec)
            vals = {var_idx: var_value, unknown_idx[0]: sol[0], unknown_idx[1]: sol[1]}
            x, y, z = vals[0], vals[1], vals[2]
            return np.array([x, y, z], dtype=float)

        # 依次尝试固定 z=0, y=0, x=0 求解一个交点
        point = solve_with_fixed(2, 0.0)
        if point is None:
            point = solve_with_fixed(1, 0.0)
        if point is None:
            point = solve_with_fixed(0, 0.0)
        if point is None:
            return False, None, None

        return True, point, direction

    def get_line_point_and_dir(self):
        """返回两装甲板法平面的交线的一点和方向向量（相机坐标系）。

        使用前两块装甲板：
        1. 计算每块装甲板的法向量；
        2. 计算每块装甲板的中心点；
        3. 由中心点和法向量构造两个平面方程；
        4. 求出两平面交线。

        返回
        ------
        (point, direction)
            point : ndarray(3,), 交线上任意一点;
            direction : ndarray(3,), 交线方向单位向量。
        """
        # 只保留前两块装甲板
        if len(self.armor_plate) >= 2:
            armors = self.armor_plate[:2]
            self.armor_plate = armors

        self.find_armor_normal_vector()
        self.find_armor_center()
        self.find_armor_normal_plane()

        has_line, point_on_line, direction = self.intersect_two_planes(
            self.normal_plane[0], self.normal_plane[1]
        )
        if not has_line or point_on_line is None or direction is None:
            raise ValueError("两平面没有唯一交线，无法计算交线")

        self.line_point = point_on_line
        self.line_dir = direction
        return self.line_point, self.line_dir

    def get_center_from_normals(self):
        """在 xz 平面中用两块装甲板法向量的投影求交点作为小车中心。"""
        # 只保留前两块装甲板
        if len(self.armor_plate) >= 2:
            armors = self.armor_plate[:2]
            self.armor_plate = armors
        self.find_armor_normal_vector()
        self.find_armor_center()

        if len(self.armor_plate_center) < 2 or len(self.normal_vector) < 2:
            raise ValueError("至少需要两块装甲板来计算中心")

        C1 = np.asarray(self.armor_plate_center[0], dtype=float).reshape(3)
        C2 = np.asarray(self.armor_plate_center[1], dtype=float).reshape(3)
        n1 = np.asarray(self.normal_vector[0], dtype=float).reshape(3)
        n2 = np.asarray(self.normal_vector[1], dtype=float).reshape(3)

        # 投影到 xz 平面: 只保留 x,z 分量
        p1 = C1[[0, 2]]  # (x1, z1)
        p2 = C2[[0, 2]]  # (x2, z2)
        v1 = n1[[0, 2]]  # (vx1, vz1)
        v2 = n2[[0, 2]]  # (vx2, vz2)

        # 避免方向退化
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            raise ValueError("法向量在 xz 平面上的投影长度过小，无法定义二维直线")

        # 二维中两条直线:
        # L1: p1 + t1 * v1
        # L2: p2 + t2 * v2
        # 解 p1 + t1*v1 = p2 + t2*v2
        A = np.array([[v1[0], -v2[0]],
                      [v1[1], -v2[1]]], dtype=float)
        b = (p2 - p1).astype(float)

        det = np.linalg.det(A)
        if abs(det) < 1e-8:
            # 二维方向近似平行，退化为两中心点在 xz 平面投影的中点
            inter_xz = 0.5 * (p1 + p2)
        else:
            t1, t2 = np.linalg.solve(A, b)
            inter_xz = p1 + t1 * v1

        center = np.array([inter_xz[0], inter_xz[1]], dtype=np.float32)

        self.center_point = center
        return self.center_point

    def predict_center_from_single_armor(self, visible_armor_index=0):
        """
        当只看到一块装甲板时，根据装甲板位置和记录的半径预测小车中心点
        
        参数:
        visible_armor_index: 当前可见的装甲板索引，默认为0
        
        返回:
        ndarray: 预测的小车中心点 [x, z]
        """
        # 检查是否已记录初始半径
        if len(self.recorded_radii) < 2:
            raise ValueError("尚未记录初始半径，请先调用record_initial_radii方法")
            
        # 确保有足够的装甲板信息
        if len(self.armor_plate) < 1:
            raise ValueError("至少需要一块装甲板来进行预测")
            
        # 直接计算装甲板中心点（而不是获取已有的中心点）
        visible_armor = self.armor_plate[visible_armor_index]
        pts = np.asarray(visible_armor.camera_pos, dtype=float).reshape(-1, 3)
        if pts.shape[0] != 4:
            raise ValueError("ArmorPlate.camera_pos 必须是4个3D角点")
            
        # 计算装甲板中心点
        visible_center = np.mean(pts, axis=0)
        
        # 计算装甲板法向量（指向车外）
        if len(self.normal_vector) <= visible_armor_index:
            self.find_armor_normal_vector()
            
        normal = self.normal_vector[visible_armor_index]
        
        # 获取装甲板高度（直接从当前装甲板中心点获取高度）
        armor_height = visible_center[1]
        
        # 根据装甲板高度选择匹配的半径（与记录的装甲板高度进行比较）
        matched_radius = None
        min_height_diff = float('inf')
        for height, radius in self.height_to_radius.items():
            height_diff = abs(height - armor_height)
            if height_diff < min_height_diff:
                min_height_diff = height_diff
                matched_radius = radius
                
        # 如果没有找到匹配的高度，则使用第一个记录的半径
        if matched_radius is None:
            matched_radius = self.recorded_radii[0]
            
        # 计算小车中心点（装甲板中心沿着法向量反方向移动半径距离）
        center_x = visible_center[0] + normal[0] * matched_radius
        center_z = visible_center[2] + normal[2] * matched_radius
        
        # 更新中心点
        self.center_point = np.array([center_x, center_z], dtype=np.float32)
        
        return self.center_point

    # === SENSITIVITY IMPROVEMENT SUGGESTION ===
    # ISSUE: 中心点预测完全依赖于单一装甲板的信息和预估的半径
    # POTENTIAL PROBLEMS:
    # 1. 单一装甲板可能因为遮挡或识别错误导致预测偏差较大
    # 2. 半径估算的误差会直接影响中心点位置的准确性
    # SOLUTION SUGGESTIONS:
    # 1. 引入多帧数据融合，增加预测的稳定性
    # 2. 添加异常值检测机制，过滤明显错误的预测结果
    # 3. 实现自适应半径校准机制，根据历史预测结果动态调整

    def calculate_height_to_radius_mapping(self):
        """
        根据装甲板的高度计算旋转半径。
        相邻装甲板的高度不同，对侧装甲板的高度相同。
        生成一个键值对，键为装甲板的高度，值为对应方向的旋转半径。
        """
        self.height_to_radius.clear()
        
        if len(self.armor_plate_center) == 0:
            self.find_armor_center()
            
        # 计算小车中心点
        if self.center_point is None:
            self.get_center_from_normals()
            
        center_x, center_z = self.center_point[0], self.center_point[1]
        
        # 为每块装甲板计算高度和对应的旋转半径
        for i, armor in enumerate(self.armor_plate):
            # 获取装甲板中心点
            armor_center = self.armor_plate_center[i] if i < len(self.armor_plate_center) else None
            if armor_center is None:
                self.find_armor_center()
                armor_center = self.armor_plate_center[i]
                
            # 装甲板高度（y坐标）
            armor_height = armor_center[1]
            
            # 计算装甲板到中心点的距离（旋转半径）
            dx = armor_center[0] - center_x
            dz = armor_center[2] - center_z
            radius = np.sqrt(dx*dx + dz*dz)
            
            # 将高度和半径添加到映射中
            self.height_to_radius[armor_height] = radius
            
        return self.height_to_radius

    # === SENSITIVITY IMPROVEMENT SUGGESTION ===
    # ISSUE: 当前高度匹配使用绝对差值比较，可能导致精度问题
    # SOLUTION SUGGESTION: 
    # 1. 添加可配置的高度容差阈值
    # 2. 使用相对误差而非绝对误差进行匹配
    # 3. 实现更复杂的匹配策略（如聚类算法）

    def record_initial_radii(self):
        """
        在第一次出现两块装甲板时记录它们的半径
        """
        # 确保至少有两块装甲板
        if len(self.armor_plate) < 2:
            raise ValueError("至少需要两块装甲板来记录初始半径")
            
        # 计算装甲板中心
        if len(self.armor_plate_center) == 0:
            self.find_armor_center()
            
        # 计算小车中心点
        if self.center_point is None:
            self.get_center_from_normals()
            
        center_x, center_z = self.center_point[0], self.center_point[1]
        
        # 记录前两块装甲板的半径
        self.recorded_radii.clear()
        self.armor_dimensions.clear()
        for i in range(min(2, len(self.armor_plate))):
            armor = self.armor_plate[i]
            armor_center = self.armor_plate_center[i]
            dx = armor_center[0] - center_x
            dz = armor_center[2] - center_z
            radius = np.sqrt(dx*dx + dz*dz)
            self.recorded_radii.append(radius)
            
            # 记录装甲板尺寸信息
            armor_height = armor_center[1]
            armor_type = armor.armor_type
            
            # 计算装甲板的长度、宽度和z方向的差异
            pts = np.asarray(armor.camera_pos, dtype=float).reshape(-1, 3)
            top_points = pts[[0, 2]]  # top_left, top_right
            bottom_points = pts[[1, 3]]  # bottom_left, bottom_right
            
            # 计算装甲板在z方向上的差异
            top_z_avg = np.mean(top_points[:, 2])
            bottom_z_avg = np.mean(bottom_points[:, 2])
            z_diff = abs(top_z_avg - bottom_z_avg)
            
            # 计算装甲板的长度和宽度
            length = np.linalg.norm(pts[2] - pts[0])  # top_right to top_left
            width = np.linalg.norm(pts[1] - pts[0])   # bottom_left to top_left
            
            # 存储装甲板尺寸信息
            self.armor_dimensions[armor_height] = (armor_type, length, width, z_diff)
            
        # 同时计算高度到半径的映射
        self.calculate_height_to_radius_mapping()
            
        return self.recorded_radii

    # === SENSITIVITY IMPROVEMENT SUGGESTION ===
    # ISSUE: 当前代码中装甲板高度匹配使用简单的最近邻匹配
    # SOLUTION SUGGESTION:
    # 1. 引入高度容差阈值，避免因微小测量误差导致错误匹配
    # 2. 添加更稳健的统计方法（如滑动窗口平均）来提高稳定性
    # 3. 实现缓存机制以减少重复计算

    def predict_other_armors(self, visible_armor_index=0):
        """
        当只看到一块装甲板时，根据之前记录的两个半径预测其他装甲板的位置
        
        参数:
        visible_armor_index: 当前可见的装甲板索引，默认为0
        
        返回:
        list: 预测的装甲板列表（不包括当前可见装甲板）
        """
        # 检查是否已记录初始半径
        if len(self.recorded_radii) < 2:
            raise ValueError("尚未记录初始半径，请先调用record_initial_radii方法")
            
        # 确保有足够的装甲板信息
        if len(self.armor_plate) < 1:
            raise ValueError("至少需要一块装甲板来进行预测")
            
        # 计算装甲板中心
        if len(self.armor_plate_center) == 0:
            self.find_armor_center()
            
        # 当只有一块装甲板时，根据装甲板位置和记录的半径推算小车中心点
        visible_armor = self.armor_plate[visible_armor_index]
        # 直接计算装甲板中心点（而不是获取已有的中心点）
        pts = np.asarray(visible_armor.camera_pos, dtype=float).reshape(-1, 3)
        if pts.shape[0] != 4:
            raise ValueError("ArmorPlate.camera_pos 必须是4个3D角点")
            
        # 计算装甲板中心点
        visible_center = np.mean(pts, axis=0)
        
        # 计算装甲板法向量（指向车外）
        if len(self.normal_vector) <= visible_armor_index:
            self.find_armor_normal_vector()
            
        normal = self.normal_vector[visible_armor_index]
        
        # 获取装甲板高度（直接从当前装甲板中心点获取高度）
        armor_height = visible_center[1]
        
        # 根据装甲板高度选择匹配的半径（与记录的装甲板高度进行比较）
        matched_radius = None
        min_height_diff = float('inf')
        for height, radius in self.height_to_radius.items():
            height_diff = abs(height - armor_height)
            if height_diff < min_height_diff:
                min_height_diff = height_diff
                matched_radius = radius
                
        # 如果没有找到匹配的高度，则使用第一个记录的半径
        if matched_radius is None:
            matched_radius = self.recorded_radii[0]
            
        # 计算小车中心点（装甲板中心沿着法向量反方向移动半径距离）
        center_x = visible_center[0] + normal[0] * matched_radius
        center_z = visible_center[2] + normal[2] * matched_radius
        
        # 更新中心点
        self.center_point = np.array([center_x, center_z], dtype=np.float32)
        
        # 计算可见装甲板的角度（相对于推算出的中心点）
        dx_visible = visible_center[0] - center_x
        dz_visible = visible_center[2] - center_z
        angle_visible = np.arctan2(dz_visible, dx_visible)
        
        # 预测其他装甲板的位置
        predicted_armors = []
        
        # 对于RM机器人，通常有4个装甲板，所以我们预测3个其他装甲板
        # 使用记录的两个半径进行预测
        for i in range(3):  # 预测3个装甲板
            # 计算预测角度（假设装甲板按标准布局分布）
            # RM机器人装甲板分布: 前(0°), 左(90°), 后(180°), 右(270°)
            angle_offset = np.pi / 2 * (i + 1)  # 90度间隔
            pred_angle = angle_visible + angle_offset
            
            # 根据装甲板布局规律选择正确的半径
            # 相邻装甲板(90度和270度方向)使用另一个半径（高度不同）
            # 对面装甲板(180度方向)使用相同的半径（高度相同）
            # 使用在计算中心点时记录的高度-半径映射关系
            current_height = visible_center[1]
            
            # 根据预测装甲板的位置选择合适的半径和高度
            if i == 0 or i == 2:  # 90度和270度方向（相邻装甲板，高度不同）
                # 查找与当前装甲板高度不同的半径和高度
                best_radius = self.recorded_radii[0]
                best_height = current_height  # 默认使用当前高度
                height_diff_max = 0
                
                for height, radius in self.height_to_radius.items():
                    height_diff = abs(height - current_height)
                    if height_diff > height_diff_max and height_diff > 1e-6:  # 确保不是同一个高度
                        height_diff_max = height_diff
                        best_radius = radius
                        best_height = height  # 使用匹配到的不同高度
                        
                radius = best_radius
                pred_y = best_height  # 使用不同高度
            else:  # 180度方向（对面装甲板，高度相同）
                # 查找与当前装甲板高度相同的半径
                best_radius = self.recorded_radii[0]
                best_height = current_height  # 默认使用当前高度
                min_height_diff = float('inf')
                
                for height, radius in self.height_to_radius.items():
                    height_diff = abs(height - current_height)
                    if height_diff < min_height_diff:
                        min_height_diff = height_diff
                        best_radius = radius
                        best_height = height  # 确认使用相同高度
                        
                radius = best_radius
                pred_y = best_height  # 使用相同高度
            
            # 计算预测位置
            pred_x = center_x + radius * np.cos(pred_angle)
            pred_z = center_z + radius * np.sin(pred_angle)
            
            # 根据装甲板类型重构装甲板角点
            pred_center_3d = np.array([pred_x, pred_y, pred_z])
            
            # 对于相邻装甲板（i == 0 or i == 2），使用新的重构函数
            # 对于对面装甲板（i == 1），可以使用原来的镜像方法
            if i == 0 or i == 2:  # 相邻装甲板
                # 使用新的函数根据中心点和装甲板尺寸重构角点
                predicted_pts = self._reconstruct_armor_from_center_and_dimensions(pred_center_3d, pred_y)
            else:  # 对面装甲板
                # 对面装甲板可以使用原来的镜像方法
                pred_center_2d = np.array([pred_x, pred_z])
                predicted_pts = self._reconstruct_corners_from_center(visible_armor, pred_center_2d)
            
            # 创建新的装甲板对象
            new_armor = ArmorPlate(
                points=predicted_pts,
                color=visible_armor.color,
                troop_type=visible_armor.troop_type,
                area=visible_armor.area,
                confident=visible_armor.confident * 0.7,  # 预测的装甲板置信度略低
            )
            predicted_armors.append(new_armor)
            
        return predicted_armors

    # === SENSITIVITY IMPROVEMENT SUGGESTION ===
    # ISSUE: 半径选择算法过于简化，仅基于高度差进行匹配
    # POTENTIAL PROBLEMS:
    # 1. 当两个装甲板高度相近时，容易产生误判
    # 2. 缺乏对噪声和异常值的鲁棒性
    # 3. 没有考虑装甲板之间的几何约束关系
    # SOLUTION SUGGESTIONS:
    # 1. 增加高度容差检查，避免过度敏感
    # 2. 实现基于几何模型的验证机制
    # 3. 添加历史数据平滑处理，提高稳定性

    def _reconstruct_armor_from_center_and_dimensions(self, center_3d: np.ndarray, armor_height: float) -> np.ndarray:
        """
        根据装甲板中心点坐标和装甲板尺寸信息重构装甲板四个角点坐标
        
        参数:
        center_3d: 装甲板中心点的3D坐标 [x, y, z]
        armor_height: 装甲板的高度，用于查找匹配的装甲板尺寸
        
        返回:
        np.ndarray: 装甲板四个角点的3D坐标 [[top_left, bottom_left, top_right, bottom_right]]
        """
        # 查找与给定高度最匹配的装甲板尺寸
        matched_height = None
        min_height_diff = float('inf')
        matched_dimensions = None
        
        for height, dimensions in self.armor_dimensions.items():
            height_diff = abs(height - armor_height)
            if height_diff < min_height_diff:
                min_height_diff = height_diff
                matched_height = height
                matched_dimensions = dimensions
                
        # 如果没有找到匹配的高度，抛出异常或使用默认处理
        if matched_height is None or matched_dimensions is None:
            # 如果没有记录的尺寸信息，我们无法准确重构装甲板
            raise ValueError(f"No recorded armor dimensions for height {armor_height}. "
                             f"Available heights: {list(self.armor_dimensions.keys())}")
            
        # 使用实际测量的装甲板尺寸
        armor_type, length, width, z_diff = matched_dimensions
            
        # 根据装甲板尺寸和中心点计算四个角点
        center_x, center_y, center_z = center_3d
        
        # 假设装甲板正面朝向z轴正方向，x轴方向为长度，y轴方向为宽度
        half_length = length / 2
        half_width = width / 2
        half_z_diff = z_diff / 2
        # 计算四个角点坐标
        # 注意：这里的坐标系是相机坐标系：x向右，y向下，z向前
        top_left = [center_x + half_z_diff, center_y + half_width, center_z+half_length]
        bottom_left = [center_x - half_z_diff, center_y - half_width, center_z+half_length]
        top_right = [center_x + half_z_diff, center_y + half_width, center_z-half_length]
        bottom_right = [center_x - half_z_diff, center_y - half_width, center_z-half_length]
        
        return np.array([top_left, bottom_left, top_right, bottom_right], dtype=np.float32)

    # === SENSITIVITY IMPROVEMENT SUGGESTION ===
    # ISSUE: 角点重构依赖于预记录的尺寸信息，如果初始测量不准确会影响后续预测
    # SOLUTION SUGGESTION:
    # 1. 实现尺寸信息的动态更新机制
    # 2. 添加尺寸测量的置信度评估
    # 3. 使用多个样本的统计平均值而非单一测量值

    def _reconstruct_corners_from_center(self, armor: ArmorPlate, car_center_2d: np.ndarray) -> np.ndarray:
        """根据小车二维中心 (x,z) 对原装甲板四角点在 x/z 平面做关于车心的对称映射，y 不变。"""
        pts = np.asarray(armor.camera_pos, dtype=float).reshape(-1, 3)
        if pts.shape[0] != 4:
            raise ValueError("ArmorPlate.camera_pos 必须是4个3D角点")

        car_center_2d = np.asarray(car_center_2d, dtype=float).reshape(2)
        car_x, car_z = car_center_2d[0], car_center_2d[1]

        new_pts = np.empty_like(pts)
        for i in range(len(pts)):
            new_x = 2.0 * car_x - pts[i][0]
            new_z = 2.0 * car_z - pts[i][2]
            new_y = pts[i][1]  # 保持原 y
            new_pts[i] = np.array([new_x, new_y, new_z], dtype=float)

        # 返回顺序仍为 [top_left, bottom_left, top_right, bottom_right]
        return new_pts

    def calculate_another_armor_by_center(self):
        """根据当前观测装甲板和小车二维中心, 构造对面装甲板并追加到 self.armor_plate。"""
        # 1. 计算车中心与每块装甲板中心
        center_cam = self.get_center_from_normals()  # center_cam: [x, z]
        self.find_armor_center()  # 更新 self.armor_plate_center

        new_armors = []

        for armor in self.armor_plate:
            # 直接用车的二维中心对原装甲板四角点做对称映射
            opposite_pts = self._reconstruct_corners_from_center(
                armor=armor,
                car_center_2d=center_cam,
            )

            new_armor = ArmorPlate(
                points=opposite_pts,
                color=armor.color,
                troop_type=armor.troop_type,
                area=armor.area,
                confident=armor.confident,
            )
            new_armors.append(new_armor)

        self.armor_plate.extend(new_armors)

    def get_center_x(self, armor: ArmorPlate) -> float:
        """计算装甲板中心的x坐标，用于匹配追踪器"""
        pts = np.array(armor.camera_pos).reshape(-1, 3)
        return float(np.mean(pts[:, 0]))

    def update_armor_trackers(self, dt: float, h: int, w: int, camera2xy_func=None, angular_velocity_info=None):
        """
        更新装甲板追踪器
        
        参数:
        dt: 时间间隔
        h: 图像高度
        w: 图像宽度
        camera2xy_func: 相机坐标到像素坐标的转换函数
        angular_velocity_info: 角速度信息 (angular_velocity, rotation_axis)
        """
        # 重置所有追踪器的丢失计数
        for armor_id in self.armor_trackers:
            self.armor_trackers[armor_id]["miss_cnt"] = 0
            
        # 为每个装甲板创建或更新追踪器
        for armor_idx, armor in enumerate(self.armor_plate):
            armor_center_x = self.get_center_x(armor)
            
            # 简单的装甲板匹配逻辑（基于中心x坐标）
            matched_armor_id = None
            for armor_id, state in self.armor_trackers.items():
                if abs(state["center_x"] - armor_center_x) < 50:  # 阈值可根据需要调整
                    matched_armor_id = armor_id
                    # 更新装甲板属性
                    state["center_x"] = armor_center_x
                    state["color"] = armor.color
                    state["troop_type"] = armor.troop_type
                    break
            
            # 如果没有匹配的装甲板，则创建新的ID
            if matched_armor_id is None:
                matched_armor_id = len(self.armor_trackers)
                self.armor_trackers[matched_armor_id] = {
                    "kfs": [None] * 4,
                    "inited": [False] * 4,
                    "miss_cnt": 0,
                    "center_x": armor_center_x,
                    "color": armor.color,
                    "troop_type": armor.troop_type,
                    "smooth_pixels": None
                }
            else:
                # 如果匹配到了，更新中心x坐标
                self.armor_trackers[matched_armor_id]["center_x"] = armor_center_x
            
            # 为4个角点应用卡尔曼滤波
            kfs = self.armor_trackers[matched_armor_id]["kfs"]
            inited = self.armor_trackers[matched_armor_id]["inited"]
            
            filtered_pixels = []
            for idx, p in enumerate(armor.camera_pos):
                px, py, pz = map(float, p)
                
                # 计算点的半径和角度（相对于机器人中心点）
                r = 0.0
                theta = 0.0
                omega = 0.0
                
                # 如果有中心点信息，计算相对半径和角度
                if self.center_point is not None:
                    center_x, center_z = self.center_point[0], self.center_point[1]
                    # 在xz平面上计算相对位置
                    dx = px - center_x
                    dz = pz - center_z
                    r = np.sqrt(dx*dx + dz*dz)
                    theta = np.arctan2(dz, dx)
                
                # 如果有角速度信息，使用它
                if angular_velocity_info is not None:
                    omega, _ = angular_velocity_info
                
                if not inited[idx]:
                    kf_point = KF(
                        state_dim=9,  # 位置(3) + 半径(1) + 角度(1) + 角速度(1) + 速度(3) = 9维
                        init_cov=self.corner_kf_init_cov,
                        measure_noise=self.corner_kf_measure_noise,
                        process_noise=self.corner_kf_process_noise,
                        x=px, y=py, z=pz,
                        r=r, theta=theta, omega=omega,
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
                
                # 使用传入的转换函数或默认函数
                if camera2xy_func:
                    u_f, v_f = camera2xy_func(pos_post)
                else:
                    u_f, v_f = self._camera2xy(pos_post, h, w)
                    
                u_f = int(max(0, min(w - 1, u_f)))
                v_f = int(max(0, min(h - 1, v_f)))
                filtered_pixels.append((u_f, v_f))
            
            # 保存滤波后的像素坐标
            self.armor_trackers[matched_armor_id]["smooth_pixels"] = filtered_pixels

    def predict_armor_positions(self, dt: float, angular_velocity_info=None):
        """
        使用卡尔曼滤波器预测装甲板位置
        
        参数:
        dt: 时间间隔
        angular_velocity_info: 角速度信息 (angular_velocity, rotation_axis)
        
        返回:
        predicted_armors: 预测的装甲板列表
        """
        predicted_armors = []
        
        # 遍历所有追踪器
        for armor_id, state in self.armor_trackers.items():
            kfs = state["kfs"]
            predicted_points = []
            
            # 对每个角点进行预测
            for idx, kf in enumerate(kfs):
                if kf is not None:
                    # 预测下一个状态
                    pred = kf.predict_next(dt)
                    # 获取预测的位置
                    pred_pos = pred[:3].reshape(-1)
                    predicted_points.append(pred_pos)
            
            # 创建预测的装甲板对象
            if len(predicted_points) == 4:
                predicted_armor = ArmorPlate(
                    points=np.array(predicted_points),
                    color=state["color"],
                    troop_type=state["troop_type"],
                    area=0,  # 面积暂时设为0
                    confident=0.5  # 置信度设为中等
                )
                predicted_armors.append(predicted_armor)
                
        return predicted_armors

    def _camera2xy(self, pos, h, w):
        """
        简化的相机坐标到像素坐标的转换函数
        注意：这是一个简化的实现，实际应用中应该使用更精确的相机参数
        """
        x, y, z = pos
        # 简单的透视投影，假设焦距为w/2，主点在图像中心
        if z != 0:
            u = (x / z) * (w / 2) + (w / 2)
            v = (y / z) * (w / 2) + (h / 2)
        else:
            u, v = 0, 0
        return u, v

    def cleanup_trackers(self):
        """清理长时间未匹配的追踪器"""
        for armor_id, state in list(self.armor_trackers.items()):
            # 增加丢失计数
            state["miss_cnt"] = state.get("miss_cnt", 0) + 1
            
            # 如果超过最大丢失帧数，则删除追踪器
            if state["miss_cnt"] > self.max_miss_frames:
                del self.armor_trackers[armor_id]

    def get_tracked_armors(self) -> List[Dict[str, Any]]:
        """
        获取所有正在追踪的装甲板信息
        
        返回:
        List[Dict]: 包含所有追踪器信息的列表
        """
        return list(self.armor_trackers.values())