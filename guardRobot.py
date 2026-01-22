from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# 假设这些类定义在 all_type 模块中，为了代码完整性保留引用
from all_type import ArmorPlate, Color, TroopType


class TestRobotCenter:
    # 注意：使用类属性存储半径历史数据。
    # 警告：如果在同一程序中同时检测多台不同类型的机器人（如同时有步兵和哨兵），
    # 这种写法会导致半径数据污染。建议在外部 Tracker 中维护每个 ID 对应的半径字典。
    predict_r_h = {}

    def __init__(self, robot_armor_coordinate=None):
        if robot_armor_coordinate is None:
            robot_armor_coordinate = []
        # 四个装甲板的四个角点对应的坐标，shape=(N,4,3)
        self.robot_armor_coordinate = robot_armor_coordinate
        self.armor_center_point = []

    def get_robot_center_by_two_armor(self, idx1=0, idx2=1):
        """通过前后两块装甲板的法线，在 xz 平面上求它们交点，作为机器人中心的二维坐标。

        返回
        ------
        center_3d: ndarray, shape=(3,)
            机器人中心 [x, y, z]，y取两板平均高度。若直线平行或退化则返回 None。
        """
        self.armor_center_point.clear()
        c1, n1, _ = self.get_armor_normal_vector(self.robot_armor_coordinate[idx1])
        c2, n2, _ = self.get_armor_normal_vector(self.robot_armor_coordinate[idx2])

        # 若某一块装甲板无法计算法线，直接返回 None
        if c1 is None or c2 is None:
            return None

        # 将三维直线投影到 xz 平面: 只保留 x 和 z 分量 (索引 0 和 2)
        # 三维参数直线: L(t) = c + t * n
        # 投影到 xz 后: L_xz(t) = [c_x, c_z] + t * [n_x, n_z]
        p1_2d = np.array([c1[0], c1[2]], dtype=float)
        d1_2d = np.array([n1[0], n1[2]], dtype=float)

        p2_2d = np.array([c2[0], c2[2]], dtype=float)
        d2_2d = np.array([n2[0], n2[2]], dtype=float)

        # 求两条二维直线的交点
        # p1 + t*d1 = p2 + s*d2
        # d1_x * t - d2_x * s = p2_x - p1_x
        # d1_y * t - d2_y * s = p2_y - p1_y (这里的y其实是z轴)
        A = np.array([
            [d1_2d[0], -d2_2d[0]],
            [d1_2d[1], -d2_2d[1]],
        ], dtype=float)
        b = p2_2d - p1_2d

        detA = np.linalg.det(A)
        eps = 1e-4  # 稍微放宽容差
        if abs(detA) < eps:
            # 两条线在 xz 平面上平行或几乎平行，无唯一交点
            return None

        try:
            t, s = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        center_xz = p1_2d + t * d1_2d

        # 计算并更新半径历史数据
        r1 = np.linalg.norm(center_xz - p1_2d)
        r2 = np.linalg.norm(center_xz - p2_2d)
        TestRobotCenter.predict_r_h[float(c1[1])] = r1
        TestRobotCenter.predict_r_h[float(c2[1])] = r2

        # [修改] 不再硬编码 0.7，而是取两块装甲板高度的平均值
        avg_height = (c1[1] + c2[1]) / 2.0

        return np.array([center_xz[0], avg_height, center_xz[1]], dtype=float)

    def get_robot_center_by_one_armor(self, idx=0):
        """通过单块装甲板计算机器人中心。"""
        self.armor_center_point.clear()
        center_armor, normal_unit, _ = self.get_armor_normal_vector(self.robot_armor_coordinate[idx])

        if center_armor is None or normal_unit is None:
            return None

        # 获取历史半径
        if not TestRobotCenter.predict_r_h:
            return None

        armor_r = get_value_by_closest_key(TestRobotCenter.predict_r_h, center_armor[1])

        # 使用法向量反向推算中心点： Center = Armor - Normal * Radius
        # 仅在XZ平面进行计算，忽略法向量的Y分量
        normal_xz = np.array([normal_unit[0], normal_unit[2]], dtype=float)
        norm_xz = np.linalg.norm(normal_xz)

        if norm_xz > 1e-6:
            normal_xz = normal_xz / norm_xz
            # 计算逻辑：中心点 = 装甲板中心 - 法向量 * 半径
            # 假设法向量是指向装甲板外部的
            center_xz = np.array([center_armor[0], center_armor[2]]) - normal_xz * armor_r
            # 返回3D坐标，高度保持与装甲板一致
            return np.array([center_xz[0], center_armor[1], center_xz[1]])
        else:
            return None

    def get_armor_normal_vector(self, four_armor_points):
        """计算装甲板的法向量以及经过装甲板中心且方向为法向量的直线参数方程。"""
        p1, p2, p3, p4 = four_armor_points
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        p3 = np.array(p3, dtype=float)
        p4 = np.array(p4, dtype=float)

        # 装甲板中心点
        center_point = (p1 + p2 + p3 + p4) / 4
        self.armor_center_point.append(center_point.tolist())

        # 计算法向量（由装甲板上的两条边叉乘得到）
        # 注意点序：通常 p1-p2-p3-p4 是逆时针或顺时针
        # 这里假设 p2-p1 和 p3-p1 构成的叉乘指向车外
        v1 = p2 - p1
        v2 = p3 - p1
        normal_vector = np.cross(v1, v2)

        norm = np.linalg.norm(normal_vector)
        if norm < 1e-6:
            return None, None, None

        # 归一化法向量
        normal_unit = normal_vector / norm

        # 直线参数方程: L(t) = center_point + t * normal_unit
        line_func = lambda t: center_point + t * normal_unit
        return center_point, normal_unit, line_func


class GuardRobot:
    def __init__(self, armor_plates=None, color: Color = None, troop_type: TroopType = None):
        if armor_plates is None:
            armor_plates = []

        # 确保armor_plates是ArmorPlate对象列表或坐标列表
        if armor_plates and hasattr(armor_plates[0], 'camera_pos'):
            self.armor_plates_camera_positions = [armor_plate.camera_pos for armor_plate in armor_plates]
        else:
            self.armor_plates_camera_positions = armor_plates

        self.color = color
        self.troop_type = troop_type
        self.armor_length = None
        self.armor_width = None
        self.armor_height = None
        self.test_robot_center = TestRobotCenter(self.armor_plates_camera_positions)
        self.center = None
        self.armor_center_point = []

    def cal_armor(self):
        """计算装甲板尺寸，添加安全检查和正确的角点索引"""
        if len(self.armor_plates_camera_positions) == 0:
            return False

        if len(self.armor_plates_camera_positions[0]) < 4:
            return False

        try:
            # 装甲板长度：左上角到右上角的距离
            top_left = np.array(self.armor_plates_camera_positions[0][0], dtype=float)
            top_right = np.array(self.armor_plates_camera_positions[0][2],
                                 dtype=float)  # 注意：通常索引0是左上，1是左下，2是右下，3是右上。或者是顺时针。请根据实际点序确认索引。
            # 假设点序：0:左上, 1:左下, 2:右下, 3:右上 (OpenCV常见) 或者 0:左上, 1:右上...
            # 这里沿用原代码逻辑：0->2 计算长度? 原代码可能是对角线?
            # 修正逻辑：通常 ArmorPlate 顺序是 左上->左下->右下->右上
            # 或者是 左上->右上->右下->左下
            # 这里保留原代码逻辑，但请注意确认相机坐标系下的点序。
            self.armor_length = np.linalg.norm(top_left - top_right)

            # 装甲板宽度：右上角到右下角在x,z平面的投影距离
            # 假设 2和3 是同一侧
            top_right_2d = np.array([self.armor_plates_camera_positions[0][2][0],
                                     self.armor_plates_camera_positions[0][2][2]], dtype=float)
            bottom_right_2d = np.array([self.armor_plates_camera_positions[0][3][0],
                                        self.armor_plates_camera_positions[0][3][2]], dtype=float)
            self.armor_width = np.linalg.norm(top_right_2d - bottom_right_2d)

            # 装甲板高度：Y轴差值
            self.armor_height = abs(self.armor_plates_camera_positions[0][0][1] -
                                    self.armor_plates_camera_positions[0][1][1])

            return True
        except (IndexError, ValueError):
            return False

    def find_robot_center(self):
        """查找机器人中心点"""
        self.armor_center_point.clear()

        # 获取静态半径字典的引用
        predict_r_h = self.test_robot_center.predict_r_h

        if len(self.armor_plates_camera_positions) > 1:
            # 两板解算
            robot_center_3d = self.test_robot_center.get_robot_center_by_two_armor()
        elif len(self.armor_plates_camera_positions) == 1 and len(predict_r_h) >= 1:
            # 单板解算，只要有历史半径数据即可
            robot_center_3d = self.test_robot_center.get_robot_center_by_one_armor()
        else:
            robot_center_3d = None

        self.center = robot_center_3d
        self.armor_center_point = self.test_robot_center.armor_center_point
        return self.center

    def get_another_armor_plate_center_by_center(self):
        """根据机器人中心计算其他装甲板中心点"""
        if self.center is None:
            return

        predict_r_h = self.test_robot_center.predict_r_h

        if len(self.armor_plates_camera_positions) == 2:
            # 有两块板，补全另外两块（中心对称）
            if len(self.armor_center_point) < 2:
                return

            armor_centers = self.armor_center_point[:2]
            new_centers = []

            for i in range(2):
                center_point = np.array(armor_centers[i], dtype=float)
                symmetric_center = np.array([
                    2 * self.center[0] - center_point[0],
                    center_point[1],  # 保持相同高度
                    2 * self.center[2] - center_point[2]
                ], dtype=float)
                new_centers.append(symmetric_center.tolist())

            self.armor_center_point.extend(new_centers)

        elif len(self.armor_plates_camera_positions) == 1:
            # 单板模式
            if len(self.armor_center_point) < 1:
                return

            # 如果没有足够的半径数据（可能刚开始只看到这一块板，且是第一次），无法推算侧板
            if not predict_r_h:
                return

            known_center = np.array(self.armor_center_point[0], dtype=float)

            # 1. 计算对面装甲板中心点（中心对称）
            opposite_center = np.array([
                2 * self.center[0] - known_center[0],
                known_center[1],
                2 * self.center[2] - known_center[2]
            ])
            self.armor_center_point.append(opposite_center.tolist())

            # 2. 获取半径数据以计算侧板
            known_height = known_center[1]
            radius_face_camera = get_key_by_closest_value(predict_r_h, known_height)

            # [修改] 尝试找到另一个高度不同的半径（针对平衡步兵）
            radius_not_face_camera = None
            for key in predict_r_h.keys():
                try:
                    if abs(float(key) - float(known_height)) > 0.01:
                        radius_not_face_camera = key
                        break
                except (ValueError, TypeError):
                    continue

            # [修改] 如果找不到不同高度的半径，说明可能是标准步兵（4板同半径），降级使用当前半径
            if radius_not_face_camera is None and radius_face_camera is not None:
                radius_not_face_camera = radius_face_camera

            if radius_not_face_camera is None:
                # 依然没有有效半径，退出
                return

            # 3. 计算相邻两个装甲板中心点
            # 向量：中心 -> 已知板
            vec_to_known = known_center - self.center
            vec_to_known_2d = np.array([vec_to_known[0], vec_to_known[2]])

            # 防止零向量
            norm_vec = np.linalg.norm(vec_to_known_2d)
            if norm_vec < 1e-6:
                return

            # 旋转90度得到相邻方向 (x, z) -> (-z, x)
            vec_adjacent_2d = np.array([-vec_to_known_2d[1], vec_to_known_2d[0]])
            vec_adjacent_2d = vec_adjacent_2d / norm_vec

            other_radius = predict_r_h[radius_not_face_camera]
            other_height = float(radius_not_face_camera)

            # 计算侧板中心
            adj_1_xz = np.array([self.center[0], self.center[2]]) + vec_adjacent_2d * other_radius
            adj_2_xz = np.array([self.center[0], self.center[2]]) - vec_adjacent_2d * other_radius

            adjacent_center1 = [adj_1_xz[0], other_height, adj_1_xz[1]]
            adjacent_center2 = [adj_2_xz[0], other_height, adj_2_xz[1]]

            self.armor_center_point.append(adjacent_center1)
            self.armor_center_point.append(adjacent_center2)

    def get_armor_by_armor_center(self):
        """根据装甲板中心点生成装甲板角点"""
        if len(self.armor_center_point) < 4:
            return

        # 如果只有1块或者2块原始数据，需要补全 self.armor_plates_camera_positions
        current_plate_count = len(self.armor_plates_camera_positions)

        # 需要补全的装甲板数量
        plates_to_add = 4 - current_plate_count

        if plates_to_add > 0:
            # 我们可以直接利用计算出的 armor_center_point (共4个) 来生成所有板的角点
            # 但通常保留原始观测数据，只生成虚拟数据

            # 这里简化逻辑：遍历armor_center_point中多出来的部分
            for i in range(current_plate_count, 4):
                center_pos = np.array(self.armor_center_point[i], dtype=float)

                # 区分 长/宽 使用
                # 如果是平衡步兵，侧板的长宽可能不同，这里简单假设所有板尺寸一致
                # 或者根据高度判断：如果高度和观测板一致，用观测板尺寸；否则可能需要另一套尺寸（代码中暂无）

                # 构建虚拟装甲板：假设面向中心，垂直放置
                # 需要计算该板的法向：从中心指向板
                vec = center_pos - self.center
                vec_xz = np.array([vec[0], vec[2]])
                norm = np.linalg.norm(vec_xz)
                if norm < 1e-6:
                    continue

                # 单位法向量 (xz平面)
                n_xz = vec_xz / norm
                # 这里的法向是指向板外

                # 板的切向 (水平方向)
                t_xz = np.array([-n_xz[1], n_xz[0]])

                half_w = self.armor_width / 2
                half_h = self.armor_height / 2
                # 这里假设 armor_length 对应 3D 空间中的水平宽度(width), armor_width 对应深度?
                # 根据 cal_armor 的逻辑:
                # armor_length 是 TopLeft 到 TopRight (3D距离) -> 物理宽度
                # armor_width 是 TopRight 到 BottomRight (2D投影) -> 物理高度? 不对
                # 通常：
                # Length = 物理宽 (左右)
                # Height = 物理高 (上下)

                # 使用 cal_armor 算出的 armor_length (水平) 和 armor_height (垂直)
                w = self.armor_length
                h = self.armor_height

                # 构建4个点：
                # 0: TL = Center - Right*w/2 + Up*h/2
                # 1: BL = Center - Right*w/2 - Up*h/2
                # 2: BR = Center + Right*w/2 - Up*h/2
                # 3: TR = Center + Right*w/2 + Up*h/2
                # 注意：t_xz 是右向量还是左向量取决于旋转方向，这里 t_xz = (-z, x) 是逆时针90度

                cx, cy, cz = center_pos

                # 简易生成逻辑，基于轴对齐或切向
                # 水平偏移
                dx = t_xz[0] * (w / 2)
                dz = t_xz[1] * (w / 2)

                dy = h / 2

                # 顺序：TL, BL, BR, TR (根据cal_armor里 0-1是高度)
                # 假设 cal_armor: 0(TL), 1(BL), 2(BR), 3(TR)

                # 左上
                p0 = [cx - dx, cy + dy, cz - dz]
                # 左下
                p1 = [cx - dx, cy - dy, cz - dz]
                # 右下
                p2 = [cx + dx, cy - dy, cz + dz]
                # 右上
                p3 = [cx + dx, cy + dy, cz + dz]

                self.armor_plates_camera_positions.append([p0, p1, p2, p3])

    def use_robot_prediction(self):
        """执行机器人预测流程"""
        if not self.cal_armor():
            return

        self.find_robot_center()

        if self.center is not None:
            # print(f"机器人中心点: {self.center}")
            self.get_another_armor_plate_center_by_center()
            self.get_armor_by_armor_center()


# ---------------- 工具函数 ----------------

def find_center_symmetric_point_with_2d_center(center_2d, point_3d):
    center = np.array(center_2d, dtype=float)
    point = np.array(point_3d, dtype=float)
    center_3d = np.array([center[0], point[1], center[1]], dtype=float)
    return 2 * center_3d - point


def get_value_by_closest_key(dict_data, target_key, default_value=0.0):
    target = target_key
    dict_keys = list(dict_data.keys())
    if not dict_keys:
        return default_value

    closest_key = min(dict_keys, key=lambda k: abs(float(k) - float(target)))
    return dict_data[closest_key]


def get_key_by_closest_value(dict_data, target_value, default_key=None):
    target = target_value
    dict_values = list(dict_data.values())
    if not dict_values:
        return default_key

    closest_value = min(dict_values, key=lambda v: abs(float(v) - float(target)))
    closest_keys = [key for key, value in dict_data.items() if value == closest_value]
    return closest_keys[0] if closest_keys else default_key