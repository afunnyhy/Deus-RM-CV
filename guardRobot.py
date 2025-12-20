from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from all_type import ArmorPlate, Color, TroopType
from KalmanFilter import KalmanFilter as KF


def find_center_symmetric_point_with_2d_center(center_2d, point_3d):
    """
    根据二维中心点（x,z）和三维点，找到中心对称点，中心的y值采用三维点的y值

    参数
    ------
    center_2d: tuple/list/ndarray, shape=(2,)
        中心点的二维坐标 [x, z]
    point_3d: tuple/list/ndarray, shape=(3,)
        目标点的三维坐标 [x, y, z]

    返回
    ------
    symmetric_point: ndarray, shape=(3,)
        中心对称点的三维坐标 [x, y, z]
    """
    # 确保输入为numpy数组
    center = np.array(center_2d, dtype=float)
    point = np.array(point_3d, dtype=float)

    # 检查输入维度
    if center.shape != (2,):
        raise ValueError("中心点必须是二维坐标 [x, z]")
    if point.shape != (3,):
        raise ValueError("目标点必须是三维坐标 [x, y, z]")

    # 构建三维中心点：使用二维中心点的x,z坐标，y值采用三维点的y值
    center_3d = np.array([center[0], point[1], center[1]], dtype=float)

    # 计算中心对称点：symmetric_point = 2 * center_3d - point
    symmetric_point = 2 * center_3d - point

    return symmetric_point

def get_value_by_closest_key(dict_data, target_key, default_value=0.0):
    """
    从字典中找到与目标键最接近的键，并返回其对应的值
    兼容字典键为：浮点数、整数、字符串类型的数值（如"0.1088"）

    Args:
        dict_data (dict): 待查找的字典（键为数值或字符串类型的数值）
        target_key (float/int/str): 目标键（会被转换为浮点数计算）
        default_value (any): 字典为空时返回的默认值（可自定义，如0.0、None等）

    Returns:
        any: 最接近键对应的值，或字典为空时返回默认值
    """
    target=target_key
    # 2. 获取字典的键列表，若字典为空则返回默认值
    dict_keys = list(dict_data.keys())
    if not dict_keys:
        # print(f"警告：输入的字典为空，返回默认值 {default_value}")
        return default_value

    # 3. 找到与目标键最接近的键（核心逻辑：计算差值的绝对值，取最小值对应的键）
    def calculate_diff(key):
        """内部函数：计算键与目标键的差值绝对值"""
        try:
            return abs(float(key) - target)
        except (ValueError, TypeError):
            # 若字典中的键无法转换为浮点数，返回极大值（使其不被选中）
            return float('inf')

    closest_key = min(dict_keys, key=calculate_diff)

    # 4. 打印调试信息（可选，可注释掉）
    diff = calculate_diff(closest_key)
    # print(f"提示：目标键 {target_key} 不存在，使用最接近的键 {closest_key}（差值：{diff:.6f}）")

    # 5. 返回最接近键对应的值
    return dict_data[closest_key]


def get_key_by_closest_value(dict_data, target_value, default_key=None):
    """
    从字典中找到与目标值最接近的值，并返回其对应的键
    兼容字典值为：浮点数、整数、字符串类型的数值（如"0.1088"）

    Args:
        dict_data (dict): 待查找的字典（值为数值或字符串类型的数值）
        target_value (float/int/str): 目标值（会被转换为浮点数计算）
        default_key (any): 字典为空时返回的默认键（可自定义，如None、0等）

    Returns:
        any: 最接近值对应的键，或字典为空时返回默认键
    """
    target = target_value

    # 2. 获取字典的值列表，若字典为空则返回默认键
    dict_values = list(dict_data.values())
    if not dict_values:
        # print(f"警告：输入的字典为空，返回默认键 {default_key}")
        return default_key

    # 3. 找到与目标值最接近的值（核心逻辑：计算差值的绝对值，取最小值对应的值）
    def calculate_diff(value):
        """内部函数：计算值与目标值的差值绝对值"""
        try:
            return abs(float(value) - target)
        except (ValueError, TypeError):
            # 若字典中的值无法转换为浮点数，返回极大值（使其不被选中）
            return float('inf')

    # 找到最接近的值
    closest_value = min(dict_values, key=calculate_diff)

    # 4. 根据最接近的值找到对应的键（可能有多个键对应同一个值）
    closest_keys = [key for key, value in dict_data.items() if value == closest_value]

    # 如果有多个键对应同一个值，返回第一个找到的键
    closest_key = closest_keys[0] if closest_keys else default_key

    # 5. 打印调试信息（可选，可注释掉）
    diff = calculate_diff(closest_value)
    # print(f"提示：目标值 {target_value} 不存在，使用最接近的值 {closest_value}（差值：{diff:.6f}），对应键：{closest_key}")

    # 6. 返回最接近值对应的键
    return closest_key

class TestRobotCenter:
    predict_r_h = {}

    def __init__(self, robot_armor_coordinate=None):
        if robot_armor_coordinate is None:
            robot_armor_coordinate = []
        self.robot_armor_coordinate = robot_armor_coordinate  # 四个装甲板的四个角点对应的坐标，shape=(4,4,3)
        self.armor_center_point=[]

    def get_robot_center_by_two_armor(self, idx1=0, idx2=1):
        """通过前后两块装甲板的法线，在 xz 平面上求它们交点，作为机器人中心的二维坐标。

        返回
        ------
        center_xz: ndarray, shape=(2,)
            机器人在 xz 平面上的估计中心 [x, z]，若直线平行或退化则返回 None。
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
        A = np.array([
            [d1_2d[0], -d2_2d[0]],
            [d1_2d[1], -d2_2d[1]],
        ], dtype=float)
        b = p2_2d - p1_2d

        detA = np.linalg.det(A)
        eps = 1e-6
        if abs(detA) < eps:
            # 两条线在 xz 平面上平行或几乎平行，无唯一交点
            return None

        t, s = np.linalg.solve(A, b)
        center_xz = p1_2d + t * d1_2d
        r1 = np.linalg.norm(center_xz - p1_2d)
        r2 = np.linalg.norm(center_xz - p2_2d)
        TestRobotCenter.predict_r_h[float(c1[1])] = r1
        TestRobotCenter.predict_r_h[float(c2[1])] = r2
        # print("predict_r_h的所有键（装甲板中心点高度）：", list(self.predict_r_h.keys()))
        print(self.predict_r_h)
        # print("center_xz:", center_xz)
        return np.array([center_xz[0],0.7,center_xz[1]], dtype=float)

    def get_robot_center_by_one_armor(self, idx=0):
        """通过单块装甲板计算机器人中心。"""
        self.armor_center_point.clear()
        center_armor,_,_=self.get_armor_normal_vector(self.robot_armor_coordinate[idx])
        print(self.predict_r_h)
        armor_r = get_value_by_closest_key(TestRobotCenter.predict_r_h, center_armor[1])
        center_armor = center_armor + np.array([0, 0, armor_r])
        # print("center_armor:", center_armor)
        return center_armor

    def get_armor_normal_vector(self, four_armor_points):
        """计算装甲板的法向量以及经过装甲板中心且方向为法向量的直线参数方程。

        参数
        ------
        four_armor_points: ndarray, shape=(4, 3)
            四个角点的三维坐标，顺序为 p1, p2, p3, p4。

        返回
        ------
        center_point: ndarray, shape=(3,)
            装甲板中心点坐标。
        normal_unit: ndarray, shape=(3,)
            归一化后的法向量。
        line_func: callable
            直线参数方程 L(t) = center_point + t * normal_unit。
        """
        p1, p2, p3, p4 = four_armor_points
        p1=np.array(p1)
        p2=np.array(p2)
        p3=np.array(p3)
        p4=np.array(p4)
        # 装甲板中心点
        center_point = (p1 + p2 + p3 + p4) / 4
        self.armor_center_point.append(center_point.tolist())
        # 未归一化的法向量（由装甲板上的两条边叉乘得到）
        normal_vector = np.cross(p2 - p1, p3 - p1)
        # 防止除零，若四点共线或退化则返回 None
        norm = np.linalg.norm(normal_vector)
        if norm == 0:
            return None, None, None
        # 归一化法向量
        normal_unit = normal_vector / norm
        # 直线参数方程: L(t) = center_point + t * normal_unit
        line_func = lambda t: center_point + t * normal_unit
        return center_point, normal_unit, line_func


class GuardRobot:
    predict_r_h = TestRobotCenter.predict_r_h

    def __init__(self, armor_plates=None, color: Color = None, troop_type: TroopType = None):
        if armor_plates is None:
            armor_plates = []

        # 确保armor_plates是ArmorPlate对象列表
        if armor_plates and hasattr(armor_plates[0], 'camera_pos'):
            self.armor_plates_camera_positions = [armor_plate.camera_pos for armor_plate in armor_plates]
        else:
            # 如果传入的是角点坐标数组，直接使用
            self.armor_plates_camera_positions = armor_plates

        self.color = color
        self.troop_type = troop_type
        self.armor_length = None
        self.armor_width = None
        self.armor_height = None
        self.test_robot_center = TestRobotCenter(self.armor_plates_camera_positions)
        self.center = None
        self.armor_center_point = []  # 格式：列表的列表 [[装甲板中心点1], [装甲板中心点2], ...]

    def cal_armor(self):
        """计算装甲板尺寸，添加安全检查和正确的角点索引"""
        if len(self.armor_plates_camera_positions) == 0:
            # print("警告：没有装甲板数据")
            return False

        # 检查第一个装甲板是否有足够的角点数据
        if len(self.armor_plates_camera_positions[0]) < 4:
            # print(f"警告：装甲板角点数据不足，需要4个角点，实际有{len(self.armor_plates_camera_positions[0])}个")
            return False

        try:
            # 装甲板长度：左上角到右上角的距离
            top_left = np.array(self.armor_plates_camera_positions[0][0], dtype=float)
            top_right = np.array(self.armor_plates_camera_positions[0][2], dtype=float)
            self.armor_length = np.linalg.norm(top_left - top_right)

            # 装甲板宽度：右上角到右下角在x,z平面的投影距离
            top_right_2d = np.array([self.armor_plates_camera_positions[0][2][0],
                                     self.armor_plates_camera_positions[0][2][2]], dtype=float)
            bottom_right_2d = np.array([self.armor_plates_camera_positions[0][3][0],
                                        self.armor_plates_camera_positions[0][3][2]], dtype=float)
            self.armor_width = np.linalg.norm(top_right_2d - bottom_right_2d)

            # 装甲板高度：左上角到左下角y坐标的距离
            self.armor_height = abs(self.armor_plates_camera_positions[0][0][1] -
                                    self.armor_plates_camera_positions[0][1][1])

            return True
        except (IndexError, ValueError) as e:
            # print(f"计算装甲板尺寸时出错: {e}")
            return False

    def find_robot_center(self):
        """查找机器人中心点"""
        # 清空装甲板中心点列表
        self.armor_center_point.clear()

        if len(self.armor_plates_camera_positions) > 1:
            TestRobotCenter.predict_r_h.clear()
            robot_center_3d = self.test_robot_center.get_robot_center_by_two_armor()
            # print(f"使用两块装甲板计算中心点: {robot_center_3d}")
        elif len(self.armor_plates_camera_positions) == 1 and len(self.test_robot_center.predict_r_h) >= 2:
            robot_center_3d = self.test_robot_center.get_robot_center_by_one_armor()
            # print(f"使用一块装甲板计算中心点: {robot_center_3d}")
        else:
            robot_center_3d = None
            # print("无法计算中心点")

        self.center = robot_center_3d

        # 从TestRobotCenter获取装甲板中心点
        self.armor_center_point = self.test_robot_center.armor_center_point
        # print(f"GuardRobot.armor_center_point: {self.armor_center_point}")
        return self.center

    def get_another_armor_plate_center_by_center(self):
        """根据机器人中心计算其他装甲板中心点"""
        if self.center is None:
            # print("警告：没有中心点数据")
            return

        # print(f"机器人中心: {self.center}")
        # print(f"当前装甲板数量: {len(self.armor_plates_camera_positions)}")
        # print(f"预测半径字典: {self.predict_r_h}")

        if len(self.armor_plates_camera_positions) == 2:
            # print("有两块装甲板数据")

            # 确保armor_center_point有足够的装甲板中心点
            if len(self.armor_center_point) < 2:
                # print("警告：装甲板中心点数据不足")
                return

            # 获取已知的两个装甲板中心点
            armor_centers = self.armor_center_point[:2]

            # 计算另外两个装甲板中心点（关于中心对称）
            new_centers = []

            for i in range(2):
                # 确保装甲板中心点是列表或数组
                if isinstance(armor_centers[i], (list, np.ndarray)):
                    center_point = np.array(armor_centers[i], dtype=float)
                else:
                    # print(f"警告：装甲板中心点{i}不是列表或数组: {armor_centers[i]}")
                    continue

                # 计算关于中心的对称点
                symmetric_center = np.array([
                    2 * self.center[0] - center_point[0],
                    center_point[1],  # 保持相同高度
                    2 * self.center[2] - center_point[2]
                ], dtype=float)
                new_centers.append(symmetric_center.tolist())

            # 将所有中心点保存到列表中
            self.armor_center_point.extend(new_centers)
            # print(f"计算得到4个装甲板中心点: {self.armor_center_point}")

        elif len(self.armor_plates_camera_positions) == 1:
            # print("有一块装甲板数据")

            # 确保有足够的装甲板中心点
            if len(self.armor_center_point) < 1:
                # print("警告：装甲板中心点数据不足")
                return

            if len(self.predict_r_h) < 2:
                # print(f"警告：预测半径数据不足，当前有{len(self.predict_r_h)}个半径")
                return

            # 获取已知装甲板中心点
            if isinstance(self.armor_center_point[0], (list, np.ndarray)):
                known_center = np.array(self.armor_center_point[0], dtype=float)
            else:
                # print(f"警告：装甲板中心点0不是列表或数组: {self.armor_center_point[0]}")
                return

            # 1. 计算对面装甲板中心点（关于中心对称）
            opposite_center = np.array([
                2 * self.center[0] - known_center[0],
                known_center[1],  # 相同高度
                2 * self.center[2] - known_center[2]
            ])

            # 保存到armor_center_point中
            self.armor_center_point.append(opposite_center.tolist())

            # 2. 获取半径数据
            # 找到与已知装甲板高度最接近的半径键
            known_height = known_center[1]
            radius_face_camera = get_key_by_closest_value(self.predict_r_h, known_height)

            # 找到另一个半径键
            radius_not_face_camera = None
            for key in self.predict_r_h.keys():
                # 将key转换为float进行比较
                try:
                    key_float = float(key)
                    known_height_float = float(known_height)
                    if abs(key_float - known_height_float) > 0.01:  # 允许微小误差
                        radius_not_face_camera = key
                        break
                except (ValueError, TypeError):
                    continue

            if radius_not_face_camera is None:
                # print("警告：无法找到非面向相机的半径键")
                return

            # print(f"查找到的面向相机半径键: {radius_face_camera}")
            # print(f"查找到的非面向相机半径键: {radius_not_face_camera}")

            # 3. 计算相邻两个装甲板中心点
            # 获取机器人中心到已知装甲板中心的向量
            vec_to_known = known_center - self.center
            vec_to_known_2d = np.array([vec_to_known[0], vec_to_known[2]])

            # 旋转90度得到相邻装甲板的方向
            vec_adjacent_2d = np.array([-vec_to_known_2d[1], vec_to_known_2d[0]])
            vec_adjacent_2d = vec_adjacent_2d / np.linalg.norm(vec_adjacent_2d)

            # 使用非面向相机的半径
            other_radius = self.predict_r_h[radius_not_face_camera]
            other_height = float(radius_not_face_camera)

            # 计算两个相邻装甲板中心点
            adjacent_center1_2d = np.array([self.center[0], self.center[2]]) + vec_adjacent_2d * other_radius
            adjacent_center2_2d = np.array([self.center[0], self.center[2]]) - vec_adjacent_2d * other_radius

            # 构建三维点并保存
            adjacent_center1 = [adjacent_center1_2d[0], other_height, adjacent_center1_2d[1]]
            adjacent_center2 = [adjacent_center2_2d[0], other_height, adjacent_center2_2d[1]]

            self.armor_center_point.append(adjacent_center1)
            self.armor_center_point.append(adjacent_center2)

            # print(f"成功添加3个装甲板中心点，当前armor_centers长度: {len(self.armor_center_point)}")

    def get_armor_by_armor_center(self):
        """根据装甲板中心点生成装甲板角点"""
        # print(f"armor_center_point长度: {len(self.armor_center_point)}")
        # print(f"armor_plates_camera_positions长度: {len(self.armor_plates_camera_positions)}")

        if len(self.armor_center_point) < 4:
            # print("警告：装甲板中心点不足4个")
            return

        if len(self.armor_plates_camera_positions) == 2:
            # 使用中心对称方法生成另外两个装甲板
            for i in range(2):
                armor_plate = [
                    find_center_symmetric_point_with_2d_center(
                        (self.center[0], self.center[2]),
                        self.armor_plates_camera_positions[i][0]
                    ),
                    find_center_symmetric_point_with_2d_center(
                        (self.center[0], self.center[2]),
                        self.armor_plates_camera_positions[i][1]
                    ),
                    find_center_symmetric_point_with_2d_center(
                        (self.center[0], self.center[2]),
                        self.armor_plates_camera_positions[i][2]
                    ),
                    find_center_symmetric_point_with_2d_center(
                        (self.center[0], self.center[2]),
                        self.armor_plates_camera_positions[i][3]
                    )
                ]
                self.armor_plates_camera_positions.append(armor_plate)

        elif len(self.armor_plates_camera_positions) == 1:
            # 确保有4个装甲板中心点
            if len(self.armor_center_point) < 4:
                # print("警告：需要4个装甲板中心点")
                return

            # 使用装甲板中心点和尺寸生成装甲板角点
            # 中心点1：已知的装甲板中心点（索引0）
            # 中心点2：对面装甲板中心点（索引1）
            # 中心点3：相邻装甲板中心点1（索引2）
            # 中心点4：相邻装甲板中心点2（索引3）

            # 生成对面装甲板（索引1）
            if isinstance(self.armor_center_point[1], (list, np.ndarray)):
                center_point_1 = np.array(self.armor_center_point[1], dtype=float)
                armor_plate_1 = [
                    [center_point_1[0] - self.armor_width / 2,
                     center_point_1[1] + self.armor_height / 2,
                     center_point_1[2] - self.armor_length / 2],
                    [center_point_1[0] - self.armor_width / 2,
                     center_point_1[1] - self.armor_height / 2,
                     center_point_1[2] - self.armor_length / 2],
                    [center_point_1[0] + self.armor_width / 2,
                     center_point_1[1] + self.armor_height / 2,
                     center_point_1[2] + self.armor_length / 2],
                    [center_point_1[0] + self.armor_width / 2,
                     center_point_1[1] - self.armor_height / 2,
                     center_point_1[2] + self.armor_length / 2]
                ]
                self.armor_plates_camera_positions.append(armor_plate_1)
            else:
                # print(f"警告：装甲板中心点1不是列表: {self.armor_center_point[1]}")
                return

            # 生成相邻装甲板1（索引2）
            if isinstance(self.armor_center_point[2], (list, np.ndarray)):
                center_point_2 = np.array(self.armor_center_point[2], dtype=float)
                armor_plate_2 = [
                    [center_point_2[0] - self.armor_length / 2,
                     center_point_2[1] + self.armor_height / 2,
                     center_point_2[2] - self.armor_width / 2],
                    [center_point_2[0] - self.armor_length / 2,
                     center_point_2[1] - self.armor_height / 2,
                     center_point_2[2] - self.armor_width / 2],
                    [center_point_2[0] + self.armor_length / 2,
                     center_point_2[1] + self.armor_height / 2,
                     center_point_2[2] + self.armor_width / 2],
                    [center_point_2[0] + self.armor_length / 2,
                     center_point_2[1] - self.armor_height / 2,
                     center_point_2[2] + self.armor_width / 2]
                ]
                self.armor_plates_camera_positions.append(armor_plate_2)
            else:
                # print(f"警告：装甲板中心点2不是列表: {self.armor_center_point[2]}")
                return

            # 生成相邻装甲板2（索引3）
            if isinstance(self.armor_center_point[3], (list, np.ndarray)):
                center_point_3 = np.array(self.armor_center_point[3], dtype=float)
                armor_plate_3 = [
                    [center_point_3[0] - self.armor_width / 2,
                     center_point_3[1] + self.armor_height / 2,
                     center_point_3[2] - self.armor_length / 2],
                    [center_point_3[0] - self.armor_width / 2,
                     center_point_3[1] - self.armor_height / 2,
                     center_point_3[2] - self.armor_length / 2],
                    [center_point_3[0] + self.armor_width / 2,
                     center_point_3[1] + self.armor_height / 2,
                     center_point_3[2] + self.armor_length / 2],
                    [center_point_3[0] + self.armor_width / 2,
                     center_point_3[1] - self.armor_height / 2,
                     center_point_3[2] + self.armor_length / 2]
                ]
                self.armor_plates_camera_positions.append(armor_plate_3)
            # else:
                # print(f"警告：装甲板中心点3不是列表: {self.armor_center_point[3]}")

    # 其他方法保持不变...
    def use_robot_prediction(self):
        """执行机器人预测流程"""
        # print("=== 开始机器人预测流程 ===")

        if not self.cal_armor():
            # print("装甲板计算失败，提前返回")
            return

        self.find_robot_center()
        print(f"机器人中心点: {self.center}")

        self.get_another_armor_plate_center_by_center()
        # print(f"计算后armor_center_point: {self.armor_center_point}")

        self.get_armor_by_armor_center()
        # print("机器人预测流程完成")

    def get_line_info_and_normal_vector(self, point1, point2):
        """
        给定两个二维点（x,z坐标），求出直线信息和这条直线对应的法向量
        """
        # 保持原有代码不变
        p1 = np.array(point1, dtype=float)
        p2 = np.array(point2, dtype=float)

        if p1.shape != (2,) or p2.shape != (2,):
            raise ValueError("输入点必须是二维坐标 [x, z]")

        direction_vector = p2 - p1
        line_length = np.linalg.norm(direction_vector)

        if line_length > 0:
            direction_unit = direction_vector / line_length
        else:
            direction_unit = np.zeros(2)

        midpoint = (p1 + p2) / 2
        line_equation = lambda t: p1 + t * direction_unit

        normal_vectors = []
        if line_length > 0:
            normal_vector = np.array([-direction_unit[1], direction_unit[0]], dtype=float)
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            normal_vectors.append(normal_vector)

        line_info = {
            'direction_vector': direction_unit,
            'line_equation': line_equation,
            'line_length': line_length,
            'midpoint': midpoint,
            'point1': p1,
            'point2': p2
        }

        return line_info, normal_vectors

    def find_point_by_vector_and_length(self, start_point, direction_vector, length, normalize_vector=True):
        """
        根据一个二维点、一个二维向量和一个长度值，找到具体的坐标点
        """
        # 保持原有代码不变
        start = np.array(start_point, dtype=float)
        direction = np.array(direction_vector, dtype=float)

        if start.shape != (2,):
            raise ValueError("起始点必须是二维坐标 [x, z]")
        if direction.shape != (2,):
            raise ValueError("方向向量必须是二维向量 [dx, dz]")

        vector_norm = np.linalg.norm(direction)
        if vector_norm == 0:
            raise ValueError("方向向量不能为零向量")

        if normalize_vector:
            direction_unit = direction / vector_norm
        else:
            direction_unit = direction

        target_point = start + length * direction_unit

        return target_point