import math
import cv2
from all_function import *
from all_type import *
from setting import camera_matrix, dist_coefficients, used_predict


class PnPSolver:
    """
    PnP解算器类，用于计算相机姿态和装甲板在云台坐标系下的位置。
    """

    def __init__(self):
        """
        初始化PnP解算器。
        包括相机内参矩阵和畸变系数。
        """

        # 相机内参矩阵
        self.camera_matrix = camera_matrix
        # 相机畸变系数
        self.dist_coefficients = dist_coefficients

        self.pitch_angle = 0  # 云台与相机之间的俯仰角
        self.gimbal_yaw = 0  # 云台与相机之间的偏航角

    def solve_pnp(self, armor_plate) -> tuple:
        """
        使用PnP算法计算相机位姿和物体在相机坐标系下的坐标。

        :param armor_plate: 装甲板对象，类型为 `ArmorPlate`.

        :return:
            - ret: PnP解算是否成功，类型为 `bool`。
            - rotation_vector: 相机的旋转向量，类型为 `np.ndarray`，形状为 `(3, 1)`。
            - translation_vector: 相机的平移向量，类型为 `np.ndarray`，形状为 `(3, 1)`。
            - armor: 装甲板对象，类型为 `ArmorPlate`。
        """
        # 定义现实世界中灯条区域的四个角点坐标
        light_length = armor_plate.light_size[0]  # 灯条长度
        light_width = armor_plate.light_size[1]  # 灯条宽度
        object_points = np.array([
            [0, 0, 0],  # 左上角
            [light_width, 0, 0],  # 右上角
            [0, light_length, 0],  # 左下角
            [light_width, light_length, 0]  # 右下角
        ], dtype=np.float32)

        # 使用solvePnP函数求解相机位姿, 返回值为是否成功、旋转向量、平移向量
        ret, rotation_vector, translation_vector = cv2.solvePnP(object_points, armor_plate.camera_pos,
                                                                self.camera_matrix, self.dist_coefficients)

        # 将世界坐标系下的坐标转换为相机坐标系下的坐标
        translation_matrix = np.eye(4)  # 创建一个4x4的单位矩阵
        translation_matrix[:3, 3] = translation_vector.flatten()  # 将平移向量赋值给平移矩阵的第4列，嵌入到齐次变换矩阵中
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)  # 旋转向量转换为旋转矩阵
        object_points_camera = []
        for point in object_points:
            object_points_camera.append(world2crema(point, rotation_matrix, translation_vector))

        return ret, rotation_vector, translation_vector, object_points_camera

    def get_armor_target(self, armor_plate, out_img, angle, gimbal_yaw) -> tuple:
        """
        获取装甲板的目标点。
        :param armor_plate: 装甲板对象，类型为 `ArmorPlate`。
        :param out_img: 输出图像，类型为 `np.ndarray`。
        :param angle: 云台与相机之间的俯仰角(弧度)，类型为 `float`。
        :param gimbal_yaw: 云台与相机之间的偏航角(弧度)，类型为 `float`。
        """
        self.set_pitch_angle(angle)
        self.set_gimbal_yaw(gimbal_yaw)
        point_list = armor_plate.camera_pos
        out_img = self.draw_point(out_img, point_list)
        if len(armor_plate.camera_pos) == 0:
            # 返回值：是否成功，装甲板对象，输出图像
            return False, None, out_img
        ret, rvec, tvec, object_points_cam = self.solve_pnp(armor_plate)  # 使用PnP算法求解相机与正方形的相对位置

        if ret:
            if used_predict:
                # 将相机坐标系下的坐标转换为固定云台坐标系下的坐标
                transformed_points = [rotate_around_y(camera2gimbal(point, self.pitch_angle), self.gimbal_yaw) for point
                                      in object_points_cam]
                # 计算装甲板中心点坐标
                pos = [sum(coord) / len(transformed_points) for coord in zip(*transformed_points)]
                armor = ArmorTargetPoint(pos, get_theta(transformed_points), armor_plate.color,
                                         armor_plate.troop_type, armor_plate.area, armor_plate.confident)
            else:
                pos = [sum(coord) / 4 for coord in zip(*object_points_cam)]
                pos = camera2gimbal(pos, self.pitch_angle)  # 将相机坐标系下的坐标转换为云台坐标系下的坐标
                points = [camera2gimbal(point, self.pitch_angle) for point in object_points_cam]
                armor = ArmorTargetPoint(pos, get_theta(points), armor_plate.color, armor_plate.troop_type,
                                         armor_plate.area,
                                         armor_plate.confident)
        else:  # 如果PnP求解失败
            armor = None
        # 返回值：是否成功，装甲板对象，输出图像
        return ret, armor, out_img

    def set_pitch_angle(self, angle):
        """
        设置云台与相机之间的俯仰角。
        :param angle: 云台与相机之间的俯仰角(弧度)，类型为 `float`。
        """
        if angle is not None:
            self.pitch_angle = angle
        else:
            self.pitch_angle = 0

    def set_gimbal_yaw(self, angle):
        """
        设置云台与相机之间的俯仰角。
        :param angle: 云台与相机之间的俯仰角(弧度)，类型为 `float`。
        """
        if angle is not None:
            while angle < -math.pi:
                angle += 2 * math.pi
            while angle > math.pi:
                angle -= 2 * math.pi
            self.gimbal_yaw = angle
        else:
            self.gimbal_yaw = 0

    # 在图像上标记特定的点，用于调试或显示检测结果
    def draw_point(self, out_img, points):
        for point in points:
            cv2.circle(out_img, (int(point[0]), int(point[1])), 4, [100, 100, 200], -1)
        return out_img
