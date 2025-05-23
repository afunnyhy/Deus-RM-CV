"""
通用函数
"""
import math
import numpy as np
from setting import origin_gimbal, camera_matrix


# 将世界坐标系转换为相机坐标系，pos为世界坐标系下的坐标，rmat为旋转矩阵，tvec为平移向量
def world2crema(pos, rmat, tvec):
    x, y, z = pos  # 提取世界坐标系下的坐标值
    original_coord = np.array([[x], [y], [z]])  # 构造坐标向量

    # 将坐标向量应用旋转矩阵和平移向量
    rotated_translated_coord = np.dot(rmat, original_coord) + tvec
    # 提取旋转和平移后的坐标值
    new_x = rotated_translated_coord[0][0]
    new_y = rotated_translated_coord[1][0]
    new_z = rotated_translated_coord[2][0]
    new_pos = [new_x, new_y, new_z]
    return new_pos


# 将相机坐标系转换为云台坐标系，pos为相机坐标系下的坐标，angel为云台与相机的夹角
def camera2gimbal(pos, angel):
    origin_frame1 = np.array([0, 0, 0])
    x_axis_frame1 = np.array([1, 0, 0])  # 假设x轴方向
    y_axis_frame1 = np.array([0, 1, 0])  # 假设y轴方向
    z_axis_frame1 = np.array([0, 0, 1])  # 假设z轴方向

    # 定义第二个坐标系的原点和轴方向
    origin_frame2 = origin_gimbal  # 假设第二个坐标系的原点相对于第一个坐标系进行了平移
    x_axis_frame2 = np.array([-1, 0, 0])  # 假设x轴方向相对于第一个坐标系进行了旋转
    y_axis_frame2 = np.array([0, -math.cos(angel), math.sin(angel)])  # 假设y轴方向相对于第一个坐标系进行了旋转
    z_axis_frame2 = np.array([0, math.sin(angel), math.cos(angel)])  # 假设z轴方向相对于第一个坐标系进行了旋转
    # 计算旋转矩阵
    rotation_matrix = np.array([x_axis_frame2, y_axis_frame2, z_axis_frame2]).T  # 构造旋转矩阵
    translation_vector = origin_frame2 - origin_frame1  # 平移向量

    # 将质点的坐标从第一个坐标系变换到第二个坐标系
    point_in_frame2 = rotation_matrix @ (pos - origin_frame1) + translation_vector
    return point_in_frame2  # 返回云台坐标系下的坐标


# 将云台坐标系转换为相机坐标系，pos为云台坐标系下的坐标，angel为云台与相机的夹角
def gimbal2camera(pos, angel):
    origin_frame1 = np.array([0, 0, 0])
    x_axis_frame1 = np.array([1, 0, 0])  # 假设x轴方向
    y_axis_frame1 = np.array([0, 1, 0])  # 假设y轴方向
    z_axis_frame1 = np.array([0, 0, 1])  # 假设z轴方向

    # 定义第二个坐标系的原点和轴方向
    origin_frame2 = origin_gimbal  # 假设第二个坐标系的原点相对于第一个坐标系进行了平移
    x_axis_frame2 = np.array([-1, 0, 0])  # 假设x轴方向相对于第一个坐标系进行了旋转
    y_axis_frame2 = np.array([0, -math.cos(angel), math.sin(angel)])  # 假设y轴方向相对于第一个坐标系进行了旋转
    z_axis_frame2 = np.array([0, math.sin(angel), math.cos(angel)])  # 假设z轴方向相对于第一个坐标系进行了旋转

    # 计算旋转矩阵
    rotation_matrix_inv = np.linalg.inv(np.array([x_axis_frame2, y_axis_frame2, z_axis_frame2])).T  # 逆矩阵
    translation_vector = origin_frame2 - origin_frame1  # 平移向量
    translation_vector_inv = -rotation_matrix_inv @ translation_vector
    # 将质点的坐标从第一个坐标系变换到第二个坐标系
    point_in_frame1 = rotation_matrix_inv @ (pos - translation_vector) + origin_frame1
    return point_in_frame1  # 返回相机坐标系下的坐标


# 将相机坐标系转换为像素坐标系，用于展示三维坐标在二维平面的位置，pos为相机坐标系下的坐标
def camera2xy(pos):
    x, y, z = pos
    pos1 = np.dot(camera_matrix, [[x], [y], [z]])
    pos1 /= pos1[2]
    pixel_coords = tuple(pos1[:2].astype(int))
    x1 = pixel_coords[0][0]
    y1 = pixel_coords[1][0]
    pos2 = [x1, y1]
    return pos2


# 计算弹道补偿角度,即pitch角度，水平为0度
def ballistic_compensation(pos, projectile_velocity=22):  # 默认弹丸速度22m/s
    g = 9.83  # 上海的重力加速度
    # 使用云台坐标系下的坐标计算补偿角度
    v2 = projectile_velocity * projectile_velocity
    x, y, z = pos  # 提取目标点的坐标值，x为水平向右方向，y为竖直向上方向，z为深度向外(向目标)方向
    d2 = x * x + z * z
    d = math.sqrt(d2)
    delta_theta = v2 * v2 - 2 * g * y * v2 - g * g * d2
    if delta_theta < 0:
        return 0
    tan_theta1 = (v2 - math.sqrt(delta_theta)) / (g * d)
    tan_theta2 = (v2 + math.sqrt(delta_theta)) / (g * d)
    theta1 = math.atan(tan_theta1)
    theta2 = math.atan(tan_theta2)
    if abs(theta1) <= abs(theta2):
        return theta1
    else:
        return theta2


# 从一个旋转向量（rvec）计算对应的旋转矩阵，并从中提取偏航角（yaw）
def get_yal(rvec):
    theta = np.linalg.norm(rvec)  # 旋转角度
    if theta == 0:
        return np.eye(3)  # 如果没有旋转，返回单位矩阵
    axis = rvec / theta  # 归一化旋转轴
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    # 基于 Rodrigues 公式计算旋转矩阵
    RO = np.array([
        [cos_theta + axis[0] ** 2 * (1 - cos_theta),
         axis[0] * axis[1] * (1 - cos_theta) - axis[2] * sin_theta,
         axis[0] * axis[2] * (1 - cos_theta) + axis[1] * sin_theta],

        [axis[1] * axis[0] * (1 - cos_theta) + axis[2] * sin_theta,
         cos_theta + axis[1] ** 2 * (1 - cos_theta),
         axis[1] * axis[2] * (1 - cos_theta) - axis[0] * sin_theta],

        [axis[2] * axis[0] * (1 - cos_theta) - axis[1] * sin_theta,
         axis[2] * axis[1] * (1 - cos_theta) + axis[0] * sin_theta,
         cos_theta + axis[2] ** 2 * (1 - cos_theta)
         ]
    ])
    RO = np.squeeze(RO)  # 去掉维度为1的维度
    yaw = np.arctan2(RO[1, 0], RO[0, 0])  # 提取yaw角
    return yaw


# 计算PnP解算后装甲板平面的法向量并投影到xoz平面上，返回与xoz平面的夹角
def get_theta(armor_points):
    p1, p2, p3, _ = armor_points
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    normal_vector = np.cross(v1, v2)
    normal_vector /= np.linalg.norm(normal_vector)
    projection = np.array([normal_vector[0], 0, normal_vector[2]])
    projection /= np.linalg.norm(projection)
    if projection[2] > 0:
        projection = -projection
    yoz_normal = np.array([1, 0, 0])  # yoz平面的法向量
    dot_product = np.dot(projection, yoz_normal)
    angle = np.arccos(dot_product)
    return angle


# 逆时针旋转坐标系中的点
def rotate_around_y(pos, yaw):
    """
    让一个3D点围绕Y轴逆时针旋转yaw角度（弧度）
    :param pos: list or tuple, [x, y, z]
    :param yaw: float, 旋转角度（单位：弧度）
    :return: list, 旋转后的[x', y', z']坐标
    """
    yaw = math.pi - yaw
    x, y, z = pos
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    x_new = x * cos_yaw - z * sin_yaw
    z_new = x * sin_yaw + z * cos_yaw
    return [x_new, y, z_new]
