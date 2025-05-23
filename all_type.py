import math
from enum import Enum
import numpy as np
import cv2

small_armor_size = (0.135, 0.0555)  # 小装甲板灯条区域的长宽(单位: m)
big_armor_size = (0.230, 0.0555)  # 大装甲板灯条区域的长宽(单位: m)


# 枚举类：颜色
class Color(Enum):
    RED = 0
    BLUE = 1


# 枚举类：类型
class ArmorType(Enum):
    LARGE = "大装甲板"
    SMALL = "小装甲板"


# 枚举类：兵种
class TroopType(Enum):
    SENTINEL = "哨兵"
    HERO = "英雄"
    INFANTRY = "步兵"


# 枚举类：相机品牌
class CameraType(Enum):
    DAHENG = "大恒"
    HAIKANG = "海康"


# 枚举类：相机种类
class CameraID(Enum):
    DAHENG_0 = "大恒0"
    HAIKANG_1 = "海康1"
    HAIKANG_2 = "海康2"


# 枚举类：跟踪状态
class TracState(Enum):
    LOST = 0
    DETECTING = 1
    TRACKING = 2
    TEMP_LOST = 3


class ArmorTargetPoint:
    """
    装甲板中心点类。
    """

    def __init__(self, gimbal_pos, rotate_yaw, color: Color, troop_type: TroopType, area=0, confident=0):
        """
        初始化装甲板中心对象。
        :param gimbal_pos: 云台坐标系下的位置 (x, y, z)。
        :param rotate_yaw: 静态云台坐标系下装甲板平面法向量在xOz平面投影与z轴的夹角 (单位: 弧度)。
        :param color: 装甲板颜色。
        :param troop_type: 兵种类型。
        :param area: 装甲板面积, 用于后续装甲板选择策略
        :param confident: 置信度
        """
        self._gimbal_pos = list(gimbal_pos)
        self.yaw = rotate_yaw
        self.color = color
        self.troop_type = troop_type
        self.area = area
        self.confident = confident
        self.armor_type = ArmorType.SMALL
        self.light_size = small_armor_size  # 小装甲板的灯条区域长宽
        if troop_type == TroopType.HERO:
            self.armor_type = ArmorType.LARGE
            self.light_size = big_armor_size  # 大装甲板的灯条区域长宽

    @property
    def gimbal_pos(self):  # 访问云台坐标系下的位置
        return self._gimbal_pos

    @gimbal_pos.setter
    def gimbal_pos(self, value):
        """ 确保 xyz 与 gimbal_pos 保持同步 """
        self._gimbal_pos = list(value)

    @property
    def x(self):
        return self._gimbal_pos[0]

    @x.setter
    def x(self, value):
        self._gimbal_pos[0] = value

    @property
    def y(self):
        return self._gimbal_pos[1]

    @y.setter
    def y(self, value):
        self._gimbal_pos[1] = value

    @property
    def z(self):
        return self._gimbal_pos[2]

    @z.setter
    def z(self, value):
        self._gimbal_pos[2] = value

    @property
    def camera_yaw(self):  # 访问装甲板中心点在相机视野下的yaw
        return math.atan(self.z / self.x) + math.pi / 2

    def __str__(self):
        if self.color.value == 0:
            color_str = "红色"
        else:
            color_str = "蓝色"
        return f"{self.armor_type.value}，颜色: {color_str}, 兵种: {self.troop_type.value}, " \
               f"装甲板平面旋转角: ({self.x:.3f}, {self.y:.3f}, {self.z:.3f})，" \
               f"yaw: {self.yaw:.3f}, 面积: {self.area:.3f}, 置信度: {self.confident:.3f}"


class ArmorPlate:
    """
    装甲板类。
    """

    def __init__(self, points, color: Color, troop_type: TroopType, area=0, confident=0):
        """
        初始化装甲板对象。
        :param points: 四个顶点坐标,
         格式 np.array([top_left, bottom_left, top_right, bottom_right], dtype=np.float32), points=(x, y)

        :param color: 装甲板颜色。
        :param troop_type: 兵种类型。
        :param area: 装甲板面积, 用于后续装甲板选择策略
        :param confident: 置信度
        """
        self.camera_pos = points
        self.color = color
        self.troop_type = troop_type
        self.area = area
        self.confident = confident
        self.armor_type = ArmorType.SMALL
        self.light_size = small_armor_size  # 小装甲板的灯条区域长宽
        if troop_type == TroopType.HERO:  # 英雄用大装甲板
            self.armor_type = ArmorType.LARGE
            self.light_size = big_armor_size  # 大装甲板的灯条区域长宽

    def __str__(self):
        if self.color.value == 0:
            color_str = "红色"
        else:
            color_str = "蓝色"
        return f"{self.armor_type.value}，颜色: {color_str}, 兵种: {self.troop_type.value}, " \
               f"相机坐标系下的位置: {self.camera_pos}, 面积: {self.area:.3f}, 置信度: {self.confident:.3f}"


# 这个类是君喵纯视觉无模型使用的类
class LightCV:
    def __init__(self, rotated_rect):
        self.bounding_rect = rotated_rect
        self.center = rotated_rect[0]  # 中心点
        self.length = max(rotated_rect[1])  # 长度
        self.width = min(rotated_rect[1])  # 宽度
        self.tilt_angle = rotated_rect[2]  # 旋转角度
        self.color = None  # 颜色将通过统计确定
        self.top_point = None
        self.bottom_point = None

    def boundingRect(self):
        return cv2.boxPoints(self.bounding_rect)  # 获取包围矩形的四个顶点

    def top(self):
        # 获取灯条顶部点
        return self.bounding_rect[0][0], self.bounding_rect[0][1] - self.length / 2

    def bottom(self):
        # 获取灯条底部点
        return (self.bounding_rect[0][0], self.bounding_rect[0][1] + self.length / 2)
