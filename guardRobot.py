from typing import List, Tuple, Dict, Set
from detect_armor import ArmorDetector
from pnp_solver import PnPSolver
import numpy as np
class GuardRobot:
    def __init__(self,all_armor_plate: List):
        """
        :param all_armor_plate: 4个装甲板
        """
        self.center_point = None
        self.normal_plane = [] # 存储四个装甲板的法向量平面方程
        self.armor_plate_center = []
        assert len(all_armor_plate) == 4, "all_armor_plate 必须是长度为 4 的列表"
        self.normal_vector=[]
        self.armor_plate = all_armor_plate
        self.front_armor = all_armor_plate[0]
        self.left_armor = all_armor_plate[1]
        self.back_armor = all_armor_plate[2]
        self.right_armor = all_armor_plate[3]

    def find_armor_normal_vector(self):
        for i in range(4):
           top_left=np.asarray(self.armor_plate[i].camera_pos[0],dtype=np.float32)
           top_right=np.asarray(self.armor_plate[i].camera_pos[2],dtype=np.float32)
           bottom_left=np.asarray(self.armor_plate[i].camera_pos[1],dtype=np.float32)
           bottom_right=np.asarray(self.armor_plate[i].camera_pos[3],dtype=np.float32)
           n=np.cross(top_right-top_left,bottom_left-top_left)
           if np.linalg.norm(n)<=1e-5:
               raise ValueError("法向量计算异常，可能是点共线")
           n=n/np.linalg.norm(n)
           self.normal_vector.append(n)

    def find_armor_center(self):
        for i in range(4):
            top_left = np.asarray(self.armor_plate[i].camera_pos[0], dtype=np.float32)
            top_right = np.asarray(self.armor_plate[i].camera_pos[2], dtype=np.float32)
            bottom_left = np.asarray(self.armor_plate[i].camera_pos[1], dtype=np.float32)
            bottom_right = np.asarray(self.armor_plate[i].camera_pos[3], dtype=np.float32)
            center = (top_left + top_right + bottom_left + bottom_right) / 4.0
            self.armor_plate_center.append(center)

    def find_armor_normal_plane(self):

        def plane_from_point_normal(point, normal, normalize=True):
            """
            根据平面上一点和法向量求平面方程 ax + by + cz + d = 0 的系数。
            :param point: 平面上一点 (x0, y0, z0) 或长度为 3 的数组
            :param normal: 法向量 (a, b, c) 或长度为 3 的数组
            :param normalize: 是否对法向量进行归一化（不影响平面几何位置）
            :return: (a, b, c, d) 平面方程系数
            """
            point = np.asarray(point, dtype=float).reshape(3)
            normal = np.asarray(normal, dtype=float).reshape(3)

            if normalize:
                n_norm = np.linalg.norm(normal)
                if n_norm < 1e-12:
                    raise ValueError("法向量长度为 0，无法确定平面")
                normal = normal / n_norm

            a, b, c = normal
            d = -np.dot(normal, point)
            return [a, b, c, d]

        def is_point_on_plane(point, plane_coeff, tol=1e-6):
            """
            判断点是否在平面上。
            :param point: (x, y, z)
            :param plane_coeff: (a, b, c, d)
            :param tol: 允许的数值误差
            """
            x, y, z = point
            a, b, c, d = plane_coeff
            val = a * x + b * y + c * z + d
            return abs(val) < tol

        def point_to_plane_distance(point, plane_coeff):
            """
            计算点到平面的有向距离。
            :param point: (x, y, z)
            :param plane_coeff: (a, b, c, d)
            :return: 距离（法向量方向上为正）
            """
            x, y, z = point
            a, b, c, d = plane_coeff
            num = a * x + b * y + c * z + d
            den = np.sqrt(a * a + b * b + c * c)
            if den < 1e-12:
                raise ValueError("非法平面系数")
            return num / den

        for i in range(4):
            self.normal_plane.append(plane_from_point_normal(self.armor_plate_center[i],self.normal_vector[i]))

    def intersect_two_planes(self, p1, p2, eps=1e-8):
        """
        p1, p2: 各为长度为4的可迭代对象 [a, b, c, d]，表示 a x + b y + c z + d = 0
        返回:
          has_line: bool，是否有唯一交线
          point: np.ndarray shape (3,)  交线上的一个点
          direction: np.ndarray shape (3,)  交线方向向量(已归一化)
        """
        a1, b1, c1, d1 = p1
        a2, b2, c2, d2 = p2

        n1 = np.array([a1, b1, c1], dtype=float)
        n2 = np.array([a2, b2, c2], dtype=float)

        # 方向向量 = 两法向量叉乘
        direction = np.cross(n1, n2)
        if np.linalg.norm(direction) < eps:
            # 法向量平行 -> 平面平行或重合，没有唯一交线
            return False, None, None

        # 归一化方向
        direction = direction / np.linalg.norm(direction)

        # 为了求交线上的一点，任选一个坐标作为自由变量（比如 z=0），
        # 解一个 2x2 的线性方程组得到 x, y。
        # 若 z=0 无法解(方程组奇异)，再尝试 y=0 或 x=0。
        def solve_with_fixed(var_idx, var_value):
            # var_idx: 0->x, 1->y, 2->z
            # 固定一个变量，解剩下两个
            A = []
            b = []
            # 平面1: a1 x + b1 y + c1 z + d1 = 0
            # 平面2: a2 x + b2 y + c2 z + d2 = 0

            # 用未知量索引列表
            unknown_idx = [i for i in range(3) if i != var_idx]

            # 构造两条方程，仅含 unknown_idx 中的变量
            coeffs1 = [n1[unknown_idx[0]], n1[unknown_idx[1]]]
            rhs1 = -d1 - n1[var_idx] * var_value
            coeffs2 = [n2[unknown_idx[0]], n2[unknown_idx[1]]]
            rhs2 = -d2 - n2[var_idx] * var_value

            A = np.array([coeffs1, coeffs2], dtype=float)
            b_vec = np.array([rhs1, rhs2], dtype=float)

            if abs(np.linalg.det(A)) < eps:
                return None  # 此固定方式不可行

            sol = np.linalg.solve(A, b_vec)
            x, y, z = 0.0, 0.0, 0.0
            vals = {var_idx: var_value,
                    unknown_idx[0]: sol[0],
                    unknown_idx[1]: sol[1]}
            x, y, z = vals[0], vals[1], vals[2]
            return np.array([x, y, z], dtype=float)

        point = solve_with_fixed(2, 0.0)  # 先试 z = 0
        if point is None:
            point = solve_with_fixed(1, 0.0)  # 再试 y = 0
        if point is None:
            point = solve_with_fixed(0, 0.0)  # 最后试 x = 0

        if point is None:
            # 极端退化情况
            return False, None, None

        return True, point, direction

    def find_center_point(self):
        self.find_armor_normal_vector()
        self.find_armor_center()
        self.find_armor_normal_plane()
        _,self.center_point,_=self.intersect_two_planes(self.normal_plane[0], self.normal_plane[1])
        self.center_point=self.center_point[:2]