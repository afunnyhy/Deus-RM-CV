from typing import List
import numpy as np


class GuardRobot:
    def __init__(self, all_armor_plate: List):
        """根据若干装甲板估计小车相关几何信息（相机坐标系）。

        约定：至少提供 2 块装甲板即可；内部主要使用前两块装甲板。
        """
        assert len(all_armor_plate) >= 2, "all_armor_plate 至少需要 2 块装甲板"
        self.armor_plate = all_armor_plate

        self.center_point = None        # 代表小车中心的3D点
        self.line_point = None          # 平面交线上的一点
        self.line_dir = None            # 平面交线方向向量（单位向量）
        self.normal_plane = []          # 每块装甲板的平面方程 [a,b,c,d]
        self.armor_plate_center = []    # 每块装甲板中心 3D
        self.normal_vector = []         # 每块装甲板法向量

    def find_armor_normal_vector(self):
        self.normal_vector.clear()
        for armor in self.armor_plate:
            top_left = np.asarray(armor.camera_pos[0], dtype=np.float32)
            top_right = np.asarray(armor.camera_pos[2], dtype=np.float32)
            bottom_left = np.asarray(armor.camera_pos[1], dtype=np.float32)
            # bottom_right = np.asarray(armor.camera_pos[3], dtype=np.float32)
            n = np.cross(top_right - top_left, bottom_left - top_left)
            if np.linalg.norm(n) <= 1e-5:
                raise ValueError("法向量计算异常，可能是点共线")
            n = n / np.linalg.norm(n)
            self.normal_vector.append(n)

    def find_armor_center(self):
        self.armor_plate_center.clear()
        for armor in self.armor_plate:
            top_left = np.asarray(armor.camera_pos[0], dtype=np.float32)
            top_right = np.asarray(armor.camera_pos[2], dtype=np.float32)
            bottom_left = np.asarray(armor.camera_pos[1], dtype=np.float32)
            bottom_right = np.asarray(armor.camera_pos[3], dtype=np.float32)
            center = (top_left + top_right + bottom_left + bottom_right) / 4.0
            self.armor_plate_center.append(center)

    def find_armor_normal_plane(self):
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
            d = -np.dot(normal, point)
            return [a, b, c, d]

        for center, n in zip(self.armor_plate_center, self.normal_vector):
            self.normal_plane.append(plane_from_point_normal(center, n))

    def intersect_two_planes(self, p1, p2, eps=1e-8):
        a1, b1, c1, d1 = p1
        a2, b2, c2, d2 = p2

        n1 = np.array([a1, b1, c1], dtype=float)
        n2 = np.array([a2, b2, c2], dtype=float)

        direction = np.cross(n1, n2)
        if np.linalg.norm(direction) < eps:
            return False, None, None

        direction = direction / np.linalg.norm(direction)

        def solve_with_fixed(var_idx, var_value):
            unknown_idx = [i for i in range(3) if i != var_idx]
            coeffs1 = [n1[unknown_idx[0]], n1[unknown_idx[1]]]
            rhs1 = -d1 - n1[var_idx] * var_value
            coeffs2 = [n2[unknown_idx[0]], n2[unknown_idx[1]]]
            rhs2 = -d2 - n2[var_idx] * var_value

            A = np.array([coeffs1, coeffs2], dtype=float)
            b_vec = np.array([rhs1, rhs2], dtype=float)

            if abs(np.linalg.det(A)) < eps:
                return None

            sol = np.linalg.solve(A, b_vec)
            vals = {var_idx: var_value, unknown_idx[0]: sol[0], unknown_idx[1]: sol[1]}
            x, y, z = vals[0], vals[1], vals[2]
            return np.array([x, y, z], dtype=float)

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

        :return: (point, direction)，如果无唯一交线则抛异常。
        """
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
        """使用两块装甲板的法向量和中心点定义直线，并求“x,y 尽量接近”的最近点中点作为小车中心。

        对于每块装甲板 i：
          - 直线 Li(t) = Ci + t * ni，其中 Ci 为装甲板中心，ni 为法向量（单位向量）。
        这里优先让两直线在水平投影（x,y）上尽量靠近：
          - 先按 3D 最近连线公式求 P1,P2；
          - 若两直线几乎平行，则退化为两中心点 C1,C2 的中点；
        最终返回 (P1+P2)/2 作为小车中心点。
        """
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

        # 避免法向量退化
        if np.linalg.norm(n1) < 1e-6 or np.linalg.norm(n2) < 1e-6:
            raise ValueError("法向量长度过小，无法定义直线")
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)

        # 3D 最近连线公式（L1: C1+t1*n1, L2: C2+t2*n2）
        w0 = C1 - C2
        a = np.dot(n1, n1)  # =1
        b = np.dot(n1, n2)
        c = np.dot(n2, n2)  # =1
        d = np.dot(n1, w0)
        e = np.dot(n2, w0)
        denom = a * c - b * b
        if abs(denom) < 1e-6:
            # 两直线近似平行：退化为两中心点中点，更稳定
            mid = 0.5 * (C1 + C2)
            self.center_point = mid
            return self.center_point

        t1 = (b * e - c * d) / denom
        t2 = (a * e - b * d) / denom

        P1 = C1 + t1 * n1
        P2 = C2 + t2 * n2

        # 中点作为小车中心
        center = 0.5 * (P1 + P2)
        self.center_point = center
        return self.center_point

    def find_center_point(self):
        """兼容旧接口：仍然返回基于平面交线的中心点。"""
        armors = self.armor_plate[:2]
        self.armor_plate = armors
        self.find_armor_normal_vector()
        self.find_armor_center()
        self.find_armor_normal_plane()
        has_line, point_on_line, _ = self.intersect_two_planes(self.normal_plane[0], self.normal_plane[1])
        if not has_line or point_on_line is None:
            raise ValueError("两平面没有唯一交线，无法计算中心点")
        x, y = point_on_line[0], point_on_line[1]
        z_mean = 0.5 * (self.armor_plate_center[0][2] + self.armor_plate_center[1][2])
        self.center_point = np.array([x, y, z_mean], dtype=float)
        return self.center_point
