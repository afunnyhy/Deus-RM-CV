from typing import List
import numpy as np

from all_type import ArmorPlate


class GuardRobot:
    def __init__(self, all_armor_plate: List):
        """根据若干装甲板估计小车相关几何信息（相机坐标系）。

        参数
        ------
        all_armor_plate : List[ArmorPlate]
            当前帧中可用的装甲板列表，每个 ArmorPlate 中应包含 4 个在相机坐标系下的 3D 角点 camera_pos。

        约定
        ------
        - 至少需要 2 块装甲板，几何上才能稳定估计出“小车中心”等信息；
        - 目前 get_center_from_normals 和 calculate_another_armor_by_center 只使用前两块装甲板；
        - 坐标系为相机坐标系：x 向右, y 向下, z 朝前（与工程中 PnP 得到的一致）。
        """
        assert len(all_armor_plate) >= 2, "all_armor_plate 至少需要 2 块装甲板"
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
        - 该中心常被用作“装甲板中心 -> 车中心”的连线起点。
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

    # def find_center_point(self):
    #     """兼容旧接口：仍然返回基于“法平面交线”的中心点估计。
    #
    #     流程:
    #     1. 由每块装甲板的中心+法向量构造平面;
    #     2. 求两平面交线, 取该交线在 y=0 平面上的交点的 (x,z);
    #     3. y 取两装甲板中心 y 坐标的平均值。
    #
    #     该方法与 get_center_from_normals 略有不同, 保留以防旧代码依赖。
    #     """
    #     # 只使用前两块装甲板
    #     armors = self.armor_plate[:2]
    #     self.armor_plate = armors
    #     self.find_armor_normal_vector()
    #     self.find_armor_center()
    #     self.find_armor_normal_plane()
    #
    #     has_line, point_on_line, direction = self.intersect_two_planes(
    #         self.normal_plane[0], self.normal_plane[1]
    #     )
    #     if not has_line or point_on_line is None or direction is None:
    #         raise ValueError("两平面没有唯一交线，无法计算中心点")
    #
    #     # 交线参数方程: L(t) = point_on_line + t * direction
    #     p = np.asarray(point_on_line, dtype=float).reshape(3)
    #     d = np.asarray(direction, dtype=float).reshape(3)
    #
    #     # 目标是在 y=0 平面上的交点, 即寻找 t 使得 p_y + t * d_y = 0
    #     if abs(d[1]) > 1e-8:
    #         t = -p[1] / d[1]
    #         inter = p + t * d
    #         x_c, z_c = inter[0], inter[2]
    #     else:
    #         # 交线几乎平行于 y 轴平面, 无法稳定求与 y=0 的交点
    #         # 退化为直接使用交线上给定点的 x,z
    #         x_c, z_c = p[0], p[2]
    #
    #     # y 坐标取两块装甲板中心 y 的平均值
    #     C1 = np.asarray(self.armor_plate_center[0], dtype=float).reshape(3)
    #     C2 = np.asarray(self.armor_plate_center[1], dtype=float).reshape(3)
    #     y_c = 0.5 * (C1[1] + C2[1])
    #
    #     self.center_point = np.array([x_c, y_c, z_c], dtype=float)
    #     return self.center_point

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



