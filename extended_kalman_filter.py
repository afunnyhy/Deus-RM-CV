import numpy as np


class ExtendedKalmanFilter:
    """
    扩展卡尔曼滤波器类
    """

    def __init__(self, P0):
        self.P_post = P0
        self.n = P0.shape[0]
        self.I = np.eye(self.n)
        self.x_pri = np.zeros(self.n)
        self.x_post = np.zeros(self.n)

        self.s2qxyz = 0.2  # 位置过程噪声（原0.5→0.2）
        self.s2qyaw = 0.1  # 航向过程噪声（原0.05→0.1）
        self.s2qr = 0.05  # 半径过程噪声（原0.1→0.05）
        self.r_xyz_factor = 0.1  # 位置测量噪声系数（原0.05→0.1）
        self.r_yaw = 0.1  # 航向测量噪声（原3→0.5）

        # 异常值检测相关
        self.innovation_threshold = 10.0
        self.outlier_count = 0
        self.total_updates = 0

        # 最小和最大噪声值
        self.min_noise = 1e-3

        # 半径约束
        self.min_radius = 0.2
        self.max_radius = 0.4  # 最大半径约束

    def h(self, x):
        """观测函数 - 将状态映射到观测空间"""
        xc, yc, zc, yaw, r = x[0], x[2], x[4], x[6], x[8]
        z = np.zeros(4)
        z[0] = xc + r * np.cos(yaw)  # xa - 装甲板x坐标
        z[1] = yc  # ya - 装甲板y坐标等于圆心y坐标
        z[2] = zc - r * np.sin(yaw)  # za - 装甲板z坐标
        z[3] = yaw  # 航向角
        return z

    def f(self, x, dt):
        """状态转移函数 - 预测下一状态"""
        x_new = np.copy(x)
        x_new[0] += x[1] * dt  # xc - 圆心x坐标
        x_new[2] += x[3] * dt  # yc - 圆心y坐标
        x_new[4] += x[5] * dt  # zc - 圆心z坐标
        x_new[6] += x[7] * dt  # yaw - 航向角

        # 标准化航向角到[-π, π]
        x_new[6] = np.mod(x_new[6] + np.pi, 2 * np.pi) - np.pi
        return x_new

    def jacobian_f(self, x, dt):
        """计算状态转移函数的雅可比矩阵"""
        F = np.zeros((9, 9))
        F[0, 0] = 1
        F[0, 1] = dt
        F[1, 1] = 1
        F[2, 2] = 1
        F[2, 3] = dt
        F[3, 3] = 1
        F[4, 4] = 1
        F[4, 5] = dt
        F[5, 5] = 1
        F[6, 6] = 1
        F[6, 7] = dt
        F[7, 7] = 1
        F[8, 8] = 1
        return F

    def jacobian_h(self, x):
        """计算观测函数的雅可比矩阵"""
        yaw, r = x[6], x[8]
        H = np.zeros((4, 9))
        H[0, 0] = 1  # dz[0]/dxc
        H[0, 6] = -r * np.sin(yaw)  # dz[0]/dyaw
        H[0, 8] = np.cos(yaw)  # dz[0]/dr

        H[1, 2] = 1  # dz[1]/dyc

        H[2, 4] = 1  # dz[2]/dzc
        H[2, 6] = -r * np.cos(yaw)  # dz[2]/dyaw
        H[2, 8] = -np.sin(yaw)  # dz[2]/dr

        H[3, 6] = 1  # dz[3]/dyaw
        return H

    def update_Q(self, dt):
        """更新过程噪声协方差矩阵"""
        Q = np.zeros((9, 9))
        t = dt

        # 位置噪声 (x, y, z)
        q_pos_pos = (t ** 4 / 4) * self.s2qxyz
        q_pos_vel = (t ** 3 / 2) * self.s2qxyz
        q_vel_vel = (t ** 2) * self.s2qxyz

        # 航向角噪声
        q_yaw_yaw = (t ** 4 / 4) * self.s2qyaw
        q_yaw_vyaw = (t ** 3 / 2) * self.s2qyaw  # 正确使用航向角噪声参数
        q_vyaw_vyaw = (t ** 2) * self.s2qyaw

        # 半径噪声
        q_r = (t ** 4 / 4) * self.s2qr

        # X位置和速度
        Q[0, 0] = q_pos_pos
        Q[0, 1] = q_pos_vel
        Q[1, 0] = q_pos_vel
        Q[1, 1] = q_vel_vel

        # Y位置和速度
        Q[2, 2] = q_pos_pos
        Q[2, 3] = q_pos_vel
        Q[3, 2] = q_pos_vel
        Q[3, 3] = q_vel_vel

        # Z位置和速度
        Q[4, 4] = q_pos_pos
        Q[4, 5] = q_pos_vel
        Q[5, 4] = q_pos_vel
        Q[5, 5] = q_vel_vel

        # 航向角和角速度
        Q[6, 6] = q_yaw_yaw
        Q[6, 7] = q_yaw_vyaw
        Q[7, 6] = q_yaw_vyaw
        Q[7, 7] = q_vyaw_vyaw

        # 半径
        Q[8, 8] = q_r

        return Q

    def update_R(self, z):
        """更新测量噪声协方差矩阵"""
        # 确保测量噪声不会过小
        R = np.diag([
            max(abs(self.r_xyz_factor * z[0]), self.min_noise),
            max(abs(self.r_xyz_factor * z[1]), self.min_noise),
            max(abs(self.r_xyz_factor * z[2]), self.min_noise),
            max(self.r_yaw, self.min_noise)
        ])
        return R

    def set_state(self, x0):
        """设置滤波器的初始状态"""
        self.x_post = x0

    def predict(self, dt):
        """预测步骤 - 估计下一状态"""
        # 确保dt是有效值
        dt = max(0.001, dt)  # 防止dt为零或负值

        # 计算状态转移矩阵和过程噪声
        F = self.jacobian_f(self.x_post, dt)
        Q = self.update_Q(dt)

        # 状态预测
        self.x_pri = self.f(self.x_post, dt)

        # 协方差预测
        self.P_pri = F @ self.P_post @ F.T + Q

        return self.x_pri

    def update(self, z):
        """更新步骤 - 使用测量值修正状态估计"""
        # 计算观测雅可比矩阵和测量噪声
        H = self.jacobian_h(self.x_pri)
        R = self.update_R(z)

        # 计算创新向量（实际测量与预测测量的差值）
        innovation = z - self.h(self.x_pri)

        # 标准化角度差异到[-π, π]
        innovation[3] = np.mod(innovation[3] + np.pi, 2 * np.pi) - np.pi

        # 计算创新协方差
        S = H @ self.P_pri @ H.T + R

        # 异常值检测 - 马氏距离
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_distance = np.sqrt(innovation.T @ S_inv @ innovation)

            # 动态阈值调整
            if mahalanobis_distance > self.innovation_threshold:
                self.outlier_count += 1
                # 如果异常率过高，适当放宽阈值
                if self.outlier_count > 5 and self.total_updates > 20:
                    outlier_ratio = self.outlier_count / self.total_updates
                    if outlier_ratio > 0.3:
                        self.innovation_threshold *= 1.1
                return self.x_pri

            self.total_updates += 1
        except np.linalg.LinAlgError:
            # 如果计算失败，继续处理
            pass

        # 计算卡尔曼增益
        try:
            K = self.P_pri @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # 数值稳定性保护
            S += np.eye(S.shape[0]) * 1e-8
            K = self.P_pri @ H.T @ np.linalg.inv(S)

        # 更新状态
        self.x_post = self.x_pri + K @ innovation

        # 使用Joseph形式更新协方差矩阵，提高数值稳定性
        I_KH = self.I - K @ H
        self.P_post = I_KH @ self.P_pri @ I_KH.T + K @ R @ K.T

        # 强制协方差矩阵对称化
        self.P_post = (self.P_post + self.P_post.T) / 2

        # 应用物理约束
        self.x_post[8] = min(self.max_radius, max(self.min_radius, self.x_post[8]))  # 确保半径在有效范围内

        # 标准化角度
        self.x_post[6] = np.mod(self.x_post[6] + np.pi, 2 * np.pi) - np.pi

        return self.x_post
