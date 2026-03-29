import math
from setting import g, defaults_bullet_speed, cd, my_TroopType


class BallisticCompensator:
    def __init__(self, rho=1.225, dt=0.004, tolerance=0.003, max_iterations=12):
        """
        初始化弹道补偿器

        :param rho: 空气密度 (kg/m^3)
        :param dt: 积分步长 (s)
        :param tolerance: 容许的高度误差 (m)
        :param max_iterations: 最大迭代修正次数
        """
        self.g = g
        self.rho = rho
        self.default_bullet_speed = defaults_bullet_speed
        self.my_TroopType = my_TroopType
        self.max_iterations = max_iterations
        self.dt = dt
        self.tolerance = tolerance

        # 弹丸参数
        mass = 0.041 if self.my_TroopType == my_TroopType.HERO else 0.0032
        radius = 0.02125 if self.my_TroopType == my_TroopType.HERO else 0.0084

        # 预先计算空气阻力常数
        cross_sectional_area = math.pi * (radius * radius)
        self.k = (0.5 * rho * cd * cross_sectional_area) / mass

    def recorrect_bullet_speed(self, v0):
        if v0 < self.default_bullet_speed * 0.75:
            return self.default_bullet_speed
        return v0

    def calculate_ideal_pitch(self, d, y, v0):
        """计算真空理想弹道角度作为初始猜测值"""
        v2 = v0 * v0
        delta = v2 * v2 - 2 * self.g * y * v2 - self.g * self.g * d * d

        if delta < 0:
            return None  # 超出物理射程

        tan_theta = (v2 - math.sqrt(delta)) / (self.g * d)
        return math.atan(tan_theta)

    def _simulate_trajectory(self, theta, d_target, v0):
        """使用欧拉法模拟弹丸飞行，返回到达目标水平距离时的竖直高度"""
        d = 0.0
        y = 0.0

        vd = v0 * math.cos(theta)
        vy = v0 * math.sin(theta)

        # 性能优化：将实例属性提取为局部变量，避免 while 循环内高频的字典查找
        _k = self.k
        _g = self.g
        _dt = self.dt

        while d < d_target:
            # 性能优化：用乘法替代求幂运算 (**2)
            v_mag = math.sqrt(vd * vd + vy * vy)

            ad = -_k * v_mag * vd
            ay = -_g - _k * v_mag * vy

            vd += ad * _dt
            vy += ay * _dt
            d += vd * _dt
            y += vy * _dt

            if y < -10.0 or vd <= 0:
                break

        return y

    def calculate_angle(self, pos_in, v0):
        """
        计算包含空气阻力的补偿角度 (Pitch)
        """
        v0 = self.recorrect_bullet_speed(v0)
        x, y_target, z = pos_in
        # 性能优化：用乘法替代求幂
        d_target = math.sqrt(x * x + z * z)

        # 1. 获取初始猜测角度
        theta = self.calculate_ideal_pitch(d_target, y_target, v0)
        if theta is None:
            return 0

        # 2. 经验导引法迭代求精确解
        for _ in range(self.max_iterations):
            y_simulated = self._simulate_trajectory(theta, d_target, v0)
            error = y_target - y_simulated

            if abs(error) < self.tolerance:
                break

            theta += math.atan(error / d_target)

        return theta


if __name__ == '__main__':
    my_TroopType = my_TroopType.SENTINEL
    defaults_bullet_speed = 23.0

    ballistic_compensator = BallisticCompensator()
    pos = [0.2, -0.1, 2]
    theta0 = ballistic_compensator.calculate_ideal_pitch(math.sqrt(pos[0] * pos[0] + pos[2] * pos[2]), pos[1],
                                                         defaults_bullet_speed)
    theta1 = ballistic_compensator.calculate_angle(pos, 0)
    print(f"理想弹道角度: {math.degrees(theta0):.2f}°, 补偿后弹道角度: {math.degrees(theta1):.2f}°")
