import math
import cv2
from all_function import *
from all_type import TracState
from extended_kalman_filter import ExtendedKalmanFilter
from setting import used_predict

# LOST 丢失
# DETECTING 检测中
# TRACKING 追踪中
# TEMP_LOST 临时丢失
class Tracker:
    def __init__(self, max_match_dis=1.5, max_match_yaw=1.0, tracking_thres=5.0, loss_thres=15.0):
        # 初始化卡尔曼滤波器
        self.max_match_dis = max_match_dis  # 匹配最大距离阈值
        self.max_match_yaw = max_match_yaw  # 匹配最大角度阈值
        self.tracking_thres = tracking_thres  # 切换到 TRACKING 状态的检测次数阈值
        self.loss_thres = loss_thres  # 从 TEMP_LOST 或 DETECTING 切换到 LOST 状态的次数阈值
        self.state = TracState.LOST  # 状态
        self.armor_type = None  # 装甲板类型
        self.armor_color = None
        self.EKF = None
        self.dy = 0  # 记录装甲板在y方向的跳变量
        self.another_r = 0  # 备用的旋转半径值
        self.detect_count = 0
        self.loss_count = 0
        self.last_armor = None
        self.last_time = 0
        self.last_state = None

        # 角度/旋转角度判断
        self.expected_angle_change = 0.5 * math.pi
        self.angle_threshold = self.expected_angle_change * 0.3  # 判断新装甲板的角度阈值
        self.last_yaw_rate = 0  # 上一次的yaw角速度
        self.rotation_direction = 0  # 1为顺时针，-1为逆时针，0默认

    # 计算目标的初始状态
    def initial(self, armor):
        self.last_armor = armor
        self.state = TracState.DETECTING
        self.armor_type = armor.armor_type

        # 初始化EKF
        # P0 = np.eye(9) * 100  # 初始协方差矩阵原先
        P0 = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 更小的初始不确定性
        self.EKF = ExtendedKalmanFilter(P0)

        # 计算初始状态向量
        r = 0.26  # 初始半径，实际应根据装甲板类型调整
        self.another_r = 0.26  # 备用半径

        x0 = np.zeros(9)
        x0[0] = armor.x - r * math.cos(armor.yaw)  # xc
        x0[1] = 0  # vx
        x0[2] = armor.y  # yc
        x0[3] = 0  # vy
        x0[4] = armor.z + r * math.sin(armor.yaw)  # zc
        x0[5] = 0  # vz
        x0[6] = armor.yaw  # yaw
        x0[7] = 0  # v_yaw
        x0[8] = r  # r (radius)

        self.EKF.set_state(x0)
        self.last_time = 0
        self.rotation_direction = 0  # 重置旋转方向

    # 计算装甲板当前位置与圆心之间的直线距离，用于验证目标是否仍然在预期的旋转轨迹上
    def calculate_distance(self, state):
        xc = state[0]  # 圆心在x轴上的坐标
        zc = state[4]  # 圆心在z轴上的坐标
        yaw = state[6]  # 装甲板当前的航向角（弧度制）
        r = state[8]  # 装甲板绕机器人旋转的半径
        # 计算装甲板的位置
        xa = xc - r * math.cos(yaw)  # 装甲板在x轴上的坐标
        za = zc - r * math.sin(yaw)  # 装甲板在z轴上的坐标
        return math.sqrt(xa ** 2 + za ** 2)  # 返回装甲板与圆心之间的直线距离

    # 计算两位置之间的欧氏距离，用于判断是否匹配
    def calculate_diff_distance(self, new_pos, pre_pos):
        return math.sqrt(abs(new_pos[0] - pre_pos[0]) ** 2 +
                         abs(new_pos[1] - pre_pos[1]) ** 2 +
                         abs(new_pos[2] - pre_pos[2]) ** 2)

    # 判断是否为新装甲板
    def is_new_armor(self, armor, predict_state):
        if self.last_armor is None:
            return False
        yaw_diff = armor.yaw - self.last_armor.yaw
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
        if self.rotation_direction == 0 and abs(yaw_diff) > 0.1:  # 旋转方向判断
            self.rotation_direction = 1 if yaw_diff > 0 else -1

        # 角度判断
        if self.rotation_direction != 0:
            # 考虑旋转方向的角度变化
            expected_direction = self.rotation_direction * abs(yaw_diff)
            abs_yaw_diff = abs(yaw_diff)

            # 是否是新装甲板？90°180°270°都有可能
            if self.expected_angle_change - self.angle_threshold < abs_yaw_diff < self.expected_angle_change + self.angle_threshold:
                return True
            if 2 * self.expected_angle_change - self.angle_threshold < abs_yaw_diff < 2 * self.expected_angle_change + self.angle_threshold:
                return True
            if 3 * self.expected_angle_change - self.angle_threshold < abs_yaw_diff < 3 * self.expected_angle_change + self.angle_threshold:
                return True

        return False

    # 修改判断匹配的函数，考虑新装甲板的情况
    def is_match(self, armor):
        # 首先检查装甲板类型是否匹配
        if armor.armor_type != self.armor_type:
            return False

        # 对于同一装甲板的匹配，使用原有逻辑
        # 提取预测位置pre_pos和新位置new_pos
        pre_pos = [self.last_armor.x, self.last_armor.y, self.last_armor.z]
        new_pos = [armor.x, armor.y, armor.z]

        # 计算两者的距离和角度差
        diff_distance = self.calculate_diff_distance(new_pos, pre_pos)
        diff_yaw = abs(armor.yaw - self.last_armor.yaw)
        diff_yaw = min(diff_yaw, 2 * math.pi - diff_yaw)  # 确保角度差在[0, π]范围内

        # 如果距离和角度差均在阈值范围内，则认为匹配成功
        if diff_distance <= self.max_match_dis:
            return True

        return False

    # 根据装甲板跳变的情况修正 target_state，确保后续预测和更新的准确性
    def handleArmorJump(self, armor, target_state, t):
        # 检查是否为新装甲板
        is_new = self.is_new_armor(armor, target_state)

        self.dy = target_state[2] - armor.y
        target_state[2] = armor.y
        # 如果是新装甲板，可能需要调整半径
        target_state[8], self.another_r = self.another_r, target_state[8]

        # 根据观测更新目标状态
        r = target_state[8]
        target_state[0] = armor.x - r * math.cos(armor.yaw)
        target_state[2] = armor.y
        target_state[4] = armor.z + r * math.sin(armor.yaw)
        target_state[6] = armor.yaw

        # 更新角速度
        if self.last_armor is not None and self.last_time != 0:
            dt = t - self.last_time
            if dt > 0:
                # 计算角速度，考虑角度环绕的情况
                yaw_diff = armor.yaw - self.last_armor.yaw
                yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi  # 存疑

                # 如果是新装甲板，可能需要调整角速度计算
                if is_new:
                    # 对于新装甲板，使用预期的装甲板间隔角度计算平均角速度
                    if self.rotation_direction != 0:
                        # 使用上一次估计的角速度方向
                        target_state[7] = self.last_yaw_rate
                else:
                    # 对于同一装甲板，直接计算角速度
                    target_state[7] = yaw_diff / dt
                    self.last_yaw_rate = target_state[7]

        return target_state

    def predict(self, dt, armor, out_img):
        predict_state = self.EKF.predict(dt)
        cv2.putText(out_img,
                    f"pre x:{predict_state[0]:<6.3f} vx:{predict_state[1]:<6.3f} y:{predict_state[2]:<6.3f} vy:{predict_state[3]:<9.3f}z:{predict_state[4]:<6.3f} vz:{predict_state[5]:<6.3f} yaw:{predict_state[6] * 180.0 / math.pi:<6.3f} v_yaw:{predict_state[7] * 180.0 / math.pi:<6.3f}r:{predict_state[8]:<6.3f}",
                    (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

        armor.x = predict_state[0] + predict_state[8] * math.cos(predict_state[6])
        armor.y = predict_state[2]
        armor.z = predict_state[4] - predict_state[8] * math.sin(predict_state[6])
        armor.yaw = predict_state[6]
        return armor, out_img

    def update(self, armor, t, out_img):
        dt = t - self.last_time if self.last_time != 0 else 0
        self.last_time = t

        # 如果处于LOST状态且检测到装甲板，则初始化
        if self.state == TracState.LOST and armor is not None:
            self.initial(armor)
            return armor, out_img

        # 如果EKF尚未初始化，则无法进行预测和更新
        if self.EKF is None:
            if armor is not None:
                self.initial(armor)
            return armor, out_img

        if dt == 0:
            dt = 0.00001  # 3.6==dt=0.0001

        # 使用EKF预测下一状态
        if dt > 0:  # 确保时间步长有效
            predict_state = self.EKF.predict(dt)
            self.last_state = predict_state
        else:
            predict_state = self.last_state if self.last_state is not None else self.EKF.x_post

        # 检查是否匹配
        matched = False
        is_new_armor = False

        if armor is not None:
            # 检查是否为新装甲板
            cv2.putText(out_img,
                        f"detecting x:{self.last_armor.x:<9.3f} y:{self.last_armor.y:<9.3f} z:{self.last_armor.z:<9.3f} yaw:{self.last_armor.yaw * 180.0 / math.pi:<9.3f}",
                        (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

            is_new_armor = self.is_new_armor(armor, predict_state)
            # print("is_new:", is_new_armor)

            # 匹配检查
            matched = self.is_match(armor)

            if matched:
                # 准备测量向量
                z = np.array([
                    armor.x,
                    armor.y,
                    armor.z,
                    armor.yaw
                ])

                # 更新EKF
                updated_state = self.EKF.update(z)

                cv2.putText(out_img,
                            f"upd x:{updated_state[0]:<6.3f} vx:{updated_state[1]:<6.3f} y:{updated_state[2]:<6.3f} vy:{updated_state[3]:<9.3f}z:{updated_state[4]:<6.3f} vz:{updated_state[5]:<6.3f} yaw:{updated_state[6] * 180.0 / math.pi:<6.3f} v_yaw:{updated_state[7] * 180.0 / math.pi:<6.3f}v_yaw:{updated_state[8]:<6.3f}",
                            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

                self.last_armor = armor
                if used_predict:
                    armor, out_img = self.predict(0.35, armor, out_img)

            else:
                if armor is not None and is_new_armor:
                    # 新装甲板直接tracking
                    z = np.array([
                        self.last_armor.x,
                        self.last_armor.y,
                        self.last_armor.z,
                        self.last_armor.yaw
                    ])
                    updated_state = self.EKF.update(z)
                    cv2.putText(out_img,
                                f"upd x:{updated_state[0]:<6.3f} vx:{updated_state[1]:<6.3f} y:{updated_state[2]:<6.3f} vy:{updated_state[3]:<9.3f}z:{updated_state[4]:<6.3f} vz:{updated_state[5]:<6.3f} yaw:{updated_state[6] * 180.0 / math.pi:<6.3f} v_yaw:{updated_state[7] * 180.0 / math.pi:<6.3f}v_yaw:{updated_state[8]:<6.3f}",
                                (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

                    updated_state = self.handleArmorJump(armor, updated_state.copy(), t)
                    cv2.putText(out_img,
                                f"upd x:{updated_state[0]:<6.3f} vx:{updated_state[1]:<6.3f} y:{updated_state[2]:<6.3f} vy:{updated_state[3]:<9.3f}z:{updated_state[4]:<6.3f} vz:{updated_state[5]:<6.3f} yaw:{updated_state[6] * 180.0 / math.pi:<6.3f} v_yaw:{updated_state[7] * 180.0 / math.pi:<6.3f}v_yaw:{updated_state[8]:<6.3f}",
                                (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

                    self.EKF.set_state(updated_state)
                    self.last_armor = armor
                    if used_predict:
                        armor, out_img = self.predict(0.35, armor, out_img)

                else:
                    cv2.putText(out_img,
                                f"我测你老母",
                                (660, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
                    #self.initial(armor)
        cv2.putText(out_img,
                    f"is_match:{matched}  is_new:{is_new_armor}",
                    (660, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.putText(out_img, f"state:{self.state}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)




        if self.state == TracState.DETECTING:
            if matched:
                self.detect_count += 1
                self.loss_count = 0
                # 如果是新装甲板，加速进入TRACKING状态
                if is_new_armor or self.detect_count >= self.tracking_thres:
                    self.detect_count = 0
                    self.state = TracState.TRACKING
            else:
                self.loss_count += 1
                if self.loss_count >= self.loss_thres:
                    self.state = TracState.LOST
                    self.EKF = None  # 重置EKF
        elif self.state == TracState.TRACKING:
            if matched:
                self.loss_count = 0
            elif is_new_armor:
                self.loss_count = 0
            else:
                self.state = TracState.TEMP_LOST
                self.loss_count = 1

        elif self.state == TracState.TEMP_LOST:
            if matched:
                self.loss_count = 0
                self.state = TracState.TRACKING
            elif is_new_armor:
                self.state = TracState.TRACKING
                self.loss_count = 0
            else:
                self.loss_count += 1
                if self.loss_count >= self.loss_thres:
                    self.state = TracState.LOST
                    self.EKF = None  # 重置EKF

        # 返回结果
        if self.state == TracState.TEMP_LOST:
            return self.last_armor, out_img

        if matched:
            return armor, out_img
        elif is_new_armor:
            return armor, out_img
        else:
            return None, out_img
