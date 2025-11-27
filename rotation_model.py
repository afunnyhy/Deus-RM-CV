import time
import cv2
from guardRobot import GuardRobot
from all_type import ArmorPlate
import numpy as np

def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

class GetAngularVelocity:
    def __init__(self,
                 initial_rvec=None,
                 initial_yaw=None,
                 initial_time=None,
                 smoothing_alpha=0.5,
                 yaw_window=8):
        self.prev_rvec = None if initial_rvec is None else np.asarray(initial_rvec, dtype=float).reshape(3)
        self.prev_yaw = None if initial_yaw is None else float(initial_yaw)
        self.prev_time = None if initial_time is None else float(initial_time)

        self.smoothing_alpha = None if smoothing_alpha is None else float(smoothing_alpha)
        # 存储上一次平滑结果
        self.smoothed_omega = None  # 3D 向量
        self.smoothed_yaw_rate = None

        # yaw 最小二乘窗口
        self.yaw_times = []
        self.yaw_values = []
        self.yaw_window = int(yaw_window)

        # 上次输出
        self.last_omega = np.zeros(3, dtype=float)
        self.last_yaw_rate = 0.0 # 用于卡尔曼滤波的角速度

    @staticmethod
    def _rvec_to_R(rvec):
        rvec = np.asarray(rvec, dtype=float).reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        return R

    @staticmethod
    def _rotation_matrix_to_rotvec(R):
        rvec, _ = cv2.Rodrigues(R)
        return rvec.flatten()

    @staticmethod
    def _yaw_from_rvec(rvec):
        R = GetAngularVelocity._rvec_to_R(rvec)
        # 提取 yaw 的常用近似（工程中需校准符号/坐标系）
        yaw = np.arctan2(R[0, 2], R[2, 2])
        return _wrap_angle(float(yaw))

    # ---- 主功能 ----
    def update_with_rvec(self, rvec, timestamp=None):
        """
        输入当前帧 rvec（3,）和可选时间戳（秒）。
        返回 (omega_vec, omega_mag, yaw_rate)：
          - omega_vec: 3D 角速度估计 (rad/s)（相机坐标系）
          - omega_mag: 角速度模长 (rad/s)
          - yaw_rate: 偏航角速度 (rad/s)（从 rvec 提取 yaw 后差分）
        """
        now = time.time() if timestamp is None else float(timestamp)
        rvec = np.asarray(rvec, dtype=float).reshape(3)

        if self.prev_rvec is None or self.prev_time is None:
            # 初始化，不产生速率
            self.prev_rvec = rvec.copy()
            self.prev_time = now
            yaw = self._yaw_from_rvec(rvec)
            self.prev_yaw = yaw
            self.last_omega = np.zeros(3, dtype=float)
            self.last_yaw_rate = 0.0
            return self.last_omega, 0.0, 0.0

        dt = max(1e-6, now - self.prev_time)
        # 计算相对旋转: R_rel = R_curr * R_prev^T
        R_prev = self._rvec_to_R(self.prev_rvec)
        R_curr = self._rvec_to_R(rvec)
        R_rel = R_curr @ R_prev.T
        rotvec = self._rotation_matrix_to_rotvec(R_rel)  # 轴角向量，方向*角度
        omega_vec = rotvec / float(dt)
        omega_mag = float(np.linalg.norm(omega_vec))

        # yaw 速率（从 rvec 提取 yaw）
        yaw_curr = self._yaw_from_rvec(rvec)
        yaw_diff = _wrap_angle(yaw_curr - self.prev_yaw)
        yaw_rate = float(yaw_diff / dt)

        # 指数平滑（若配置）
        if self.smoothing_alpha is not None:
            if self.smoothed_omega is None:
                self.smoothed_omega = omega_vec
            else:
                self.smoothed_omega = self.smoothing_alpha * omega_vec + (
                            1.0 - self.smoothing_alpha) * self.smoothed_omega
            if self.smoothed_yaw_rate is None:
                self.smoothed_yaw_rate = yaw_rate
            else:
                self.smoothed_yaw_rate = self.smoothing_alpha * yaw_rate + (
                            1.0 - self.smoothing_alpha) * self.smoothed_yaw_rate

            out_omega = self.smoothed_omega
            out_yaw_rate = float(self.smoothed_yaw_rate)
        else:
            out_omega = omega_vec
            out_yaw_rate = yaw_rate

        # 保存状态
        self.prev_rvec = rvec.copy()
        self.prev_time = now
        self.prev_yaw = yaw_curr

        self.last_omega = out_omega
        self.last_yaw_rate = out_yaw_rate

        return out_omega, float(np.linalg.norm(out_omega)), out_yaw_rate

    def update_with_yaw(self, yaw, timestamp=None):
        """
        仅用 yaw（rad）序列估计偏航角速度。
        支持窗口最小二乘拟合（更鲁棒）以及差分。
        返回 yaw_rate (rad/s)。
        """
        now = time.time() if timestamp is None else float(timestamp)
        yaw = float(_wrap_angle(yaw))

        # 插入窗口
        self.yaw_times.append(now)
        self.yaw_values.append(yaw)
        if len(self.yaw_times) > self.yaw_window:
            self.yaw_times.pop(0)
            self.yaw_values.pop(0)

        # 如果窗口长度 >= 3 使用最小二乘拟合（先 unwrap）
        if len(self.yaw_times) >= 3:
            t = np.asarray(self.yaw_times, dtype=float)
            th = np.asarray(self.yaw_values, dtype=float)
            # 展开角度序列以避免 2pi 跳变
            for i in range(1, len(th)):
                diff = _wrap_angle(th[i] - th[i - 1])
                th[i] = th[i - 1] + diff
            # 线性拟合 th = k * t + b
            A = np.vstack([t, np.ones_like(t)]).T
            k, _ = np.linalg.lstsq(A, th, rcond=None)[0]
            yaw_rate = float(k)
        else:
            # 使用差分
            if self.prev_yaw is None or self.prev_time is None:
                yaw_rate = 0.0
            else:
                dt = max(1e-6, now - self.prev_time)
                dy = _wrap_angle(yaw - self.prev_yaw)
                yaw_rate = float(dy / dt)

        # 平滑
        if self.smoothing_alpha is not None:
            if self.smoothed_yaw_rate is None:
                self.smoothed_yaw_rate = yaw_rate
            else:
                self.smoothed_yaw_rate = self.smoothing_alpha * yaw_rate + (
                            1.0 - self.smoothing_alpha) * self.smoothed_yaw_rate
            out_yaw_rate = float(self.smoothed_yaw_rate)
        else:
            out_yaw_rate = float(yaw_rate)

        # 保存
        self.prev_yaw = yaw
        self.prev_time = now
        self.last_yaw_rate = out_yaw_rate

        return out_yaw_rate

    def get_state(self):
        """返回 (last_omega_vec, last_omega_mag, last_yaw_rate)。"""
        return np.asarray(self.last_omega, dtype=float), float(np.linalg.norm(self.last_omega)), float(
            self.last_yaw_rate)

class YawRateKalmanCV:
    def __init__(self, q_angle=1e-3, q_gyro=1e-2, r_meas=1e-1):
        # 状态: [yaw, yaw_rate]^T
        self.kf = cv2.KalmanFilter(2, 1, 0, cv2.CV_32F)
        # F 矩阵中的 dt 后面每次更新
        self.kf.transitionMatrix = np.array([[1, 0],
                                             [0, 1]], np.float32)
        # 测量矩阵: 只观测 yaw
        self.kf.measurementMatrix = np.array([[1, 0]], np.float32)
        # 过程噪声协方差 Q
        self.kf.processNoiseCov = np.array([[q_angle, 0],
                                            [0, q_gyro]], np.float32)
        # 测量噪声协方差 R
        self.kf.measurementNoiseCov = np.array([[r_meas]], np.float32)
        self.kf.errorCovPost = np.eye(2, dtype=np.float32)
        self.kf.statePost = np.zeros((2, 1), np.float32)

        self.last_time = None

    def _update_F(self, dt):
        # yaw_k   = yaw_{k-1} + dt * yaw_rate
        # rate_k  = rate_{k-1}
        self.kf.transitionMatrix[0, 1] = dt

    def step(self, measured_yaw, timestamp=None, dt_predict=0.0):
        now = time.time() if timestamp is None else float(timestamp)
        if self.last_time is None:
            self.last_time = now
            self.kf.statePost = np.array([[measured_yaw],
                                          [0.0]], np.float32)
            return 0.0, 0.0  # 初次角速度未知

        dt = now - self.last_time
        self.last_time = now
        if dt <= 0:
            dt = 1e-3

        self._update_F(dt)

        # 预测
        pred_state = self.kf.predict()  # shape: (2,1)

        # 校正
        meas = np.array([[np.float32(measured_yaw)]])
        corr_state = self.kf.correct(meas)

        yaw_est = float(corr_state[0, 0])
        yaw_rate_est = float(corr_state[1, 0])

        # 额外再预测 dt_predict 之后的 yaw_rate（常速度模型就没变）
        yaw_rate_pred = yaw_rate_est
        if dt_predict > 0.0:
            # 可以构造一个临时 F 做更进一步预测 yaw，如需要的话
            pass

        return yaw_rate_est, yaw_rate_pred


def _yaw_from_rvec(rvec):
    """内部封装一下 yaw 提取，方便以后改坐标系。"""
    return GetAngularVelocity._yaw_from_rvec(rvec)


class RotationModel:
    def __init__(self,
                 initial_rvec=None,
                 initial_yaw=None,
                 initial_time=None,
                 smoothing_alpha=0.5,
                 yaw_window=8,
                 q_angle=1e-3,
                 q_gyro=1e-2,
                 r_meas=1e-1):
        # 基于 rvec 的角速度和 yaw_rate 估计（差分 + 指数平滑）
        self.get_angular_velocity = GetAngularVelocity(
            initial_rvec=initial_rvec,
            initial_yaw=initial_yaw,
            initial_time=initial_time,
            smoothing_alpha=smoothing_alpha,
            yaw_window=yaw_window
        )

        # 基于 OpenCV 的卡尔曼滤波器，对 yaw / yaw_rate 做时序滤波与预测
        self.yaw_kf = YawRateKalmanCV(
            q_angle=q_angle,
            q_gyro=q_gyro,
            r_meas=r_meas
        )

    def update_with_rvec(self, rvec, timestamp=None, dt_predict=0.0):
        """
        使用当前帧 rvec 更新旋转模型。

        参数:
          \- rvec: 当前帧的旋转向量 (3,)
          \- timestamp: 当前时间戳（秒），None 则用 time.time()
          \- dt_predict: 需要向前预测的时间（秒），0 表示不做额外预测

        返回:
          \- omega_vec: 3D 角速度向量 (rad/s)
          \- omega_mag: 角速度模长 (rad/s)
          \- yaw_curr: 当前帧 yaw 角 (rad)
          \- yaw_rate_raw: 差分/平滑得到的原始 yaw 角速度 (rad/s)
          \- yaw_rate_filt: 卡尔曼滤波后的 yaw 角速度估计 (rad/s)
          \- yaw_rate_pred: 预测 dt_predict 之后的 yaw 角速度 (rad/s)
        """
        # 1\. 用 rvec 估计 3D 角速度 + 原始 yaw_rate（差分/平滑）
        omega_vec, omega_mag, yaw_rate_raw = self.get_angular_velocity.update_with_rvec(
            rvec, timestamp=timestamp
        )

        # 2\. 从 rvec 提取当前 yaw
        yaw_curr = _yaw_from_rvec(rvec)

        # 3\. 用 OpenCV 卡尔曼滤波 yaw / yaw_rate
        yaw_rate_filt, yaw_rate_pred = self.yaw_kf.step(
            measured_yaw=yaw_curr,
            timestamp=timestamp,
            dt_predict=dt_predict
        )

        return omega_vec, omega_mag, yaw_curr, yaw_rate_raw, yaw_rate_filt, yaw_rate_pred

    def get_state(self):
        """
        返回当前内部状态的一个简单快照:
          \- omega_vec, omega_mag, yaw_rate_raw: 来自 GetAngularVelocity
          \- yaw, yaw_rate_filt: 来自卡尔曼滤波器的当前估计
        """
        omega_vec, omega_mag, yaw_rate_raw = self.get_angular_velocity.get_state()

        # 从卡尔曼中取当前 yaw / yaw_rate 估计
        state_post = self.yaw_kf.kf.statePost
        yaw = float(state_post[0, 0])
        yaw_rate_filt = float(state_post[1, 0])

        return omega_vec, omega_mag, yaw_rate_raw, yaw, yaw_rate_filt

