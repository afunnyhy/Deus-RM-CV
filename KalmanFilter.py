import cv2
import numpy as np


class KalmanFilter:
    """
    3D 常速度模型卡尔曼滤波封装
    状态: x=[x,y,z,r,theta,omega,vx,vy,vz]^T （针对单个点的位置、半径、角度、角速度和线速度）
    量测: z=[x,y,z]^T （单个点的位置测量）
    只按 9 维状态处理，不再支持 6 维等其他变体。
    """

    def __init__(self,
                 state_dim=9,  # 固定为 9 维: 位置(3) + 半径(1) + 角度(1) + 角速度(1) + 速度(3)
                 measure_dim=3,
                 init_cov=100.0,
                 measure_noise=1.0,
                 process_noise=1.0,
                 x=0, y=0, z=0,
                 vx=0, vy=0, vz=0,
                 r=0, theta=0, omega=0):
        # 强制 9 维
        self.state_dim = 9
        self.measure_dim = int(measure_dim)
        self.init_cov = float(init_cov)
        self.measure_noise = float(measure_noise)
        self.process_noise = float(process_noise)

        # 默认参数字典
        self.parameters = {
            'state_dim': self.state_dim,
            'measure_dim': self.measure_dim,
            'init_cov': self.init_cov,
            'measure_noise': self.measure_noise,
            'process_noise': self.process_noise
        }

        # 量测缓存
        self.z = None

        # 状态与协方差
        self.X = self._make_state_vector(x, y, z, r, theta, omega, vx, vy, vz)
        self.P = self._make_state_cov(self.init_cov, self.state_dim)

        # 矩阵初始化（固定 9 维）
        self.F = np.eye(self.state_dim, dtype=np.float32)
        self.H = self._make_H(self.measure_dim, self.state_dim)
        self.R = self._make_R(self.measure_noise, self.measure_dim)
        # 默认 dt 略微调大一点: 5ms
        self.default_dt = 5e-3
        self.GAMMA = self._make_Gamma(dt=self.default_dt)
        self.Q = self._make_Q_from_Gamma(self.GAMMA, self.process_noise)
        self.K = np.zeros((self.state_dim, self.measure_dim), dtype=np.float32)

        # OpenCV KalmanFilter
        self.kf = None
        self._ensure_kf_created()

    # ---------------- 公用矩阵构造 ----------------
    def _make_state_vector(self, x, y, z, r, theta, omega, vx, vy, vz):
        # 固定 9x1
        return np.array([[x], [y], [z], [r], [theta], [omega], [vx], [vy], [vz]], dtype=np.float32)

    def _make_state_cov(self, init_cov, n):
        return np.eye(n, dtype=np.float32) * float(init_cov)

    def _make_H(self, m, n):
        H = np.zeros((m, n), dtype=np.float32)
        # 仅观测 xyz 位置
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        return H

    def _make_R(self, measure_noise, m):
        return np.eye(m, dtype=np.float32) * float(measure_noise)

    def _make_F(self, dt):
        # dt 下限 1e-4，上限 0.2，避免极端值
        dt = float(np.clip(dt, 1e-4, 0.2))
        F = np.eye(self.state_dim, dtype=np.float32)

        # 状态顺序固定: [x,y,z,r,theta,omega,vx,vy,vz]
        # 位置由速度积分
        F[0, 6] = dt
        F[1, 7] = dt
        F[2, 8] = dt
        # 角度由角速度积分
        F[4, 5] = dt
        return F

    def _make_Gamma(self, dt):
        # 过程噪声仅针对 xyz 加速度（隐含），固定 9x3
        dt = float(np.clip(dt, 1e-4, 0.2))
        G = np.zeros((self.state_dim, self.measure_dim), dtype=np.float32)
        half_dt2 = 0.5 * dt * dt

        # 位置维度
        G[0, 0] = half_dt2
        G[1, 1] = half_dt2
        G[2, 2] = half_dt2
        # 速度维度
        G[6, 0] = dt
        G[7, 1] = dt
        G[8, 2] = dt
        return G

    def _make_Q_from_Gamma(self, Gamma, q):
        q = float(q)
        Qi = np.eye(self.measure_dim, dtype=np.float32) * q
        return (Gamma @ Qi @ Gamma.T).astype(np.float32)

    def upgrade_Q(self):
        # 重新依据当前 GAMMA 和 process_noise 构造 Q
        self.Q = self._make_Q_from_Gamma(self.GAMMA, self.process_noise)
        return self.Q.copy()

    # 根据 dt 动态调节噪声（供外部可选调用）
    def adjust_for_frame_rate(self, dt):
        """根据帧率动态调整过程噪声，在低帧率时增加过程噪声，减小飘移。"""
        base_process_noise = float(self.parameters.get('process_noise', self.process_noise))
        # 以 60 FPS 为基准，dt ~ 1/60 ≈ 0.016s
        base_dt = 1.0 / 60.0
        dt_factor = float(np.clip(dt / base_dt, 0.5, 4.0))
        adjusted_process_noise = base_process_noise * dt_factor
        adjusted_process_noise = float(np.clip(adjusted_process_noise, 1e-6, 1e3))

        self.process_noise = adjusted_process_noise
        self.Q = self._make_Q_from_Gamma(self.GAMMA, self.process_noise)

    # ---------------- OpenCV 同步 ----------------
    def _ensure_kf_created(self):
        if self.kf is None:
            self.kf = cv2.KalmanFilter(self.state_dim, self.measure_dim, 0, cv2.CV_32F)
        self.kf.transitionMatrix = self.F.copy()
        self.kf.measurementMatrix = self.H.copy()
        self.kf.processNoiseCov = self.Q.copy()
        self.kf.measurementNoiseCov = self.R.copy()
        self.kf.statePost = self.X.copy()
        self.kf.errorCovPost = self.P.copy()

    # ---------------- 外部接口(保持原名) ----------------
    def init_kf(self, dt=None):
        # 如果外部没传 dt，就用略大一点的默认值
        if dt is None:
            dt = self.default_dt
        self.F = self._make_F(dt)
        self.GAMMA = self._make_Gamma(dt)
        self.Q = self._make_Q_from_Gamma(self.GAMMA, self.process_noise)
        self._ensure_kf_created()

    def reset_state(self, x=0, y=0, z=0, r=0, theta=0, omega=0, vx=0, vy=0, vz=0, init_cov=None):
        self.X = self._make_state_vector(x, y, z, r, theta, omega, vx, vy, vz)
        if init_cov is not None:
            self.P = self._make_state_cov(init_cov, self.state_dim)
        self._ensure_kf_created()

    def build_F_Q(self, dt=None):
        if dt is None:
            dt = self.default_dt
        self.F = self._make_F(dt)
        self.GAMMA = self._make_Gamma(dt)
        self.Q = self.upgrade_Q()
        if self.kf is not None:
            self.kf.transitionMatrix = self.F.copy()
            self.kf.processNoiseCov = self.Q.copy()
        return self.F.copy(), self.GAMMA.copy(), self.Q.copy()

    def _format_measurement(self, z):
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            z = z.reshape(self.measure_dim, 1)
        elif z.shape == (self.measure_dim,):
            z = z.reshape(self.measure_dim, 1)
        assert z.shape == (self.measure_dim, 1), "量测 z 形状必须为 (measure_dim,1)"
        return z

    # ===== OpenCV 预测 =====
    def predict_next(self, dt=None):
        # 每帧根据真实 dt 更新 F/Q，保证 9 维一致
        if dt is None:
            dt = self.default_dt
        self.build_F_Q(dt)
        # 可以选择性地根据 dt 调整过程噪声强度
        self.adjust_for_frame_rate(dt)
        pred = self.kf.predict()
        self.X = self.kf.statePre.copy()
        self.P = self.kf.errorCovPre.copy()
        return pred.copy()

    # ===== OpenCV 校正 =====
    def correct_by_sensor(self, z, R_override=None):
        z = self._format_measurement(z)
        self.z = z.copy()
        if R_override is not None:
            R_new = np.asarray(R_override, dtype=np.float32)
            self.R = R_new
            self.kf.measurementNoiseCov = R_new
        corr = self.kf.correct(z)
        self.X = self.kf.statePost.copy()
        self.P = self.kf.errorCovPost.copy()
        return corr.copy()

    def get_state(self):
        return self.X.copy(), self.P.copy()

    # ---------------- 手写路径辅助函数(仅 numpy) ----------------
    def cal_X_next(self):
        self.X = self.F @ self.X
        return self.X.copy()

    def cal_P_next(self):
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.P.copy()

    def cal_z_pred(self):
        noise = np.random.multivariate_normal(
            mean=np.zeros(self.measure_dim, dtype=np.float64),
            cov=self.R.astype(np.float64)
        ).astype(np.float32).reshape(self.measure_dim, 1)
        self.z = self.H @ self.X + noise
        return self.z.copy()

    def cal_K(self):
        S = self.H @ self.P @ self.H.T + self.R
        PHt = self.P @ self.H.T
        self.K = PHt @ np.linalg.inv(S.astype(np.float64)).astype(np.float32)
        return self.K.copy()

    def upgrade_X(self):
        self.X = self.X + self.K @ (self.z - self.H @ self.X)
        return self.X.copy()

    def upgrade_P(self):
        I = np.eye(self.state_dim, dtype=np.float32)
        IKH = I - self.K @ self.H
        self.P = IKH @ self.P @ IKH.T + self.K @ self.R @ self.K.T
        return self.P.copy()

    # ---------------- 统一一步接口 ----------------
    def filter_once(self, z, use_manual=True, dt=None, R_override=None):
        z = self._format_measurement(z)
        if dt is None:
            dt = self.default_dt
        if R_override is not None:
            self.R = np.asarray(R_override, dtype=np.float32)
            if not use_manual and self.kf is not None:
                self.kf.measurementNoiseCov = self.R.copy()

        if use_manual:
            self.build_F_Q(dt)
            self.cal_X_next()
            self.cal_P_next()
            self.z = z
            self.cal_K()
            self.upgrade_X()
            self.upgrade_P()
        else:
            self.predict_next(dt)
            self.correct_by_sensor(z)
            self.cal_K()

        return self.X.copy(), self.P.copy(), self.K.copy()


import cv2
import numpy as np


class KalmanFilter6D:
    """
    3D 常速度模型 (Constant Velocity) 卡尔曼滤波封装 - 6维版本
    状态: x=[x, y, z, vx, vy, vz]^T
    量测: z=[x, y, z]^T
    """

    def __init__(self,
                 measure_dim=3,
                 init_cov=100.0,
                 measure_noise=0.1,
                 process_noise=100.0,
                 x=0, y=0, z=0,
                 vx=0, vy=0, vz=0):
        # 固定 6 维: 位置(3) + 速度(3)
        self.state_dim = 6
        self.measure_dim = int(measure_dim)
        self.init_cov = float(init_cov)
        self.measure_noise = float(measure_noise)
        self.process_noise = float(process_noise)

        # 默认参数字典
        self.parameters = {
            'state_dim': self.state_dim,
            'measure_dim': self.measure_dim,
            'init_cov': self.init_cov,
            'measure_noise': self.measure_noise,
            'process_noise': self.process_noise
        }

        # 量测缓存
        self.z = None

        # 状态与协方差
        self.X = self._make_state_vector(x, y, z, vx, vy, vz)
        self.P = self._make_state_cov(self.init_cov, self.state_dim)

        # 矩阵初始化
        self.F = np.eye(self.state_dim, dtype=np.float32)
        self.H = self._make_H(self.measure_dim, self.state_dim)
        self.R = self._make_R(self.measure_noise, self.measure_dim)

        # 默认 dt
        self.default_dt = 5e-3
        self.GAMMA = self._make_Gamma(dt=self.default_dt)
        self.Q = self._make_Q_from_Gamma(self.GAMMA, self.process_noise)
        self.K = np.zeros((self.state_dim, self.measure_dim), dtype=np.float32)

        # OpenCV KalmanFilter
        self.kf = None
        self._ensure_kf_created()

    # ---------------- 公用矩阵构造 ----------------
    def _make_state_vector(self, x, y, z, vx, vy, vz):
        # 固定 6x1
        return np.array([[x], [y], [z], [vx], [vy], [vz]], dtype=np.float32)

    def _make_state_cov(self, init_cov, n):
        return np.eye(n, dtype=np.float32) * float(init_cov)

    def _make_H(self, m, n):
        # m=3, n=6
        H = np.zeros((m, n), dtype=np.float32)
        # 仅观测 xyz 位置 (索引 0, 1, 2)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        return H

    def _make_R(self, measure_noise, m):
        return np.eye(m, dtype=np.float32) * float(measure_noise)

    def _make_F(self, dt):
        # dt 下限 1e-4，上限 0.2，避免极端值
        dt = float(np.clip(dt, 1e-4, 0.2))
        F = np.eye(self.state_dim, dtype=np.float32)

        # 状态顺序: [x, y, z, vx, vy, vz]
        # 索引对应: 0, 1, 2, 3,  4,  5
        # 位置 = 位置 + 速度 * dt
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt
        return F

    def _make_Gamma(self, dt):
        # 过程噪声针对 xyz 的加速度（隐含），将加速度映射到状态
        # 状态: 6维, 噪声源: 3维 (ax, ay, az)
        dt = float(np.clip(dt, 1e-4, 0.2))
        G = np.zeros((self.state_dim, self.measure_dim), dtype=np.float32)
        half_dt2 = 0.5 * dt * dt

        # 位置受加速度影响 (1/2 * a * t^2)
        G[0, 0] = half_dt2
        G[1, 1] = half_dt2
        G[2, 2] = half_dt2

        # 速度受加速度影响 (a * t)
        G[3, 0] = dt
        G[4, 1] = dt
        G[5, 2] = dt
        return G

    def _make_Q_from_Gamma(self, Gamma, q):
        q = float(q)
        Qi = np.eye(self.measure_dim, dtype=np.float32) * q
        return (Gamma @ Qi @ Gamma.T).astype(np.float32)

    def upgrade_Q(self):
        self.Q = self._make_Q_from_Gamma(self.GAMMA, self.process_noise)
        return self.Q.copy()

    def adjust_for_frame_rate(self, dt):
        """根据帧率动态调整过程噪声"""
        base_process_noise = float(self.parameters.get('process_noise', self.process_noise))
        # 以 60 FPS 为基准
        base_dt = 1.0 / 60.0
        dt_factor = float(np.clip(dt / base_dt, 0.5, 4.0))
        adjusted_process_noise = base_process_noise * dt_factor
        adjusted_process_noise = float(np.clip(adjusted_process_noise, 1e-6, 1e3))

        self.process_noise = adjusted_process_noise
        self.Q = self._make_Q_from_Gamma(self.GAMMA, self.process_noise)

    # ---------------- OpenCV 同步 ----------------
    def _ensure_kf_created(self):
        if self.kf is None:
            self.kf = cv2.KalmanFilter(self.state_dim, self.measure_dim, 0, cv2.CV_32F)
        self.kf.transitionMatrix = self.F.copy()
        self.kf.measurementMatrix = self.H.copy()
        self.kf.processNoiseCov = self.Q.copy()
        self.kf.measurementNoiseCov = self.R.copy()
        self.kf.statePost = self.X.copy()
        self.kf.errorCovPost = self.P.copy()

    # ---------------- 外部接口 ----------------
    def init_kf(self, dt=None):
        if dt is None:
            dt = self.default_dt
        self.F = self._make_F(dt)
        self.GAMMA = self._make_Gamma(dt)
        self.Q = self._make_Q_from_Gamma(self.GAMMA, self.process_noise)
        self._ensure_kf_created()

    def reset_state(self, x=0, y=0, z=0, vx=0, vy=0, vz=0, init_cov=None):
        """重置状态，参数去除了 r, theta, omega"""
        self.X = self._make_state_vector(x, y, z, vx, vy, vz)
        if init_cov is not None:
            self.P = self._make_state_cov(init_cov, self.state_dim)
        self._ensure_kf_created()

    def build_F_Q(self, dt=None):
        if dt is None:
            dt = self.default_dt
        self.F = self._make_F(dt)
        self.GAMMA = self._make_Gamma(dt)
        self.Q = self.upgrade_Q()
        if self.kf is not None:
            self.kf.transitionMatrix = self.F.copy()
            self.kf.processNoiseCov = self.Q.copy()
        return self.F.copy(), self.GAMMA.copy(), self.Q.copy()

    def _format_measurement(self, z):
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            z = z.reshape(self.measure_dim, 1)
        elif z.shape == (self.measure_dim,):
            z = z.reshape(self.measure_dim, 1)
        assert z.shape == (self.measure_dim, 1), "量测 z 形状必须为 (3, 1)"
        return z

    # ===== OpenCV 预测 =====
    def predict_next(self, dt=None):
        if dt is None:
            dt = self.default_dt
        self.build_F_Q(dt)
        self.adjust_for_frame_rate(dt)
        pred = self.kf.predict()
        self.X = self.kf.statePre.copy()
        self.P = self.kf.errorCovPre.copy()
        return pred.copy()

    # ===== OpenCV 校正 =====
    def correct_by_sensor(self, z, R_override=None):
        z = self._format_measurement(z)
        self.z = z.copy()
        if R_override is not None:
            R_new = np.asarray(R_override, dtype=np.float32)
            self.R = R_new
            self.kf.measurementNoiseCov = R_new
        corr = self.kf.correct(z)
        self.X = self.kf.statePost.copy()
        self.P = self.kf.errorCovPost.copy()
        return corr.copy()

    def get_state(self):
        return self.X.copy(), self.P.copy()

    # ---------------- 手写路径辅助函数(仅 numpy) ----------------
    def cal_X_next(self):
        self.X = self.F @ self.X
        return self.X.copy()

    def cal_P_next(self):
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.P.copy()

    def cal_z_pred(self):
        noise = np.random.multivariate_normal(
            mean=np.zeros(self.measure_dim, dtype=np.float64),
            cov=self.R.astype(np.float64)
        ).astype(np.float32).reshape(self.measure_dim, 1)
        self.z = self.H @ self.X + noise
        return self.z.copy()

    def cal_K(self):
        S = self.H @ self.P @ self.H.T + self.R
        PHt = self.P @ self.H.T
        self.K = PHt @ np.linalg.inv(S.astype(np.float64)).astype(np.float32)
        return self.K.copy()

    def upgrade_X(self):
        self.X = self.X + self.K @ (self.z - self.H @ self.X)
        return self.X.copy()

    def upgrade_P(self):
        I = np.eye(self.state_dim, dtype=np.float32)
        IKH = I - self.K @ self.H
        self.P = IKH @ self.P @ IKH.T + self.K @ self.R @ self.K.T
        return self.P.copy()

    # ---------------- 统一一步接口 ----------------
    def filter_once(self, z, use_manual=True, dt=None, R_override=None):
        z = self._format_measurement(z)
        if dt is None:
            dt = self.default_dt
        if R_override is not None:
            self.R = np.asarray(R_override, dtype=np.float32)
            if not use_manual and self.kf is not None:
                self.kf.measurementNoiseCov = self.R.copy()

        if use_manual:
            self.build_F_Q(dt)
            self.cal_X_next()
            self.cal_P_next()
            self.z = z
            self.cal_K()
            self.upgrade_X()
            self.upgrade_P()
        else:
            self.predict_next(dt)
            self.correct_by_sensor(z)
            self.cal_K()

        return self.X.copy(), self.P.copy(), self.K.copy()