import cv2
import numpy as np


class KalmanFilter:
    """
    3D 常速度模型卡尔曼滤波封装
    状态: x=[x,y,z,vx,vy,vz]^T
    量测: z=[x,y,z]^T
    包含两种实现:
      1) OpenCV 内部 predict/correct
      2) 手写 numpy 版本 (单步调试用)
    """

    def __init__(self,
                 state_dim=6,
                 measure_dim=3,
                 init_cov=100.0,
                 measure_noise=1.0,
                 process_noise=1.0,
                 x=0, y=0, z=0,
                 vx=0, vy=0, vz=0):
        # 基本参数
        self.z = None
        self.state_dim = int(state_dim)
        self.measure_dim = int(measure_dim)
        self.init_cov = float(init_cov)
        self.measure_noise = float(measure_noise)
        self.process_noise = float(process_noise)

        # 状态与协方差
        self.X = self._make_state_vector(x, y, z, vx, vy, vz)
        self.P = self._make_state_cov(self.init_cov, self.state_dim)

        # 模型矩阵 (初始占位)
        self.F = np.eye(self.state_dim, dtype=np.float32)
        self.H = self._make_H(self.measure_dim, self.state_dim)
        self.R = self._make_R(self.measure_noise, self.measure_dim)
        self.GAMMA = self._make_Gamma(dt=1e-3)
        self.Q = self._make_Q_from_Gamma(self.GAMMA, self.process_noise)
        self.K = np.zeros((self.state_dim, self.measure_dim), dtype=np.float32)

        # OpenCV KalmanFilter
        self.kf = None
        self._ensure_kf_created()

    # ---------------- 矩阵构造(公用) ----------------
    def _make_state_vector(self, x, y, z, vx, vy, vz):
        return np.array([[x], [y], [z], [vx], [vy], [vz]], dtype=np.float32)

    def _make_state_cov(self, init_cov, n):
        return np.eye(n, dtype=np.float32) * float(init_cov)

    def _make_H(self, m, n):
        H = np.zeros((m, n), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        return H

    def _make_R(self, r, m):
        return np.eye(m, dtype=np.float32) * float(r)

    def _make_F(self, dt):
        dt = float(max(dt, 1e-6))
        F = np.eye(self.state_dim, dtype=np.float32)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def _make_Gamma(self, dt):
        dt = float(max(dt, 1e-6))
        G = np.zeros((self.state_dim, self.measure_dim), dtype=np.float32)
        half_dt2 = 0.5 * dt * dt
        G[0, 0] = half_dt2
        G[1, 1] = half_dt2
        G[2, 2] = half_dt2
        G[3, 0] = dt
        G[4, 1] = dt
        G[5, 2] = dt
        return G

    def _make_Q_from_Gamma(self, Gamma, q):
        q = float(q)
        Qi = np.eye(self.measure_dim, dtype=np.float32) * q
        return (Gamma @ Qi @ Gamma.T).astype(np.float32)

    def upgrade_Q(self):
        # 重新依据当前 GAMMA 和 process_noise 构造 Q
        self.Q = self._make_Q_from_Gamma(self.GAMMA, self.process_noise)
        return self.Q.copy()

    # 添加动态调整参数的方法
    def adjust_for_frame_rate(self, dt):
        """
        根据帧率动态调整滤波器参数
        在低帧率时增加过程噪声，使预测更依赖测量值
        """
        # 基于 dt 动态调整过程噪声
        # dt 越大（帧率越低），过程噪声应该越大
        base_process_noise = self.process_noise
        dt_factor = min(10.0, max(1.0, dt / (1.0 / 30.0)))  # 以30FPS为基准
        adjusted_process_noise = base_process_noise * dt_factor

        # 更新过程噪声
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
    def init_kf(self, dt=1e-3):
        self.F = self._make_F(dt)
        self.GAMMA = self._make_Gamma(dt)
        self.Q = self._make_Q_from_Gamma(self.GAMMA, self.process_noise)
        self._ensure_kf_created()

    def reset_state(self, x=0, y=0, z=0, vx=0, vy=0, vz=0, init_cov=None):
        self.X = self._make_state_vector(x, y, z, vx, vy, vz)
        if init_cov is not None:
            self.P = self._make_state_cov(init_cov, self.state_dim)
        self._ensure_kf_created()

    def build_F_Q(self, dt=1e-3):
        self.F = self._make_F(dt)
        self.GAMMA = self._make_Gamma(dt)
        self.Q = self.upgrade_Q()
        # 同步到 OpenCV (仅供 cv 路径使用)
        if self.kf is not None:
            self.kf.transitionMatrix = self.F.copy()
            self.kf.processNoiseCov = self.Q.copy()
        return self.F.copy(), self.GAMMA.copy(), self.Q.copy()

    # ===== OpenCV 预测 =====
    def predict_next(self, dt=1e-3):
        self.build_F_Q(dt)
        pred = self.kf.predict()
        # 使用 statePre / errorCovPre (更符合标准流程)
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
    # ===== 手写: 预测状态 =====
    def cal_X_next(self):
        self.X = self.F @ self.X
        return self.X.copy()

    # ===== 手写: 预测协方差 =====
    def cal_P_next(self):
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.P.copy()

    # ===== 手写: 生成模拟量测(可选) =====
    def cal_z_pred(self):
        noise = np.random.multivariate_normal(
            mean=np.zeros(self.measure_dim, dtype=np.float64),
            cov=self.R.astype(np.float64)
        ).astype(np.float32).reshape(self.measure_dim, 1)
        self.z = self.H @ self.X + noise
        return self.z.copy()

    # ===== 手写: 卡尔曼增益 =====
    def cal_K(self):
        S = self.H @ self.P @ self.H.T + self.R
        # solve 优于直接逆
        PHt = self.P @ self.H.T
        self.K = PHt @ np.linalg.inv(S.astype(np.float64)).astype(np.float32)
        return self.K.copy()

    # ===== 手写: 状态更新 =====
    def upgrade_X(self):
        # self.z 需已设置 (外部传入或 cal_z_pred)
        self.X = self.X + self.K @ (self.z - self.H @ self.X)
        return self.X.copy()

    # ===== 手写: 协方差更新 (Joseph 形式稳定) =====
    def upgrade_P(self):
        I = np.eye(self.state_dim, dtype=np.float32)
        IKH = I - self.K @ self.H
        self.P = IKH @ self.P @ IKH.T + self.K @ self.R @ self.K.T
        return self.P.copy()

    # ---------------- 新增统一接口 ----------------
    def filter_once(self, z, use_manual=True, dt=1e-3, R_override=None):
        """
        执行一次完整卡尔曼滤波 (预测 + 校正)
        参数:
          z: 外部量测 (长度=measure_dim 或列向量)
          use_manual: True 使用手写 numpy 实现；False 使用 OpenCV
          dt: 时间步长
          R_override: 可选替换量测噪声协方差
        返回: (X, P, K)
        """
        z = self._format_measurement(z)
        if R_override is not None:
            self.R = np.asarray(R_override, dtype=np.float32)
            if not use_manual and self.kf is not None:
                self.kf.measurementNoiseCov = self.R.copy()

        if use_manual:
            # 手写路径
            self.build_F_Q(dt)  # 重建 F,Q
            self.cal_X_next()  # 预测状态
            self.cal_P_next()  # 预测协方差
            self.z = z  # 设置当前量测
            self.cal_K()  # 计算增益
            self.upgrade_X()  # 更新状态
            self.upgrade_P()  # 更新协方差
        else:
            # OpenCV 路径
            self.predict_next(dt)
            self.correct_by_sensor(z)

            # 为统一输出 K (可计算一次，但不用于 OpenCV 内部)
            self.cal_K()

        return self.X.copy(), self.P.copy(), self.K.copy()

    # ---------------- 内部辅助 ----------------
    def _format_measurement(self, z):
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            z = z.reshape(self.measure_dim, 1)
        elif z.shape == (self.measure_dim,):
            z = z.reshape(self.measure_dim, 1)
        assert z.shape == (self.measure_dim, 1), "量测 z 形状必须为 (measure_dim,1)"
        return z
