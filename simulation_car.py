import numpy as np

class SimulationCar:
    def __init__(self,center:list=float):
        self.center=np.array(center, dtype=np.float32)
        self.r1=5
        self.r2=2.5
        self.h1=2.0
        self.h2=4.0
        self.armor_length=2 #长度
        self.armor_width=0.4 #装甲板倾斜导致的厚度
        self.armor_height=1 #高度
        self.car_coordinate=np.array([
            [self.r1,self.h1,0], #右
            [-self.r1,self.h1,0], #左
            [0,self.h2,self.r2], #背向相机
            [0,self.h2,-self.r2]]) #正向相机
        self.armor_place=np.array([[[self.car_coordinate[0][0]-self.armor_width/2,self.car_coordinate[0][1]+self.armor_height/2,self.car_coordinate[0][2]+self.armor_length/2],
                                    [self.car_coordinate[0][0]-self.armor_width/2,self.car_coordinate[0][1]+self.armor_height/2,self.car_coordinate[0][2]-self.armor_length/2],
                                    [self.car_coordinate[0][0]+self.armor_width/2,self.car_coordinate[0][1]-self.armor_height/2,self.car_coordinate[0][2]-self.armor_length/2],
                                    [self.car_coordinate[0][0]+self.armor_width/2,self.car_coordinate[0][1]-self.armor_height/2,self.car_coordinate[0][2]+self.armor_length/2]],
                                   [[self.car_coordinate[1][0]+self.armor_width/2,self.car_coordinate[1][1]+self.armor_height/2,self.car_coordinate[1][2]-self.armor_length/2],
                                    [self.car_coordinate[1][0]+self.armor_width/2,self.car_coordinate[1][1]+self.armor_height/2,self.car_coordinate[1][2]+self.armor_length/2],
                                    [self.car_coordinate[1][0] - self.armor_width / 2,self.car_coordinate[1][1] - self.armor_height / 2,self.car_coordinate[1][2] + self.armor_length / 2] ,
                                    [self.car_coordinate[1][0]-self.armor_width/2,self.car_coordinate[1][1]-self.armor_height/2,self.car_coordinate[1][2]-self.armor_length/2],],
                                   [[self.car_coordinate[2][0]+self.armor_length/2,self.car_coordinate[2][1]+self.armor_height/2,self.car_coordinate[2][2]-self.armor_width/2],
                                    [self.car_coordinate[2][0]-self.armor_length/2,self.car_coordinate[2][1]+self.armor_height/2,self.car_coordinate[2][2]-self.armor_width/2],
                                    [self.car_coordinate[2][0]-self.armor_length/2,self.car_coordinate[2][1]-self.armor_height/2,self.car_coordinate[2][2]+self.armor_width/2],
                                    [self.car_coordinate[2][0]+self.armor_length/2,self.car_coordinate[2][1]-self.armor_height/2,self.car_coordinate[2][2]+self.armor_width/2]],
                                   [[self.car_coordinate[3][0]+self.armor_length/2,self.car_coordinate[3][1]+self.armor_height/2,self.car_coordinate[3][2]+self.armor_width/2],
                                    [self.car_coordinate[3][0]-self.armor_length/2,self.car_coordinate[3][1]+self.armor_height/2,self.car_coordinate[3][2]+self.armor_width/2],
                                    [self.car_coordinate[3][0]-self.armor_length/2,self.car_coordinate[3][1]-self.armor_height/2,self.car_coordinate[3][2]-self.armor_width/2],
                                    [self.car_coordinate[3][0]+self.armor_length/2,self.car_coordinate[3][1]-self.armor_height/2,self.car_coordinate[3][2]-self.armor_width/2]]]) #装甲板位置
        self.w=0.5 #旋转步长，弧度制
        self.fps=30
        self.t=3.0
    def rotate_closewise(self):
        theta = -self.w
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Rotation matrix around Y-axis
        R = np.array([
            [cos_t, 0, -sin_t],
            [0, 1, 0],
            [sin_t, 0, cos_t]
        ])

        self.car_coordinate = np.dot(self.car_coordinate, R)
        self.armor_place = np.dot(self.armor_place, R)

    def rotate_counterclosewise(self):
        """模拟小车绕 Y 轴逆时针匀角速度旋转 t 秒的整个过程。

        匀角速度已由 self.w 给出（弧度/帧），fps 为每秒帧数，
        因此总帧数为 int(self.fps * self.t)。

        本函数会在内部按帧更新小车和装甲板的位置，
        并返回整个旋转过程中每一帧的 car_coordinate 和 armor_place 轨迹。
        """
        # 总帧数
        total_frames = int(self.fps * self.t)
        if total_frames <= 0:
            return [], []

        # 用于记录轨迹：列表中每个元素是一帧的坐标快照
        car_traj = []
        armor_traj = []

        # 逆时针：这里取正的 self.w
        theta = self.w
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # 绕 Y 轴的旋转矩阵（与 rotate_closewise 相反方向）
        R = np.array([
            [cos_t, 0, sin_t],
            [0, 1, 0],
            [-sin_t, 0, cos_t]
        ])

        for _ in range(total_frames):
            # 先更新当前帧坐标
            self.car_coordinate = np.dot(self.car_coordinate, R)
            self.armor_place = np.dot(self.armor_place, R)

            # 记录当前帧的快照（拷贝，避免后续修改影响历史数据）
            car_traj.append(self.car_coordinate.copy())
            armor_traj.append(self.armor_place.copy())

        return car_traj, armor_traj

    def car_to_world(self):
        """将小车坐标系下的坐标转换到世界坐标系。

        约定：
        - 小车坐标系中，小车中心在 (0, 0, 0)。
        - 世界坐标系中，小车中心在 self.center。

        因此转换就是对所有点加上 self.center 的平移。

        返回：
        - world_car_coordinate: shape (4, 3)，车身关键点在世界坐标系下的坐标
        - world_armor_place:    shape (4, 4, 3)，四块装甲板顶点在世界坐标系下的坐标
        """
        # 保证中心是 (3,) 形状，方便广播
        center_vec = self.center.reshape(1, 3).astype(np.float32)

        # 对车身关键点做平移：每个点 + center
        world_car_coordinate = self.car_coordinate.astype(np.float32) + center_vec

        # 对装甲板顶点做平移：利用 numpy 广播，在最后一维加上 center
        world_armor_place = self.armor_place.astype(np.float32) + center_vec

        return world_car_coordinate, world_armor_place

    def get_world_rotation_trajectory(self, clockwise: bool = True):
        """获取小车在世界坐标系下的旋转过程轨迹。

        参数：
        - clockwise: True 表示使用顺时针旋转（rotate_closewise 的方向）；
                     False 表示使用逆时针旋转（与 rotate_closewise 反方向）。

        返回：
        - world_car_traj:   长度为 N 的列表，每个元素是 shape (4, 3) 的 ndarray，表示车身关键点在世界坐标系下的坐标；
        - world_armor_traj: 长度为 N 的列表，每个元素是 shape (4, 4, 3) 的 ndarray，表示装甲板顶点在世界坐标系下的坐标。

        注意：
        - 本函数内部会从当前 car_coordinate / armor_place 状态出发，
          模拟 self.t 秒内的旋转过程，并在每一帧都做一次 car_to_world 转换。
        - 为避免修改原始状态，函数结束后会将 car_coordinate 和 armor_place 恢复成调用前的值。
        """
        # 备份当前局部坐标系下的状态，避免外部状态被破坏
        car_coord_backup = self.car_coordinate.copy()
        armor_place_backup = self.armor_place.copy()

        total_frames = int(self.fps * self.t)
        if total_frames <= 0:
            return [], []

        world_car_traj = []
        world_armor_traj = []

        # 根据方向构造一次性的旋转矩阵
        if clockwise:
            theta = -self.w
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            R = np.array([
                [cos_t, 0, -sin_t],
                [0, 1, 0],
                [sin_t, 0, cos_t]
            ])
        else:
            theta = self.w
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            R = np.array([
                [cos_t, 0, sin_t],
                [0, 1, 0],
                [-sin_t, 0, cos_t]
            ])

        for _ in range(total_frames):
            # 先在小车坐标系下旋转一帧
            self.car_coordinate = np.dot(self.car_coordinate, R)
            self.armor_place = np.dot(self.armor_place, R)

            # 再转换到世界坐标系并记录
            world_car, world_armor = self.car_to_world()
            world_car_traj.append(world_car)
            world_armor_traj.append(world_armor)

        # 恢复原始状态
        self.car_coordinate = car_coord_backup
        self.armor_place = armor_place_backup

        return world_car_traj, world_armor_traj
