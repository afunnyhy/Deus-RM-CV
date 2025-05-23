"""
配置基本参数
"""
from all_type import *

# 云台坐标系原点相对相机镜头中心点的平移向量, 由机器人尺寸测量得到(单位: m)------------------------------------------------------------
# 老烧饼
origin_gimbal_old_sb = np.array([0, -0.04475, -0.131705])

# 新烧饼相机上置
origin_gimbal_new_sb_up = np.array([0, 0.04902, -0.08932])

# 新烧饼相机下置
origin_gimbal_new_sb_down = np.array([0, -0.04509, -0.18435])

# 新步兵相机上置
origin_gimbal_bb_up = np.array([0, 0.05052, -0.16674])

# 新英雄相机下置
origin_gimbal_yx_down = np.array([0, -0.0563, -0.23985])

# 相机内参矩阵------------------------------------------------------------------------------------------------------
# self.camera_matrix = np.array([[self.fx, 0, self.cx],
#                                [0, self.fy, self.cy],
#                                [0, 0, 1]])
# 相机畸变系数
# self.dist_coefficients = np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

# 大恒老相机
daheng_1_camera_matrix = np.array([[7.716379127971313e+02, 0, 5.183947142266674e+02],
                                   [0, 7.716767700989153e+02, 3.102502727089079e+02],
                                   [0, 0, 1]])
daheng_1_dist_coefficients = np.array(
    [-0.083388858626141, 0.152721313501007, -3.919473027550272e-04, -5.753236297258918e-06, -0.048868652840575])

# 海康相机1
haikang_1_camera_matrix = np.array([[2395.8701, 0, 698.3712],
                                    [0, 2395.5855, 572.1990],
                                    [0, 0, 1]])
haikang_1_dist_coefficients = np.array([-0.0322, 0.1497, 0.0008109, 6.716e-05, -0.1226])

# 海康相机2
haikang_2_camera_matrix = np.array([[2415.7, 0, 716.1212],
                                    [0, 2416, 556.5836],
                                    [0, 0, 1]])
haikang_2_dist_coefficients = np.array([-0.0451, 0.4553, -1.4647e-4, 9.4951e-4, -2.0458])

# 模型参数--------------------------------------------------------------------------------------------------------
model_path = "./weight/"
model_name = "20250318_normal"

# CV提点参数------------------------------------------------------------------------------------------------------
# CV提点后微小修正灯条角点位置,延长或缩小的百分比
recorrect_pixel = 0.92

# 弹道计算与补偿参数------------------------------------------------------------------------------------------------
# 重力加速度
g = 9.79460
# 默认初始弹速(m/s)
defaults_bullet_speed = 25
# yaw动态时补偿参数t0,由不同机器人暴力测试得出
t0 = {
    TroopType.SENTINEL: 0.5,  # 哨兵的yaw运动参数
    TroopType.HERO: 0.5,  # 英雄的yaw运动参数
    TroopType.INFANTRY: 0.5  # 步兵的yaw运动参数
}

# 对局需要手动设置的重要参数------------------------------------------------------------------------------------------
# 我方颜色,后根据通信自动设置
friend_color = Color.BLUE
# 我方兵种 (哨兵SENTINEL 英雄HERO 步兵INFANTRY)
my_TroopType = TroopType.INFANTRY

# 调试参数--------------------------------------------------------------------------------------------------------
# 保存视频的时间，单位: 秒，0表示不保存
save_video_time = 0
# 是否展示视频
is_show_video = True
# 是否展示3D绘图
is_show_3d = False
# 是否瞄准预测后的装甲板,False为不开启预测
used_predict = False
# 是否使用模型推理
used_yolo = True
if my_TroopType == TroopType.HERO:
    # 英雄使用CV
    used_yolo = False

# 自动选择相机
cameraID = CameraID.HAIKANG_2  # 相机ID
if my_TroopType == TroopType.INFANTRY:  # 步兵用海康2相机
    cameraID = CameraID.HAIKANG_2
elif my_TroopType == TroopType.HERO:  # 英雄用海康1相机
    cameraID = CameraID.HAIKANG_1
elif my_TroopType == TroopType.SENTINEL:  # 哨兵用大恒相机
    cameraID = CameraID.DAHENG_0
# cameraID = CameraID.HAIKANG_1  # 强制选择相机ID

# 自动选择平移向量
origin_gimbal = origin_gimbal_bb_up
if my_TroopType == TroopType.INFANTRY:  # 步兵
    origin_gimbal = origin_gimbal_bb_up  # 步兵相机上置
elif my_TroopType == TroopType.HERO:  # 英雄
    origin_gimbal = origin_gimbal_yx_down  # 英雄相机下置
elif my_TroopType == TroopType.SENTINEL:  # 哨兵
    # origin_gimbal = origin_gimbal_new_sb_up  # 新哨兵相机上置
    origin_gimbal = origin_gimbal_new_sb_down  # 新哨兵相机下置
# origin_gimbal = origin_gimbal_bb_up  # 强制选择平移向量

# 自动配置相机翻转
camera_flip = False  # 相机翻转
if my_TroopType in [TroopType.INFANTRY, TroopType.HERO, TroopType.SENTINEL]:
    camera_flip = True
# camera_flip = False  # 强制选择相机翻转

# 自动配置相机内参矩阵和相机类型
camera_matrix = haikang_2_camera_matrix  # 相机内参矩阵
dist_coefficients = haikang_2_dist_coefficients  # 相机畸变系数
cameraType = CameraType.HAIKANG  # 相机品牌
if cameraID == CameraID.DAHENG_0:  # 大恒相机
    camera_matrix = daheng_1_camera_matrix
    dist_coefficients = daheng_1_dist_coefficients
    cameraType = CameraType.DAHENG
elif cameraID == CameraID.HAIKANG_1:  # 海康相机1
    camera_matrix = haikang_1_camera_matrix
    dist_coefficients = haikang_1_dist_coefficients
    cameraType = CameraType.HAIKANG
elif cameraID == CameraID.HAIKANG_2:  # 海康相机2
    camera_matrix = haikang_2_camera_matrix
    dist_coefficients = haikang_2_dist_coefficients
    cameraType = CameraType.HAIKANG

# 自动配置yaw运动参数t0
t0 = t0[my_TroopType]
