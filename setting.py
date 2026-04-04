"""
配置基本参数
"""
from all_type import *

# 电控调试参数-----------------------------------------------------------------------------------------------------
# 通信波特率
baud_rate = 115200
# yaw和pitch的缓冲系数,0.1-1之间，数值越小越平滑，但也会增加响应时间
yaw_buffer_factor = 0.80  # 0.80
pitch_buffer_factor = 0.75  # 0.75
# 是否给电控发弧度差值(True为差值 False为目标绝对值)
send_radian_diff = True
# 最大lock误差角度
miss_yaw_angle = 2.5  # 电控yaw与目标yaw的最大误差角度，单位:度，超过这个角度认为未锁定
miss_pitch_angle = 2.0  # 电控pitch与目标pitch的最大误差角度，单位:度，超过这个角度认为未锁定

# 是否显示识别和瞄准结果窗口
is_show_video = False

# 对局需要手动设置的重要参数------------------------------------------------------------------------------------------
# 我方颜色,与电控通信后根据裁判系统自动设置
friend_color = Color.RED
# 我方兵种 (哨兵SENTINEL 英雄HERO 步兵INFANTRY)
my_TroopType = TroopType.SENTINEL

# 导航通信调试-----------------------------------------------------------------------------------------------------
send_chase = False
if my_TroopType == TroopType.SENTINEL:
    send_chase = True  # 哨兵需要发送追踪指令给导航，步兵和英雄不需要
send_target_ip = "192.168.1.5"  # 导航nuc的IP
send_target_port = 8964  # 导航nuc监听的端口

# 云台坐标系原点相对相机镜头中心点的平移向量, 由机器人尺寸测量得到(单位: m)---------------------------------------------------
# 烧饼相机下置
origin_gimbal_sb = np.array([0, -0.04509, -0.18435])

# 步兵相机下置
origin_gimbal_bb = np.array([0, -0.04959, -0.17185])

# 英雄相机下置
origin_gimbal_yx = np.array([0, -0.05850, -0.22561])

# 相机内参矩阵------------------------------------------------------------------------------------------------------
# self.camera_matrix = np.array([[self.fx, 0, self.cx],
#                                [0, self.fy, self.cy],
#                                [0, 0, 1]])
# 相机畸变系数
# self.dist_coefficients = np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

# 大恒相机
daheng_0_camera_matrix = np.array([[773.9795, 0.0, 515.3171],
                                   [0.0, 773.9609, 309.1733],
                                   [0.0, 0.0, 1.0]])
daheng_0_dist_coefficients = np.array([-0.080853, 0.125599, 0.0, 0.0, 0.0])

# 海康相机1
haikang_1_camera_matrix = np.array([[2.4010857e+03, 0.0000000e+00, 6.9597490e+02],
                                    [0.0000000e+00, 2.4002448e+03, 5.6805220e+02],
                                    [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])
haikang_1_dist_coefficients = np.array([-0.034071, 0.169365, 0.0, 0.0, 0.0])

# 海康相机2
haikang_2_camera_matrix = np.array([[1.7895407e+03, 0.0000000e+00, 7.1850660e+02],
                                    [0.0000000e+00, 1.7865637e+03, 5.5832830e+02],
                                    [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])
haikang_2_dist_coefficients = np.array([-0.078121, 0.144656, 0.0, 0.0, 0.0])

# CV灯条提点参数---------------------------------------------------------------------------------------------------
# CV提点后微小修正灯条角点位置,延长或缩小的百分比,如果PnP偏远可以适当增加这个值，反之可以适当减小这个值
recorrect_pixel = 0.93

# 弹道计算与补偿参数------------------------------------------------------------------------------------------------
# 重力加速度
g = 9.79460
# 默认初始弹速(m/s)
defaults_bullet_speed = 23.0  # 哨兵步兵小弹丸
cd = 0.47  # 小弹丸风阻系数
if my_TroopType == TroopType.HERO:
    defaults_bullet_speed = 11.5  # 英雄大弹丸
    cd = 0.22  # 大弹丸风阻系数

# 调试参数--------------------------------------------------------------------------------------------------------
# 保存视频的时间，单位: 秒，0表示不保存
save_video_time = 0

# 以下参数已经废弃，修改无效
# 是否展示3D绘图
is_show_3d = False
# 是否瞄准预测后的装甲板,False为不开启预测
used_predict = False
# 是否使用模型推理
used_yolo = True

# 模型参数--------------------------------------------------------------------------------------------------------
model_path = "models"
model_name = "best-cv.onnx"
if my_TroopType == TroopType.INFANTRY:
    model_name = "infantry.engine"
elif my_TroopType == TroopType.HERO:
    model_name = "hero.engine"
elif my_TroopType == TroopType.SENTINEL:
    model_name = "sentinel.engine"
# model_name = "best-cv.onnx" # 强制选择模型

# 自动选择相机
cameraID = CameraID.HAIKANG_1  # 相机ID
if my_TroopType == TroopType.INFANTRY:  # 步兵用海康1相机
    cameraID = CameraID.HAIKANG_1
elif my_TroopType == TroopType.HERO:  # 英雄用海康2相机
    cameraID = CameraID.HAIKANG_2
elif my_TroopType == TroopType.SENTINEL:  # 哨兵用大恒相机
    cameraID = CameraID.DAHENG_0
# cameraID = CameraID.HAIKANG_1 # 强制选择相机ID

# 自动选择平移向量
origin_gimbal = origin_gimbal_bb
if my_TroopType == TroopType.INFANTRY:  # 步兵
    origin_gimbal = origin_gimbal_bb  # 步兵相机
elif my_TroopType == TroopType.HERO:  # 英雄
    origin_gimbal = origin_gimbal_yx  # 英雄相机
elif my_TroopType == TroopType.SENTINEL:  # 哨兵
    origin_gimbal = origin_gimbal_sb  # 哨兵相机
# origin_gimbal = origin_gimbal_bb_up  # 强制选择平移向量

# 自动配置相机翻转
camera_flip = False  # 相机翻转
if my_TroopType in [TroopType.SENTINEL]:
    camera_flip = True
# camera_flip = False  # 强制选择相机翻转

# 自动配置相机内参矩阵和相机类型
camera_matrix = haikang_1_camera_matrix  # 相机内参矩阵
dist_coefficients = haikang_1_dist_coefficients  # 相机畸变系数
cameraType = CameraType.HAIKANG  # 相机品牌
if cameraID == CameraID.DAHENG_0:  # 大恒相机
    camera_matrix = daheng_0_camera_matrix
    dist_coefficients = daheng_0_dist_coefficients
    cameraType = CameraType.DAHENG
elif cameraID == CameraID.HAIKANG_1:  # 海康相机1
    camera_matrix = haikang_1_camera_matrix
    dist_coefficients = haikang_1_dist_coefficients
    cameraType = CameraType.HAIKANG
elif cameraID == CameraID.HAIKANG_2:  # 海康相机2
    camera_matrix = haikang_2_camera_matrix
    dist_coefficients = haikang_2_dist_coefficients
    cameraType = CameraType.HAIKANG
