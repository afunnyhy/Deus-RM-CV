import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from extended_kalman_filter import ExtendedKalmanFilter


# 模拟装甲板类
class ArmorSimulation:
    def __init__(self, x, y, z, yaw, armor_type="small"):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.armor_type = armor_type


def main():
    """测试卡尔曼滤波器的预测性能"""
    print("开始测试扩展卡尔曼滤波器...")

    # 1. 模拟真实场景: 一个机器人绕着圆心旋转，有4个装甲板
    xc, yc, zc = 0.0, 0.5, 0.0  # 圆心坐标
    r_small = 0.25  # 小装甲板半径
    r_large = 0.4  # 大装甲板半径
    omega = 2.0  # 角速度 (rad/s)

    # 时间参数
    dt = 0.033  # 时间步长 (约30Hz)
    total_time = 3.0  # 总模拟时间

    # 噪声参数
    pos_noise_std = 0.02  # 位置观测噪声标准差
    yaw_noise_std = 0.01  # 航向角观测噪声标准差

    # 生成时间序列
    num_steps = int(total_time / dt)
    time_steps = [i * dt for i in range(num_steps)]

    # 装甲板初始角度(4个装甲板等间隔分布)
    armor_angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    armor_types = ["small", "large", "small", "large"]
    armor_radii = [r_small, r_large, r_small, r_large]

    # 生成所有装甲板的真实轨迹
    true_armors = []
    for t in time_steps:
        current_armors = []
        for i in range(4):
            angle = armor_angles[i] + omega * t
            true_xa = xc - armor_radii[i] * np.cos(angle)
            true_ya = yc
            true_za = zc - armor_radii[i] * np.sin(angle)
            current_armors.append([true_xa, true_ya, true_za, angle, armor_types[i], armor_radii[i]])
        true_armors.append(current_armors)

    # 2. 模拟装甲板检测过程：实际应用中只能看到部分装甲板
    detected_armors = []
    for t_idx, t in enumerate(time_steps):
        visible_armors = []

        # 模拟只有在前方90度视野内的装甲板会被检测到
        for armor in true_armors[t_idx]:
            x, y, z = armor[0], armor[1], armor[2]
            # 计算装甲板相对于原点的方位角
            azimuth = math.atan2(-z, x)
            # 模拟视野限制（-π/4到π/4）内的装甲板会被检测到
            if -np.pi / 4 <= azimuth <= np.pi / 4:
                # 添加测量噪声
                noisy_x = x + np.random.normal(0, pos_noise_std)
                noisy_y = y + np.random.normal(0, pos_noise_std)
                noisy_z = z + np.random.normal(0, pos_noise_std)
                noisy_yaw = armor[3] + np.random.normal(0, yaw_noise_std)
                # 创建带噪声的装甲板对象
                noisy_armor = ArmorSimulation(noisy_x, noisy_y, noisy_z, noisy_yaw, armor[4])
                visible_armors.append(noisy_armor)

        # 如果有多个可见装甲板，只保留最接近中心的那个（模拟追踪单个装甲板）
        if visible_armors:
            # 根据到原点的距离排序
            visible_armors.sort(key=lambda a: a.x ** 2 + a.z ** 2)
            detected_armors.append(visible_armors[0])
        else:
            detected_armors.append(None)

    # 3. 初始化EKF
    P0 = np.eye(9) * 100
    ekf = ExtendedKalmanFilter(P0)

    # 等待第一个有效检测
    first_valid_idx = next((i for i, a in enumerate(detected_armors) if a is not None), None)
    if first_valid_idx is None:
        print("没有检测到任何装甲板，测试终止")
        return

    # 初始化状态
    first_armor = detected_armors[first_valid_idx]
    first_r = r_small if first_armor.armor_type == "small" else r_large

    x0 = np.zeros(9)
    x0[0] = first_armor.x + first_r * np.cos(first_armor.yaw)  # xc
    x0[1] = 0  # vx
    x0[2] = first_armor.y  # yc
    x0[3] = 0  # vy
    x0[4] = first_armor.z + first_r * np.sin(first_armor.yaw)  # zc
    x0[5] = 0  # vz
    x0[6] = first_armor.yaw  # yaw
    x0[7] = 0  # v_yaw
    x0[8] = first_r  # r

    ekf.set_state(x0)

    # 4. 运行滤波器
    filtered_states = []
    predicted_states = []
    true_positions = []  # 存储对应时刻的真实位置
    last_tracked_idx = -1  # 记录上一次成功跟踪的索引

    for i in range(first_valid_idx + 1, len(time_steps)):
        # 计算实际时间步长
        actual_dt = time_steps[i] - time_steps[i - 1]

        # 预测步骤
        predicted_state = ekf.predict(actual_dt)

        # 存储预测的装甲板位置
        pred_x = predicted_state[0] - predicted_state[8] * np.cos(predicted_state[6])
        pred_y = predicted_state[2]
        pred_z = predicted_state[4] - predicted_state[8] * np.sin(predicted_state[6])
        predicted_states.append([pred_x, pred_y, pred_z, predicted_state[6]])

        # 如果有检测到装甲板，则更新滤波器
        if detected_armors[i] is not None:
            armor = detected_armors[i]

            # 检查是否是同类型装甲板（简化处理）
            current_r = predicted_state[8]

            # 更新步骤
            z = np.array([armor.x, armor.y, armor.z, armor.yaw])
            updated_state = ekf.update(z)

            # 如果装甲板类型变化，调整半径
            if (armor.armor_type == "small" and abs(current_r - r_small) > 0.1) or \
                    (armor.armor_type == "large" and abs(current_r - r_large) > 0.1):
                new_r = r_small if armor.armor_type == "small" else r_large
                updated_state[8] = new_r
                # 更新圆心位置
                updated_state[0] = armor.x + new_r * np.cos(armor.yaw)
                updated_state[4] = armor.z + new_r * np.sin(armor.yaw)
                ekf.set_state(updated_state)

            # 从状态中提取装甲板位置
            filtered_x = updated_state[0] - updated_state[8] * np.cos(updated_state[6])
            filtered_y = updated_state[2]
            filtered_z = updated_state[4] - updated_state[8] * np.sin(updated_state[6])
            filtered_states.append([filtered_x, filtered_y, filtered_z, updated_state[6]])

            # 找到真实装甲板
            found_true_position = False
            for true_armor in true_armors[i]:
                if true_armor[4] == armor.armor_type:
                    # 简单的距离匹配
                    dist = np.sqrt((true_armor[0] - armor.x) ** 2 + (true_armor[2] - armor.z) ** 2)
                    if dist < 0.1:  # 接近度阈值
                        true_positions.append(true_armor[:4])
                        found_true_position = True
                        break

            if not found_true_position:
                # 如果找不到匹配的真实装甲板，使用上一个记录的真实位置
                if true_positions:
                    true_positions.append(true_positions[-1])
                else:
                    # 如果没有记录，使用检测到的位置
                    true_positions.append([armor.x, armor.y, armor.z, armor.yaw])

            last_tracked_idx = i
        else:
            # 如果没有检测到，使用预测值作为滤波值
            filtered_states.append([pred_x, pred_y, pred_z, predicted_state[6]])

            # 使用上一个真实位置
            if true_positions:
                true_positions.append(true_positions[-1])
            else:
                # 不应该发生这种情况，但以防万一
                true_positions.append([0, 0, 0, 0])

    # 5. 计算误差
    prediction_errors = []
    filtered_errors = []

    for i in range(min(len(predicted_states), len(true_positions))):
        # 计算预测误差
        true_pos = np.array(true_positions[i][:3])
        pred_pos = np.array(predicted_states[i][:3])
        pred_error = np.linalg.norm(true_pos - pred_pos)
        prediction_errors.append(pred_error)

        # 计算滤波后误差
        filtered_pos = np.array(filtered_states[i][:3])
        filter_error = np.linalg.norm(true_pos - filtered_pos)
        filtered_errors.append(filter_error)

    # 6. 输出结果
    avg_pred_error = np.mean(prediction_errors)
    avg_filter_error = np.mean(filtered_errors)

    print(f"平均预测误差: {avg_pred_error:.4f} m")
    print(f"平均滤波误差: {avg_filter_error:.4f} m")

    # 7. 可视化结果
    try:
        # 转换为numpy数组
        true_positions_np = np.array(true_positions)
        filtered_states_np = np.array(filtered_states)
        predicted_states_np = np.array(predicted_states)

        # 提取检测到的装甲板位置
        detected_positions = []
        for armor in detected_armors[first_valid_idx + 1:]:
            if armor is not None:
                detected_positions.append([armor.x, armor.y, armor.z, armor.yaw])
            else:
                # 没有检测到装甲板的帧，填充NaN
                detected_positions.append([np.nan, np.nan, np.nan, np.nan])
        detected_positions_np = np.array(detected_positions)

        # 3D轨迹图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制轨迹
        ax.plot(true_positions_np[:, 0], true_positions_np[:, 2], true_positions_np[:, 1], 'b-', label='真实轨迹')

        # 只绘制有效的检测点（即非NaN值）
        valid_mask = ~np.isnan(detected_positions_np[:, 0])
        ax.scatter(detected_positions_np[valid_mask, 0],
                   detected_positions_np[valid_mask, 2],
                   detected_positions_np[valid_mask, 1],
                   c='r', marker='.', label='检测')

        ax.plot(filtered_states_np[:, 0], filtered_states_np[:, 2], filtered_states_np[:, 1], 'g-', label='滤波轨迹')

        # 只绘制部分预测点，避免图像过于拥挤
        stride = 3
        ax.scatter(predicted_states_np[::stride, 0],
                   predicted_states_np[::stride, 2],
                   predicted_states_np[::stride, 1],
                   c='k', marker='o', s=20, alpha=0.5, label='预测位置')

        # 设置坐标轴标签
        ax.set_xlabel('X轴')
        ax.set_ylabel('Z轴')
        ax.set_zlabel('Y轴')
        ax.set_title('装甲板轨迹与EKF滤波结果')
        ax.legend()

        # 误差图
        plt.figure(figsize=(10, 6))
        time_axis = time_steps[first_valid_idx + 1:first_valid_idx + 1 + len(prediction_errors)]
        plt.plot(time_axis, prediction_errors, 'r-', label='预测误差')
        plt.plot(time_axis, filtered_errors, 'g-', label='滤波误差')
        plt.xlabel('时间 (s)')
        plt.ylabel('误差 (m)')
        plt.title('EKF预测与滤波误差')
        plt.legend()
        plt.grid(True)

        # 顶视图（XZ平面）
        plt.figure(figsize=(10, 10))
        plt.plot(true_positions_np[:, 0], true_positions_np[:, 2], 'b-', label='真实轨迹')
        plt.scatter(detected_positions_np[valid_mask, 0], detected_positions_np[valid_mask, 2],
                    c='r', marker='.', label='检测')
        plt.plot(filtered_states_np[:, 0], filtered_states_np[:, 2], 'g-', label='滤波轨迹')
        plt.scatter(predicted_states_np[::stride, 0], predicted_states_np[::stride, 2],
                    c='k', marker='o', s=20, alpha=0.5, label='预测位置')

        # 画出机器人圆心
        plt.scatter([xc], [zc], c='m', marker='*', s=200, label='机器人圆心')

        # 画出所有装甲板轨迹
        for i, armor_type in enumerate(armor_types):
            r = armor_radii[i]
            circle_angles = np.linspace(0, 2 * np.pi, 100)
            circle_x = xc - r * np.cos(circle_angles)
            circle_z = zc - r * np.sin(circle_angles)
            plt.plot(circle_x, circle_z, 'k--', alpha=0.3)

        plt.xlabel('X轴')
        plt.ylabel('Z轴')
        plt.title('装甲板轨迹顶视图')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

    except ImportError:
        print("matplotlib未安装，跳过可视化步骤")
    print("测试完成!")


if __name__ == "__main__":
    main()
