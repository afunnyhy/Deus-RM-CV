import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional


class RotationVelocityEstimator:
    """旋转角速度估计器
    
    通过装甲板出现和消失时的法向量计算机器人的角速度
    """
    
    def __init__(self, history_length: int = 10):
        """初始化角速度估计器
        
        Args:
            history_length: 为每个装甲板ID保存的历史记录长度
        """
        self.history_length = history_length
        # 存储每个装甲板的历史记录
        # 格式: {armor_id: [(timestamp, normal_vector), ...]}
        self.armor_normals: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=history_length)
        )
        
    def update_armor_normal(self, armor_id: int, timestamp: float, normal_vector: np.ndarray):
        """更新装甲板的法向量
        
        Args:
            armor_id: 装甲板ID
            timestamp: 时间戳
            normal_vector: 法向量 (3D向量)
        """
        # 确保法向量是单位向量
        normal_vector = np.array(normal_vector, dtype=np.float64)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        
        # 添加到历史记录
        self.armor_normals[armor_id].append((timestamp, normal_vector))
        
    def estimate_angular_velocity(self, armor_id: int) -> Optional[Tuple[float, np.ndarray]]:
        """估计指定装甲板的角速度
        
        Args:
            armor_id: 装甲板ID
            
        Returns:
            (angular_velocity_magnitude, rotation_axis) 或 None
            - angular_velocity_magnitude: 角速度大小 (rad/s)
            - rotation_axis: 旋转轴向量
        """
        history = self.armor_normals.get(armor_id, [])
        
        # 需要至少两个时间点的数据
        if len(history) < 2:
            return None
            
        # 获取最早和最晚的记录
        first_timestamp, first_normal = history[0]
        last_timestamp, last_normal = history[-1]
        
        # 计算时间差
        dt = last_timestamp - first_timestamp
        if dt <= 0:
            return None
            
        # 计算旋转角度和轴
        # 使用点积计算夹角
        dot_product = np.clip(np.dot(first_normal, last_normal), -1.0, 1.0)
        angle = np.arccos(dot_product)
        
        # 使用叉积计算旋转轴
        rotation_axis = np.cross(first_normal, last_normal)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        # 如果旋转轴长度接近0，说明几乎没有旋转
        if rotation_axis_norm < 1e-6:
            return 0.0, np.array([0, 0, 1])  # 默认旋转轴
            
        # 归一化旋转轴
        rotation_axis = rotation_axis / rotation_axis_norm
        
        # 计算角速度
        angular_velocity = angle / dt
        
        return angular_velocity, rotation_axis
        
    def estimate_robot_angular_velocity(self, armor_ids: List[int]) -> Optional[Tuple[float, np.ndarray]]:
        """估计整个机器人的角速度
        
        Args:
            armor_ids: 当前可见的装甲板ID列表
            
        Returns:
            (angular_velocity_magnitude, rotation_axis) 或 None
        """
        if not armor_ids:
            return None
            
        # 计算所有装甲板的平均角速度
        angular_velocities = []
        rotation_axes = []
        
        for armor_id in armor_ids:
            result = self.estimate_angular_velocity(armor_id)
            if result is not None:
                angular_vel, rotation_axis = result
                angular_velocities.append(angular_vel)
                rotation_axes.append(rotation_axis)
                
        if not angular_velocities:
            return None
            
        # 计算平均角速度和旋转轴
        avg_angular_velocity = np.mean(angular_velocities)
        
        if rotation_axes:
            # 平均旋转轴
            avg_rotation_axis = np.mean(rotation_axes, axis=0)
            avg_rotation_axis = avg_rotation_axis / np.linalg.norm(avg_rotation_axis)
        else:
            avg_rotation_axis = np.array([0, 1, 0])  # 默认旋转轴
            
        return float(avg_angular_velocity), avg_rotation_axis
        
    def clear_history(self, armor_id: int):
        """清除指定装甲板的历史记录
        
        Args:
            armor_id: 装甲板ID
        """
        if armor_id in self.armor_normals:
            self.armor_normals[armor_id].clear()
            
    def clear_all_history(self):
        """清除所有装甲板的历史记录"""
        self.armor_normals.clear()