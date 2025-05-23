from all_type import *
import math
from collections import deque, Counter


class TargetSelector:
    """
    目标选择类，从一组ArmorTargetPoint中选择最优目标。
    """

    def __init__(self, history_size=10):
        """
        初始化目标选择器。
        :param history_size: 记录历史选择的最大数量。
        """
        self.history_size = history_size
        self.history = deque(maxlen=history_size)  # 存储历史选择的目标
        # 初始化队列为全空对象
        for _ in range(history_size):
            self.history.append(None)

    def select_best_target(self, candidates):
        """
        选择最优的目标。
        :param candidates: 可选目标的列表 (List[ArmorTargetPoint])。
        :return: 最优的 ArmorTargetPoint 对象。
        """
        if not candidates:  # 如果没有候选目标
            self.add_empty_entry()
            return None
        if len(candidates) == 1:  # 如果只有一个候选目标
            best_target = candidates[0]
            self.history.append(best_target)
            return best_target
        elif len(candidates) == 2:  # 如果有两个候选目标
            if candidates[0].troop_type == candidates[1].troop_type:  # 如果两个候选目标的兵种类型相同
                # 使用小陀螺选择策略
                candidates.sort(key=lambda c: c.area, reverse=True)  # 按照面积从大到小排序
                if candidates[0].area / (candidates[1].area + 1e-6) >= 1.8:  # 小面积候选目标即将转走
                    best_target = candidates[0]
                    self.history.append(best_target)
                    return best_target
                else:
                    pass  # 直接进下面策略
            else:
                pass  # 直接进下面策略

        if all(target is None for target in self.history):  # 如果历史记录全为None
            # # 统计兵种出现次数
            # troop_counts = Counter(c.troop_type for c in candidates)
            # # 计算每个兵种到原点的距离
            # troop_distances = {troop: min(self._distance_to_origin(c) for c in candidates if c.troop_type == troop) for
            #                    troop in troop_counts}
            # # 选择出现次数最多的兵种，如果有多个兵种出现次数相同，则选择距离原点最近的兵种
            # most_common_troop, max_count = max(troop_counts.items(), key=lambda x: (x[1], -troop_distances[x[0]]))
            # # 过滤出该兵种的所有目标
            # troop_candidates = [c for c in candidates if c.troop_type == most_common_troop]
            # # 选择面积最大的装甲板
            # best_target = max(troop_candidates, key=lambda c: (c.area, -self._distance_to_origin(c)))
            # 选择距离原点最近的目标
            # best_target = min(candidates, key=lambda c: self._distance_to_origin(c))
            best_target = self.select_priority_in_range(candidates)  # 按照一定优先级加权锁定目标
        else:
            # 计算各个兵种的加权平均坐标
            troop_positions_data = self._calculate_weighted_troop_positions()
            # 从权重从高到低遍历加权平均坐标字典
            for troop_type, (weight_sum, position) in troop_positions_data:
                # 过滤出该兵种的所有目标
                troop_candidates = [c for c in candidates if c.troop_type == troop_type]
                # 如果该兵种有候选目标
                if troop_candidates:
                    # 计算到加权平均坐标的距离
                    distances = [self._distance_to_point(c, position) for c in troop_candidates]
                    # 选择距离加权平均坐标最近的目标
                    best_target = min(troop_candidates, key=lambda c: self._distance_to_point(c, position))
                    break
            else:  # 如果没有找到任何候选目标
                best_target = self.select_priority_in_range(candidates)  # 按照一定优先级加权锁定目标
                # best_target = min(candidates, key=lambda c: self._distance_to_origin(c))  # 选择距离原点最近的目标
                # best_target = max(candidates, key=lambda c: (c.area, -self._distance_to_origin(c)))  # 选择面积最大的目标

        self.history.append(best_target)  # 更新历史记录
        return best_target

    def select_priority_in_range(self, candidates):
        """
        在一定距离内寻找高优先级目标
        :param candidates: 可选目标的列表 (List[ArmorTargetPoint])。
        """
        # 筛选距离2m内的目标
        candidates_in3m = [c for c in candidates if self._distance_to_origin(c) <= 3]
        if len(candidates_in3m) > 0:  # 如果有候选目标,按优先级降序距离升序排序
            candidates_in3m.sort(key=lambda t: (-TargetSelector.get_priority(t), TargetSelector._distance_to_origin(t)))
            # 选择优先级最高的目标
            best_target = candidates_in3m[0]
            return best_target
        return max(candidates, key=lambda c: (c.area, -self._distance_to_origin(c)))  # # 选择面积最大的目标

    @staticmethod
    def get_priority(target):
        """
        获取目标的优先级
        """
        # 定义优先级
        priority_map = {TroopType.HERO: 3, TroopType.INFANTRY: 2, TroopType.SENTINEL: 1}
        # 获取目标的优先级
        return priority_map.get(target.troop_type, 0)

    def add_empty_entry(self):
        """
        在历史队列中加入空对象。
        """
        self.history.append(None)

    def _calculate_weighted_troop_positions(self):
        """
        计算各个兵种的加权平均坐标, 权重按入队顺序等差递减, 忽略空对象。
        """
        troop_positions_data = {}
        # 从队首往队尾遍历历史记录
        for i, target in enumerate(self.history):
            if target is None:
                continue
            troop_type = target.troop_type
            if troop_type not in troop_positions_data:
                troop_positions_data[troop_type] = [0, [0, 0, 0]]
            troop_positions_data[troop_type][0] += i + 1
            troop_positions_data[troop_type][1][0] += target.x * (i + 1)
            troop_positions_data[troop_type][1][1] += target.y * (i + 1)
            troop_positions_data[troop_type][1][2] += target.z * (i + 1)
        # 计算加权平均坐标
        for troop_type, (weight_sum, position) in troop_positions_data.items():
            troop_positions_data[troop_type][1] = tuple(coord / weight_sum for coord in position)
        # 按照权重从大到小排序
        troop_positions_data = sorted(troop_positions_data.items(), key=lambda x: x[1][0], reverse=True)
        # print(troop_positions_data)
        return troop_positions_data

    @staticmethod
    def _distance(target1, target2):
        """
        计算两个目标在三维空间的欧几里得距离。
        """
        return math.sqrt(
            (target1.x - target2.x) ** 2 +
            (target1.y - target2.y) ** 2 +
            (target1.z - target2.z) ** 2
        )

    @staticmethod
    def _distance_to_origin(target):
        """
        计算目标到原点的距离。
        """
        return math.sqrt(target.x ** 2 + target.y ** 2 + target.z ** 2)

    @staticmethod
    def _distance_to_point(target, point):
        """
        计算目标到给定点的距离。
        """
        return math.sqrt((target.x - point[0]) ** 2 + (target.y - point[1]) ** 2 + (target.z - point[2]) ** 2)
