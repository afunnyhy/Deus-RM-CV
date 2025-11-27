import time
from collections import deque
from typing import Dict, Deque, Tuple, Optional


class MotionStateDetector:
    """运动状态检测器（Motion State Detection Module）。

    设计目标：
        - 检测机器人车辆的运动状态（平动/旋转）
        - 利用装甲板数量随时间的变化模式来判断运动状态
        - 平动状态：装甲板数量稳定为1或2个
        - 旋转状态：装甲板数量在1和2之间周期性切换（如1-2-1-2-1-2...）

    主要数据结构：
        - self._histories[robot_id]:
            deque[(armor_count, timestamp)]
            保存最近若干次装甲板数量变化的记录；
        - self._last_count[robot_id]:
            记录上一帧装甲板的数量，用于判断"是否真的发生变化"；
        - self._motion_state[robot_id]:
            记录当前的运动状态（平动TRANSLATION或旋转ROTATION）
    """

    # 运动状态枚举
    TRANSLATION = "translation"  # 平动状态
    ROTATION = "rotation"        # 旋转状态
    UNKNOWN = "unknown"          # 未知状态

    # 默认参数配置
    DEFAULT_PARAMS = {
        "history_len": 10,              # 历史记录长度 - 保存多少帧的装甲板数量变化记录
        "min_period": 0.05,             # 最小周期时间（秒） - 两次装甲板数量切换的最小时间间隔
        "max_period": 1.0,              # 最大周期时间（秒） - 两次装甲板数量切换的最大时间间隔
        "state_confirm_threshold": 3,   # 状态确认阈值 - 连续多少帧确认同一状态才最终判定
        "min_clear_pattern_len": 4,     # 明确模式检测的最小长度 - 检测明确旋转模式所需的最小样本数
        "min_alternating_score": 0.6,   # 最小交替性评分 - 判定为旋转状态所需的最小交替性得分(0-1)
        "min_valid_interval_ratio": 0.6, # 最小有效时间间隔比例 - 有效时间间隔占总间隔数的最小比例
        "max_interval_stability_ratio": 4.0, # 最大时间间隔稳定性比值 - 最大时间间隔与最小时间间隔的最大比值
        "default_armor_counts": [1, 2], # 默认装甲板数量范围 - 认为合理的装甲板数量列表
    }

    def __init__(self, params=None):
        """初始化检测器及其参数。

        Args:
            params (dict, optional): 参数字典，可覆盖默认参数
                - history_len (int): 历史记录长度，影响状态判断的稳定性
                - min_period (float): 最小周期时间（秒），过小的值会被认为是噪声
                - max_period (float): 最大周期时间（秒），过大的值会被认为不是周期性变化
                - state_confirm_threshold (int): 状态确认阈值，避免状态频繁切换
                - min_clear_pattern_len (int): 明确模式检测的最小长度，确保模式的可靠性
                - min_alternating_score (float): 最小交替性评分，0-1之间，越高要求越严格
                - min_valid_interval_ratio (float): 最小有效时间间隔比例，0-1之间
                - max_interval_stability_ratio (float): 最大时间间隔稳定性比值，越大容忍度越高
                - default_armor_counts (list): 默认装甲板数量范围，合理的装甲板数量列表
        """
        # 合并参数
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

        # 参数提取
        self.history_len = int(self.params["history_len"])
        self.min_period = float(self.params["min_period"])
        self.max_period = float(self.params["max_period"])
        self.state_confirm_threshold = int(self.params["state_confirm_threshold"])
        self.min_clear_pattern_len = int(self.params["min_clear_pattern_len"])
        self.min_alternating_score = float(self.params["min_alternating_score"])
        self.min_valid_interval_ratio = float(self.params["min_valid_interval_ratio"])
        self.max_interval_stability_ratio = float(self.params["max_interval_stability_ratio"])
        self.default_armor_counts = list(self.params["default_armor_counts"])

        # 按 robot_id 存储：
        #   - _histories: 装甲板数量变化历史队列
        #   - _last_count: 上一次记录的装甲板数量
        #   - _state_counter: 状态计数器（用于确认状态）
        #   - _motion_state: 当前运动状态
        self._histories: Dict[int, Deque[Tuple[int, float]]] = {}
        self._last_count: Dict[int, Optional[int]] = {}
        self._state_counter: Dict[int, int] = {}
        self._motion_state: Dict[int, str] = {}

    def _get_history(self, robot_id: int) -> Deque[Tuple[int, float]]:
        """确保给定 robot_id 的内部结构已初始化，并返回其历史队列。

        该函数在第一次访问某个 robot_id 时，会：
            - 创建一个固定长度的 deque 作为历史记录；
            - 将上一帧装甲板数量设为 None；
            - 将状态计数器初始化为 0；
            - 将运动状态初始化为 UNKNOWN。

        Args:
            robot_id: 机器人 ID（通常是整数）。

        Returns:
            当前 ID 对应的历史记录队列（deque[(armor_count, timestamp)]）。
        """
        if robot_id not in self._histories:
            # 使用 maxlen 限定历史长度，超出时自动弹出最老元素
            self._histories[robot_id] = deque(maxlen=self.history_len)
            self._last_count[robot_id] = None
            self._state_counter[robot_id] = 0
            self._motion_state[robot_id] = self.UNKNOWN
        return self._histories[robot_id]

    def update(self, robot_id: int, armor_count: int, timestamp: Optional[float] = None):
        """在每帧（或每次检测更新）调用，用当前信息刷新某个机器人的运动状态。

        通常在检测 / 跟踪逻辑中，对每个机器人 ID 都会拿到：
            - 当前帧与该 ID 关联的装甲板数量 armor_count；
            - 当前时间戳 timestamp（若为空则内部用 time.time() 补齐）。

        该函数会：
            1. 若这是该 ID 第一次出现，则进行初始化；
            2. 若本帧装甲板数量与上一帧不同，则记录一条 (armor_count, timestamp)；
            3. 使用全部历史记录判断当前运动状态；
            4. 更新状态计数器和最终状态。

        Args:
            robot_id: 机器人 ID。
            armor_count: 当前与该机器人关联的装甲板数量。
            timestamp: 当前时间（秒）。若为 None，则使用 time.time()。
        """
        if timestamp is None:
            timestamp = time.time()

        # 获取 / 初始化该 ID 的历史结构
        hist = self._get_history(robot_id)
        last_count = self._last_count[robot_id]

        # 只有当装甲板数量真正发生变化时，才追加一条记录，避免填满无效重复数据
        if last_count is None or last_count != armor_count:
            # 记录此次变化：装甲板数量 + 时间
            hist.append((armor_count, float(timestamp)))
            self._last_count[robot_id] = armor_count

            # 基于更新后的历史记录判断当前运动状态
            state = self._determine_motion_state(hist)
            
            # 更新状态计数器
            if state == self._motion_state[robot_id]:
                # 状态一致，增加计数器
                self._state_counter[robot_id] = min(
                    self._state_counter[robot_id] + 1, 
                    self.state_confirm_threshold
                )
            else:
                # 状态改变，重置计数器
                self._state_counter[robot_id] = 1
                self._motion_state[robot_id] = state

    def _calculate_alternating_score(self, counts):
        """计算交替性得分，衡量序列在1和2之间交替的程度。
        
        Args:
            counts: 装甲板数量序列
            
        Returns:
            交替性得分，范围0-1，1表示完美交替
        """
        if len(counts) < 2:
            return 0
            
        alternating_changes = 0
        total_changes = 0
        
        for i in range(1, len(counts)):
            # 如果数值发生了变化
            if counts[i] != counts[i-1]:
                total_changes += 1
                # 如果是1和2之间的变化，则认为是交替变化
                if set([counts[i], counts[i-1]]) == {1, 2}:
                    alternating_changes += 1
        
        # 如果没有变化，检查是否全是1或全是2
        if total_changes == 0:
            # 全是1或全是2，交替性为0（完全不交替）
            return 0
            
        return alternating_changes / total_changes

    def _calculate_intervals(self, counts, times):
        """计算装甲板数量变化之间的时间间隔。
        
        Args:
            counts: 装甲板数量序列
            times: 对应的时间戳序列
            
        Returns:
            时间间隔列表
        """
        intervals = []
        for i in range(1, len(counts)):
            # 只有当装甲板数量发生变化时才计算间隔
            if counts[i] != counts[i-1]:
                intervals.append(times[i] - times[i-1])
        return intervals

    def _determine_motion_state(self, hist: Deque[Tuple[int, float]]) -> str:
        """根据历史记录判断运动状态。

        判据说明：
            1. 如果装甲板数量始终为1或2，则为平动状态；
            2. 如果装甲板数量在1和2之间周期性切换，则为旋转状态；
            3. 其他情况为未知状态。

        Args:
            hist: 某个 robot_id 的装甲板数量变化记录队列。

        Returns:
            运动状态字符串（TRANSLATION/ROTATION/UNKNOWN）。
        """
        # 样本过少，难以判断，但如果有装甲板检测到，至少可以判定为平动
        if len(hist) < 4:
            if len(hist) > 0:
                # 如果有记录，且装甲板数量在合理范围内，返回平动
                last_count = hist[-1][0]
                if last_count in self.default_armor_counts:
                    return self.TRANSLATION
            return self.UNKNOWN

        # 分别提取装甲板数量和时间序列
        counts = [c for (c, t) in hist]
        times = [t for (c, t) in hist]

        # 检查是否只有1和2两种数量
        unique_counts = sorted(set(counts))
        
        # 如果只有1个装甲板或只有2个装甲板，判断为平动
        if len(unique_counts) == 1 and unique_counts[0] in self.default_armor_counts:
            return self.TRANSLATION
            
        # 检查明确的旋转模式（要求更严格的条件）
        if len(counts) >= self.min_clear_pattern_len:
            # 检查连续的2-1-2-1-2-1模式
            pattern_len = min(len(counts), self.min_clear_pattern_len)
            start_idx = len(counts) - pattern_len
            
            if pattern_len >= 6:
                pattern_matches = True
                for i in range(pattern_len):
                    expected_value = 2 if i % 2 == 0 else 1
                    if counts[start_idx + i] != expected_value:
                        pattern_matches = False
                        break
                
                if pattern_matches:
                    # 还需要检查时间间隔是否规律
                    pattern_counts = counts[start_idx:start_idx+pattern_len]
                    pattern_times = times[start_idx:start_idx+pattern_len]
                    intervals = self._calculate_intervals(pattern_counts, pattern_times)
                    if len(intervals) >= pattern_len - 1:
                        # 检查时间间隔是否相对稳定
                        valid_intervals = 0
                        for dt in intervals:
                            if self.min_period <= dt <= self.max_period:
                                valid_intervals += 1
                        # 只有当时间间隔大部分有效且相对稳定时才判定为旋转
                        if valid_intervals >= pattern_len - 2 and len(intervals) > 1:
                            dt_min = min(intervals)
                            dt_max = max(intervals)
                            if dt_min > 0 and dt_max / dt_min <= self.max_interval_stability_ratio:
                                return self.ROTATION
            
            # 检查连续的1-2-1-2-1-2模式
            if pattern_len >= 6:
                pattern_matches = True
                for i in range(pattern_len):
                    expected_value = 1 if i % 2 == 0 else 2
                    if counts[start_idx + i] != expected_value:
                        pattern_matches = False
                        break
                
                if pattern_matches:
                    # 还需要检查时间间隔是否规律
                    pattern_counts = counts[start_idx:start_idx+pattern_len]
                    pattern_times = times[start_idx:start_idx+pattern_len]
                    intervals = self._calculate_intervals(pattern_counts, pattern_times)
                    if len(intervals) >= pattern_len - 1:
                        # 检查时间间隔是否相对稳定
                        valid_intervals = 0
                        for dt in intervals:
                            if self.min_period <= dt <= self.max_period:
                                valid_intervals += 1
                        # 只有当时间间隔大部分有效且相对稳定时才判定为旋转
                        if valid_intervals >= pattern_len - 2 and len(intervals) > 1:
                            dt_min = min(intervals)
                            dt_max = max(intervals)
                            if dt_min > 0 and dt_max / dt_min <= self.max_interval_stability_ratio:
                                return self.ROTATION
        
        # 如果装甲板数量始终在默认范围内
        if all(c in self.default_armor_counts for c in counts):
            # 检查是否在1和2之间交替出现
            alternating_score = self._calculate_alternating_score(counts)
            
            # 只有当交替性非常显著时才判定为旋转
            if alternating_score >= self.min_alternating_score:
                # 检查时间间隔是否在合理范围内
                intervals = self._calculate_intervals(counts, times)
                
                # 检查每个间隔是否在 [min_period, max_period] 范围内
                valid_intervals = 0
                for dt in intervals:
                    if self.min_period <= dt <= self.max_period:
                        valid_intervals += 1

                # 只有当大部分时间间隔都有效且稳定时才判定为旋转
                if len(intervals) > 0 and valid_intervals / len(intervals) >= self.min_valid_interval_ratio:
                    if len(intervals) > 1:
                        dt_min = min(intervals)
                        dt_max = max(intervals)
                        if dt_min > 0 and dt_max / dt_min <= self.max_interval_stability_ratio:
                            return self.ROTATION
            
            # 默认返回平动状态
            return self.TRANSLATION

        # 如果装甲板数量在合理范围内（扩展范围），默认为平动而不是未知
        extended_counts = self.default_armor_counts + [3]  # 扩展到包含3
        if all(c in extended_counts for c in counts):
            return self.TRANSLATION

        # 其他情况返回未知状态
        return self.UNKNOWN

    def get_motion_state(self, robot_id: int) -> str:
        """查询给定 robot_id 当前的运动状态。

        Args:
            robot_id: 机器人 ID。

        Returns:
            运动状态字符串（TRANSLATION/ROTATION/UNKNOWN）。
        """
        # 如果有记录且装甲板数量在合理范围内，返回当前状态或默认平动状态
        hist = self._histories.get(robot_id, deque())
        if len(hist) > 0:
            last_count = hist[-1][0]
            # 如果装甲板数量在合理范围内，至少返回平动状态而不是未知
            if last_count in self.default_armor_counts:
                current_state = self._motion_state.get(robot_id, self.UNKNOWN)
                if current_state != self.UNKNOWN:
                    return current_state
                else:
                    return self.TRANSLATION  # 默认返回平动
                    
        # 只有当状态计数器达到阈值时才确认状态，否则返回未知
        if self._state_counter.get(robot_id, 0) >= self.state_confirm_threshold:
            return self._motion_state.get(robot_id, self.UNKNOWN)
        else:
            return self.UNKNOWN

    def get_state_counter(self, robot_id: int) -> int:
        """获取给定 robot_id 的状态计数器数值，用于调试或日志记录。

        Args:
            robot_id: 机器人 ID。

        Returns:
            当前记录的状态计数器值，若 ID 不存在则返回 0。
        """
        return self._state_counter.get(robot_id, 0)

    def get_params(self) -> dict:
        """获取当前参数配置。

        Returns:
            当前参数配置字典。
                - history_len (int): 历史记录长度
                - min_period (float): 最小周期时间（秒）
                - max_period (float): 最大周期时间（秒）
                - state_confirm_threshold (int): 状态确认阈值
                - min_clear_pattern_len (int): 明确模式检测的最小长度
                - min_alternating_score (float): 最小交替性评分
                - min_valid_interval_ratio (float): 最小有效时间间隔比例
                - max_interval_stability_ratio (float): 最大时间间隔稳定性比值
                - default_armor_counts (list): 默认装甲板数量范围
        """
        return self.params.copy()

    def set_params(self, params: dict):
        """设置参数配置。

        Args:
            params (dict): 参数配置字典，可包含以下任意参数：
                - history_len (int): 历史记录长度
                - min_period (float): 最小周期时间（秒）
                - max_period (float): 最大周期时间（秒）
                - state_confirm_threshold (int): 状态确认阈值
                - min_clear_pattern_len (int): 明确模式检测的最小长度
                - min_alternating_score (float): 最小交替性评分
                - min_valid_interval_ratio (float): 最小有效时间间隔比例
                - max_interval_stability_ratio (float): 最大时间间隔稳定性比值
                - default_armor_counts (list): 默认装甲板数量范围
        """
        self.params.update(params)
        # 重新初始化参数
        self.history_len = int(self.params["history_len"])
        self.min_period = float(self.params["min_period"])
        self.max_period = float(self.params["max_period"])
        self.state_confirm_threshold = int(self.params["state_confirm_threshold"])
        self.min_clear_pattern_len = int(self.params["min_clear_pattern_len"])
        self.min_alternating_score = float(self.params["min_alternating_score"])
        self.min_valid_interval_ratio = float(self.params["min_valid_interval_ratio"])
        self.max_interval_stability_ratio = float(self.params["max_interval_stability_ratio"])
        self.default_armor_counts = list(self.params["default_armor_counts"])
        
        # 重新初始化历史记录队列
        for robot_id in self._histories:
            self._histories[robot_id] = deque(maxlen=self.history_len)