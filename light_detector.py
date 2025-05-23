import cv2
from all_type import *
from setting import recorrect_pixel


class LightDetector:
    """
    灯条检测器类, 用于检测模型推断出的装甲板中灯条的四个角点。
    """

    def __init__(self, correct_constant=recorrect_pixel):
        """
        灯条检测器初始化。
        """
        self.scale_constant = correct_constant  # 灯条检测器的修正常数

    @staticmethod
    def detect_vertex(binary_image) -> tuple:
        """
        用于在单侧灯条二值图像中检测顶部点和底部点。
        :param binary_image: 灯条的二值化掩膜图像
        :return: 顶部点和底部点坐标 (x, y)
        """
        height, width = binary_image.shape  # 获取灯条二值图像的高宽
        # 初始化顶部点和底部点
        top_point = None
        bottom_point = None
        # 查找轮廓（RETR_EXTERNAL 仅获取外轮廓，CHAIN_APPROX_SIMPLE 仅保留关键点）
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:  # 如果没有找到轮廓,返回None
            return top_point, bottom_point
        # 创建一个空白画布，与原图同尺寸
        filled_img = np.zeros_like(binary_image)
        # 在空白画布上绘制填充的轮廓
        cv2.drawContours(filled_img, contours, -1, [255], thickness=cv2.FILLED)
        # cv2.imshow("filled_img", filled_img)
        binary_image = filled_img
        # 截取中间三分之一
        third_height = height // 3
        mid_bin = binary_image[third_height:2 * third_height, :]
        # cv2.imshow("mid", mid_bin)
        # 选择图中最大联通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mid_bin, connectivity=8)
        max_area = 0
        max_label = 0
        # 如果没有联通区域
        if num_labels == 1:
            return top_point, bottom_point
        # 遍历所有联通区域，找到最大面积的区域
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area:
                max_area = area
                max_label = i
        # 获取最大联通区域的坐标
        max_area_mask = np.zeros_like(mid_bin)
        max_area_mask[labels == max_label] = 255
        # cv2.imshow("max_area_mask", max_area_mask)
        # 拟合直线方程
        y_indices, x_indices = np.where(max_area_mask == 255)  # 找到白色像素 (255) 的位置
        # 使用最小二乘法拟合直线方程x = ky + b
        A = np.vstack([y_indices, np.ones(len(y_indices))]).T  # 设计矩阵
        k, b = np.linalg.lstsq(A, x_indices, rcond=None)[0]  # 计算最优 k 和 b
        # 将直线变换到原图像素坐标系
        b = b - k * third_height
        # 绘制黑色直线
        # show_line = binary_image
        # for y in range(height):
        #     x = round(k * y + b)
        #     if 0 <= x < width:
        #         show_line[y, x] = 0
        # cv2.imshow("show_line", show_line)
        # cv2.waitKey(0)
        x_top = round(b)
        x_bottom = round(k * (height - 1) + b)
        small_num = 0.0001
        if abs(k) > small_num:
            y_left = round(-b / k)
            y_right = round((width - 1 - b) / k)
        else:
            y_left = round(-b * (1 / small_num))
            y_right = round((width - 1 - b) * (1 / small_num))
        # 计算直线与图像边界交点
        begin_point = (0, x_top)
        end_point = (height - 1, x_bottom)
        if 0 <= x_top < width:
            if x_bottom < 0:
                end_point = (y_left, 0)
            elif x_bottom >= width:
                end_point = (y_right, width - 1)
        elif x_top < 0:
            begin_point = (y_left, 0)
            if x_bottom >= width:
                end_point = (y_right, width - 1)
        else:
            begin_point = (y_right, width - 1)
            if x_bottom < 0:
                end_point = (y_left, 0)
        begin_point = LightDetector.correct_point(begin_point, binary_image)  # 防止点超出图像边界
        end_point = LightDetector.correct_point(end_point, binary_image)
        # 生成直线上均匀分布的像素点
        num_points = end_point[0] - begin_point[0] + 1  # 采样点数
        # 计算 binary_image 每行白点直方图
        line_fitness_hist = np.sum(binary_image == 255, axis=1)
        line_fitness = np.sum(line_fitness_hist) / np.count_nonzero(line_fitness_hist)  # 灯条平均宽度
        kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # 宽 5，高 1 的核
        if line_fitness < 5:  # 灯条太细
            binary_image = cv2.dilate(binary_image, kernel_x, iterations=2)  # x方向膨胀，加粗灯条
            # cv2.imshow("dilate fitness", binary_image)
            # cv2.waitKey(0)
        # 计算采样点的 x 和 y 坐标
        x_vals = np.linspace(begin_point[1], end_point[1], num_points).astype(int)
        y_vals = np.linspace(begin_point[0], end_point[0], num_points).astype(int)
        # 确保索引在图像范围内
        mask = (x_vals >= 0) & (x_vals < width) & (y_vals >= 0) & (y_vals < height)
        x_vals, y_vals = x_vals[mask], y_vals[mask]  # 过滤掉超出图像范围的点
        # 获取所有直线上的像素值
        pixel_values = binary_image[y_vals, x_vals]
        # 找到白色像素的位置
        white_indices = np.where(pixel_values == 255)[0]
        if len(white_indices) == 0:  # 如果没有找到白色像素
            return top_point, bottom_point
        # 寻找连续数列
        white_ranges = []
        white_start = white_indices[0]  # 当前连续区间的起始值
        for i in range(1, len(white_indices)):
            if white_indices[i] != white_indices[i - 1] + 1:  # 断开连续数列
                white_ranges.append((white_start, white_indices[i - 1]))  # 记录当前区间
                white_start = white_indices[i]  # 更新新的起始值
        white_ranges.append((white_start, white_indices[-1]))  # 添加最后一个区间
        # 选择最长的连续区间
        longest_range = max(white_ranges, key=lambda x: x[1] - x[0])
        top_point = (x_vals[longest_range[0]], y_vals[longest_range[0]])  # 靠近begin_point的交点
        bottom_point = (x_vals[longest_range[1]], y_vals[longest_range[1]])  # 靠近end_point的交点
        return top_point, bottom_point

    @staticmethod
    def correct_point(point, binary_image) -> tuple:
        """
        对检测到的点进行修正, 防止点超出图像边界。
        :param point: 检测到的点坐标
        :param binary_image: 灯条的二值化掩膜图像
        :return: 修正后的点坐标
        """
        height, width = binary_image.shape
        # 检查点是否在图像范围内
        y, x = point
        y = max(0, min(height - 1, y))
        x = max(0, min(width - 1, x))
        return y, x

    @staticmethod
    def scale_points(top_point, bottom_point, scale) -> tuple:
        """
        对检测到的点根据两点连线进行缩放
        :param top_point: 检测到的顶部点坐标
        :param bottom_point: 检测到的底部点坐标
        :param scale: 缩放比例
        :return: 修正后的点坐标
        """
        line_vector = (top_point[0] - bottom_point[0], top_point[1] - bottom_point[1])  # 计算两点连线的向量
        line_vector = (line_vector[0] * (1 - scale), line_vector[1] * (1 - scale))  # 缩放向量
        # 计算缩放后的点坐标
        top_point = (top_point[0] - line_vector[0], top_point[1] - line_vector[1])
        bottom_point = (bottom_point[0] + line_vector[0], bottom_point[1] + line_vector[1])
        return top_point, bottom_point

    @staticmethod
    def calculate_length(top_point, bottom_point) -> float:
        """
        计算两点之间的距离
        :param top_point: 顶部点坐标
        :param bottom_point: 底部点坐标
        :return: 两点之间的距离
        """
        return ((top_point[0] - bottom_point[0]) ** 2 + (top_point[1] - bottom_point[1]) ** 2) ** 0.5

    @staticmethod
    def refine_vertex(gray_image, top_point, bottom_point) -> tuple:
        """
        使用灰度图像精细化顶部点和底部点的位置, 通过角点周围的亮度归一化在一像素范围内估算精确坐标。
        :param gray_image: 灯条区域的灰度图像
        :param top_point: 粗略检测到的顶部点
        :param bottom_point: 粗略检测到的底部点
        :return: 顶部点亮度归一化值, 底部点亮度归一化值
        """
        height, width = gray_image.shape  # 获取灰度图像的高宽
        top_brightness = 0.0  # 表示在 top 点上方区域的亮度归一化值
        bottom_brightness = 0.0  # 表示在 bot 点下方区域的亮度归一化值

        # 调整顶部点亮度归一化值
        if top_point and top_point[1] > 0:  # 如果顶部点存在且不在图像顶部边界
            # 确保不超出图像宽度范围
            x_start = max(0, top_point[0] - 3)
            x_end = min(width - 1, top_point[0] + 3)
            # 获取 top_point[0]（x 坐标）左右的 −3,+3 范围内的最大亮度值
            top_brightness = max(gray_image[top_point[1] - 1, x_start:x_end + 1]) / 255

        # 调整底部点亮度归一化值
        if bottom_point and bottom_point[1] < height - 1:  # 如果底部点存在且不在图像底部边界
            # 确保不超出图像宽度范围
            x_start = max(0, bottom_point[0] - 3)
            x_end = min(width - 1, bottom_point[0] + 3)
            # 获取 bottom_point[0]（x 坐标）左右的 −3,+3 范围内的最大亮度值
            bottom_brightness = max(gray_image[bottom_point[1] + 1, x_start:x_end + 1]) / 255

        return top_brightness, bottom_brightness

    def extract_light_points(self, image, detection_data, out_img) -> tuple:
        """
        根据输入图像和检测框提取灯条的四个角点。
        :param image: 输入原始大小图像 (BGR 格式, 未经缩放)
        :param detection_data: 为一个ArmorPlate对象
        :param out_img: 输出图像, 用于绘制检测到的灯条
        :return: 一个ArmorPlate对象
        """
        bbox = detection_data.camera_pos  # 获取四个顶点坐标
        color_cls = detection_data.color  # 获取颜色类别
        troop_cls = detection_data.troop_type  # 获取兵种类别
        x1, y1, x2, y2 = bbox
        # 确保边界框不超出图像范围
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1] - 1, x2), min(image.shape[0] - 1, y2)
        roi = image[y1:y2, x1:x2]  # 提取感兴趣区域 (ROI)
        # cv2.imshow("roi", roi)  # 显示roi
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  # 转换为 HSV 空间
        # cv2.imshow("hsv_roi", hsv_roi)  # 显示HSV图像
        gray = hsv_roi[:, :, 2]
        # cv2.imshow("gray", gray)  # 显示灰度图像
        th, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 动态阈值二值化
        # cv2.imshow("binary_image", binary_image)  # 显示二值化图像
        # cv2.waitKey(0)
        # 分割二值化图像的左右两灯条
        half_width = binary_image.shape[1] // 2
        left_mask, right_mask = binary_image[:, :half_width], binary_image[:, half_width:]
        # cv2.imshow("left_mask", left_mask)  # 显示左侧灯条掩膜
        # cv2.imshow("right_mask", right_mask)  # 显示右侧灯条掩膜
        # 检测左半部分和右半部分的边缘点
        top_left, bottom_left = self.detect_vertex(left_mask)
        top_right, bottom_right = self.detect_vertex(right_mask)
        if not top_left or not bottom_left or not top_right or not bottom_right:  # 如果没有检测到边缘点
            return False, ArmorPlate(None, color_cls, troop_cls), out_img  # 返回空装甲板对象

        # 使用颜色分量图像进一步精细化边缘点位置
        # left_color, right_color = color_threshold[:, :half_width], color_threshold[:, half_width:]  # 左右两灯条的颜色分量
        # top_left_refined, bottom_left_refined = self.refine_vertex(left_color, top_left, bottom_left)
        # top_right_refined, bottom_right_refined = self.refine_vertex(right_color, top_right, bottom_right)

        # 缩放点坐标
        top_left, bottom_left = self.scale_points(top_left, bottom_left, self.scale_constant)
        top_right, bottom_right = self.scale_points(top_right, bottom_right, self.scale_constant)
        # 对检测到的点进行修正, 防止点超出图像边界
        top_left = self.correct_point(top_left, binary_image)
        bottom_left = self.correct_point(bottom_left, binary_image)
        top_right = self.correct_point(top_right, binary_image)
        bottom_right = self.correct_point(bottom_right, binary_image)
        # 计算灯条长度
        left_light_length = self.calculate_length(top_left, bottom_left)  # 左灯条长度
        right_light_length = self.calculate_length(top_right, bottom_right)  # 右灯条长度
        if max(left_light_length, right_light_length) / min(left_light_length, right_light_length) > 2.5:  # 无效装甲板
            return False, ArmorPlate(None, color_cls, troop_cls), out_img  # 返回空装甲板对象

        # 转换坐标到原图空间
        top_left = (top_left[0] + x1, top_left[1] + y1)  # - top_left_refined)
        bottom_left = (bottom_left[0] + x1, bottom_left[1] + y1)  # + bottom_left_refined)
        top_right = (top_right[0] + x1 + half_width, top_right[1] + y1)  # - top_right_refined)
        bottom_right = (bottom_right[0] + x1 + half_width, bottom_right[1] + y1)  # + bottom_right_refined)
        # 构造输出点集
        output_points = np.array([top_left, bottom_left, top_right, bottom_right], dtype=np.float32)
        # 计算灯条区域四边形面积
        light_area = cv2.contourArea(np.array([top_left, bottom_left, bottom_right, top_right], dtype=np.int32))

        # 将点集重塑为适合 cv2.polylines 的形状
        out_img = np.ascontiguousarray(out_img)
        top_left = (round(top_left[0]), round(top_left[1]))
        bottom_left = (round(bottom_left[0]), round(bottom_left[1]))
        top_right = (round(top_right[0]), round(top_right[1]))
        bottom_right = (round(bottom_right[0]), round(bottom_right[1]))
        # 绘制四个角点
        cv2.line(out_img, top_left, bottom_left, color=(0, 255, 0), thickness=2)
        cv2.line(out_img, bottom_left, top_right, color=(0, 255, 0), thickness=2)
        cv2.line(out_img, top_right, bottom_right, color=(0, 255, 0), thickness=2)
        cv2.line(out_img, bottom_right, top_left, color=(0, 255, 0), thickness=2)
        return True, ArmorPlate(output_points, color_cls, troop_cls, light_area,
                                detection_data.confident), out_img  # 返回装甲板对象


if __name__ == "__main__":
    # 测试代码
    test_photo_path = "./test_data/photo2/"
    test_color = Color.BLUE  # 要识别的颜色

    from detect_armor import ArmorDetector
    from setting import *
    import torch
    import os

    if test_color == Color.RED:
        my_color = Color.BLUE
    else:
        my_color = Color.RED
    # 初始化模型推断类
    armor_de = ArmorDetector(model_path, model_name, torch.cuda.is_available(), my_color, ".pt")
    # 初始化灯条解算类
    light_pos = LightDetector()
    # 获取目录的所有.jpg或.png文件
    test_photo_list = os.listdir(test_photo_path)
    test_photo_list = [os.path.join(test_photo_path, photo) for photo in test_photo_list if
                       photo.endswith(".png") or photo.endswith(".jpg")]
    # 开始测试
    print("test photo path:", test_photo_path)
    print("test color:", test_color)
    for photo_path in test_photo_list:
        # 读取图片
        orig_frame = cv2.imread(photo_path)
        all_detect_armor, out_img = armor_de.detect_armor(orig_frame)
        find = False
        for detected_armor in all_detect_armor:
            ret, detected_armor, out_img = light_pos.extract_light_points(orig_frame, detected_armor, out_img)
            if ret:
                find = True
        if find:
            cv2.imshow(photo_path + " output", out_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
