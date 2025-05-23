import os
import time
import threading
import queue
import cv2
import numpy as np
from all_type import LightCV
import matplotlib.pyplot as plt
from enum import Enum
from all_type import Color, ArmorType, TroopType, ArmorTargetPoint, ArmorPlate


class armor_getter():
    def __init__(self, my_color):
        detect_color = Color.BLUE
        if my_color == Color.BLUE:
            detect_color = Color.RED
        if detect_color == detect_color.RED:
            self.color = [0, 0, 255]  # OpenCV uses BGR format
            # 下面的可能要删
            self.color_name = Color.BLUE
        elif detect_color == detect_color.BLUE:
            self.color = [255, 0, 0]
            # 下面的可能要删
            self.color_name = Color.RED

        self.binary_thres = 180
        self.l_params = {
            'min_ratio': 0.1,
            'max_ratio': 0.55,
            'max_angle': 40.0
        }
        self.a_params = {
            'min_light_ratio': 0.7,
            'min_small_center_distance': 0.8,
            'max_small_center_distance': 3.2,
            'min_large_center_distance': 3.2,
            'max_large_center_distance': 5.5,
            'max_angle': 15.0
        }
        self.classifier_threshold = 0.7
        self.ignore_classes = ['negative']

    def preprocessImage(self, img):
        # 将图像从 BGR 转换为 HSV 颜色空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 定义红色的HSV范围
        lower_red1 = np.array([0, 100, 175])
        upper_red1 = np.array([20, 255, 255])
        lower_red2 = np.array([160, 100, 175])
        upper_red2 = np.array([180, 255, 255])

        # 定义蓝色的HSV范围
        lower_blue = np.array([100, 100, 175])
        upper_blue = np.array([140, 255, 255])

        # 创建红色和蓝色的掩码
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        # 合并红色掩码
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # 合并红色和蓝色掩码
        combined_mask = cv2.bitwise_or(mask_red, mask_blue)

        # 形态学操作：膨胀和腐蚀
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(combined_mask, kernel, iterations=3)
        binary = cv2.erode(binary, kernel, iterations=2)
        # 显示处理结果
        resized_img = cv2.resize(binary, (500, 500))
        # cv2.imshow('Preprocessed Image', resized_img)

        return binary

    def preprocessImageForWhite(self, img):
        # 将图像从 BGR 转换为灰度图像（如果需要的话）
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 应用二值化
        # 使用cv2.THRESH_BINARY作为类型，表示高于阈值的部分将被设为max_value，其余部分设为0
        ret, binary_img = cv2.threshold(gray, 32, 255, cv2.THRESH_BINARY)

        # 形态学操作：膨胀和腐蚀，以去除噪声
        kernel = np.ones((3, 3), np.uint8)
        binary_white = cv2.dilate(binary_img, kernel, iterations=2)
        binary_white = cv2.erode(binary_white, kernel, iterations=1)

        # resized_img = cv2.resize(binary_white, (500, 500))
        # cv2.imshow('white', resized_img)

        return binary_white

    def isLight(self, light):
        ratio = light.width / light.length
        ratio_ok = self.l_params['min_ratio'] < ratio < self.l_params['max_ratio']
        # 调整角度条件，允许竖直矩形和左右偏移不超过40度的矩形
        adjusted_angle = light.tilt_angle % 180  # 将角度转换到[0, 180)范围内
        angle_ok = (0 <= light.tilt_angle < 40) or \
                   (90 >= adjusted_angle >= 50)
        # print(
        #   f"Light: width={light.width}, length={light.length}, ratio={ratio}, tilt_angle={light.tilt_angle}, ratio_ok={ratio_ok}, angle_ok={angle_ok}")
        return ratio_ok and angle_ok

    def find_all_light(self, img):
        # print("find_all_light刚开始")
        binary_img = self.preprocessImage(img)

        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lights = []

        for contour in contours:
            # print(len(contour))
            if len(contour) < 10:
                continue
            r_rect = cv2.minAreaRect(contour)
            light = LightCV(r_rect)

            # 在这里计算最上面点和最下面点，并将其添加为light对象的属性
            if contour is not None and len(contour) > 0:
                top_point = tuple(contour[contour[:, :, 1].argmin()][0])
                bottom_point = tuple(contour[contour[:, :, 1].argmax()][0])
                # print("top_point=", top_point)
                # print("bottom_point=", bottom_point)
                light.top_point = top_point
                light.bottom_point = bottom_point

            # print(self.isLight(light))  # 测试使用

            if self.isLight(light):
                rect = cv2.boundingRect(contour)
                if (0 <= rect[0] < img.shape[1] and 0 <= rect[1] < img.shape[0]):
                    sum_r = 0
                    sum_b = 0
                    roi = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                    # cv2.imshow("roi",roi)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    for i in range(roi.shape[0]):
                        for j in range(roi.shape[1]):
                            if cv2.pointPolygonTest(contour, (j + rect[0], i + rect[1]), False) >= 0:
                                sum_r += roi[i, j][2]
                                sum_b += roi[i, j][0]

                    # 测试使用
                    # print("sum_r=", sum_r)
                    # print("sum_b=", sum_b)

                    light.color = Color.RED if sum_r > sum_b else Color.BLUE
                    # 测试使用
                    # print(light.color)
                    # print(Color.RED)
                    # print(Color.BLUE)
                    # if light.color == self.color_name:
                    # cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2) # 测试使用
                    # print("light")
                    # lights.append(light)
                    lights.append(light)
        # print(lights)

        return lights

    def containLight(self, light_1, light_2, lights):
        points = [light_1.top(), light_1.bottom(), light_2.top(), light_2.bottom()]
        bounding_rect = cv2.boundingRect(np.array(points, dtype=np.float32))

        for test_light in lights:
            if test_light.center == light_1.center or test_light.center == light_2.center:
                continue

            if (bounding_rect[0] <= test_light.top()[0] <= bounding_rect[0] + bounding_rect[2] and
                bounding_rect[1] <= test_light.top()[1] <= bounding_rect[1] + bounding_rect[3]) or \
                    (bounding_rect[0] <= test_light.bottom()[0] <= bounding_rect[0] + bounding_rect[2] and
                     bounding_rect[1] <= test_light.bottom()[1] <= bounding_rect[1] + bounding_rect[3]) or \
                    (bounding_rect[0] <= test_light.center[0] <= bounding_rect[0] + bounding_rect[2] and
                     bounding_rect[1] <= test_light.center[1] <= bounding_rect[1] + bounding_rect[3]):
                return True

        return False

    def detect_white_region(self, roi, quad):
        # 转换为 int32
        quad = quad.astype(np.int32)

        # 创建一个轮廓（确保点是按照顺时针或逆时针顺序排列）
        contour = quad.reshape((-1, 1, 2))

        # 计算整个画面大小面积
        original_area = roi.shape[0] * roi.shape[1]

        # 计算四边形的面积
        cropped_area = cv2.contourArea(contour)

        # 创建一个与原始图像大小相同的黑色掩模
        image_shape = (roi.shape[0], roi.shape[1])  # 根据你的图像尺寸调整这里
        mask = np.zeros(image_shape, dtype=np.uint8)

        # 在掩模中绘制填充的四边形
        cv2.fillConvexPoly(mask, quad, 255)

        # 对灰度图像进行二值化处理，假设白色接近255
        binary_image = self.preprocessImageForWhite(roi)

        # 应用掩模到二值图像上
        masked_image = cv2.bitwise_and(binary_image, mask)

        # 计算白色像素点的面积
        white_pixels = cv2.countNonZero(masked_image)

        area_ratio = cropped_area / original_area if original_area > 0 else 0

        # 动态调整白色区域阈值，这里假设800是针对全图的一个合理阈值
        threshold = 700000 * area_ratio  # 根据面积比调整阈值

        # 判断是否存在白色区域
        has_white_region = white_pixels > threshold

        return has_white_region, white_pixels, threshold

    def isArmor(self, light_1, light_2, img):  # 添加 img 参数
        # print("is_Armor!!!!!!")
        # 计算光条长度比
        light_length_ratio = min(light_1.length, light_2.length) / max(light_1.length, light_2.length)
        light_ratio_ok = light_length_ratio > self.a_params['min_light_ratio']

        # 计算中心距离
        avg_light_length = (light_1.length + light_2.length) / 2
        center_distance = cv2.norm(np.array(light_1.center) - np.array(light_2.center)) / avg_light_length
        center_distance_ok = (
                (self.a_params['min_small_center_distance'] <= center_distance < self.a_params[
                    'max_small_center_distance']) or
                (self.a_params['min_large_center_distance'] <= center_distance < self.a_params[
                    'max_large_center_distance']))

        # 计算两光条中心连线的角度
        diff_center = np.array(light_1.center) - np.array(light_2.center)
        angle_center = np.abs(np.arctan2(diff_center[1], diff_center[0])) * 180 / np.pi
        angle_ok = (0 <= angle_center < 15) or (180 >= angle_center >= 165)

        # 新增：计算两个灯条上下端点连线的角度，并计算两者之间的角度差
        if hasattr(light_1, 'top_point') and hasattr(light_2, 'top_point') \
                and hasattr(light_1, 'bottom_point') and hasattr(light_2, 'bottom_point'):
            top_diff_1 = np.array(light_1.bottom_point) - np.array(light_1.top_point)
            top_diff_2 = np.array(light_2.bottom_point) - np.array(light_2.top_point)

            # 计算每个灯条上下端点连线的角度
            angle_light_1 = np.arctan2(top_diff_1[1], top_diff_1[0]) * 180 / np.pi
            angle_light_2 = np.arctan2(top_diff_2[1], top_diff_2[0]) * 180 / np.pi

            # 将角度调整到 [0, 180] 范围内
            angle_light_1 = angle_light_1 % 180
            angle_light_2 = angle_light_2 % 180

            # 计算两个角度之间的最小差值
            angle_difference = abs(angle_light_1 - angle_light_2)
            angle_difference_adjusted = min(angle_difference, 180 - angle_difference)  # 处理跨越180度的情况
        else:
            angle_light_1 = None
            angle_light_2 = None
            angle_difference_adjusted = None
            print("Error: Light objects do not have top_point or bottom_point attributes.")

        angles_difference_ok = angle_light_1 >= 90 and angle_light_2 >= 90 and angle_difference_adjusted <= 8 or angle_light_1 <= 90 and angle_light_2 <= 90 and angle_difference_adjusted <= 8 or angle_difference_adjusted <= 3

        # 新增：检查四边形内是否有白色部分
        points = np.array([light_1.top_point, light_1.bottom_point, light_2.top_point, light_2.bottom_point],
                          dtype=np.float32)
        binary_white_img = self.preprocessImageForWhite(img)
        has_white_region, whitearea, threshold = self.detect_white_region(binary_white_img, points)

        # 判断装甲板有效性，新增条件：has_white_region
        is_armor = light_ratio_ok and center_distance_ok and angle_ok and angles_difference_ok and has_white_region

        if is_armor:
            # 此处已修改
            type_armor = ArmorType.LARGE if center_distance > self.a_params[
                'min_large_center_distance'] else ArmorType.SMALL
        else:
            type_armor = "Invalid"

        # print("cancan nengbunengshuchu3")
        # print(f"cancan nengbunengshuchu")
        # print("cancan nengbunengshuchu2")
        # print(
        # f"Armor: light_length_ratio={light_length_ratio}, center_distance={center_distance}, # angle_center={angle_center}, "
        # f"light_ratio_ok={light_ratio_ok}, center_distance_ok={center_distance_ok}, # angle_ok={angle_ok}, "
        # f"angle_light_1={angle_light_1}, angle_light_2={angle_light_2}, angle_difference_adjusted={angle_difference_adjusted},"
        # f"has_white_region={has_white_region}, is_armor={is_armor}")
        # f"is_armor={is_armor}")
        # ****************************************************
        # print(
        #     f"Armor: light_length_ratio={light_length_ratio}, center_distance={center_distance}, angle_center={angle_center}, "
        #     f"light_ratio_ok={light_ratio_ok}, center_distance_ok={center_distance_ok}, angle_ok={angle_ok}, "
        #     f"angle_light_1={angle_light_1}, angle_light_2={angle_light_2}, angle_difference_adjusted={angle_difference_adjusted}, "
        #     f"is_armor={is_armor},"
        #     f"whitearea={whitearea},"
        #     f"threshold={threshold}")

        return type_armor

    def match_armors(self, lights, img):
        # match_armor可以进循环
        armors = []
        points = []  # 装甲板点集

        for i, light_1 in enumerate(lights):
            for light_2 in lights[i + 1:]:
                # print(light_1.color, light_2.color)
                # 判断颜色是否匹配，如果不匹配则跳过
                if light_1.color == self.color_name or light_2.color == self.color_name:
                    continue

                # 如果这两个灯条包含在其他灯条中，则跳过
                if self.containLight(light_1, light_2, lights):
                    continue

                # 如果不是无效的装甲板，检查是否符合装甲板的条件
                armor_type = self.isArmor(light_1, light_2, img)
                # 排序左右装甲板
                light_1, light_2 = sorted([light_1, light_2], key=lambda x: x.top_point[0])
                if armor_type != "Invalid":
                    # 创建 NumPy 数组
                    points = np.array(
                        [light_1.top_point, light_1.bottom_point, light_2.top_point, light_2.bottom_point],
                        dtype=np.float32)
                    if armor_type == ArmorType.LARGE:
                        armor_plate = ArmorPlate(points, self.color_name, TroopType.HERO)
                    else:
                        armor_plate = ArmorPlate(points, self.color_name, TroopType.INFANTRY)
                    armors.append(armor_plate)

                    # armor_plate = ArmorPlate(points, light_1.color , TroopType.SENTINEL)
                    # armors.append(armor_plate)

        # print("match_armor")
        return armors

    def draw_Armors(self, img, armors):
        armor_color = [0, 255, 0]  # 绿色，BGR格式
        for armor in armors:
            if isinstance(armor, ArmorPlate):
                top_left = (int(armor.camera_pos[0][0]), int(armor.camera_pos[0][1]))
                top_right = (int(armor.camera_pos[2][0]), int(armor.camera_pos[2][1]))
                bottom_left = (int(armor.camera_pos[1][0]), int(armor.camera_pos[1][1]))
                bottom_right = (int(armor.camera_pos[3][0]), int(armor.camera_pos[3][1]))

                # 绘制上边
                cv2.line(img, top_left, top_right, armor_color, 2)
                # 绘制下边
                cv2.line(img, bottom_left, bottom_right, armor_color, 2)
        return img

    # 修改 draw_Lights 函数中的相关部分
    def draw_Lights(self, img, lights):
        for light in lights:
            if hasattr(light, 'top_point') and hasattr(light, 'bottom_point'):
                top = [int(light.top_point[0]), int(light.top_point[1])]
                bottom = [int(light.bottom_point[0]), int(light.bottom_point[1])]
                cv2.line(img, tuple(top), tuple(bottom), self.color, 4)
        return img

    def get_armors_by_img(self, img):
        # result_Armors = []
        out_img = img.copy()
        points = []  # 点集 新增的

        lights = self.find_all_light(img)
        # armors = self.match_armors(lights,img) # 原来的
        armors = self.match_armors(lights, img)  # 新增的

        out_img = self.draw_Lights(out_img, lights)
        out_img = self.draw_Armors(out_img, armors)
        return len(armors) != 0, armors, out_img
