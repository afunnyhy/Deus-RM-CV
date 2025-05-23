import os
from ultralytics import YOLO
from all_type import *
import time
import cv2


class ArmorDetector:  # 模型推理类

    def __init__(self, model_path, model_name, CUDA, CmdID, model_type=".engine"):
        self.photo = None
        self.CUDA = CUDA  # 是否使用GPU
        self.CmdID = CmdID  # 我方装甲板颜色id
        self.resize_shape = 320  # 图片缩放尺寸
        self.min_confidence = 0.7  # 最低置信度
        model_name += model_type
        self.model_path = os.path.abspath(os.path.join(model_path, model_name))  # 模型路径
        print("model path:", self.model_path)
        self.model = YOLO(self.model_path, task="detect")  # 初始化模型
        # 初始化颜色和装甲板类型
        self.label_index = {
            "blue3": (Color.BLUE, TroopType.INFANTRY),
            "red3": (Color.RED, TroopType.INFANTRY),
            "bluesb": (Color.BLUE, TroopType.SENTINEL),
            "redsb": (Color.RED, TroopType.SENTINEL),
            "blue1": (Color.BLUE, TroopType.HERO),
            "red1": (Color.RED, TroopType.HERO),
        }
        # 初始化要攻击的装甲板颜色
        self.pos_cls = Color.RED
        if self.CmdID == Color.RED:
            self.pos_cls = Color.BLUE

    def detect_armor(self, orig_img, detect_color=None):
        if detect_color is None:
            detect_color = self.pos_cls
        else:
            if detect_color == 1:
                detect_color = Color.BLUE
            else:  # 0
                detect_color = Color.RED
        frame_img = orig_img.copy()  # 复制原图
        out_img = orig_img.copy()

        ori_h, ori_w, ori_c = frame_img.shape
        frame_img = cv2.resize(frame_img, (ori_w // 2, ori_h // 2), interpolation=cv2.INTER_NEAREST)

        # 运行推理
        start = time.time()
        output = self.model(frame_img, imgsz=self.resize_shape, device="0" if self.CUDA else "cpu", verbose=False)
        # print(time.time()-start)

        # 解析输出
        detected = []  # 检测到的装甲板
        for i, result in enumerate(output):
            boxes = result.boxes
            names = result.names
            for box in boxes:
                confidence = box.conf[0].item()
                if confidence < self.min_confidence:
                    continue  # 置信度过低
                label_name = names[int(box.cls[0].item())]
                # print("label_name", label_name)
                color_type, troop_type = self.label_index[label_name]
                if color_type != detect_color:
                    continue  # 颜色不符合
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                xyxy = [x1 * 2, y1 * 2, x2 * 2, y2 * 2]
                # 提取边界框颜色
                color_print = (0, 0, 255)  # 默认红色
                if color_type == Color.BLUE:  # 根据类别设置不同颜色
                    color_print = (255, 0, 0)  # 蓝色
                # 绘制边界框
                cv2.rectangle(out_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color_print, 2)
                # 绘制标签
                cv2.putText(out_img, f"{names[int(box.cls[0].item())]} {confidence:.2f}",
                            (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_print, 2)

                detect_armor = ArmorPlate(xyxy, color_type, troop_type, confidence)  # 构建装甲板对象
                detected.append(detect_armor)  # 添加到检测到的装甲板列表

        # 按照置信度排序
        detected.sort(key=lambda x: x.confident, reverse=True)

        return detected, out_img  # 返回检测到的装甲板列表
