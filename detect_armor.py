import os
from ultralytics import YOLO
import torch
from all_type import *
from setting import is_show_video


class ArmorDetector:  # 模型推理类

    def __init__(self, model_path: str, model_name: str, CmdID, CUDA=True):
        print("torch.cuda.is_available:", torch.cuda.is_available())
        self.photo = None
        self.CUDA = CUDA  # 是否使用GPU
        self.CmdID = CmdID  # 我方装甲板颜色id
        self.resize_shape = 640  # 图片缩放尺寸
        self.min_confidence = 0.7  # 最低置信度
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, model_path)
        self.model_path = os.path.abspath(os.path.join(model_dir, model_name))
        print("model path:", self.model_path)
        self.model = YOLO(self.model_path, task="detect")  # 初始化模型
        # print(self.model.device)
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
        # 颜色判断逻辑
        if detect_color is None:
            detect_color = self.pos_cls
        else:
            detect_color = Color.BLUE if detect_color == 1 else Color.RED

        out_img = orig_img
        # 运行推理
        results = self.model(orig_img, imgsz=self.resize_shape, device="cuda:0" if self.CUDA else "cpu", verbose=False,
                             stream=True)
        detected = []
        for result in results:  # stream=True 返回的是生成器
            boxes = result.boxes
            if len(boxes) == 0:
                continue
            # 一次性将所有框的数据拉回 CPU，转换为 numpy 数组进行批量操作
            confs = boxes.conf.cpu().numpy()
            clses = boxes.cls.cpu().numpy()
            xyxys = boxes.xyxy.cpu().numpy()
            for i in range(len(boxes)):
                confidence = confs[i]
                if confidence < self.min_confidence:
                    continue
                label_name = result.names[int(clses[i])]
                color_type, troop_type = self.label_index[label_name]
                if color_type != detect_color:
                    continue
                x1, y1, x2, y2 = map(int, xyxys[i])
                detect_armor = ArmorPlate([x1, y1, x2, y2], color_type, troop_type, confidence)
                detected.append(detect_armor)

                if is_show_video:
                    # 绘制边界框
                    color_print = (255, 0, 0) if color_type == Color.BLUE else (0, 0, 255)
                    cv2.rectangle(out_img, (x1, y1), (x2, y2), color_print, 2)
                    cv2.putText(out_img, f"{label_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                color_print, 2)

        # 按照置信度排序
        detected.sort(key=lambda x: x.confident, reverse=True)
        return detected, out_img
