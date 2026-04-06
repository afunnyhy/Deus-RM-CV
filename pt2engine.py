from ultralytics import YOLO

# 载入模型
model = YOLO("models/best-cv.pt", task="detect")
# model = YOLO("models/best-pose.pt", task="pose")

# 导出模型
model.export(
    format="engine",  # TensorRT引擎格式
    imgsz=640,  # 模型输入图像大小
    batch=1,  # 批处理大小，在实时视频流中通常设为1
    device=0,  # 使用GPU设备0进行推理
    workspace=4,  # TensorRT工作空间大小，单位为GB，增加此值可以提高性能，但需要更多的GPU内存
    # simplify=True,  # 简化 ONNX 图结构
    # half=True,  # 开启半精度推理以提高性能和减少内存使用
    int8=False,
    # data="ultralytics/cfg/datasets/robomaster.yaml",
    # dynamic=True,
)
