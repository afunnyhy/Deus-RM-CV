from ultralytics import YOLO

# 载入模型
model = YOLO("weight/20250318_normal.pt", task="detect")

# 导出模型
model.export(
    format="engine",
    imgsz=320,
    # dynamic=True,
    # batch=1,
    workspace=2,
    int8=False,
    # half=True,
    # data="ultralytics/cfg/datasets/robomaster.yaml",
    device=0,
)
