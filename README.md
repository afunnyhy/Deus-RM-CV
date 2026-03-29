# Deus-RM-CV

ZJUT Deus team RoboMaster CV and auto aim code

## 核心特性

- **🚀 极速多进程架构**：采用 `multiprocessing` 彻底分离相机取流与推理计算。利用 `Shared Memory`（无锁 `mp.Array`
  ）实现图像零拷贝传输，极大降低了 IPC 延迟，充分发挥多核 CPU 性能。
- **🎯 深度学习目标检测**：支持 **YOLO** 系列模型，通过 **TensorRT** 推理加速（支持 `.engine` 格式），在 Jetson
  平台上实现超高帧率识别。同时保留传统 OpenCV 识别作为备选。
- **📐 精准姿态解算**：集成 **PnPSolver**，基于装甲板几何特征与相机内参解算目标在三维空间中的相对位置与角度。
- **🛡️ 全兵种支持**：针对步兵、英雄、哨兵不同机型进行适配，支持多相机视角（上置/下置）与相机内参自动切换。
- **📷 工业驱动集成**：原生支持主流工业相机驱动（海康威视 HikVision、大恒图像 Daheng）。

## 项目结构

```text
Deus-RM-CV/
├── main_cam.py               # 主运行程序（多进程调度中心）
├── setting.py                # 全局配置参数（兵种、相机内参、波特率等）
├── uart.py                   # 串口通信模块（与控制板交互）
├── chase_sender.py           # 与导航联动的 UDP 通信发送端
├── detect_armor.py           # 装甲板检测类（深度学习推理入口）
├── get_armor_points_cv.py    # 传统 OpenCV 视觉装甲板检测算法
├── pnp_solver.py             # PnP 位姿解算法
├── coord_converter.py        # 坐标系转换工具（相机、云台、世界坐标）
├── pre_armor.py              # 目标跟踪与 EKF 预测逻辑
├── extended_kalman_filter.py # EKF 算法底层实现
├── armor_chose.py            # 目标选择与打击策略决策
├── ballistic_compensation.py # 弹道补偿计算（空气阻力与重力下坠）
├── light_detector.py         # 灯条提取与角点微调
├── all_type.py               # 通用数据类型与枚举定义
├── camera_get_photo.py       # 相机抽象类与初始化逻辑
├── video_capture.py          # 视频采集与保存工具
├── test_camera.py            # 摄像头辅助测试与参数调节工具
├── hkcamera/                 # 海康相机驱动封装
├── dhcamera/                 # 大恒相机驱动封装
├── models/                   # 存放训练好的模型文件 (.engine .onnx .pt)
├── pt2engine.py              # 工具：导出 TensorRT 模型引擎
└── auto_label.py             # 工具：自动化数据集标注
```

## 快速开始

1. **配置参数**：
   修改 `setting.py` 中的 `my_TroopType`（我方兵种）和 `friend_color`（我方颜色）。

2. **调试说明**：
   可在 `setting.py` 中设置yaw和pitch的缓冲系数、最大瞄准允许误差等参数。

3. **启动系统**：

   ```bash
   python main_cam.py
   ```
