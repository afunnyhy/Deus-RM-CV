import cv2
import time


# 空回调函数，trackbar 必须需要
def nothing(x):
    pass


def record_with_settings(duration_sec=30):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # --- 阶段 1：设置窗口和滑块 ---
    window_name = 'Camera Setup'
    cv2.namedWindow(window_name)

    # 尝试获取当前摄像头的默认值，作为滑块初始位置
    # 注意：不同摄像头获取到的值范围差异极大，如果报错或不准，可以手动改为 50
    try:
        init_bright = int(cap.get(cv2.CAP_PROP_BRIGHTNESS))
        init_contrast = int(cap.get(cv2.CAP_PROP_CONTRAST))
        # 曝光通常很特殊，先设个中间值试试
        init_exposure = 50
    except:
        init_bright = 50
        init_contrast = 50
        init_exposure = 50

    # 创建滑块
    # 参数: 滑块名称, 窗口名称, 默认值, 最大值, 回调函数
    # 大部分摄像头亮度和对比度最大值是 255 或 100。这里设为 255。
    cv2.createTrackbar('Brightness', window_name, init_bright, 255, nothing)
    cv2.createTrackbar('Contrast', window_name, init_contrast, 255, nothing)

    # 曝光比较特殊，有些摄像头需要先关闭自动曝光才能手动调节。
    # 0.25 或 0.75 在很多 OpenCV 版本中代表“手动模式”
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    # 曝光滑块范围设大一点，因为有些摄像头是负数，有些是正大数。
    # 这里演示用 0-100 的通用范围尝试，如果无效需针对硬件调整。
    cv2.createTrackbar('Exposure (Try vary)', window_name, init_exposure, 100, nothing)

    print("---------------------------------------------------------")
    print("【准备阶段】请拖动滑块调整画面。")
    print("调整完毕后，按键盘 's' 键开始 30秒录制。")
    print("按 'q' 键直接退出。")
    print("---------------------------------------------------------")

    recording_started = False
    start_time = None
    out = None

    # 获取画面尺寸用于录制初始化
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret: break

        key = cv2.waitKey(1) & 0xFF

        if not recording_started:
            # --- 预览调节模式 ---

            # 1. 读取滑块当前位置
            br_val = cv2.getTrackbarPos('Brightness', window_name)
            co_val = cv2.getTrackbarPos('Contrast', window_name)
            ex_val = cv2.getTrackbarPos('Exposure (Try vary)', window_name)

            # 2. 将设置应用到摄像头
            # 注意：对于曝光，不同摄像头需要的值很不一样。
            # 如果滑块无效，可能需要尝试将 ex_val 映射到负数区间，例如 (ex_val - 100)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, br_val)
            cap.set(cv2.CAP_PROP_CONTRAST, co_val)
            cap.set(cv2.CAP_PROP_EXPOSURE, ex_val)  # 尝试直接设置正数

            # 在画面上显示提示
            cv2.putText(frame, "Adjust Settings. Press 's' to Start Record", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow(window_name, frame)

            # 按 's' 开始录制
            if key == ord('s'):
                print("设置已锁定，开始录制...")
                recording_started = True
                start_time = time.time()
                # 初始化视频写入器，使用当前时间作为文件名防止覆盖
                time_str = time.strftime("%Y%m%d_%H%M%S")
                filename = f'video_{time_str}.avi'
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

            # 按 'q' 退出
            elif key == ord('q'):
                break

        else:
            # --- 录制模式 (沿用之前的逻辑) ---
            elapsed_time = time.time() - start_time
            remaining_time = duration_sec - elapsed_time

            if elapsed_time > duration_sec:
                print("时间到，录制结束。")
                break

            # 显示倒计时并录制
            timer_text = f"Recording: {int(remaining_time)}s left"
            # 为了不把倒计时文字录进去，我们 copy 一份用来显示
            display_frame = frame.copy()
            cv2.putText(display_frame, timer_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow(window_name, display_frame)
            # 写入原始的纯净画面
            out.write(frame)

            if key == ord('q'):
                print("中途停止录制")
                break

    # 清理资源
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    record_with_settings(30)
