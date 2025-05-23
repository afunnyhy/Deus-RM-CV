import time
import cv2
from camera_get_photo import InitCamera
from setting import *


def video_capture(capture_time=30):
    """
    视频采集函数
    :param capture_time: 采集时间，单位秒，默认30秒
    """
    # 初始化相机类
    print("Camera type:", cameraType, "    ID:", cameraID)
    camera = InitCamera(cameraType)
    print(cameraID, "init success.")
    output_file = time.strftime("%Y%m%d_%H%M%S") + "_captured.mp4"  # 导出文件名为时间
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器（MP4格式）
    fps = 30  # 帧率
    ret, orig_frame = camera.get_photo()
    frame_size = (orig_frame.shape[1], orig_frame.shape[0])  # 视频帧大小（宽度, 高度）
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
    start_time = time.time()
    print("Start capturing...")
    while True:
        # 读取视频流的一帧
        ret, orig_frame = camera.get_photo()
        orig_frame = cv2.flip(orig_frame, -1)
        if not ret:
            continue
        if is_show_video:
            cv2.imshow("capturing", orig_frame)
            cv2.waitKey(1)
        if capture_time > 0:
            video_writer.write(orig_frame)
        if 0 < capture_time < time.time() - start_time:
            video_writer.release()
            cv2.destroyAllWindows()
            camera.delete()
            print("video write over")
            break


if __name__ == "__main__":
    video_capture(capture_time=30)
