from all_type import *
import time
import cv2  # 用于保存图片
from datetime import datetime  # 用于获取当前时间


class InitCamera:  # 初始化相机
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.cap_init_flag = False
        if camera_id == CameraType.DAHENG:
            from dhcamera.GxVideoCupture import GxVideoCupture  # 大恒相机
            self.cap = GxVideoCupture(exposure_time=3500, gain=0, Binning=[2, 2], BinningMode=[0, 0])
            time.sleep(0.2)
            self.cap_init_flag = True
        elif camera_id == CameraType.HAIKANG:
            from hkcamera.HkVideoCupture import HkCaptureVideo  # 海康相机
            self.cap = HkCaptureVideo(exposure_time=4500, gain=0)
            self.cap.start_grabbing()
            time.sleep(0.2)
            self.cap_init_flag = True

    def get_photo(self):
        if not self.cap_init_flag:
            print("相机未初始化，无法获取图像")
            return False, None
        if self.camera_id == CameraType.DAHENG:
            ret, orig_frame = self.cap.read()
        elif self.camera_id == CameraType.HAIKANG:
            ret, orig_frame = self.cap.read()
        else:
            print("相机类型不支持")
            return False, None
        return ret, orig_frame

    def __del__(self):
        print("释放相机资源...")
        self.cap_init_flag = False
        if self.camera_id == CameraType.DAHENG:
            self.cap.release()
        elif self.camera_id == CameraType.HAIKANG:
            self.cap.delete_came()


if __name__ == '__main__':
    test_camera_type = CameraType.DAHENG

    # 1. 实例化相机对象
    camera = InitCamera(test_camera_type)

    # 2. 检查相机是否初始化成功
    if camera.cap_init_flag:
        print("相机初始化成功！")
        print("--> 按 's' 键保存当前画面")
        print("--> 按 'q' 键或 'ESC' 键退出程序")

        while True:
            # 3. 持续获取图像帧
            ret, frame = camera.get_photo()

            if not ret or frame is None:
                print("无法获取画面，退出循环...")
                break

            frame = cv2.flip(frame, -1)
            # 4. 实时显示画面
            cv2.imshow("Real-time Camera Feed", frame)

            # 5. 监听键盘按键 (延迟1毫秒)
            key = cv2.waitKey(1) & 0xFF

            # 6. 处理按键逻辑
            if key == ord('s'):
                # 按 's' 保存图片
                # 生成以当前时间为文件名的字符串，精确到毫秒
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"image_{current_time}.jpg"

                # 保存图片到当前目录
                cv2.imwrite(filename, frame)
                print(f"成功保存图片: {filename}")

            elif key == ord('q') or key == 27:
                # 按 'q' 或者是 ESC 键 (ASCII码27) 退出程序
                print("退出相机显示...")
                break

        # 7. 销毁所有OpenCV窗口
        cv2.destroyAllWindows()
        # 显式删除相机对象，触发 __del__ 释放资源
        del camera
    else:
        print("相机初始化失败，请检查连接或配置。")
