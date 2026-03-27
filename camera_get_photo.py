from all_type import *
import time


class InitCamera:  # 初始化相机
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.cap_init_flag = False
        if camera_id == CameraType.DAHENG:
            from dhcamera.GxVideoCupture import GxVideoCupture  # 大恒相机
            self.cap = GxVideoCupture(exposure_time=4000, gain=10, Binning=[2, 2], BinningMode=[0, 0])
            time.sleep(0.3)
            self.cap_init_flag = True
        elif camera_id == CameraType.HAIKANG:
            from hkcamera.test import HkCaptureVedio  # 海康相机
            self.cap = HkCaptureVedio(exposure_time=5000, gain=10)
            self.cap.start_grabbing()
            time.sleep(0.3)
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
