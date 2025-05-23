from all_type import *
from GxVideoCupture import GxVideoCupture  # 大恒相机
from hkcamera.test import HkCaptureVedio  # 海康相机


class InitCamera:  # 初始化相机
    def __init__(self, camera_id):
        self.camera_id = camera_id
        if camera_id == CameraType.DAHENG:
            self.cap = GxVideoCupture(exposure_time=4000, gain=0, Binning=[2, 2], BinningMode=[0, 0])
        elif camera_id == CameraType.HAIKANG:
            self.cap = HkCaptureVedio(exposure_time=5000, gain=16)
            self.cap.start_grabbing()

    def get_photo(self):
        if self.camera_id == CameraType.DAHENG:
            ret, orig_frame = self.cap.read()
        elif self.camera_id == CameraType.HAIKANG:
            ret, orig_frame = self.cap.read()
        else:
            print("相机类型不支持")
            return False, None
        return ret, orig_frame

    def delete(self):
        print("释放相机资源...")
        if self.camera_id == CameraType.DAHENG:
            self.cap.release()
        elif self.camera_id == CameraType.HAIKANG:
            self.cap.delete_came()
