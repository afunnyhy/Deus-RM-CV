# -*- coding: utf-8 -*-
import sys
# import MvImport

from .MvCameraControl_class import *
from .MvErrorDefine_const import *
from .CameraParams_header import *

import ctypes
import numpy as np
import cv2
import time
import datetime

# # -*- coding: utf-8 -*-
# import sys
# import time
# from CamOperation_class import CameraOperation
# import datetime
# # import MvImport

# from MvCameraControl_class import *
# from MvErrorDefine_const import *
# from CameraParams_header import *

# import ctypes
# import numpy as np
# import cv2


def ToHexStr(num):
    """将错误码转换为十六进制"""
    chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
    hexStr = ""
    if num < 0:
        num = num + 2 ** 32
    while num >= 16:
        digit = num % 16
        hexStr = chaDic.get(digit, str(digit)) + hexStr
        num //= 16
    hexStr = chaDic.get(num, str(num)) + hexStr
    return hexStr



class HkCaptureVedio:
    def __init__(self, dev_num: int = 1, exposure_time: float = 5000, gain: float = 10, size=None, Binning=0,balancewiteauto=1):
        self.cam = MvCamera()
        self.deviceList = MV_CC_DEVICE_INFO_LIST()
        self.n_layer_type = MV_GIGE_DEVICE | MV_USB_DEVICE
        self.dev_num = dev_num
        # self.exposure_time = exposure_time
        # self.gain = gain
        # self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        # self.set_exposure(exposure_time)
        # self.set_gain(gain)
        # self.Set_parameter(exposure_time, gain)
        self.size = size
        self.Binning = Binning
        self.is_open = False

        self.data_size = None
        self.data = None

        if self.initialize_camera():
            print("打开相机成功")
            self.Set_parameter(exposure_time, gain,Binning,balancewiteauto)


    def set_exposure(self, exposure_time):
        """设置曝光时间"""
        ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", exposure_time)
        if ret != 0:
            print(f"Set exposure time failed! Error: {ToHexStr(ret)}")
        else:
            print(f"Exposure time set to {exposure_time} μs")

    def set_gain(self, gain):
        """设置增益"""
        ret = self.cam.MV_CC_SetEnumValue("GainAuto", 0)
        ret = self.cam.MV_CC_SetFloatValue("Gain", gain)
        if ret != 0:
            print(f"Set gain failed! Error: {ToHexStr(ret)}")
        else:
            print(f"Gain set to {gain} dB")
    def Set_parameter(self, exposureTime, gain,binning,balancewiteauto):
        if  '' == exposureTime or '' == gain:
            print('show info', 'please type in the text box !')
            return MV_E_PARAMETER
        if self.is_open:
            ret = self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposureTime))
            if ret != 0:
                print('show error', 'set exposure time fail! ret = ' + ToHexStr(ret))
                return ret

            ret = self.cam.MV_CC_SetFloatValue("Gain", float(gain))
            if ret != 0:
                print('show error', 'set gain fail! ret = ' + ToHexStr(ret))
                return ret
            ret = self.cam.MV_CC_SetEnumValue("DecimationHorizontal", binning)
            if ret != 0:
                print('show error', 'set binning fail! ret = ' + ToHexStr(ret))
                return ret
            ret = self.cam.MV_CC_SetEnumValue("BalanceWhiteAuto", balancewiteauto)
            #ret = self.cam.MV_CC_SetEnumValue("BalanceRatioSelector", 0)
            #ret = self.cam.MV_CC_SetIntValue("BalanceRatio", 458)
            if ret != 0:
                print('show error', 'set white fail! ret = ' + ToHexStr(ret))
                return ret
            # ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frameRate))
            # if ret != 0:
            #     print('show error', 'set acquistion frame rate fail! ret = ' + To_hex_str(ret))
            #     return ret

            print('show info', 'set parameter success!')

            return MV_OK

    def initialize_camera(self):
        """初始化相机"""
        # 初始化SDK
        MvCamera.MV_CC_Initialize()

        # 枚举设备
        ret = self.cam.MV_CC_EnumDevices(self.n_layer_type, self.deviceList)
        if ret != 0 or self.deviceList.nDeviceNum == 0:
            print("No devices found! Error: ", ToHexStr(ret))
            return False

        print(f"Find {self.deviceList.nDeviceNum} device(s).")

        # 初始化相机对象
        device_info = cast(self.deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents  # 默认使用第一个设备
        ret = self.cam.MV_CC_CreateHandle(device_info)
        if ret != 0:
            print("Create handle failed! Error: ", ToHexStr(ret))
            return False

        # 打开设备
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("Open device failed! Error: ", ToHexStr(ret))
            return False

        # 设置像素格式为 RGB8
        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BGR8_Packed)
        if ret != 0:
            print("Set pixel format failed! Error: ", ToHexStr(ret))
            return False

        # 设置触发模式为连续
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        # ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_SOURCE_SOFTWARE)
        if ret != 0:
            print("Set trigger mode failed! Error: ", ToHexStr(ret))
            return False

        self.is_open = True

        # cam.Set_parameter(5000, 14)
        return True

    def start_grabbing(self):
        """开始取流并实时显示画面"""
        if not self.is_open:
            print("Camera is not initialized.")
            return

        # 开始采集
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("Start grabbing failed! Error: ", ToHexStr(ret))
            return

        # 分配缓存
        self.data_size = MV_FRAME_OUT_INFO_EX()
        self.data = (c_ubyte * (1024 * 1024 * 5))()  # 假定最大帧大小为5MB

        print("Camera is running. Press 'q' to quit. Press '3' to save image.")

    def read(self):
        ret = self.cam.MV_CC_GetOneFrameTimeout(self.data, len(self.data), self.data_size, 1000)
        if ret == 0:
            frame = np.frombuffer(self.data, dtype=np.uint8, count=self.data_size.nFrameLen)
            frame = frame.reshape((self.data_size.nHeight, self.data_size.nWidth, 3))  # 转换为图像格式
            return True, frame
        else:
            print("Failed to get frame! Error: ", ToHexStr(ret))
            return False, None


    def delete_came(self):
        # 释放资源
        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        MvCamera.MV_CC_Finalize()


if __name__ == "__main__":
    cam = HkCaptureVedio(exposure_time=5000,gain=16)
    cam.start_grabbing()
    time_start = time.time()
    cnt = 0
    while True:
        start = time.time()
        ret, frame = cam.read()
        if not ret:
            print(111)
            break
        cv2.imshow("111",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cnt+=1
        if cnt == 20:  # Calculate FPS every 20 frames
            time_end = time.time()
            fps = cnt / (time_end - time_start)
            print(f"FPS: {fps:.2f}")
            time_start = time.time()
            cnt = 0

