"""
大恒相机拍摄
"""
# version:1.0.1905.9051
import gxipy as gx
from PIL import Image
import cv2
import time
import traceback


def main():
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    print(dev_num)
    print(dev_info_list)
    cam = device_manager.open_device_by_index(1)

    # exit when the camera is a mono camera
    if cam.PixelColorFilter.is_implemented() is False:
        cam.close_device()
        return

    # set continuous acquisition
    # cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

    # set exposure
    cam.ExposureTime.set(8000.0)

    # set gain
    cam.Gain.set(16.0)

    data_stream = cam.data_stream[0]
    data_stream.StreamBufferHandlingMode.set(3)

    # get param of improving image quality
    '''
    if cam.GammaParam.is_readable():
        gamma_value = cam.GammaParam.get()
        gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
    else:
        gamma_lut = None
    if cam.ContrastParam.is_readable():
        contrast_value = cam.ContrastParam.get()
        contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
    else:
        contrast_lut = None
    if cam.ColorCorrectionParam.is_readable():
        color_correction_param = cam.ColorCorrectionParam.get()
    else:
        color_correction_param = 0
    #'''
    # start data acquisition
    cam.stream_on()

    t_s = time.time()
    p = 0
    while True:
        # get raw image
        raw_image = data_stream.get_image()
        # if raw_image is None:
        #    print("Getting image failed.")
        #    continue

        # get RGB image from raw image
        rgb_image = raw_image.convert("RGB")
        # if rgb_image is None:
        #    continue

        # improve image quality
        # rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)

        # create numpy array with data from raw image
        numpy_image = rgb_image.get_numpy_array()[..., ::-1]
        # numpy_image = cv2.blur(numpy_image,(3,3))
        # if numpy_image is None:
        #    continue

        # show acquired image

        if p == 1000:
            print((time.time() - t_s), 1000 / (time.time() - t_s))
            t_s = time.time()
            p = 0
        cv2.imshow("image", numpy_image)
        if cv2.waitKey(1) == ord('q'):
            break
        p += 1

    cv2.destroyAllWindows()
    cam.stream_off()
    cam.close_device()


class GxVideoCupture():
    def __init__(self, dev_num: int = 1, exposure_time: float = 5000, gain: float = 8, size=None, Binning=None,
                 BinningMode=[0, 0]):
        self.destoryed = True
        self.device_manager = gx.DeviceManager()
        self.cam = self.device_manager.open_device_by_index(dev_num)
        self.cam.ExposureTime.set(exposure_time)
        self.cam.Gain.set(gain)
        if size:
            self.cam.Width.set(size[0])
            self.cam.Height.set(size[1])
        if Binning:
            self.cam.BinningHorizontal.set(Binning[0])
            self.cam.BinningVertical.set(Binning[1])
        if BinningMode:
            self.cam.BinningHorizontalMode.set(BinningMode[0])
            self.cam.BinningVerticalMode.set(BinningMode[1])
        self.data_stream = self.cam.data_stream[0]
        self.data_stream.StreamBufferHandlingMode.set(3)

        self.cam.stream_on()

        self.destoryed = False
        print('GxVideoCupture start')

    def read(self):
        try:
            raw_image = self.data_stream.get_image()
            rgb_image = raw_image.convert("RGB")
            numpy_image = rgb_image.get_numpy_array()[..., ::-1]
            return True, numpy_image
        except:
            print(traceback.format_exc())
            return False, None

    def release(self):
        self.__del__()

    def __del__(self):
        if not self.destoryed:
            self.destoryed = True
            self.cam.stream_off()
            self.cam.close_device()
            print('GxVideoCupture close')

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()


def demo():
    cap = GxVideoCupture()
    for i in range(100):
        _, frame = cap.read()
        cv2.imshow("img", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
    # demo()
