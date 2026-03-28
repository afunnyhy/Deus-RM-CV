from ctypes import *
from .CameraParams_const import *

STRING = c_char_p

MV_PointCloudFile_Undefined = 0  # 未定义的点云格式
MV_EXPOSURE_MODE_TRIGGER_WIDTH = 1  # 曝光模式外触发脉宽
MV_TRIGGER_SOURCE_FrequencyConverter = 8  # 触发源变频器
MV_TRIGGER_SOURCE_SOFTWARE = 7  # 软触发
MV_TRIGGER_SOURCE_COUNTER0 = 4  # 触发源计数器
MV_BALANCEWHITE_AUTO_OFF = 0  # 白平衡自动关闭
MV_TRIGGER_SOURCE_LINE3 = 3  # LINE3 触发源
MV_TRIGGER_SOURCE_LINE2 = 2  # LINE2 触发源
MV_TRIGGER_SOURCE_LINE1 = 1  # LINE1 触发源
MV_EXPOSURE_MODE_TIMED = 0  # 曝光超时模式
MV_TRIGGER_SOURCE_LINE0 = 0  # LINE0 触发源
AM_NA = 1  # 不可用
V_Undefined = 99  # 未定义
V_Invisible = 3  # 不可见
V_Guru = 2  # 大师可见
V_Expert = 1  # 专家可见
MV_GAIN_MODE_CONTINUOUS = 2  # 连续
IFT_IString = 6  # IString接口类型
MV_ACQ_MODE_CONTINUOUS = 2  # 连续采集模式
MV_EXPOSURE_AUTO_MODE_ONCE = 1  # 单次自动曝光模式
MV_Image_Png = 3  # Png格式
MV_Image_Jpeg = 2  # Jpeg格式
MV_Image_Bmp = 1  # Bmp格式
MV_ACQ_MODE_MUTLI = 1  # 多帧模式
MV_EXPOSURE_AUTO_MODE_OFF = 0  # 关闭自动曝光模式
IFT_IPort = 11  # IPort接口类型
IFT_IEnumEntry = 10  # IEnumEntry接口类型
IFT_ICategory = 8  # ICategory接口类型
IFT_IRegister = 7  # IRegister接口类型
IFT_IFloat = 5  # IFloat接口类型
IFT_IEnumeration = 9  # IEnumeration接口类型
IFT_ICommand = 4  # ICommand接口类型
IFT_IBoolean = 3  # IBoolean接口类型
IFT_IInteger = 2  # IInteger接口类型
IFT_IBase = 1  # IBase接口类型
IFT_IValue = 0  # IValue接口类型
MV_TRIGGER_MODE_OFF = 0  # 关闭
MV_TRIGGER_MODE_ON = 1  # 开
MV_GIGE_TRANSTYPE_UNICAST = 0  # 表示单播(默认)
MV_GAIN_MODE_ONCE = 1  # 单次
MV_EXPOSURE_AUTO_MODE_CONTINUOUS = 2  # 自动连续曝光模式
MV_Image_Tif = 4  # Tif格式
AM_RW = 4  # 可读可写
MV_BALANCEWHITE_AUTO_CONTINUOUS = 1  # 白平衡自动连续
MV_BALANCEWHITE_AUTO_ONCE = 2  # 单次自动白平衡
MV_GAMMA_SELECTOR_SRGB = 2  # gamma选择器SRGB
MV_GAIN_MODE_OFF = 0  # 关闭增益模式
MV_GAMMA_SELECTOR_USER = 1  # gamma选择器User
MV_GrabStrategy_UpcomingImage = 3  # 等待下一帧图像
MV_GrabStrategy_LatestImages = 2  # 获取列表中最新的图像
MV_GrabStrategy_LatestImagesOnly = 1  # 获取列表中最新的一帧图像同时抛弃列表中的其它图像
MV_GrabStrategy_OneByOne = 0  # 从旧到新一帧一帧的获取图像
MV_PointCloudFile_OBJ = 3  # OBJ点云格式
MV_PointCloudFile_CSV = 2  # CSV点云格式
MV_PointCloudFile_PLY = 1  # PLY点云格式
MV_ACQ_MODE_SINGLE = 0  # 单帧模式
AM_CycleDetect = 6  # 内部用于AccessMode循环检测
MV_GIGE_TRANSTYPE_MULTICAST_WITHOUT_RECV = 65537  # 表示组播模式，但本实例不接收图像数据
AM_Undefined = 5  # 对象未定义初始化
MV_GIGE_TRANSTYPE_UNICAST_WITHOUT_RECV = 65536  # 表示单播服务端模式，但本实例不接收图像数据
MV_GIGE_TRANSTYPE_UNICAST_DEFINED_PORT = 5  # 表示用户自定义应用端接收图像数据Port号
AM_RO = 3  # 只读
MV_GIGE_TRANSTYPE_SUBNETBROADCAST = 3  # 表示局域网广播数据暂不支持
MV_GIGE_TRANSTYPE_LIMITEDBROADCAST = 2  # 表示受限局域网广播数据暂不支持
MV_GIGE_TRANSTYPE_MULTICAST = 1  # 表示组播
MV_GIGE_TRANSTYPE_CAMERADEFINED = 4  # 表示由相机读取数据暂不支持
AM_NI = 0  # 没有实现
MV_Image_Undefined = 0  # 未定义的图像类型
AM_WO = 2  # 只写
MV_FormatType_AVI = 1  # AVI视频格式
MV_FormatType_Undefined = 0  # 未定义的格式类型
SortMethod_SerialNumber = 0  # 按序列号排序
SortMethod_UserID = 1  # 按用户自定义名称排序
SortMethod_CurrentIP_ASC = 2  # 按当前IP地址排序（升序）
SortMethod_CurrentIP_DESC = 3  # 按当前IP地址排序（降序）
MV_IMAGE_ROTATE_90 = 1  # 旋转90度
MV_IMAGE_ROTATE_180 = 2  # 旋转180度
MV_IMAGE_ROTATE_270 = 3  # 旋转270度
MV_FLIP_VERTICAL = 1  # 垂直翻转
MV_FLIP_HORIZONTAL = 2  # 水平翻转
MV_CC_GAMMA_TYPE_NONE = 0  # 不启用
MV_CC_GAMMA_TYPE_VALUE = 1  # Gamma值
MV_CC_GAMMA_TYPE_USER_CURVE = 2  # Gamma曲线
MV_CC_GAMMA_TYPE_LRGB2SRGB = 3  # linear RGB to sRGB
MV_CC_GAMMA_TYPE_SRGB2LRGB = 4  # sRGB to linear RGB(彩色插值时支持，色彩校正时无效)
MV_CC_STREAM_EXCEPTION_ABNORMAL_IMAGE = 0x4001  # 异常的图像，该帧舍弃
MV_CC_STREAM_EXCEPTION_LIST_OVERFLOW = 0x4002  # 缓存列表溢出，清除最旧的一帧
MV_CC_STREAM_EXCEPTION_LIST_EMPTY = 0x4003  # 缓存列表为空，该帧舍弃
MV_CC_STREAM_EXCEPTION_RECONNECTION = 0x4004  # 断线恢复
MV_CC_STREAM_EXCEPTION_DISCONNECTED = 0x4005  # 断开,恢复失败,取流被终止
MV_CC_STREAM_EXCEPTION_DEVICE = 0x4006  # 设备异常,取流被终止
MV_SPLIT_BY_LINE = 1  # 源图像按行拆分成多张图像
V_Beginner = 0  # 基础可见
int8_t = c_int8
int16_t = c_int16
int32_t = c_int32
int64_t = c_int64
uint8_t = c_uint8
uint16_t = c_uint16
uint32_t = c_uint32
uint64_t = c_uint64
int_least8_t = c_byte
int_least16_t = c_short
int_least32_t = c_int
int_least64_t = c_long
uint_least8_t = c_ubyte
uint_least16_t = c_ushort
uint_least32_t = c_uint
uint_least64_t = c_ulong
int_fast8_t = c_byte
int_fast16_t = c_long
int_fast32_t = c_long
int_fast64_t = c_long
uint_fast8_t = c_ubyte
uint_fast16_t = c_ulong
uint_fast32_t = c_ulong
uint_fast64_t = c_ulong
intptr_t = c_long
uintptr_t = c_ulong
intmax_t = c_long
uintmax_t = c_ulong

# values for enumeration 'MvGvspPixelType'
MvGvspPixelType = int64_t  # enum


class _MV_GIGE_DEVICE_INFO_(Structure):
    pass


_MV_GIGE_DEVICE_INFO_._fields_ = [
    ('nIpCfgOption', c_uint),  # IP配置选项
    ('nIpCfgCurrent', c_uint),  # 当前IP地址配置:bit31-static bit30-dhcp bit29-lla
    ('nCurrentIp', c_uint),  # 当前主机IP地址
    ('nCurrentSubNetMask', c_uint),  # 当前子网掩码
    ('nDefultGateWay', c_uint),  # 默认网关
    ('chManufacturerName', c_ubyte * 32),  # 厂商名称
    ('chModelName', c_ubyte * 32),  # 型号名称
    ('chDeviceVersion', c_ubyte * 32),  # 设备固件版本
    ('chManufacturerSpecificInfo', c_ubyte * 48),  # 厂商特定信息
    ('chSerialNumber', c_ubyte * 16),  # 序列号
    ('chUserDefinedName', c_ubyte * 16),  # 用户自定义名称
    ('nNetExport', c_uint),  # 网口Ip地址
    ('nReserved', c_uint * 4),  # 保留字节
]
MV_GIGE_DEVICE_INFO = _MV_GIGE_DEVICE_INFO_


class _MV_USB3_DEVICE_INFO_(Structure):
    pass


_MV_USB3_DEVICE_INFO_._fields_ = [
    ('CrtlInEndPoint', c_ubyte),  # 控制输入端点
    ('CrtlOutEndPoint', c_ubyte),  # 控制输出端点
    ('StreamEndPoint', c_ubyte),  # 流端点
    ('EventEndPoint', c_ubyte),  # 事件端点
    ('idVendor', c_ushort),  # 供应商ID号
    ('idProduct', c_ushort),  # 产品ID号
    ('nDeviceNumber', c_uint),  # 设备序列号
    ('chDeviceGUID', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 设备GUID号
    ('chVendorName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 供应商名称
    ('chModelName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 型号名称
    ('chFamilyName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 系列名称
    ('chDeviceVersion', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 设备版本号
    ('chManufacturerName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 厂商名称
    ('chSerialNumber', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 序列号
    ('chUserDefinedName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 用户自定义名称
    ('nbcdUSB', c_uint),  # 支持的USB协议
    ('nDeviceAddress', c_uint),  # 设备地址
    ('nReserved', c_uint * 2),  # 保留字节
]
MV_USB3_DEVICE_INFO = _MV_USB3_DEVICE_INFO_


# CameraLink设备信息
class _MV_CamL_DEV_INFO_(Structure):
    pass


_MV_CamL_DEV_INFO_._fields_ = [
    ('chPortID', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 端口号
    ('chModelName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 设备型号
    ('chFamilyName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 系列名称
    ('chDeviceVersion', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 设备版本号
    ('chManufacturerName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 供应商名称
    ('chSerialNumber', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 序列号
    ('nReserved', c_uint * 38),  # 保留字节
]
MV_CamL_DEV_INFO = _MV_CamL_DEV_INFO_


# 采集卡Camera Link设备信息
class _MV_CML_DEVICE_INFO_(Structure):
    pass


_MV_CML_DEVICE_INFO_._fields_ = [
    ('chInterfaceID', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 采集卡ID
    ('chVendorName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 供应商名称
    ('chModelName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 型号名称
    ('chManufacturerInfo', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 制造信息
    ('chDeviceVersion', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 设备版本
    ('chSerialNumber', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 序列号
    ('chUserDefinedName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 用户自定义名称
    ('chDeviceID', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 设备ID
    ('nReserved', c_uint * 7),  # 保留字节
]
MV_CML_DEVICE_INFO = _MV_CML_DEVICE_INFO_


# CoaXPress设备信息
class _MV_CXP_DEVICE_INFO_(Structure):
    pass


_MV_CXP_DEVICE_INFO_._fields_ = [
    ('chInterfaceID', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 采集卡ID
    ('chVendorName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 供应商名称
    ('chModelName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 型号名称
    ('chManufacturerInfo', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 制造信息
    ('chDeviceVersion', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 设备版本
    ('chSerialNumber', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 序列号
    ('chUserDefinedName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 用户自定义名称
    ('chDeviceID', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 设备ID
    ('nReserved', c_uint * 7),  # 保留字节
]
MV_CXP_DEVICE_INFO = _MV_CXP_DEVICE_INFO_


# XoFLink设备信息
class _MV_XOF_DEVICE_INFO_(Structure):
    pass


_MV_XOF_DEVICE_INFO_._fields_ = [
    ('chInterfaceID', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 采集卡ID
    ('chVendorName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 供应商名称
    ('chModelName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 型号名称
    ('chManufacturerInfo', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 制造信息
    ('chDeviceVersion', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 设备版本
    ('chSerialNumber', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 序列号
    ('chUserDefinedName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 用户自定义名称
    ('chDeviceID', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 设备ID
    ('nReserved', c_uint * 7),  # 保留字节
]
MV_XOF_DEVICE_INFO = _MV_XOF_DEVICE_INFO_


class _MV_CC_DEVICE_INFO_(Structure):
    pass


class N19_MV_CC_DEVICE_INFO_3DOT_0E(Union):
    pass


N19_MV_CC_DEVICE_INFO_3DOT_0E._fields_ = [
    ('stGigEInfo', MV_GIGE_DEVICE_INFO),  # Gige设备信息
    ('stUsb3VInfo', MV_USB3_DEVICE_INFO),  # U3V设备信息
    ('stCamLInfo', MV_CamL_DEV_INFO),  # CamLink设备信息
    ('stCMLInfo', MV_CML_DEVICE_INFO),  # 采集卡CameraLink设备信息
    ('stCXPInfo', MV_CXP_DEVICE_INFO),  # 采集卡CoaXPress设备信息
    ('stXoFInfo', MV_XOF_DEVICE_INFO),  # 采集卡XoF设备信息
]
_MV_CC_DEVICE_INFO_._fields_ = [
    ('nMajorVer', c_ushort),  # 规范的主要版本
    ('nMinorVer', c_ushort),  # 规范的次要版本
    ('nMacAddrHigh', c_uint),  # MAC地址高位
    ('nMacAddrLow', c_uint),  # MAC地址低位
    ('nTLayerType', c_uint),  # 设备传输层协议类型
    ('nDevTypeInfo', c_uint),  # 设备类型信息
    ('nReserved', c_uint * 3),  # 保留字节
    ('SpecialInfo', N19_MV_CC_DEVICE_INFO_3DOT_0E),  # 不同设备特有信息
]
MV_CC_DEVICE_INFO = _MV_CC_DEVICE_INFO_


# 网络传输层信息
class _MV_NETTRANS_INFO_(Structure):
    pass


_MV_NETTRANS_INFO_._fields_ = [
    ('nReceiveDataSize', int64_t),  # 已接收数据大小 [统计StartGrabbing与StopGrabbing之间的数据量]
    ('nThrowFrameCount', c_int),  # 丢帧数量
    ('nNetRecvFrameCount', c_uint),  # 收到帧计数
    ('nRequestResendPacketCount', int64_t),  # 请求重发包数
    ('nResendPacketCount', int64_t),  # 重发包数
]
MV_NETTRANS_INFO = _MV_NETTRANS_INFO_


class _MV_CC_DEVICE_INFO_LIST_(Structure):
    pass


_MV_CC_DEVICE_INFO_LIST_._fields_ = [
    ('nDeviceNum', c_uint),  # 在线设备数量
    ('pDeviceInfo', POINTER(MV_CC_DEVICE_INFO) * MV_MAX_DEVICE_NUM),  # 支持最多256个设备
]
MV_CC_DEVICE_INFO_LIST = _MV_CC_DEVICE_INFO_LIST_


# 通过GenTL枚举到的Interface信息
class _MV_GENTL_IF_INFO_(Structure):
    pass


_MV_GENTL_IF_INFO_._fields_ = [
    ('chInterfaceID', c_ubyte * INFO_MAX_BUFFER_SIZE),  # GenTL接口ID
    ('chTLType', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 传输层类型
    ('chDisplayName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 设备显示名称
    ('nCtiIndex', c_uint),  # GenTL的cti文件索引
    ('nReserved', c_uint * 8),  # 保留字节
]
MV_GENTL_IF_INFO = _MV_GENTL_IF_INFO_


# 通过GenTL枚举到的设备信息列表
class _MV_GENTL_IF_INFO_LIST_(Structure):
    pass


_MV_GENTL_IF_INFO_LIST_._fields_ = [
    ('nInterfaceNum', c_uint),  # 在线设备数量
    ('pIFInfo', POINTER(MV_GENTL_IF_INFO) * MV_MAX_GENTL_IF_NUM),  # 支持最多256个设备
]
MV_GENTL_IF_INFO_LIST = _MV_GENTL_IF_INFO_LIST_


# 通过GenTL枚举到的设备信息
class _MV_GENTL_DEV_INFO_(Structure):
    pass


_MV_GENTL_DEV_INFO_._fields_ = [
    ('chInterfaceID', c_ubyte * INFO_MAX_BUFFER_SIZE),  # GenTL接口ID
    ('chDeviceID', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 设备ID
    ('chVendorName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 供应商名称
    ('chModelName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 型号名称
    ('chTLType', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 传输层类型
    ('chDisplayName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 显示名称
    ('chUserDefinedName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 用户自定义名称
    ('chSerialNumber', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 序列号
    ('chDeviceVersion', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 设备版本号
    ('nCtiIndex', c_uint),  # cti索引
    ('nReserved', c_uint * 8),  # 保留字节
]
MV_GENTL_DEV_INFO = _MV_GENTL_DEV_INFO_


# 通过GenTL枚举到的设备信息列表
class _MV_GENTL_DEV_INFO_LIST_(Structure):
    pass


_MV_GENTL_DEV_INFO_LIST_._fields_ = [
    ('nDeviceNum', c_uint),  # 在线设备数量
    ('pDeviceInfo', POINTER(MV_GENTL_DEV_INFO) * MV_MAX_GENTL_DEV_NUM),  # GenTL设备信息
]
MV_GENTL_DEV_INFO_LIST = _MV_GENTL_DEV_INFO_LIST_


# Chunk内容
class _MV_CHUNK_DATA_CONTENT_(Structure):
    pass


_MV_CHUNK_DATA_CONTENT_._fields_ = [
    ('pChunkData', POINTER(c_ubyte)),  # 数据内容
    ('nChunkID', c_uint),  # 数据块ID
    ('nChunkLen', c_uint),  # 数据块长度
    ('nReserved', c_uint * 8),  # 保留字节
]
MV_CHUNK_DATA_CONTENT = _MV_CHUNK_DATA_CONTENT_


class _MV_FRAME_OUT_INFO_(Structure):
    pass


_MV_FRAME_OUT_INFO_._fields_ = [
    ('nWidth', c_ushort),  # 图像宽
    ('nHeight', c_ushort),  # 图像高
    ('enPixelType', MvGvspPixelType),  # 像素格式
    ('nFrameNum', c_uint),  # 帧号
    ('nDevTimeStampHigh', c_uint),  # 时间戳高32位
    ('nDevTimeStampLow', c_uint),  # 时间戳低32位
    ('nReserved0', c_uint),  # 保留，8字节对齐
    ('nHostTimeStamp', int64_t),  # 主机生成的时间戳
    ('nFrameLen', c_uint),  # 帧的长度
    ('nLostPacket', c_uint),  # 丢包数量
    ('nReserved', c_uint * 2),  # 保留字节
]
MV_FRAME_OUT_INFO = _MV_FRAME_OUT_INFO_


# CameraParams.h 129
class _MV_FRAME_OUT_INFO_EX_(Structure):
    pass


class N22_MV_FRAME_OUT_INFO_EX_3DOT_1E(Union):
    pass


N22_MV_FRAME_OUT_INFO_EX_3DOT_1E._fields_ = [
    ('pUnparsedChunkContent', POINTER(MV_CHUNK_DATA_CONTENT)),  # Chunk内容
    ('nAligning', int64_t),  # 校准字段
]
_MV_FRAME_OUT_INFO_EX_._fields_ = [
    ('nWidth', c_ushort),  # 图像宽(超过65535时请使用nExtendWidth)
    ('nHeight', c_ushort),  # 图像高(超过65535时请使用nExtendHeight)
    ('enPixelType', MvGvspPixelType),  # 像素格式
    ('nFrameNum', c_uint),  # 帧号
    ('nDevTimeStampHigh', c_uint),  # 时间戳高32位
    ('nDevTimeStampLow', c_uint),  # 时间戳低32位
    ('nReserved0', c_uint),  # 保留，8字节对齐
    ('nHostTimeStamp', int64_t),  # 主机生成的时间戳
    ('nFrameLen', c_uint),  # 帧的长度
    # 以下为chunk和帧水印信息
    ('nSecondCount', c_uint),  # 秒数
    ('nCycleCount', c_uint),  # 循环计数
    ('nCycleOffset', c_uint),  # 循环偏移量
    ('fGain', c_float),  # 增益
    ('fExposureTime', c_float),  # 曝光时间
    ('nAverageBrightness', c_uint),  # 平均亮度
    # 白平衡
    ('nRed', c_uint),  # 红色
    ('nGreen', c_uint),  # 绿色
    ('nBlue', c_uint),  # 蓝色
    ('nFrameCounter', c_uint),  # 帧计数器
    ('nTriggerIndex', c_uint),  # 触发索引
    # 输入/输出
    ('nInput', c_uint),  # 输入
    ('nOutput', c_uint),  # 输出
    # ROI区域
    ('nOffsetX', c_ushort),  # 水平偏移量
    ('nOffsetY', c_ushort),  # 垂直偏移量
    ('nChunkWidth', c_ushort),  # chunk 宽
    ('nChunkHeight', c_ushort),  # chunk 高
    ('nLostPacket', c_uint),  # 本帧丢包数量
    ('nUnparsedChunkNum', c_uint),  # 未解析的Chunkdata数量
    ('UnparsedChunkList', N22_MV_FRAME_OUT_INFO_EX_3DOT_1E),  # 数据块列表
    ('nExtendWidth', c_uint),  # 图像宽(扩展字段)
    ('nExtendHeight', c_uint),  # 图像高(扩展字段)
    ('nReserved', c_uint * 34),  # 保留字节
]
MV_FRAME_OUT_INFO_EX = _MV_FRAME_OUT_INFO_EX_


class _MV_DISPLAY_FRAME_INFO_(Structure):
    pass


_MV_DISPLAY_FRAME_INFO_._fields_ = [
    ('hWnd', c_void_p),  # 窗口句柄
    ('pData', POINTER(c_ubyte)),  # 显示数据缓存
    ('nDataLen', c_uint),  # 数据长度
    ('nWidth', c_ushort),  # 图像宽
    ('nHeight', c_ushort),  # 图像高
    ('enPixelType', MvGvspPixelType),  # 像素格式
    ('enRenderMode', c_uint),  # 图像渲染模式 0-默认模式(Windows GDI/Linux OPENGL), 1-D3D模式(Windows有效)
    ('nRes', c_uint * 3),  # 保留字节
]
MV_DISPLAY_FRAME_INFO = _MV_DISPLAY_FRAME_INFO_


# 显示帧信息
class _MV_DISPLAY_FRAME_INFO_EX_(Structure):
    pass


_MV_DISPLAY_FRAME_INFO_EX_._fields_ = [
    ('nWidth', c_uint),  # 图像宽
    ('nHeight', c_uint),  # 图像高
    ('enPixelType', MvGvspPixelType),  # 像素格式
    ('pImageBuf', POINTER(c_ubyte)),  # 输入图像缓存
    ('nImageBufLen', c_uint),  # 输入图像长度
    ('enRenderMode', c_uint),  # 图像渲染模式 0-默认模式(Windows GDI/Linux OPENGL), 1-D3D模式(Windows有效)
    ('nRes', c_uint * 3),  # 保留字节
]
MV_DISPLAY_FRAME_INFO_EX = _MV_DISPLAY_FRAME_INFO_EX_


# 图像结构体，输出图像指针和特定图像信息
class _MV_FRAME_OUT_(Structure):
    pass


_MV_FRAME_OUT_._fields_ = [
    ('pBufAddr', POINTER(c_ubyte)),  # 图像指针地址
    ('stFrameInfo', MV_FRAME_OUT_INFO_EX),  # 图像信息
    ('nRes', c_uint * 16),  # 保留字节
]
MV_FRAME_OUT = _MV_FRAME_OUT_

# 取流策略枚举
_MV_GRAB_STRATEGY_ = c_int  # enum
MV_GRAB_STRATEGY = _MV_GRAB_STRATEGY_

# 保存点云文件类型枚举
MV_SAVE_POINT_CLOUD_FILE_TYPE = c_int  # enum


# 保存3D数据到缓存
class _MV_SAVE_POINT_CLOUD_PARAM_(Structure):
    pass


_MV_SAVE_POINT_CLOUD_PARAM_._fields_ = [
    ('nLinePntNum', c_uint),  # 每一行的点数，即图像宽
    ('nLineNum', c_uint),  # 行数，即图像高
    ('enSrcPixelType', MvGvspPixelType),  # 输入数据的像素格式
    ('pSrcData', POINTER(c_ubyte)),  # 输入数据缓存
    ('nSrcDataLen', c_uint),  # 输入数据大小
    ('pDstBuf', POINTER(c_ubyte)),  # 输出像素数据缓存
    ('nDstBufSize', c_uint),  # 提供的输出缓存大小(nLinePntNum * nLineNum * (16*3 + 4) + 2048)
    ('nDstBufLen', c_uint),  # 输出像素数据缓存长度
    ('enPointCloudFileType', MV_SAVE_POINT_CLOUD_FILE_TYPE),  # 提供的输出点云文件类型
    ('nReserved', c_uint * 8),  # 保留字节
]
MV_SAVE_POINT_CLOUD_PARAM = _MV_SAVE_POINT_CLOUD_PARAM_

# 保存图像类型枚举
MV_SAVE_IAMGE_TYPE = c_int  # enum


class _MV_SAVE_IMAGE_PARAM_T_(Structure):
    pass


_MV_SAVE_IMAGE_PARAM_T_._fields_ = [
    ('pData', POINTER(c_ubyte)),  # 输入数据缓存
    ('nDataLen', c_uint),  # 输入数据大小
    ('enPixelType', MvGvspPixelType),  # 输入数据像素格式
    ('nWidth', c_ushort),  # 图像宽
    ('nHeight', c_ushort),  # 图像高
    ('pImageBuffer', POINTER(c_ubyte)),  # 输出图片缓存
    ('nImageLen', c_uint),  # 输出图片大小
    ('nBufferSize', c_uint),  # 提供的输出缓存大小
    ('enImageType', MV_SAVE_IAMGE_TYPE),  # 输出图片格式
]
MV_SAVE_IMAGE_PARAM = _MV_SAVE_IMAGE_PARAM_T_


# 图片保存参数
class _MV_SAVE_IMAGE_PARAM_T_EX_(Structure):
    pass


_MV_SAVE_IMAGE_PARAM_T_EX_._fields_ = [
    ('pData', POINTER(c_ubyte)),  # 输入数据缓存
    ('nDataLen', c_uint),  # 输入数据大小
    ('enPixelType', MvGvspPixelType),  # 输入数据像素格式
    ('nWidth', c_ushort),  # 图像宽
    ('nHeight', c_ushort),  # 图像高
    ('pImageBuffer', POINTER(c_ubyte)),  # 输出图片缓存
    ('nImageLen', c_uint),  # 输出图片大小
    ('nBufferSize', c_uint),  # 提供的输出缓存大小
    ('enImageType', MV_SAVE_IAMGE_TYPE),  # 输出图片格式
    ('nJpgQuality', c_uint),  # 编码质量, (50-99]
    # Bayer格式转为RGB24的插值方法: 0-快速 1-均衡 2-最优 3-最优+
    ('iMethodValue', c_uint),
    ('nReserved', c_uint * 3),  # 保留字节
]
MV_SAVE_IMAGE_PARAM_EX = _MV_SAVE_IMAGE_PARAM_T_EX_


class _MV_SAVE_IMAGE_PARAM_EX3_(Structure):
    pass


_MV_SAVE_IMAGE_PARAM_EX3_._fields_ = [
    ('pData', POINTER(c_ubyte)),  # 输入数据缓存
    ('nDataLen', c_uint),  # 输入数据大小
    ('enPixelType', MvGvspPixelType),  # 输入数据像素格式
    ('nWidth', c_uint),  # 图像宽
    ('nHeight', c_uint),  # 图像高
    ('pImageBuffer', POINTER(c_ubyte)),  # 输出图片缓存
    ('nImageLen', c_uint),  # 输出图片大小
    ('nBufferSize', c_uint),  # 提供的输出缓存大小
    ('enImageType', MV_SAVE_IAMGE_TYPE),  # 输出图片格式
    ('nJpgQuality', c_uint),  # 编码质量, (50-99]
    # Bayer格式转为RGB24的插值方法: 0-快速 1-均衡 2-最优 3-最优+
    ('iMethodValue', c_uint),
    ('nReserved', c_uint * 3),  # 保留字节
]
MV_SAVE_IMAGE_PARAM_EX3 = _MV_SAVE_IMAGE_PARAM_EX3_


class _MV_SAVE_IMAGE_TO_FILE_PARAM_EX_(Structure):
    pass


_MV_SAVE_IMAGE_TO_FILE_PARAM_EX_._fields_ = [
    ('nWidth', c_uint),  # 图像宽
    ('nHeight', c_uint),  # 图像高
    ('enPixelType', MvGvspPixelType),  # 输入数据的像素格式
    ('pData', POINTER(c_ubyte)),  # 输入数据缓存
    ('nDataLen', c_uint),  # 输入数据大小
    ('enImageType', MV_SAVE_IAMGE_TYPE),  # 输入图片格式
    ('pcImagePath', POINTER(c_char)),  # 输入文件路径
    ('nQuality', c_uint),  # JPG编码质量(50-99]，对其它格式无效
    # Bayer格式转为RGB24的插值方法: 0-快速 1-均衡 2-最优 3-最优+
    ('iMethodValue', c_int),
    ('nReserved', c_uint * 8),  # 保留字节
]
MV_SAVE_IMAGE_TO_FILE_PARAM_EX = _MV_SAVE_IMAGE_TO_FILE_PARAM_EX_


# 图像转换结构体
class _MV_CC_PIXEL_CONVERT_PARAM_T_(Structure):
    pass


_MV_CC_PIXEL_CONVERT_PARAM_T_._fields_ = [
    ('nWidth', c_ushort),  # 图像宽
    ('nHeight', c_ushort),  # 图像高
    ('enSrcPixelType', MvGvspPixelType),  # 源像素格式
    ('pSrcData', POINTER(c_ubyte)),  # 输入数据缓存
    ('nSrcDataLen', c_uint),  # 输入数据大小
    ('enDstPixelType', MvGvspPixelType),  # 目标像素格式
    ('pDstBuffer', POINTER(c_ubyte)),  # 输出数据缓存
    ('nDstLen', c_uint),  # 输出数据大小
    ('nDstBufferSize', c_uint),  # 提供的输出缓存大小
    ('nRes', c_uint * 4),  # 保留字节
]
MV_CC_PIXEL_CONVERT_PARAM = _MV_CC_PIXEL_CONVERT_PARAM_T_


class _MV_PIXEL_CONVERT_PARAM_EX_T_(Structure):
    pass


_MV_PIXEL_CONVERT_PARAM_EX_T_._fields_ = [
    ('nWidth', c_uint),  # 图像宽
    ('nHeight', c_uint),  # 图像高
    ('enSrcPixelType', MvGvspPixelType),  # 源像素格式
    ('pSrcData', POINTER(c_ubyte)),  # 输入数据缓存
    ('nSrcDataLen', c_uint),  # 输入数据大小
    ('enDstPixelType', MvGvspPixelType),  # 目标像素格式
    ('pDstBuffer', POINTER(c_ubyte)),  # 输出数据缓存
    ('nDstLen', c_uint),  # 输出数据大小
    ('nDstBufferSize', c_uint),  # 提供的输出缓存大小
    ('nRes', c_uint * 4),  # 保留字节
]
MV_CC_PIXEL_CONVERT_PARAM_EX = _MV_PIXEL_CONVERT_PARAM_EX_T_

# 录像格式类型枚举
_MV_RECORD_FORMAT_TYPE_ = c_int  # enum
MV_RECORD_FORMAT_TYPE = _MV_RECORD_FORMAT_TYPE_


# 录像参数
class _MV_CC_RECORD_PARAM_T_(Structure):
    pass


_MV_CC_RECORD_PARAM_T_._fields_ = [
    ('enPixelType', MvGvspPixelType),  # 输入数据的像素格式
    ('nWidth', c_ushort),  # 图像宽(指定目标码流时需为2的倍数)
    ('nHeight', c_ushort),  # 图像高(指定目标码流时需为2的倍数)
    ('fFrameRate', c_float),  # 帧率fps(1/16-120)
    ('nBitRate', c_uint),  # 码率kbps(128kbps-16Mbps)
    ('enRecordFmtType', MV_RECORD_FORMAT_TYPE),  # 录制格式
    ('strFilePath', STRING),  # 录像文件保存路径(若路径中有中文，需转为utf-8)
    ('nRes', c_uint * 8),  # 保留字节
]
MV_CC_RECORD_PARAM = _MV_CC_RECORD_PARAM_T_


# 录像数据
class _MV_CC_INPUT_FRAME_INFO_T_(Structure):
    pass


_MV_CC_INPUT_FRAME_INFO_T_._fields_ = [
    ('pData', POINTER(c_ubyte)),  # 图像数据指针
    ('nDataLen', c_uint),  # 输入数据大小
    ('nRes', c_uint * 8),  # 保留字节
]
MV_CC_INPUT_FRAME_INFO = _MV_CC_INPUT_FRAME_INFO_T_

# 其他枚举类型占位
_MV_CAM_ACQUISITION_MODE_ = c_int
MV_CAM_ACQUISITION_MODE = _MV_CAM_ACQUISITION_MODE_
_MV_CAM_GAIN_MODE_ = c_int
MV_CAM_GAIN_MODE = _MV_CAM_GAIN_MODE_
_MV_CAM_EXPOSURE_MODE_ = c_int
MV_CAM_EXPOSURE_MODE = _MV_CAM_EXPOSURE_MODE_
_MV_CAM_EXPOSURE_AUTO_MODE_ = c_int
MV_CAM_EXPOSURE_AUTO_MODE = _MV_CAM_EXPOSURE_AUTO_MODE_
_MV_CAM_TRIGGER_MODE_ = c_int
MV_CAM_TRIGGER_MODE = _MV_CAM_TRIGGER_MODE_
_MV_CAM_GAMMA_SELECTOR_ = c_int
MV_CAM_GAMMA_SELECTOR = _MV_CAM_GAMMA_SELECTOR_
_MV_CAM_BALANCEWHITE_AUTO_ = c_int
MV_CAM_BALANCEWHITE_AUTO = _MV_CAM_BALANCEWHITE_AUTO_
_MV_CAM_TRIGGER_SOURCE_ = c_int
MV_CAM_TRIGGER_SOURCE = _MV_CAM_TRIGGER_SOURCE_
_MV_GIGE_TRANSMISSION_TYPE_ = c_int
MV_GIGE_TRANSMISSION_TYPE = _MV_GIGE_TRANSMISSION_TYPE_


# 全匹配信息结构体
class _MV_ALL_MATCH_INFO_(Structure):
    pass


_MV_ALL_MATCH_INFO_._fields_ = [
    ('nType', c_uint),  # 需要输出的信息类型
    ('pInfo', c_void_p),  # 输出信息缓存，由调用者分配
    ('nInfoSize', c_uint),  # 信息缓存的大小
]
MV_ALL_MATCH_INFO = _MV_ALL_MATCH_INFO_


# 网络流量和丢包信息反馈结构体，对应类型为 MV_MATCH_TYPE_NET_DETECT
class _MV_MATCH_INFO_NET_DETECT_(Structure):
    pass


_MV_MATCH_INFO_NET_DETECT_._fields_ = [
    ('nReceiveDataSize', int64_t),  # 已接收数据大小
    ('nLostPacketCount', int64_t),  # 丢失的数据包数
    ('nLostFrameCount', c_uint),  # 丢帧数量
    ('nNetRecvFrameCount', c_uint),  # 收到帧计数
    ('nRequestResendPacketCount', int64_t),  # 请求重发包数
    ('nResendPacketCount', int64_t),  # 重发包数
]
MV_MATCH_INFO_NET_DETECT = _MV_MATCH_INFO_NET_DETECT_


# host收到的u3v设备端总字节数，对应类型为 MV_MATCH_TYPE_USB_DETECT
class _MV_MATCH_INFO_USB_DETECT_(Structure):
    pass


_MV_MATCH_INFO_USB_DETECT_._fields_ = [
    ('nReceiveDataSize', int64_t),  # 已接收数据大小
    ('nReceivedFrameCount', c_uint),  # 接收到的帧数
    ('nErrorFrameCount', c_uint),  # 错误帧数
    ('nReserved', c_uint * 2),  # 保留字节
]
MV_MATCH_INFO_USB_DETECT = _MV_MATCH_INFO_USB_DETECT_


class _MV_IMAGE_BASIC_INFO_(Structure):
    pass


_MV_IMAGE_BASIC_INFO_._fields_ = [
    ('nWidthValue', c_ushort),  # 宽度
    ('nWidthMin', c_ushort),  # 宽度最小值
    ('nWidthMax', c_uint),  # 宽度最大值
    ('nWidthInc', c_uint),  # 宽度步进
    ('nHeightValue', c_uint),  # 高度
    ('nHeightMin', c_uint),  # 高度最小值
    ('nHeightMax', c_uint),  # 高度最大值
    ('nHeightInc', c_uint),  # 高度步进
    ('fFrameRateValue', c_float),  # 帧率
    ('fFrameRateMin', c_float),  # 帧率最小值
    ('fFrameRateMax', c_float),  # 帧率最大值
    ('enPixelType', c_uint),  # 当前像素格式
    ('nSupportedPixelFmtNum', c_uint),  # 支持的像素格式数量
    ('enPixelList', c_uint * 64),  # 像素格式列表
    ('nReserved', c_uint * 8),  # 保留字节
]
MV_IMAGE_BASIC_INFO = _MV_IMAGE_BASIC_INFO_

# XML 接口类型枚举
MV_XML_InterfaceType = c_int  # enum

# XML 访问模式枚举
MV_XML_AccessMode = c_int  # enum

# XML 可见性枚举
MV_XML_Visibility = c_int  # enum


class _MV_EVENT_OUT_INFO_(Structure):
    pass


_MV_EVENT_OUT_INFO_._fields_ = [
    ('EventName', c_char * MAX_EVENT_NAME_SIZE),  # 事件名称
    ('nEventID', c_ushort),  # 事件ID
    ('nStreamChannel', c_ushort),  # 流通道序号
    ('nBlockIdHigh', c_uint),  # 帧号高位 (需固件支持)
    ('nBlockIdLow', c_uint),  # 帧号低位 (需固件支持)
    ('nTimestampHigh', c_uint),  # 时间戳高位
    ('nTimestampLow', c_uint),  # 时间戳低位
    ('pEventData', c_void_p),  # 事件数据 (需固件支持)
    ('nEventDataSize', c_uint),  # 事件数据长度 (需固件支持)
    ('nReserved', c_uint * 16),  # 保留字节
]
MV_EVENT_OUT_INFO = _MV_EVENT_OUT_INFO_


class _MV_CC_FILE_ACCESS_T(Structure):
    pass


_MV_CC_FILE_ACCESS_T._fields_ = [
    ('pUserFileName', STRING),  # 用户文件名
    ('pDevFileName', STRING),  # 设备文件名
    ('nReserved', c_uint * 32),  # 保留字节
]
MV_CC_FILE_ACCESS = _MV_CC_FILE_ACCESS_T


class _MV_CC_FILE_ACCESS_PROGRESS_T(Structure):
    pass


_MV_CC_FILE_ACCESS_PROGRESS_T._fields_ = [
    ('nCompleted', int64_t),  # 已完成的长度
    ('nTotal', int64_t),  # 总长度
    ('nReserved', c_uint * 8),  # 保留字节
]
MV_CC_FILE_ACCESS_PROGRESS = _MV_CC_FILE_ACCESS_PROGRESS_T


# CameraParams.h 538
class _MV_TRANSMISSION_TYPE_T(Structure):
    pass


_MV_TRANSMISSION_TYPE_T._fields_ = [
    ('enTransmissionType', MV_GIGE_TRANSMISSION_TYPE),  # 传输模式
    ('nDestIp', c_uint),  # 目标IP（组播模式下需要）
    ('nDestPort', c_ushort),  # 目标端口（组播模式下需要）
    ('nReserved', c_uint * 32),  # 保留字节
]
MV_TRANSMISSION_TYPE = _MV_TRANSMISSION_TYPE_T


# 动作命令信息
class _MV_ACTION_CMD_INFO_T(Structure):
    pass


_MV_ACTION_CMD_INFO_T._fields_ = [
    ('nDeviceKey', c_uint),  # 设备密钥
    ('nGroupKey', c_uint),  # 组密钥
    ('nGroupMask', c_uint),  # 组掩码
    ('bActionTimeEnable', c_uint),  # 只有设置为1时Action Time才有效
    ('nActionTime', int64_t),  # 预定执行时间
    ('pBroadcastAddress', STRING),  # 广播目的地址
    ('nTimeOut', c_uint),  # 等待ACK的超时时间，设为0表示不需要ACK
    ('nReserved', c_uint * 16),  # 预留字节
]
MV_ACTION_CMD_INFO = _MV_ACTION_CMD_INFO_T


# 动作命令返回信息
class _MV_ACTION_CMD_RESULT_T(Structure):
    pass


_MV_ACTION_CMD_RESULT_T._fields_ = [
    ('strDeviceAddress', c_ubyte * 16),  # 设备的IP地址
    ('nStatus', c_int),  # 状态码 (0:成功; 0x8001:不支持; 0x8013:未同步; 0x8015:溢出; 0x8016:已过期)
    ('nReserved', c_uint * 4),  # 预留字节
]
MV_ACTION_CMD_RESULT = _MV_ACTION_CMD_RESULT_T


# 动作命令返回信息列表
class _MV_ACTION_CMD_RESULT_LIST_T(Structure):
    pass


_MV_ACTION_CMD_RESULT_LIST_T._fields_ = [
    ('nNumResults', c_uint),  # 返回值的个数
    ('pResults', POINTER(MV_ACTION_CMD_RESULT)),  # 动作命令返回信息列表
]
MV_ACTION_CMD_RESULT_LIST = _MV_ACTION_CMD_RESULT_LIST_T


# 单个节点基本属性
class _MV_XML_NODE_FEATURE_(Structure):
    pass


_MV_XML_NODE_FEATURE_._fields_ = [
    ('enType', MV_XML_InterfaceType),  # 节点类型
    ('enVisivility', MV_XML_Visibility),  # 可见性
    ('strDescription', c_char * 512),  # 节点描述 (目前暂不支持)
    ('strDisplayName', c_char * 64),  # 显示名称
    ('strName', c_char * 64),  # 节点名
    ('strToolTip', c_char * 512),  # 提示
    ('nReserved', c_uint * 4),  # 预留字节
]
MV_XML_NODE_FEATURE = _MV_XML_NODE_FEATURE_


class _MV_XML_NODES_LIST_(Structure):
    pass


_MV_XML_NODES_LIST_._fields_ = [
    ('nNodeNum', c_uint),  # 节点数量
    ('stNodes', MV_XML_NODE_FEATURE * 128),  # 节点基本属性列表
]
MV_XML_NODES_LIST = _MV_XML_NODES_LIST_


class _MV_XML_FEATURE_Value_(Structure):
    pass


_MV_XML_FEATURE_Value_._fields_ = [
    ('enType', MV_XML_InterfaceType),  # 节点类型
    ('strDescription', c_char * 512),  # 节点描述 (目前暂不支持)
    ('strDisplayName', c_char * 64),  # 显示名称
    ('strName', c_char * 64),  # 节点名
    ('strToolTip', c_char * 512),  # 提示
    ('nReserved', c_uint * 4),  # 预留字节
]
MV_XML_FEATURE_Value = _MV_XML_FEATURE_Value_


class _MV_XML_FEATURE_Base_(Structure):
    pass


_MV_XML_FEATURE_Base_._fields_ = [
    ('enAccessMode', MV_XML_AccessMode),  # 访问模式
]
MV_XML_FEATURE_Base = _MV_XML_FEATURE_Base_


class _MV_XML_FEATURE_Integer_(Structure):
    pass


_MV_XML_FEATURE_Integer_._fields_ = [
    ('strName', c_char * 64),  # 节点名
    ('strDisplayName', c_char * 64),  # 显示名称
    ('strDescription', c_char * 512),  # 节点描述 (目前暂不支持)
    ('strToolTip', c_char * 512),  # 提示
    ('enVisivility', MV_XML_Visibility),  # 是否可见
    ('enAccessMode', MV_XML_AccessMode),  # 访问模式
    ('bIsLocked', c_int),  # 是否锁定 (目前暂不支持)
    ('nValue', int64_t),  # 当前值
    ('nMinValue', int64_t),  # 最小值
    ('nMaxValue', int64_t),  # 最大值
    ('nIncrement', int64_t),  # 增量
    ('nReserved', c_uint * 4),  # 预留字节
]
MV_XML_FEATURE_Integer = _MV_XML_FEATURE_Integer_


class _MV_XML_FEATURE_Boolean_(Structure):
    pass


_MV_XML_FEATURE_Boolean_._fields_ = [
    ('strName', c_char * 64),  # 节点名
    ('strDisplayName', c_char * 64),  # 显示名称
    ('strDescription', c_char * 512),  # 节点描述 (目前暂不支持)
    ('strToolTip', c_char * 512),  # 提示
    ('enVisivility', MV_XML_Visibility),  # 是否可见
    ('enAccessMode', MV_XML_AccessMode),  # 访问模式
    ('bIsLocked', c_int),  # 是否锁定 (目前暂不支持)
    ('bValue', c_bool),  # 当前值
    ('nReserved', c_uint * 4),  # 预留字节
]
MV_XML_FEATURE_Boolean = _MV_XML_FEATURE_Boolean_


class _MV_XML_FEATURE_Command_(Structure):
    pass


_MV_XML_FEATURE_Command_._fields_ = [
    ('strName', c_char * 64),  # 节点名
    ('strDisplayName', c_char * 64),  # 显示名称
    ('strDescription', c_char * 512),  # 节点描述 (目前暂不支持)
    ('strToolTip', c_char * 512),  # 提示
    ('enVisivility', MV_XML_Visibility),  # 是否可见
    ('enAccessMode', MV_XML_AccessMode),  # 访问模式
    ('bIsLocked', c_int),  # 是否锁定 (目前暂不支持)
    ('nReserved', c_uint * 4),  # 预留字节
]
MV_XML_FEATURE_Command = _MV_XML_FEATURE_Command_


class _MV_XML_FEATURE_Float_(Structure):
    pass


_MV_XML_FEATURE_Float_._fields_ = [
    ('strName', c_char * 64),  # 节点名
    ('strDisplayName', c_char * 64),  # 显示名称
    ('strDescription', c_char * 512),  # 节点描述 (目前暂不支持)
    ('strToolTip', c_char * 512),  # 提示
    ('enVisivility', MV_XML_Visibility),  # 是否可见
    ('enAccessMode', MV_XML_AccessMode),  # 访问模式
    ('bIsLocked', c_int),  # 是否锁定 (目前暂不支持)
    ('dfValue', c_double),  # 当前值
    ('dfMinValue', c_double),  # 最小值
    ('dfMaxValue', c_double),  # 最大值
    ('dfIncrement', c_double),  # 增量
    ('nReserved', c_uint * 4),  # 预留字节
]
MV_XML_FEATURE_Float = _MV_XML_FEATURE_Float_


class _MV_XML_FEATURE_String_(Structure):
    pass


_MV_XML_FEATURE_String_._fields_ = [
    ('strName', c_char * 64),  # 节点名
    ('strDisplayName', c_char * 64),  # 显示名称
    ('strDescription', c_char * 512),  # 节点描述 (目前暂不支持)
    ('strToolTip', c_char * 512),  # 提示
    ('enVisivility', MV_XML_Visibility),  # 是否可见
    ('enAccessMode', MV_XML_AccessMode),  # 访问模式
    ('bIsLocked', c_int),  # 是否锁定 (目前暂不支持)
    ('strValue', c_char * 64),  # 当前值
    ('nReserved', c_uint * 4),  # 预留字节
]
MV_XML_FEATURE_String = _MV_XML_FEATURE_String_


class _MV_XML_FEATURE_Register_(Structure):
    pass


_MV_XML_FEATURE_Register_._fields_ = [
    ('strName', c_char * 64),  # 节点名
    ('strDisplayName', c_char * 64),  # 显示名称
    ('strDescription', c_char * 512),  # 节点描述 (目前暂不支持)
    ('strToolTip', c_char * 512),  # 提示
    ('enVisivility', MV_XML_Visibility),  # 是否可见
    ('enAccessMode', MV_XML_AccessMode),  # 访问模式
    ('bIsLocked', c_int),  # 是否锁定 (目前暂不支持)
    ('nAddrValue', int64_t),  # 当前地址值
    ('nReserved', c_uint * 4),  # 预留字节
]
MV_XML_FEATURE_Register = _MV_XML_FEATURE_Register_


class _MV_XML_FEATURE_Category_(Structure):
    pass


_MV_XML_FEATURE_Category_._fields_ = [
    ('strDescription', c_char * 512),  # 节点描述 (目前暂不支持)
    ('strDisplayName', c_char * 64),  # 显示名称
    ('strName', c_char * 64),  # 节点名
    ('strToolTip', c_char * 512),  # 提示
    ('enVisivility', MV_XML_Visibility),  # 是否可见
    ('nReserved', c_uint * 4),  # 预留字节
]
MV_XML_FEATURE_Category = _MV_XML_FEATURE_Category_


class _MV_XML_FEATURE_EnumEntry_(Structure):
    pass


_MV_XML_FEATURE_EnumEntry_._fields_ = [
    ('strName', c_char * 64),  # 节点名
    ('strDisplayName', c_char * 64),  # 显示名称
    ('strDescription', c_char * 512),  # 节点描述 (目前暂不支持)
    ('strToolTip', c_char * 512),  # 提示
    ('bIsImplemented', c_int),  # 是否已实现
    ('nParentsNum', c_int),  # 父节点数量
    ('stParentsList', MV_XML_NODE_FEATURE * 8),  # 父节点列表
    ('enVisivility', MV_XML_Visibility),  # 是否可见
    ('nValue', int64_t),  # 当前值
    ('enAccessMode', MV_XML_AccessMode),  # 访问模式
    ('bIsLocked', c_int),  # 是否锁定 (目前暂不支持)
    ('nReserved', c_int * 8),  # 预留字节
]
MV_XML_FEATURE_EnumEntry = _MV_XML_FEATURE_EnumEntry_


class _MV_XML_FEATURE_Enumeration_(Structure):
    pass


_MV_XML_FEATURE_Enumeration_._fields_ = [
    ('enVisivility', MV_XML_Visibility),  # 是否可见
    ('strDescription', c_char * 512),  # 节点描述 (目前暂不支持)
    ('strDisplayName', c_char * 64),  # 显示名称
    ('strName', c_char * 64),  # 节点名
    ('strToolTip', c_char * 512),  # 提示
    ('nSymbolicNum', c_int),  # 枚举条目(Symbolic)数量
    ('strCurrentSymbolic', c_char * 64),  # 当前显示的枚举条目名
    ('strSymbolic', c_char * 64 * 64),  # 枚举条目列表
    ('enAccessMode', MV_XML_AccessMode),  # 访问模式
    ('bIsLocked', c_int),  # 是否锁定 (目前暂不支持)
    ('nValue', int64_t),  # 当前值
    ('nReserved', c_uint * 4),  # 预留字节
]
MV_XML_FEATURE_Enumeration = _MV_XML_FEATURE_Enumeration_


class _MV_XML_FEATURE_Port_(Structure):
    pass


_MV_XML_FEATURE_Port_._fields_ = [
    ('enVisivility', MV_XML_Visibility),  # 是否可见
    ('strDescription', c_char * 512),  # 节点描述 (目前暂不支持)
    ('strDisplayName', c_char * 64),  # 显示名称
    ('strName', c_char * 64),  # 节点名
    ('strToolTip', c_char * 512),  # 提示
    ('enAccessMode', MV_XML_AccessMode),  # 访问模式
    ('bIsLocked', c_int),  # 是否锁定 (目前暂不支持)
    ('nReserved', c_uint * 4),  # 预留字节
]
MV_XML_FEATURE_Port = _MV_XML_FEATURE_Port_


class _MV_XML_CAMERA_FEATURE_(Structure):
    pass


class N23_MV_XML_CAMERA_FEATURE_3DOT_1E(Union):
    pass


N23_MV_XML_CAMERA_FEATURE_3DOT_1E._fields_ = [
    ('stIntegerFeature', MV_XML_FEATURE_Integer),  # 整型特性
    ('stFloatFeature', MV_XML_FEATURE_Float),  # 浮点型特性
    ('stEnumerationFeature', MV_XML_FEATURE_Enumeration),  # 枚举型特性
    ('stStringFeature', MV_XML_FEATURE_String),  # 字符串特性
]
_MV_XML_CAMERA_FEATURE_._fields_ = [
    ('enType', MV_XML_InterfaceType),
    ('SpecialFeature', N23_MV_XML_CAMERA_FEATURE_3DOT_1E),
]
MV_XML_CAMERA_FEATURE = _MV_XML_CAMERA_FEATURE_


class _MVCC_ENUMVALUE_T(Structure):
    pass


_MVCC_ENUMVALUE_T._fields_ = [
    ('nCurValue', c_uint),  # 当前值
    ('nSupportedNum', c_uint),  # 列表内有效数据个数
    ('nSupportValue', c_uint * MV_MAX_XML_SYMBOLIC_NUM),  # 支持的索引值列表
    ('nReserved', c_uint * 4),  # 预留字节
]
MVCC_ENUMVALUE = _MVCC_ENUMVALUE_T


class _MVCC_INTVALUE_T(Structure):
    pass


_MVCC_INTVALUE_T._fields_ = [
    ('nCurValue', c_uint),  # 当前值
    ('nMax', c_uint),  # 最大值
    ('nMin', c_uint),  # 最小值
    ('nInc', c_uint),  # 步进
    ('nReserved', c_uint * 4),  # 预留字节
]
MVCC_INTVALUE = _MVCC_INTVALUE_T


class _MVCC_INTVALUE_EX_T(Structure):
    pass


_MVCC_INTVALUE_EX_T._fields_ = [
    ('nCurValue', int64_t),  # 当前值
    ('nMax', int64_t),  # 最大值
    ('nMin', int64_t),  # 最小值
    ('nInc', int64_t),  # 步进
    ('nReserved', c_uint * 16),  # 预留字节
]
MVCC_INTVALUE_EX = _MVCC_INTVALUE_EX_T


class _MVCC_FLOATVALUE_T(Structure):
    pass


_MVCC_FLOATVALUE_T._fields_ = [
    ('fCurValue', c_float),  # 当前值
    ('fMax', c_float),  # 最大值
    ('fMin', c_float),  # 最小值
    ('nReserved', c_uint * 4),  # 预留字节
]
MVCC_FLOATVALUE = _MVCC_FLOATVALUE_T


class _MVCC_STRINGVALUE_T(Structure):
    pass


_MVCC_STRINGVALUE_T._fields_ = [
    ('chCurValue', c_char * 256),  # 当前值
    ('nMaxLength', int64_t),  # 最大长度
    ('nReserved', c_uint * 2),  # 预留字节
]
MVCC_STRINGVALUE = _MVCC_STRINGVALUE_T


# 帧水印信息
class _MV_CC_FRAME_SPEC_INFO_(Structure):
    pass


_MV_CC_FRAME_SPEC_INFO_._fields_ = [
    # 设备水印时间刻度
    ('nSecondCount', c_uint),  # 秒数
    ('nCycleCount', c_uint),  # 循环计数
    ('nCycleOffset', c_uint),  # 循环偏移量
    ('fGain', c_float),  # 增益
    ('fExposureTime', c_float),  # 曝光时间
    ('nAverageBrightness', c_uint),  # 平均亮度
    # 白平衡
    ('nRed', c_uint),  # 红色
    ('nGreen', c_uint),  # 绿色
    ('nBlue', c_uint),  # 蓝色
    ('nFrameCounter', c_uint),  # 当前帧号
    ('nTriggerIndex', c_uint),  # 触发计数
    ('nInput', c_uint),  # 输入
    ('nOutput', c_uint),  # 输出
    # ROI 区域
    ('nOffsetX', c_ushort),  # 水平偏移
    ('nOffsetY', c_ushort),  # 垂直偏移
    ('nFrameWidth', c_ushort),  # 水印宽
    ('nFrameHeight', c_ushort),  # 水印高
    ('nReserved', c_uint * 16),  # 预留字节
]
MV_CC_FRAME_SPEC_INFO = _MV_CC_FRAME_SPEC_INFO_


# 高带宽解码参数
class _MV_CC_HB_DECODE_PARAM_T_(Structure):
    pass


_MV_CC_HB_DECODE_PARAM_T_._fields_ = [
    ('pSrcBuf', POINTER(c_ubyte)),  # 输入数据缓存
    ('nSrcLen', c_uint),  # 输入数据大小
    ('nWidth', c_uint),  # 图像宽
    ('nHeight', c_uint),  # 图像高
    ('pDstBuf', POINTER(c_ubyte)),  # 输出数据缓存
    ('nDstBufSize', c_uint),  # 提供的输出缓存大小
    ('nDstBufLen', c_uint),  # 输出数据大小
    ('enDstPixelType', MvGvspPixelType),  # 输出像素格式
    ('stFrameInfo', MV_CC_FRAME_SPEC_INFO),  # 水印信息
    ('nRes', c_uint * 8),  # 保留字节
]
MV_CC_HB_DECODE_PARAM = _MV_CC_HB_DECODE_PARAM_T_

# 枚举类型定义
_MV_SORT_METHOD_ = c_int
MV_SORT_METHOD = _MV_SORT_METHOD_
_MV_IMG_ROTATION_ANGLE_ = c_int
MV_IMG_ROTATION_ANGLE = _MV_IMG_ROTATION_ANGLE_
_MV_IMG_FLIP_TYPE_ = c_int
MV_IMG_FLIP_TYPE = _MV_IMG_FLIP_TYPE_
_MV_CC_GAMMA_TYPE_ = c_int
MV_CC_GAMMA_TYPE = _MV_CC_GAMMA_TYPE_
_MV_CC_STREAM_EXCEPTION_TYPE_ = c_int
MV_CC_STREAM_EXCEPTION_TYPE = _MV_CC_STREAM_EXCEPTION_TYPE_
_MV_IMAGE_RECONSTRUCTION_METHOD_ = c_int
MV_IMAGE_RECONSTRUCTION_METHOD = _MV_IMAGE_RECONSTRUCTION_METHOD_


# 图像旋转参数
class _MV_CC_ROTATE_IMAGE_PARAM_T_(Structure):
    pass


_MV_CC_ROTATE_IMAGE_PARAM_T_._fields_ = [
    ('enPixelType', MvGvspPixelType),  # 像素格式
    ('nWidth', c_uint),  # 图像宽
    ('nHeight', c_uint),  # 图像高
    ('pSrcData', POINTER(c_ubyte)),  # 输入数据缓存
    ('nSrcDataLen', c_uint),  # 输入数据长度
    ('pDstBuf', POINTER(c_ubyte)),  # 输出数据缓存
    ('nDstBufLen', c_uint),  # 输出数据长度
    ('nDstBufSize', c_uint),  # 提供的输出缓存大小
    ('enRotationAngle', MV_IMG_ROTATION_ANGLE),  # 旋转角度
    ('nRes', c_uint * 8),  # 保留字节
]
MV_CC_ROTATE_IMAGE_PARAM = _MV_CC_ROTATE_IMAGE_PARAM_T_


# 图像翻转参数
class _MV_CC_FLIP_IMAGE_PARAM_T_(Structure):
    pass


_MV_CC_FLIP_IMAGE_PARAM_T_._fields_ = [
    ('enPixelType', MvGvspPixelType),  # 像素格式
    ('nWidth', c_uint),  # 图像宽
    ('nHeight', c_uint),  # 图像高
    ('pSrcData', POINTER(c_ubyte)),  # 输入数据缓存
    ('nSrcDataLen', c_uint),  # 输入数据长度
    ('pDstBuf', POINTER(c_ubyte)),  # 输出数据缓存
    ('nDstBufLen', c_uint),  # 输出数据长度
    ('nDstBufSize', c_uint),  # 提供的输出缓存大小
    ('enFlipType', MV_IMG_FLIP_TYPE),  # 翻转类型
    ('nRes', c_uint * 8),  # 保留字节
]
MV_CC_FLIP_IMAGE_PARAM = _MV_CC_FLIP_IMAGE_PARAM_T_


# Gamma 参数
class _MV_CC_GAMMA_PARAM_T_(Structure):
    pass


_MV_CC_GAMMA_PARAM_T_._fields_ = [
    ('enGammaType', MV_CC_GAMMA_TYPE),  # Gamma 类型
    ('fGammaValue', c_float),  # Gamma 值[0.1, 4.0]
    ('pGammaCurveBuf', POINTER(c_ubyte)),  # Gamma 曲线缓存
    ('nGammaCurveBufLen', c_uint),  # Gamma 曲线缓存长度
    ('nRes', c_uint * 8),  # 保留字节
]
MV_CC_GAMMA_PARAM = _MV_CC_GAMMA_PARAM_T_


# CCM 色彩校正矩阵参数
class _MV_CC_CCM_PARAM_T_(Structure):
    pass


_MV_CC_CCM_PARAM_T_._fields_ = [
    ('bCCMEnable', c_bool),  # 是否启用 CCM
    ('nCCMat', c_int * 9),  # CCM 矩阵[-8192~8192]
    ('nRes', c_uint * 8),  # 保留字节
]
MV_CC_CCM_PARAM = _MV_CC_CCM_PARAM_T_


# CCM 扩展参数
class _MV_CC_CCM_PARAM_EX_T_(Structure):
    pass


_MV_CC_CCM_PARAM_EX_T_._fields_ = [
    ('bCCMEnable', c_bool),  # 是否启用 CCM
    ('nCCMat', c_int * 9),  # CCM 矩阵[-65536~65536]
    ('nCCMScale', c_uint),  # 量化系数 (2 的整数幂, <= 65536)
    ('nRes', c_uint * 8),  # 保留字节
]
MV_CC_CCM_PARAM_EX = _MV_CC_CCM_PARAM_EX_T_


# 图像对比度调节
class _MV_CC_CONTRAST_PARAM_T_(Structure):
    pass


_MV_CC_CONTRAST_PARAM_T_._fields_ = [
    ('nWidth', c_uint),  # 图像宽(最小8)
    ('nHeight', c_uint),  # 图像高(最小8)
    ('pSrcBuf', POINTER(c_ubyte)),  # 输入数据缓存
    ('nSrcBufLen', c_uint),  # 输入数据长度
    ('enPixelType', MvGvspPixelType),  # 像素格式
    ('pDstBuf', POINTER(c_ubyte)),  # 输出数据缓存
    ('nDstBufSize', c_uint),  # 提供的输出缓存大小
    ('nDstBufLen', c_uint),  # 输出数据长度
    ('nContrastFactor', c_uint),  # 对比度值 [1, 10000]
    ('nRes', c_uint * 8),  # 保留字节
]
MV_CC_CONTRAST_PARAM_T = _MV_CC_CONTRAST_PARAM_T_


# 枚举条目
class _MVCC_ENUMENTRY_T(Structure):
    pass


_MVCC_ENUMENTRY_T._fields_ = [
    ('nValue', c_uint),  # 索引值
    ('chSymbolic', c_char * MV_MAX_SYMBOLIC_LEN),  # 对应的符号名
    ('nReserved', c_uint * 4),  # 预留字节
]
MVCC_ENUMENTRY = _MVCC_ENUMENTRY_T


# 辅助线颜色
class _MVCC_COLORF(Structure):
    pass


_MVCC_COLORF._fields_ = [
    ('fR', c_float),  # 红色分量 [0.0, 1.0]
    ('fG', c_float),  # 绿色分量 [0.0, 1.0]
    ('fB', c_float),  # 蓝色分量 [0.0, 1.0]
    ('fAlpha', c_float),  # 透明度 [0.0, 1.0] (目前不支持)
    ('nReserved', c_uint * 4),  # 保留字节
]
MVCC_COLORF = _MVCC_COLORF


# 点坐标定义
class _MVCC_POINTF(Structure):
    pass


_MVCC_POINTF._fields_ = [
    ('fX', c_float),  # 距离左边缘距离 [0.0, 1.0]
    ('fY', c_float),  # 距离上边缘距离 [0.0, 1.0]
    ('nReserved', c_uint * 4),  # 保留字节
]
MVCC_POINTF = _MVCC_POINTF


# 矩形框区域信息
class _MVCC_RECT_INFO(Structure):
    pass


_MVCC_RECT_INFO._fields_ = [
    ('fTop', c_float),  # 距离图像上边缘的距离，[0.0, 1.0]
    ('fBottom', c_float),  # 距离图像下边缘的距离，[0.0, 1.0]
    ('fLeft', c_float),  # 距离图像左边缘的距离，[0.0, 1.0]
    ('fRight', c_float),  # 距离图像右边缘的距离，[0.0, 1.0]
    ('stColor', MVCC_COLORF),  # 辅助线颜色信息
    ('nLineWidth', c_uint),  # 辅助线宽度，目前只支持1或2
    ('nReserved', c_uint * 4),  # 保留字节
]
MVCC_RECT_INFO = _MVCC_RECT_INFO


# 圆形框区域信息
class _MVCC_CIRCLE_INFO(Structure):
    pass


_MVCC_CIRCLE_INFO._fields_ = [
    ('stCenterPoint', MVCC_POINTF),  # 圆心信息
    ('fR1', c_float),  # 宽半径，相对于图像宽度的比例 [0.0, 1.0]
    ('fR2', c_float),  # 高半径，相对于图像高度的比例 [0.0, 1.0]
    ('stColor', MVCC_COLORF),  # 辅助线颜色信息
    ('nLineWidth', c_uint),  # 辅助线宽度，目前只支持1或2
    ('nReserved', c_uint * 4),  # 保留字节
]
MVCC_CIRCLE_INFO = _MVCC_CIRCLE_INFO


# 辅助线段信息
class _MVCC_LINES_INFO(Structure):
    pass


_MVCC_LINES_INFO._fields_ = [
    ('stStartPoint', MVCC_POINTF),  # 辅助线段的起始点坐标
    ('stEndPoint', MVCC_POINTF),  # 辅助线段的终点坐标
    ('stColor', MVCC_COLORF),  # 辅助线颜色信息
    ('nLineWidth', c_uint),  # 辅助线宽度，目前只支持1或2
    ('nReserved', c_uint * 4),  # 保留字节
]
MVCC_LINES_INFO = _MVCC_LINES_INFO


# 图像重构后的图像列表
class _MV_OUTPUT_IMAGE_INFO_(Structure):
    pass


_MV_OUTPUT_IMAGE_INFO_._fields_ = [
    ('nWidth', c_uint),  # 图像宽
    ('nHeight', c_uint),  # 图像高
    ('enPixelType', MvGvspPixelType),  # 像素格式
    ('pBuf', POINTER(c_ubyte)),  # 输出数据缓存
    ('nBufLen', c_uint),  # 输出数据长度
    ('nBufSize', c_uint),  # 提供的输出缓存大小
    ('nRes', c_uint * 8),  # 保留字节
]
MV_OUTPUT_IMAGE_INFO = _MV_OUTPUT_IMAGE_INFO_


# 重构图像参数信息
class _MV_RECONSTRUCT_IMAGE_PARAM_(Structure):
    pass


_MV_RECONSTRUCT_IMAGE_PARAM_._fields_ = [
    ('nWidth', c_uint),  # 图像宽
    ('nHeight', c_uint),  # 图像高
    ('enPixelType', MvGvspPixelType),  # 像素格式
    ('pSrcData', POINTER(c_ubyte)),  # 输入数据缓存
    ('nSrcDataLen', c_uint),  # 输入数据大小
    ('nExposureNum', c_uint),  # 曝光次数 (1-8]
    ('enReconstructMethod', MV_IMAGE_RECONSTRUCTION_METHOD),  # 图像重构方式
    ('stDstBufList', MV_OUTPUT_IMAGE_INFO * MV_MAX_SPLIT_NUM),  # 输出数据缓存信息
    ('nRes', c_uint * 4),  # 保留字节
]
MV_RECONSTRUCT_IMAGE_PARAM = _MV_RECONSTRUCT_IMAGE_PARAM_


# 文件访问扩展
class _MV_CC_FILE_ACCESS_E(Structure):
    pass


_MV_CC_FILE_ACCESS_E._fields_ = [
    ('pUserFileBuf', POINTER(c_char)),  # 用户文件数据
    ('pFileBufSize', c_uint),  # 用户数据缓存大小
    ('pFileBufLen', c_uint),  # 用户数据缓存长度
    ('pDevFileName', STRING),  # 设备文件名
    ('nReserved', c_uint * 32),  # 保留字节
]
MV_CC_FILE_ACCESS_EX = _MV_CC_FILE_ACCESS_E


# 采集卡信息
class _MV_INTERFACE_INFO_(Structure):
    pass


_MV_INTERFACE_INFO_._fields_ = [
    ('nTLayerType', c_uint),  # 传输层类型
    ('nPCIEInfo', c_uint),  # 采集卡对应PCIe插槽信息
    ('chInterfaceID', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 采集卡ID
    ('chDisplayName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 显示名称
    ('chSerialNumber', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 序列号
    ('chModelName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 型号
    ('chManufacturer', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 制造商
    ('chDeviceVersion', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 版本号
    ('chUserDefinedName', c_ubyte * INFO_MAX_BUFFER_SIZE),  # 自定义名称
    ('nReserved', c_uint * 64),  # 保留字节
]
MV_INTERFACE_INFO = _MV_INTERFACE_INFO_


# 采集卡信息列表
class _MV_INTERFACE_INFO_LIST_(Structure):
    pass


_MV_INTERFACE_INFO_LIST_._fields_ = [
    ('nInterfaceNum', c_uint),  # 在线设备数量
    ('pInterfaceInfos', POINTER(MV_INTERFACE_INFO) * MV_MAX_INTERFACE_NUM),  # 支持最多256个采集卡
]
MV_INTERFACE_INFO_LIST = _MV_INTERFACE_INFO_LIST_


# 串口信息
class _MV_CAML_SERIAL_PORT_(Structure):
    pass


_MV_CAML_SERIAL_PORT_._fields_ = [
    ('chSerialPort', c_char * INFO_MAX_BUFFER_SIZE),  # 串口号
    ('nRes', c_uint * 4),  # 保留字节
]
MV_CAML_SERIAL_PORT = _MV_CAML_SERIAL_PORT_


# 串口列表
class _MV_CAML_SERIAL_PORT_LIST_(Structure):
    pass


_MV_CAML_SERIAL_PORT_LIST_._fields_ = [
    ('nSerialPortNum', c_uint),  # 串口数量
    ('stSerialPort', MV_CAML_SERIAL_PORT * MV_MAX_SERIAL_PORT_NUM),  # 串口信息列表
    ('nRes', c_uint * 4),  # 保留字节
]
MV_CAML_SERIAL_PORT_LIST = _MV_CAML_SERIAL_PORT_LIST_
__all__ = ['_MV_ALL_MATCH_INFO_', '_MV_XML_FEATURE_Integer_',
           'MV_CC_FILE_ACCESS_PROGRESS',
           'N19_MV_CC_DEVICE_INFO_3DOT_0E',
           'MV_CAM_EXPOSURE_AUTO_MODE',
           'MV_CAM_GAIN_MODE',
           'MV_GIGE_TRANSTYPE_UNICAST_WITHOUT_RECV',
           'MV_TRIGGER_SOURCE_LINE0', 'MV_TRIGGER_SOURCE_LINE1',
           'MV_TRIGGER_SOURCE_LINE2', 'MV_TRIGGER_SOURCE_LINE3',
           'AM_CycleDetect',
           'IFT_IFloat', 'MV_TRANSMISSION_TYPE',
           '_MV_XML_FEATURE_Command_', '_MV_XML_FEATURE_String_',
           '_MV_CAM_TRIGGER_SOURCE_',
           'AM_RO', 'IFT_IPort',
           'uint_least16_t', '_MV_FRAME_OUT_INFO_EX_',
           '_MV_TRANSMISSION_TYPE_T', 'MV_SAVE_IMAGE_PARAM_EX',
           'AM_RW', 'MV_XML_InterfaceType', '_MV_XML_CAMERA_FEATURE_',
           'intptr_t', 'uint_least64_t', 'V_Guru',
           '_MV_CAM_TRIGGER_MODE_', 'MV_CAM_EXPOSURE_MODE',
           'int_least32_t', 'MV_GIGE_TRANSTYPE_SUBNETBROADCAST',
           '_MV_XML_FEATURE_Boolean_',
           'MV_BALANCEWHITE_AUTO_CONTINUOUS', 'MV_XML_NODE_FEATURE',
           '_MV_FRAME_OUT_INFO_', 'MV_ALL_MATCH_INFO',
           '_MV_XML_FEATURE_EnumEntry_', '_MV_CC_PIXEL_CONVERT_PARAM_T_',
           'MV_ACQ_MODE_SINGLE',
           'MV_TRIGGER_MODE_ON', '_MV_XML_FEATURE_Base_',
           'int_least16_t',
           'MV_GIGE_TRANSTYPE_LIMITEDBROADCAST', 'int_fast32_t',
           '_MV_CAM_GAIN_MODE_', 'uint_fast16_t',
           '_MV_DISPLAY_FRAME_INFO_', 'IFT_ICommand',
           '_MV_CAM_ACQUISITION_MODE_',
           'V_Beginner',
           'MV_GIGE_TRANSTYPE_MULTICAST_WITHOUT_RECV',
           'MVCC_INTVALUE', '_MV_XML_FEATURE_Port_',
           '_MV_GIGE_TRANSMISSION_TYPE_', 'MV_GIGE_TRANSMISSION_TYPE',
           '_MV_XML_FEATURE_Register_', 'MV_EXPOSURE_MODE_TIMED',
           'intmax_t', 'int16_t',
           'MV_DISPLAY_FRAME_INFO', 'MV_XML_FEATURE_Integer',
           '_MV_CC_FILE_ACCESS_PROGRESS_T', 'MV_XML_Visibility',
           'IFT_IEnumEntry', 'int_fast64_t',
           'MV_XML_AccessMode', 'V_Expert', 'MV_GAIN_MODE_ONCE',
           'IFT_IInteger',
           'MV_CAM_BALANCEWHITE_AUTO', 'int_least8_t', 'IFT_IBase',
           'MV_XML_NODES_LIST', 'MV_TRIGGER_MODE_OFF', 'MV_Image_Bmp',
           'MV_XML_FEATURE_String', 'MV_CC_FILE_ACCESS',
           '_MV_CAM_EXPOSURE_AUTO_MODE_',
           'uint_least8_t',
           '_MV_XML_FEATURE_Float_', '_MV_XML_NODE_FEATURE_',
           'MV_XML_FEATURE_Float',
           'MV_SAVE_IMAGE_PARAM',
           'MV_EVENT_OUT_INFO', 'IFT_IEnumeration', 'uint64_t',
           'uint8_t', 'V_Undefined',
           '_MVCC_STRINGVALUE_T',
           'MV_CAM_TRIGGER_MODE',
           'N23_MV_XML_CAMERA_FEATURE_3DOT_1E', 'uint16_t',
           'uint_fast8_t', '_MV_SAVE_IMAGE_PARAM_T_',
           '_MVCC_ENUMVALUE_T',
           '_MV_MATCH_INFO_USB_DETECT_', 'MV_XML_FEATURE_Category',
           'int32_t',
           'MV_XML_FEATURE_EnumEntry', '_MV_CC_DEVICE_INFO_',
           'IFT_IBoolean',
           'MV_MATCH_INFO_USB_DETECT',
           'MVCC_ENUMVALUE', 'IFT_IString',
           '_MV_XML_FEATURE_Value_',
           'MV_ACQ_MODE_CONTINUOUS',
           'MV_TRIGGER_SOURCE_FrequencyConverter',
           'MV_TRIGGER_SOURCE_COUNTER0',
           'MV_GAIN_MODE_OFF', 'MV_Image_Png',
           '_MV_CC_DEVICE_INFO_LIST_', 'MV_GIGE_DEVICE_INFO',
           '_MV_SAVE_IMAGE_PARAM_T_EX_',
           'uint_least32_t',
           'MV_FRAME_OUT_INFO',
           '_MVCC_INTVALUE_EX_T', 'uintptr_t',
           'MVCC_FLOATVALUE',
           'MV_GIGE_TRANSTYPE_CAMERADEFINED', '_MV_XML_NODES_LIST_',
           'MV_NETTRANS_INFO', 'IFT_IRegister', 'AM_NA',
           'MV_GIGE_TRANSTYPE_UNICAST', 'int8_t',
           '_MV_GIGE_DEVICE_INFO_', 'IFT_IValue', 'IFT_ICategory',
           'int_fast8_t',
           'MV_XML_FEATURE_Enumeration', 'MV_GAMMA_SELECTOR_SRGB',
           'int_least64_t',
           'MV_EXPOSURE_AUTO_MODE_OFF', 'MV_CC_PIXEL_CONVERT_PARAM',
           'MV_EXPOSURE_AUTO_MODE_CONTINUOUS',
           'MV_CAM_ACQUISITION_MODE', '_MVCC_INTVALUE_T',
           'MV_XML_FEATURE_Value', 'AM_Undefined',
           'MV_MATCH_INFO_NET_DETECT',
           '_MV_CC_FILE_ACCESS_T', 'AM_NI',
           'V_Invisible',
           'MV_CAM_GAMMA_SELECTOR',
           'MV_TRIGGER_SOURCE_SOFTWARE',
           'MV_BALANCEWHITE_AUTO_ONCE',
           'uintmax_t', 'int_fast16_t',
           '_MV_CAM_EXPOSURE_MODE_', 'MV_Image_Tif',
           'MV_BALANCEWHITE_AUTO_OFF',
           'int64_t', 'MV_Image_Undefined', '_MV_NETTRANS_INFO_',
           'MV_GAIN_MODE_CONTINUOUS',
           'uint_fast32_t', 'MV_CAM_TRIGGER_SOURCE',
           'MV_Image_Jpeg',
           '_MVCC_FLOATVALUE_T', 'MV_XML_FEATURE_Port',
           'MV_FRAME_OUT_INFO_EX', '_MV_IMAGE_BASIC_INFO_',
           '_MV_CAM_BALANCEWHITE_AUTO_', 'MV_XML_FEATURE_Base',
           '_MV_USB3_DEVICE_INFO_',
           'MVCC_INTVALUE_EX', 'MV_XML_FEATURE_Register', 'AM_WO',
           'MV_GIGE_TRANSTYPE_UNICAST_DEFINED_PORT',
           '_MV_XML_FEATURE_Enumeration_',
           '_MV_MATCH_INFO_NET_DETECT_', 'MV_SAVE_IAMGE_TYPE',
           'MV_EXPOSURE_AUTO_MODE_ONCE',
           'MV_GIGE_TRANSTYPE_MULTICAST', 'MV_XML_CAMERA_FEATURE',
           'MVCC_STRINGVALUE',
           'MV_CC_DEVICE_INFO',
           'MvGvspPixelType',
           'MV_IMAGE_BASIC_INFO',
           'MV_CC_DEVICE_INFO_LIST',
           'uint_fast64_t',
           '_MV_XML_FEATURE_Category_',
           'MV_PointCloudFile_Undefined', 'MV_EXPOSURE_MODE_TRIGGER_WIDTH',
           'MV_XML_FEATURE_Boolean',
           'MV_GAMMA_SELECTOR_USER',
           'uint32_t', 'MV_XML_FEATURE_Command',
           '_MV_CAM_GAMMA_SELECTOR_', 'MV_ACQ_MODE_MUTLI',
           'MV_USB3_DEVICE_INFO', '_MV_EVENT_OUT_INFO_',
           'MV_GrabStrategy_UpcomingImage', 'MV_GrabStrategy_LatestImages',
           'MV_GrabStrategy_LatestImagesOnly', 'MV_PointCloudFile_OBJ',
           'MV_PointCloudFile_CSV', 'MV_PointCloudFile_PLY',
           'MV_FormatType_Undefined', 'MV_GrabStrategy_OneByOne',
           'MV_FormatType_AVI', 'SortMethod_SerialNumber',
           'SortMethod_UserID', 'SortMethod_CurrentIP_ASC',
           'SortMethod_CurrentIP_DESC', 'MV_IMAGE_ROTATE_90',
           'MV_IMAGE_ROTATE_180', 'MV_IMAGE_ROTATE_270',
           'MV_FLIP_VERTICAL', 'MV_FLIP_HORIZONTAL',
           'MV_CC_GAMMA_TYPE_NONE', 'MV_CC_GAMMA_TYPE_VALUE',
           'MV_CC_GAMMA_TYPE_USER_CURVE', 'MV_CC_GAMMA_TYPE_LRGB2SRGB',
           'MV_CC_GAMMA_TYPE_SRGB2LRGB', 'MV_CC_STREAM_EXCEPTION_ABNORMAL_IMAGE',
           'MV_CC_STREAM_EXCEPTION_LIST_OVERFLOW', 'MV_CC_STREAM_EXCEPTION_LIST_EMPTY',
           'MV_CC_STREAM_EXCEPTION_RECONNECTION', 'MV_CC_STREAM_EXCEPTION_DISCONNECTED',
           'MV_CC_STREAM_EXCEPTION_DEVICE', 'MV_SPLIT_BY_LINE',
           'MV_CamL_DEV_INFO', '_MV_CamL_DEV_INFO_',
           'MV_GENTL_IF_INFO', '_MV_GENTL_IF_INFO_',
           'MV_GENTL_IF_INFO_LIST', '_MV_GENTL_IF_INFO_LIST_',
           'MV_GENTL_DEV_INFO', '_MV_GENTL_DEV_INFO_',
           'MV_GENTL_DEV_INFO_LIST', '_MV_GENTL_DEV_INFO_LIST_',
           'MV_CHUNK_DATA_CONTENT', '_MV_CHUNK_DATA_CONTENT_',
           'N22_MV_FRAME_OUT_INFO_EX_3DOT_1E', 'MV_DISPLAY_FRAME_INFO_EX',
           '_MV_DISPLAY_FRAME_INFO_EX_', '_MV_FRAME_OUT_',
           '_MV_GRAB_STRATEGY_', 'MV_GRAB_STRATEGY', 'MV_SAVE_POINT_CLOUD_FILE_TYPE',
           'MV_SAVE_POINT_CLOUD_PARAM', '_MV_SAVE_POINT_CLOUD_PARAM_',
           'MV_FRAME_OUT', 'MV_RECORD_FORMAT_TYPE', '_MV_RECORD_FORMAT_TYPE_',
           'MV_CC_RECORD_PARAM', '_MV_CC_RECORD_PARAM_T_',
           'MV_CC_INPUT_FRAME_INFO', '_MV_CC_INPUT_FRAME_INFO_T_',
           'MV_ACTION_CMD_INFO', '_MV_ACTION_CMD_INFO_T',
           'MV_ACTION_CMD_RESULT', '_MV_ACTION_CMD_RESULT_T',
           'MV_ACTION_CMD_RESULT_LIST', '_MV_ACTION_CMD_RESULT_LIST_T',
           'MV_SORT_METHOD', '_MV_SORT_METHOD_',
           '_MV_IMG_ROTATION_ANGLE_', 'MV_IMG_ROTATION_ANGLE',
           '_MV_IMG_FLIP_TYPE_', 'MV_IMG_FLIP_TYPE',
           '_MV_CC_GAMMA_TYPE_', 'MV_CC_GAMMA_TYPE',
           'MV_CC_STREAM_EXCEPTION_TYPE', '_MV_CC_STREAM_EXCEPTION_TYPE_',
           '_MV_IMAGE_RECONSTRUCTION_METHOD_', 'MV_IMAGE_RECONSTRUCTION_METHOD',
           'MV_CC_ROTATE_IMAGE_PARAM', '_MV_CC_ROTATE_IMAGE_PARAM_T_',
           'MV_CC_FLIP_IMAGE_PARAM', '_MV_CC_FLIP_IMAGE_PARAM_T_',
           'MV_CC_GAMMA_PARAM', '_MV_CC_GAMMA_PARAM_T_',
           'MV_CC_CCM_PARAM', '_MV_CC_CCM_PARAM_T_',
           'MV_CC_CCM_PARAM_EX', '_MV_CC_CCM_PARAM_EX_T_',
           'MV_CC_CONTRAST_PARAM_T', '_MV_CC_CONTRAST_PARAM_T_',
           'MVCC_ENUMENTRY', '_MVCC_ENUMENTRY_T',
           'MVCC_COLORF', '_MVCC_COLORF',
           'MVCC_POINTF', '_MVCC_POINTF',
           'MVCC_RECT_INFO', '_MVCC_RECT_INFO',
           'MVCC_CIRCLE_INFO', '_MVCC_CIRCLE_INFO',
           'MVCC_LINES_INFO', '_MVCC_LINES_INFO',
           'MV_OUTPUT_IMAGE_INFO', '_MV_OUTPUT_IMAGE_INFO_',
           'MV_RECONSTRUCT_IMAGE_PARAM', '_MV_RECONSTRUCT_IMAGE_PARAM_',
           'MV_CC_FILE_ACCESS_EX', '_MV_CC_FILE_ACCESS_E', '_MV_SAVE_IMAGE_PARAM_EX3_',
           'MV_SAVE_IMAGE_PARAM_EX3', 'MV_CML_DEVICE_INFO', '_MV_CML_DEVICE_INFO_',
           'MV_CXP_DEVICE_INFO', '_MV_CXP_DEVICE_INFO_', '_MV_XOF_DEVICE_INFO_', 'MV_XOF_DEVICE_INFO',
           'MV_SAVE_IMAGE_TO_FILE_PARAM_EX', '_MV_SAVE_IMAGE_TO_FILE_PARAM_EX_',
           '_MV_PIXEL_CONVERT_PARAM_EX_T_', 'MV_CC_PIXEL_CONVERT_PARAM_EX', '_MV_CC_FRAME_SPEC_INFO_',
           'MV_CC_FRAME_SPEC_INFO', 'MV_CC_HB_DECODE_PARAM',
           '_MV_INTERFACE_INFO_LIST_', 'MV_INTERFACE_INFO_LIST', '_MV_INTERFACE_INFO_', 'MV_INTERFACE_INFO',
           '_MV_CAML_SERIAL_PORT_LIST_', 'MV_CAML_SERIAL_PORT_LIST', '_MV_CAML_SERIAL_PORT_', 'MV_CAML_SERIAL_PORT',
           '_MV_CC_HB_DECODE_PARAM_T_']
