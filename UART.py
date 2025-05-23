import serial  # 串口
import struct  # 处理二进制数据打包为C语言中的结构体
import time


class VisionData_t:
    def __init__(self, PORT, BPS, TIMEOUT):
        self.PORT = PORT  # 串口号
        self.BPS = BPS  # 波特率
        self.TIMEOUT = TIMEOUT  # 超时时间

        self.uart = self.open_uart(self.PORT, self.BPS, self.TIMEOUT)  # 打开串口
        self.start_flag = False  # 开始标志
        # 头
        self.BEGIN = 0xA5  # 开始字节
        self.CmdID = 2  # 命令ID
        # 发送的数据
        self.pitch_angle = 0  # 俯仰角
        self.yaw_angle = 0  # 偏航角(水平角)
        self.distance = 0
        self.centre_lock = 0
        self.identify_target = 0
        self.identify_buff = 0
        # 接收的数据
        self.roll = 0  # 横滚角
        self.pitch = 0  # 俯仰角
        self.speed = 25  # 弹速
        self.yaw = 0  # 偏航角
        # 尾
        self.END = 0xFF

    def send(self):
        if self.uart and self.uart.is_open:
            data = struct.pack('BB', self.BEGIN, self.CmdID)
            data = data + struct.pack('fff', float(self.pitch_angle), float(self.yaw_angle), float(self.distance))
            data = data + struct.pack('BBBB', self.centre_lock, self.identify_target, self.identify_buff, self.END)
            self.uart.write(data)
        else:
            # print("UART未打开或已关闭")
            pass

    def get(self):
        try:
            if not self.uart or not self.uart.is_open:
                # print("UART未打开或已关闭")
                time.sleep(0.01)
                return

            max_attempts = 10  # 设置最大尝试次数
            attempts = 0

            while attempts < max_attempts:
                time.sleep(0.001)
                rdata = self.uart.read(1)

                if len(rdata) == 0 or rdata[0] != self.BEGIN:
                    print("无效的start字节")
                    continue

                data = self.uart.read(18)

                if len(data) != 18:
                    print("接收到的数据不完整")
                    continue

                if data[-1] != self.END:
                    print("无效的结束字节")
                    continue
                self.CmdID = struct.unpack('B', data[0:1])[0]
                self.speed = struct.unpack('f', data[1:5])[0]
                self.yaw = struct.unpack('f', data[5:9])[0]
                self.pitch = struct.unpack('f', data[9:13])[0]
                self.roll = struct.unpack('f', data[13:-1])[0]
                # print("接受到的数据:", self.CmdID, self.speed, self.yaw, self.pitch, self.roll)
                self.callback()
                break
        except Exception as e:
            print(e)
            self.close_uart()
            if not self.uart:
                time.sleep(1)
                self.uart = self.open_uart(self.PORT, self.BPS, self.TIMEOUT)

    def set_data(self, target_yaw, dif_pitch, dis, target, is_lock, buff=0):
        self.pitch_angle = dif_pitch
        self.yaw_angle = target_yaw
        self.distance = dis
        self.identify_target = target
        self.center_lock = is_lock
        self.identify_buff = buff

    def start(self):
        print("start")
        self.start_flag = True
        while self.start_flag:
            self.get()

    def stop(self):
        self.start_flag = False

    def callback(self):
        # 处理函数
        # print("接收到的数据:", self.CmdID, self.speed, self.yaw, self.pitch, self.roll)
        self.send()
        # 可以在这里添加其他处理逻辑
        pass

    # 打开端口
    def open_uart(self, port, bps, timeout):
        if int(port) < 0:
            for i in range(1000):
                try:
                    uart = serial.Serial(
                        port="/dev/ttyUSB" + str(i),
                        baudrate=bps,
                        timeout=timeout,
                        parity=serial.PARITY_NONE,
                        stopbits=1
                    )
                    opened = i
                    return uart
                except:
                    pass
        else:
            uart = serial.Serial(
                port="/dev/ttyUSB" + str(port),
                baudrate=bps,
                timeout=timeout,
                parity=serial.PARITY_NONE,
                stopbits=1
            )
            return uart
        print("open uart failed")

    def close_uart(self, uart):
        try:
            uart.close()
        except Exception as e:
            print("关闭UART时发生错误")
