import serial
import struct
import time


class VisionData_t:
    def __init__(self, BPS, TIMEOUT=5, PORT=-1):
        self.PORT = PORT
        self.BPS = BPS
        self.TIMEOUT = TIMEOUT

        # 帧结构参数
        self.BEGIN = 0xA5
        self.END = 0xFF
        self.CmdID = 2

        # 发送的数据 (Tx)
        self.pitch_angle = 0.0
        self.yaw_angle = 0.0
        self.distance = 0.0
        self.center_lock = 0
        self.identify_target = 0
        self.identify_buff = 0

        # 接收的数据 (Rx)
        self.roll = 0.0
        self.pitch = 0.0
        self.speed = 25.0
        self.yaw = 0.0

        self.start_flag = False
        self.uart = self.open_uart(self.PORT, self.BPS, self.TIMEOUT)

    def open_uart(self, port, bps, timeout):
        """自动寻找并打开串口，并清空历史积压数据"""
        uart = None
        if int(port) < 0:
            for i in range(15):
                try:
                    uart = serial.Serial(port=f"/dev/ttyCH341USB{i}", baudrate=bps, timeout=timeout)
                    print(f"成功连接至 /dev/ttyCH341USB{i}")
                    break
                except:
                    pass
            if not uart:
                for i in range(15):
                    try:
                        uart = serial.Serial(port=f"/dev/ttyUSB{i}", baudrate=bps, timeout=timeout)
                        print(f"成功连接至 /dev/ttyUSB{i}")
                        break
                    except:
                        pass
        else:
            try:
                uart = serial.Serial(port=f"/dev/ttyUSB{port}", baudrate=bps, timeout=timeout)
            except Exception as e:
                print(f"打开指定串口失败: {e}")

        if uart and uart.is_open:
            uart.reset_input_buffer()
            uart.reset_output_buffer()
            return uart
        else:
            print("未能找到或打开任何串口设备")
            return None

    def close_uart(self):
        if self.uart and self.uart.is_open:
            try:
                self.uart.close()
                print("串口已安全关闭")
            except Exception as e:
                print(f"关闭UART时发生错误: {e}")

    def send(self):
        if self.uart and self.uart.is_open:
            try:
                data = struct.pack('<BB', self.BEGIN, self.CmdID)
                data += struct.pack('<fff', float(self.pitch_angle), float(self.yaw_angle), float(self.distance))
                data += struct.pack('<BBBB', self.center_lock, self.identify_target, self.identify_buff, self.END)
                self.uart.write(data)
            except Exception as e:
                print(f"发送数据失败: {e}")

    def get(self):
        try:
            if not self.uart or not self.uart.is_open:
                return

            waiting = self.uart.in_waiting
            FRAME_LEN = 19

            # 超量积压直接清空（保留防延迟逻辑）
            if waiting > FRAME_LEN * 3:
                self.uart.reset_input_buffer()
                return

            # 主动滑动窗口寻找帧头
            while self.uart.in_waiting > 0:
                rdata = self.uart.read(1)

                # 如果摸到了包头 0xA5！
                if rdata and rdata[0] == self.BEGIN:
                    data = self.uart.read(18)
                    if len(data) == 18 and data[-1] == self.END:
                        # 成功接收完整一帧，开始解包！
                        self.CmdID = struct.unpack('<B', data[0:1])[0]
                        self.speed = struct.unpack('<f', data[1:5])[0]
                        self.yaw = struct.unpack('<f', data[5:9])[0]
                        self.pitch = struct.unpack('<f', data[9:13])[0]
                        self.roll = struct.unpack('<f', data[13:17])[0]
                        self.callback()  # 成功解包，触发发送函数 self.send()
                        return  # 处理完最新的一帧就退出，把 CPU 释放给上位机主循环
                    else:
                        # 凑不齐 18 个字节，或者帧尾不是 0xFF，说明遇到了残断帧
                        # 丢弃它，等下一轮循环重新找包头
                        return

                # 如果读出来的 1 个字节不是包头 0xA5，while 循环会继续执行 read(1)
                # 相当于把错位的脏数据全部“吞”掉，直到重新对齐！

        except serial.SerialException as e:
            print(f"串口物理断开，尝试重连: {e}")
            self.close_uart()
            time.sleep(0.5)
            self.uart = self.open_uart(self.PORT, self.BPS, self.TIMEOUT)
        except Exception as e:
            print(f"串口读取异常: {e}")

    def set_data(self, target_yaw, dif_pitch, dis, target, is_lock, buff=0):
        self.pitch_angle = dif_pitch
        self.yaw_angle = target_yaw
        self.distance = dis
        self.identify_target = target
        self.center_lock = is_lock
        self.identify_buff = buff

    def start(self):
        print("开始高速接收...")
        self.start_flag = True
        while self.start_flag:
            self.get()

    def stop(self):
        self.start_flag = False

    def callback(self):
        # 调试用：查看接收到的最新姿态
        # print(f"最新姿态 -> Yaw: {self.yaw:.2f}, Pitch: {self.pitch:.2f}")
        self.send()
