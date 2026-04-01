import serial
import struct
import time


class UartCommunication:
    def __init__(self, BPS, TIMEOUT=5, PORT=-1):
        self.PORT = PORT
        self.BPS = BPS
        self.TIMEOUT = TIMEOUT

        # 帧结构参数
        self.BEGIN = 0xA5
        self.END = 0xFF
        self.CmdID = 2

        # 发送的数据 (Tx)
        self.state = 2  # 当前程序运行的状态，0表示非正常如相机断线1表示正常
        self.pitch_angle = 0.0
        self.yaw_angle = 0.0
        self.distance = 0.0
        self.center_lock = 0
        self.identify_target = 0
        self.identify_buff = 0

        # 接收的数据 (Rx)
        self.roll = 0.0
        self.pitch = 0.0
        self.speed = 0.0
        self.yaw = 0.0

        self.start_flag = False
        self._buffer = bytearray()  # 声明字节缓冲区

        self.uart = self.open_uart(self.PORT, self.BPS, self.TIMEOUT)

    @staticmethod
    def open_uart(port, bps, timeout):
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
                data = struct.pack('<BB', self.BEGIN, self.state)
                data += struct.pack('<fff', float(self.pitch_angle), float(self.yaw_angle), float(self.distance))
                data += struct.pack('<BBBB', self.center_lock, self.identify_target, self.identify_buff, self.END)
                self.uart.write(data)
            except Exception as e:
                print(f"发送数据失败: {e}")

    def get(self):
        try:
            if not self.uart or not self.uart.is_open:
                time.sleep(0.01)
                return

            waiting = self.uart.in_waiting
            if waiting > 0:
                # 一口气读取底层全部就绪数据并追加到内存缓冲区
                self._buffer.extend(self.uart.read(waiting))
            else:
                # 如果没有新数据，主动休眠极短时间，避免 while 循环跑满 CPU 单核
                time.sleep(0.001)
                return

            FRAME_LEN = 19
            # 防止异常延迟：如果积压了过多的数据包，截断前面的旧数据，只保留最近的约10帧 (190字节)
            if len(self._buffer) > FRAME_LEN * 10:
                self._buffer = self._buffer[-FRAME_LEN * 10:]

            # 如果当前缓冲区数据还不够一帧长度，直接返回，等下一轮 get() 补齐
            if len(self._buffer) < FRAME_LEN:
                return

            # 遍历缓冲区，解析所有完整的帧
            processed_any = False
            while len(self._buffer) >= FRAME_LEN:
                if self._buffer[0] == self.BEGIN:
                    if self._buffer[FRAME_LEN - 1] == self.END:
                        # 找到完整的一帧
                        frame_data = self._buffer[:FRAME_LEN]
                        self._buffer = self._buffer[FRAME_LEN:]

                        data = frame_data[1:]  # 去掉BEGIN字节，对应后面的解包逻辑
                        self.CmdID = struct.unpack('<B', data[0:1])[0]
                        self.speed = struct.unpack('<f', data[1:5])[0]
                        self.yaw = struct.unpack('<f', data[5:9])[0]
                        self.pitch = struct.unpack('<f', data[9:13])[0]
                        self.roll = struct.unpack('<f', data[13:17])[0]

                        processed_any = True
                    else:
                        # 包头是对的，但是包尾不对，发生错位
                        self._buffer = self._buffer[1:]  # 跳过这个包头，继续向后找下一个包头
                else:
                    # 不是包头，继续向后找脏数据
                    self._buffer = self._buffer[1:]

            if processed_any:
                self.callback()  # 仅触发一次发送函数，确保回应给电控的是最新鲜的 CmdID 序列号

        except serial.SerialException as e:
            print(f"串口物理断开，尝试重连: {e}")
            self.close_uart()
            time.sleep(0.5)
            self.uart = self.open_uart(self.PORT, self.BPS, self.TIMEOUT)
        except Exception as e:
            print(f"串口读取异常: {e}")

    def set_data(self, target_yaw, dif_pitch, dis, target, is_lock, state, buff=0):
        self.pitch_angle = dif_pitch
        self.yaw_angle = target_yaw
        self.distance = dis
        self.identify_target = target
        self.center_lock = is_lock
        self.identify_buff = buff
        self.state = state

    def start(self):
        self.start_flag = True
        while self.start_flag:
            self.get()

    def stop(self):
        self.start_flag = False

    def callback(self):
        self.send()
