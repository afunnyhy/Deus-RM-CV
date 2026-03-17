#!/usr/bin/env python
# -*- coding: utf-8 -*-

import socket
import struct
import time
import math
import threading


class EnemyVisionSender:
    def __init__(self, target_ip: str = "127.0.0.1", target_port: int = 8964):
        """初始化 UDP 发送端"""
        self.target_ip = target_ip
        self.target_port = target_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # --- 数据缓存与线程锁 ---
        self.lock = threading.Lock()
        self.is_detected = False
        self.rel_x = 0.0
        self.rel_y = 0.0
        self.reserved = 0

        # --- 线程控制标志 ---
        self.is_running = False
        self.send_thread = None

        print(f"发送端已初始化，目标地址: {self.target_ip}:{self.target_port}")

    def update_data(self, is_detected: bool, rel_x: float, rel_y: float, reserved: int = 0):
        """
        [函数 1] 修改/更新要发送的信息 (主线程或视觉线程调用)
        """
        with self.lock:  # 加锁，防止数据在更新中途被发送线程读取
            self.is_detected = is_detected
            self.rel_x = rel_x
            self.rel_y = rel_y
            self.reserved = reserved

    def _send_loop(self, hz: float):
        """
        [函数 2] 实时发送函数 (在独立线程中循环运行)
        """
        sleep_interval = 1.0 / hz

        while self.is_running:
            # 1. 加锁读取最新数据
            with self.lock:
                detected_flag = 1 if self.is_detected else 0
                x = self.rel_x
                y = self.rel_y
                r = self.reserved

            # 2. 打包前 12 个字节的数据 (<BBBBff)
            payload = struct.pack("<BBBBff", 0x5A, 0xA5, detected_flag, r, x, y)

            # 3. 计算异或校验和
            checksum = 0
            for byte in payload:
                checksum ^= byte

            # 4. 追加校验和并发送 (共 13 字节)
            packet = payload + struct.pack("<B", checksum)
            try:
                # print("发送数据 是否检测到敌人:", self.is_detected, "相对坐标 X:", x, "Y:", y, "校验和:", checksum)
                self.socket.sendto(packet, (self.target_ip, self.target_port))
            except Exception as e:
                print(f"发送失败: {e}")

            # 5. 控制发送频率
            time.sleep(sleep_interval)

    def start(self, hz: float = 20.0):
        """启动后台发送线程"""
        if self.is_running:
            return

        self.is_running = True
        self.send_thread = threading.Thread(target=self._send_loop, args=(hz,), daemon=True)
        self.send_thread.start()
        print(f"后台发送线程已启动，发送频率: {hz} Hz")

    def stop(self):
        """停止发送线程并关闭 Socket"""
        self.is_running = False
        if self.send_thread:
            self.send_thread.join(timeout=2.0)
        self.socket.close()
        print("发送端已关闭。")


if __name__ == "__main__":
    # 测试环节
    sender = EnemyVisionSender(target_ip="127.0.0.1", target_port=8888)

    # 1. 启动独立线程，以 20Hz (每秒20次) 的频率实时发送当前缓存的数据
    sender.start(hz=20.0)

    try:
        print("开始模拟视觉识别更新... (按 Ctrl+C 停止)")
        counter = 0.0

        # 2. 主线程模拟视觉算法，更新数据的频率和发送频率是完全解耦的
        while True:
            # 模拟算法计算出了新的目标坐标
            sim_detected = True
            sim_x = 5.0 * math.cos(counter)
            sim_y = 5.0 * math.sin(counter)

            # 调用函数修改待发送的信息
            sender.update_data(is_detected=sim_detected, rel_x=sim_x, rel_y=sim_y)
            print(f"[主线程] 更新视觉数据 -> X: {sim_x:5.2f}m, Y: {sim_y:5.2f}m")

            counter += 0.1
            # 模拟视觉算法的耗时 (这里设为 10Hz 更新率，慢于发送线程的 20Hz)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n检测到退出信号。")
    finally:
        # 3. 退出时清理资源
        sender.stop()
