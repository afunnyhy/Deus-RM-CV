import os
import sys
import time
import threading
import multiprocessing as mp

from setting import *
from coord_converter import *
from all_type import *

from camera_get_photo import InitCamera  # 相机类
from detect_armor import ArmorDetector  # 模型推理类
from get_armor_points_cv import armor_getter  # 初始化装甲板检测类
from light_detector import LightDetector  # 导入灯条解算类
from pnp_solver import PnPSolver  # 导入PnP解算类
from armor_chose import TargetSelector  # 导入目标选择类
from pre_armor import Tracker  # 跟踪器类
from ballistic_compensation import BallisticCompensator  # 弹道解算类

from uart import UartCommunication  # 电控通信
from chase_sender import EnemyVisionSender  # 导航通信

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def camera_process(shared_buf, shared_shape, frame_ready):
    print("Camera type:", cameraType, " , ID:", cameraID)
    try:
        camera = InitCamera(cameraType)
        print(cameraID, "finish init")
        fail_count = 0
        while True:
            ret, orig_frame = camera.get_photo()
            if not ret:
                fail_count += 1
                time.sleep(0.01)
                if fail_count > 5:
                    print(
                        "[Camera Process] 连续 5 次读取失败，相机失去响应，进程主动自杀以强制触发操作系统的物理重置...")
                    # 尝试柔性释放资源
                    if hasattr(camera, '__del__'):
                        camera.__del__()
                    time.sleep(0.3)
                    # 强行终止当前进程，内核会自动回收所有被卡死的 C++ 句柄和底层内库状态
                    sys.exit(1)
                continue

            # 成功拿流，清空错误计数
            fail_count = 0

            if camera_flip:
                orig_frame = cv2.flip(orig_frame, -1)

            # 极速内存拷贝：利用零拷贝机制(Zero-Copy)通过内存连续分配更新
            shape = orig_frame.shape
            shared_shape[0], shared_shape[1], shared_shape[2] = shape
            mp_array_np = np.frombuffer(shared_buf, dtype=np.uint8)
            mp_array_np[:orig_frame.nbytes] = orig_frame.ravel()
            frame_ready.set()  # 唤醒推理进程
    except Exception as e:
        print(f"[Camera Process] 相机底层异常/掉线断开: {e}，正在执行进程级重启...")
        sys.exit(1)


def inference_process(shared_buf, shared_shape, frame_ready, state_arr):
    print("Inference process started...")

    if used_yolo:
        print("model:", model_name)
        armor_de = ArmorDetector(model_path, model_name, friend_color)
        print("Armor detector init success")
        print("Troop type:", my_TroopType, " , Friend color:", friend_color)
        print("Is show video:", is_show_video, " , Save video times:", save_video_time)
    else:
        armor_de = armor_getter(friend_color)

    light_pos = LightDetector()  # 灯条解算
    pnp_solver = PnPSolver()
    target_selector = TargetSelector()
    tra = Tracker()
    ballistic_compensation = BallisticCompensator()

    t = time.time()
    time1 = time.time()
    cnt = 0
    last_vision_yaw = 0
    video_time_limit = save_video_time

    # 阻塞等待第一帧用于初始化 VideoWriter
    frame_ready.wait()
    frame_ready.clear()
    shape = (shared_shape[0], shared_shape[1], shared_shape[2])
    size = shape[0] * shape[1] * shape[2]
    orig_frame = np.frombuffer(shared_buf, dtype=np.uint8)[:size].reshape(shape).copy()

    if video_time_limit > 0:
        start_time = time.time()
        output_file = time.strftime("%Y%m%d_%H%M%S") + "_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps_video = 30
        frame_size = (orig_frame.shape[1], orig_frame.shape[0])
        video_writer = cv2.VideoWriter(output_file, fourcc, fps_video, frame_size)
    else:
        start_time = time.time()

    armor = None
    predict_armor = None
    predicted_armor_yaw = 0

    print("Start working...")
    while True:
        # 极速读取跨进程共享状态 (绕过内核锁带来的性能激增)
        # 索引: 0:yaw, 1:pitch, 2:cmd_id, 12:speed
        curr_yaw = state_arr[0]
        curr_pitch = state_arr[1]
        curr_cmd_id = int(state_arr[2])
        curr_speed = float(state_arr[12])

        detected_point = []
        if used_yolo:
            all_detect_armor, out_img = armor_de.detect_armor(orig_frame)
        else:
            ret_flag, all_detect_armor, out_img = armor_de.get_armors_by_img(orig_frame)

        is_find = False

        # 遍历所有检测到的装甲板框，进行灯条提取和PnP解算，得到装甲板中心点的3D坐标
        for detected_armor_box in all_detect_armor:
            if used_yolo:
                ret_detected, detected_armor, out_img = light_pos.extract_light_points(orig_frame, detected_armor_box,
                                                                                       out_img)
            else:
                ret_detected = True
                detected_armor = detected_armor_box

            if ret_detected:
                ret_pnp, armor_pnp, out_img = pnp_solver.get_armor_target(detected_armor, out_img, curr_pitch, curr_yaw)
                if ret_pnp:
                    detected_point.append(armor_pnp)

        if is_show_video:
            cv2.putText(out_img,
                        f"receive yaw:{math.degrees(curr_yaw) :<9.3f} pitch:{math.degrees(curr_pitch):<9.3f} ",
                        (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

        if len(detected_point) > 0:
            armor = target_selector.select_best_target(detected_point)
            if armor is not None:
                if used_predict:
                    found_pos2d = camera2xy(gimbal2camera(rotate_around_y(armor.gimbal_pos, -curr_yaw), curr_pitch))
                else:
                    found_pos2d = camera2xy(gimbal2camera(armor.gimbal_pos, curr_pitch))
                if is_show_video:
                    cv2.circle(out_img, found_pos2d, 11, (0, 200, 200), 4)
                ax, ay, az = armor.gimbal_pos

                if is_show_video:
                    cv2.putText(out_img,
                                f"detecting x:{ax:<9.3f} y:{ay:<9.3f} z:{az:<9.3f} yaw:{math.degrees(armor.yaw):<9.3f}",
                                (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

                is_find = True
                t_n = time.time()

                if tra.state == TracState.LOST:
                    tra.initial(armor)
                    t = t_n
                else:
                    predicted_pos2d = camera2xy(gimbal2camera(armor.gimbal_pos, curr_pitch))
                    if is_show_video:
                        cv2.circle(out_img, predicted_pos2d, 14, (174, 29, 128), 4)
                    dt = t_n - t
                    predict_armor, out_img = tra.update(armor, dt, out_img)
                    t = t_n

                    if predict_armor is not None:
                        predicted_armor_yaw = predict_armor.yaw
        else:
            target_selector.add_empty_entry()

        if not is_find and tra.state != TracState.LOST:
            t_n = time.time()
            dt = t_n - t
            predict_armor, out_img = tra.update(None, dt, out_img)
            t = t_n

        if tra.state == TracState.TRACKING:
            last_vision_yaw = curr_yaw

        if tra.state == TracState.TRACKING or tra.state == TracState.TEMP_LOST:
            angle_pitch = curr_pitch
            if used_predict and predict_armor is not None:
                re_transform_pos = rotate_around_y(predict_armor.gimbal_pos, -curr_yaw)
                predict_armor.gimbal_pos = re_transform_pos
            elif not used_predict and armor is not None:
                predict_armor = armor

            if predict_armor is not None:
                # 用运动云台坐标系计算弹道
                change_angle = ballistic_compensation.calculate_angle(predict_armor.gimbal_pos, curr_speed)
                ax, ay, az = predict_armor.gimbal_pos

                if is_show_video:
                    cv2.putText(out_img,
                                f"predicted x:{ax:<9.3f} y:{ay:<9.3f} z:{az:<9.3f} yaw:{math.degrees(predicted_armor_yaw):<9.3f}",
                                (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

                if az >= 0.1:
                    # 计算发电控的弧度
                    angle_xoz = math.atan2(ax, az) * yaw_buffer_factor
                    angle_yoz = pitch_buffer_factor * (angle_pitch - change_angle)
                    if not send_radian_diff:
                        angle_xoz += curr_yaw
                        angle_yoz += angle_pitch
                    # 弧度归一化 [-pi, pi]
                    angle_xoz = (angle_xoz + math.pi) % (2 * math.pi) - math.pi
                    angle_yoz = (angle_yoz + math.pi) % (2 * math.pi) - math.pi

                    if tra.state == TracState.TEMP_LOST:
                        angle_xoz = angle_xoz - (curr_yaw - last_vision_yaw)

                    # 是否锁上目标判断逻辑
                    miss_yaw = math.radians(miss_yaw_angle)
                    miss_pitch = math.radians(miss_pitch_angle)
                    dy = angle_xoz if send_radian_diff else (angle_xoz - curr_yaw)
                    dp = angle_yoz if send_radian_diff else (angle_yoz - curr_pitch)
                    lock = 0 if abs(dy) > miss_yaw or abs(dp) > miss_pitch else 1

                    # 将计算结果写入共享无锁数组
                    # 索引: 3:cyaw, 4:cpitch, 5:dist, 6:target, 7:lock, 8:buff, 9:nav_det, 10:nav_x, 11:nav_y
                    state_arr[3] = float(angle_xoz)
                    state_arr[4] = float(angle_yoz)
                    state_arr[5] = float(math.sqrt(az * az + ax * ax))
                    state_arr[6] = 1.0
                    state_arr[7] = float(lock)
                    state_arr[8] = 0.0

                    if send_chase:
                        state_arr[9] = 1.0
                        state_arr[10] = float(az)
                        state_arr[11] = float(ax)

                    if is_show_video:
                        cv2.putText(out_img,
                                    f"sending yaw:{math.degrees(angle_xoz):<9.3f} pitch :{math.degrees(angle_yoz):<9.3f} lock:{lock}",
                                    (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        else:
            offset_y, offset_p = (0.0, 0.0) if send_radian_diff else (float(curr_yaw), float(curr_pitch))
            state_arr[3], state_arr[4], state_arr[5] = float(offset_y), float(offset_p), 0.0
            state_arr[6], state_arr[7], state_arr[8] = 0.0, 0.0, 0.0

            if send_chase:
                state_arr[9], state_arr[10], state_arr[11] = 0.0, 0.0, 0.0
        if is_show_video:
            cv2.putText(out_img, f"state:{tra.state},cmd_id:{curr_cmd_id}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)

        if video_time_limit > 0:
            video_writer.write(out_img)

        if is_show_video:
            small_img = cv2.resize(out_img, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
            cv2.imshow("vision output", small_img)
            cv2.waitKey(1)

        cnt += 1
        if cnt == 40:
            fps = 40 / (time.time() - time1)
            time1 = time.time()
            cnt = 0
            print("fps:", round(fps, 3))

        if 0 < video_time_limit < time.time() - start_time:
            video_writer.release()
            cv2.destroyAllWindows()
            print("video write to", output_file, "over")
            video_time_limit = 0

        # 请求下一帧，带超时容错机制
        got_next_frame = False
        while not got_next_frame:
            if frame_ready.wait(timeout=0.25):
                frame_ready.clear()
                shape = (shared_shape[0], shared_shape[1], shared_shape[2])
                size = shape[0] * shape[1] * shape[2]
                orig_frame = np.frombuffer(shared_buf, dtype=np.uint8)[:size].reshape(shape).copy()
                got_next_frame = True
            else:
                print("[Inference Process] 警告：超过 0.25s 未收到相机新帧，重置锁定位和指令...")
                tra.state = TracState.LOST
                curr_yaw = state_arr[0]
                curr_pitch = state_arr[1]
                offset_y, offset_p = (0.0, 0.0) if send_radian_diff else (float(curr_yaw), float(curr_pitch))

                state_arr[3] = float(offset_y)
                state_arr[4] = float(offset_p)
                state_arr[5] = 0.0
                state_arr[6] = 0.0
                state_arr[7] = 0.0
                state_arr[8] = 0.0

                if send_chase:
                    state_arr[9], state_arr[10], state_arr[11] = 0.0, 0.0, 0.0
                # 继续等待


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # 强制以 spawn 方式启动多进程（Jetson / CUDA 必须）

    print("----- Debugging parameters -----")
    print("is_sending_diff:", send_radian_diff)
    print("yaw_buffer_factor:", yaw_buffer_factor)
    print("pitch_buffer_factor:", pitch_buffer_factor)
    print("miss_yaw_angle:", miss_yaw_angle)
    print("miss_pitch_angle:", miss_pitch_angle)

    # 与电控和导航的通信
    vision = UartCommunication(baud_rate)
    print("UART communication initialized. Baud rate:", baud_rate)
    if send_chase:
        sender = EnemyVisionSender(target_ip=send_target_ip, target_port=send_target_port)
        print("Chase sending enabled. Target IP:", send_target_ip, " Target Port:", send_target_port)

    import ctypes

    # ========== 跨进程共享内存区 (彻底优化 IPC 通信) ==========
    # 1. 消除 Queue，使用无锁多进程数组传递图像，免除 Pickle 开销
    max_frame_bytes = 1920 * 1080 * 3  # 最大支持到 1080p
    shared_frame_buf = mp.Array(ctypes.c_uint8, max_frame_bytes, lock=False)
    shared_frame_shape = mp.Array('i', 3, lock=False)
    frame_ready = mp.Event()

    # 2. 消除 13 个带锁 mp.Value，使用无锁连续数组，完全避免内核态内核锁抢占开销
    # [0:yaw, 1:pitch, 2:cmd_id, 3:cyaw, 4:cpitch, 5:dist, 6:target, 7:lock, 8:buff, 9:nav_det, 10:nav_x, 11:nav_y, 12:speed]
    state_arr = mp.Array('d', 13, lock=False)

    # 启动双核双进程：将 Camera 取流，和 推理计算 彻底隔离开，跑满多核并实现 Zero-Copy
    p_camera = mp.Process(target=camera_process, args=(shared_frame_buf, shared_frame_shape, frame_ready))
    p_inference = mp.Process(target=inference_process,
                             args=(shared_frame_buf, shared_frame_shape, frame_ready, state_arr))
    # 设置为守护模式
    p_camera.daemon = True
    p_inference.daemon = True

    p_camera.start()
    p_inference.start()


    def process_monitor_loop(p_cam, buf, shape, evt):
        # 看门狗守护线程
        while True:
            time.sleep(1.0)
            if not p_cam.is_alive():
                print("[看门狗_Monitor] 检测到相机取流进程已死亡/退出，正在操作系统层面彻底重启相机进程(重置句柄)...")
                p_cam = mp.Process(target=camera_process, args=(buf, shape, evt))
                p_cam.daemon = True
                p_cam.start()


    # 启动看门狗线程
    watchdog_thread = threading.Thread(target=process_monitor_loop,
                                       args=(p_camera, shared_frame_buf, shared_frame_shape, frame_ready), daemon=True)
    watchdog_thread.start()


    def comms_sync_loop():
        # 微线程无锁同步：负责在主进程调度串口收发与状态共享
        while True:
            try:
                state_arr[0] = float(vision.yaw)
                state_arr[1] = float(vision.pitch)
                state_arr[2] = float(vision.CmdID)
                state_arr[12] = float(vision.speed)

                # 使用安全转换以防共享内存撕裂造成的 NaN 引发 ValueError
                def safe_int(val):
                    import math
                    return 0 if math.isnan(val) else int(val)

                vision.set_data(state_arr[3], state_arr[4], state_arr[5],
                                safe_int(state_arr[6]), safe_int(state_arr[7]), safe_int(state_arr[8]))

                if send_chase:
                    sender.update_data(bool(state_arr[9]), state_arr[10], state_arr[11])
            except Exception as e:
                print(f"[UART Sync Thread] 警告：数据同步线程异常 ({e})")

            time.sleep(0.005)  # ~200Hz 数据交互帧率


    # 启动主进程通讯同步协程
    sync_thread = threading.Thread(target=comms_sync_loop, daemon=True)
    sync_thread.start()

    if send_chase:
        sender.start(hz=50.0)

    print("主进程载入 UART 循环接管...")
    vision.start()  # 阻塞在读取

    # 防护
    p_camera.join()
    p_inference.join()
