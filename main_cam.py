import os
import sys
import time
import threading
import multiprocessing as mp
import queue

from setting import *
from all_function import *
from all_type import *

from camera_get_photo import InitCamera  # 相机类
from detect_armor import ArmorDetector  # 模型推理类
from get_armor_points_cv import armor_getter  # 初始化装甲板检测类
from light_detector import LightDetector  # 导入灯条解算类
from pnp_solver import PnPSolver  # 导入PnP解算类
from armor_chose import TargetSelector  # 导入目标选择类
from pre_armor import Tracker  # 跟踪器类

from UART import VisionData_t  # 电控通信
from chase_sender import EnemyVisionSender  # 导航通信

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def camera_process(frame_queue):
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
                if fail_count > 15:
                    print(
                        "[Camera Process] 连续 15 次读取失败，相机失去响应，进程主动自杀以强制触发操作系统的物理重置...")
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

            # 维持 maxsize=1 的队列，保证推理时拿到的是最新的一帧（零延迟丢帧机制）
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Exception:
                    pass
            frame_queue.put(orig_frame)
    except Exception as e:
        print(f"[Camera Process] 相机底层异常/掉线断开: {e}，正在执行进程级重启...")
        sys.exit(1)


def inference_process(frame_queue, shared_yaw, shared_pitch, shared_cmd_id,
                      cmd_yaw, cmd_pitch, cmd_dist, cmd_target, cmd_lock, cmd_buff,
                      nav_detected, nav_x, nav_y):
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

    t = time.time()
    time1 = time.time()
    cnt = 0
    last_vision_yaw = 0
    video_time_limit = save_video_time

    # 阻塞等待第一帧用于初始化 VideoWriter
    orig_frame = frame_queue.get()

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
        # 极速读取跨进程共享的状态变量 (UI和解算需要)
        curr_yaw = shared_yaw.value
        curr_pitch = shared_pitch.value
        curr_cmd_id = shared_cmd_id.value

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
                        f"receive yaw:{curr_yaw * 180 / math.pi:<9.3f} pitch:{curr_pitch * 180 / math.pi:<9.3f} ",
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
                                f"detecting x:{ax:<9.3f} y:{ay:<9.3f} z:{az:<9.3f} yaw:{armor.yaw * 180.0 / math.pi:<9.3f}",
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
                change_angle = ballistic_compensation(predict_armor.gimbal_pos)
                ax, ay, az = predict_armor.gimbal_pos

                if is_show_video:
                    cv2.putText(out_img,
                                f"predicted x:{ax:<9.3f} y:{ay:<9.3f} z:{az:<9.3f} yaw:{predicted_armor_yaw * 180.0 / math.pi:<9.3f}",
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
                    miss_yaw = miss_yaw_angle * math.pi / 180.0 / yaw_buffer_factor
                    miss_pitch = miss_pitch_angle * math.pi / 180.0 / pitch_buffer_factor
                    dy = angle_xoz if send_radian_diff else (angle_xoz - curr_yaw)
                    dp = angle_yoz if send_radian_diff else (angle_yoz - curr_pitch)
                    lock = 0 if abs(dy) > miss_yaw or abs(dp) > miss_pitch else 1

                    # 将计算结果写入跨进程共享变量，触发发包
                    cmd_yaw.value = angle_xoz
                    cmd_pitch.value = angle_yoz
                    cmd_dist.value = float(math.sqrt(az * az + ax * ax))
                    cmd_target.value = 1
                    cmd_lock.value = lock
                    cmd_buff.value = 0

                    if send_chase:
                        nav_detected.value = 1
                        nav_x.value = float(az)
                        nav_y.value = float(ax)

                    if is_show_video:
                        cv2.putText(out_img,
                                    f"sending yaw:{angle_xoz * 180 / math.pi:<9.3f} pitch :{angle_yoz * 180 / math.pi:<9.3f} lock:{lock}",
                                    (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        else:
            offset_y, offset_p = (0.0, 0.0) if send_radian_diff else (float(curr_yaw), float(curr_pitch))
            cmd_yaw.value = offset_y
            cmd_pitch.value = offset_p
            cmd_dist.value = 0.0
            cmd_target.value = 0
            cmd_lock.value = 0
            cmd_buff.value = 0

            if send_chase:
                nav_detected.value = 0
                nav_x.value = 0.0
                nav_y.value = 0.0
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

        # 请求下一帧，无缝衔接，带超时容错机制（防止相机死掉导致死等）
        got_next_frame = False
        while not got_next_frame:
            try:
                orig_frame = frame_queue.get(timeout=0.5)
                got_next_frame = True
            except queue.Empty:
                print("[Inference Process] 警告：超过 0.5s 未收到相机新帧，重置锁定位和指令...")
                # 让机器人进入安全/未识别状态，重置所有发往电控的跨进程变量，停止跟随
                tra.state = TracState.LOST
                cmd_yaw.value, cmd_pitch.value, cmd_dist.value = 0.0, 0.0, 0.0
                cmd_target.value, cmd_lock.value, cmd_buff.value = 0, 0, 0

                if send_chase:
                    nav_detected.value = 0
                    nav_x.value, nav_y.value = 0.0, 0.0
                # 让循环继续堵在等待直到相机恢复


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # 强制以 spawn 方式启动多进程（Jetson / CUDA 必须）

    # 与电控和导航的通信
    vision = VisionData_t(baud_rate)
    if send_chase:
        sender = EnemyVisionSender(target_ip=send_target_ip, target_port=send_target_port)
        print("Chase sending enabled. Target IP:", send_target_ip, " Target Port:", send_target_port)

    # ========== 跨进程共享内存区 ==========
    # 图像队列（保证零延迟）
    frame_queue = mp.Queue(maxsize=1)

    # UART接收状态 -> 推理进程
    shared_yaw = mp.Value('d', 0.0)
    shared_pitch = mp.Value('d', 0.0)
    shared_cmd_id = mp.Value('i', 0)

    # 推理进程 -> UART发送指令
    cmd_yaw = mp.Value('d', 0.0)
    cmd_pitch = mp.Value('d', 0.0)
    cmd_dist = mp.Value('d', 0.0)
    cmd_target = mp.Value('i', 0)
    cmd_lock = mp.Value('i', 0)
    cmd_buff = mp.Value('i', 0)

    # 推理进程 -> 导航发送指令
    nav_detected = mp.Value('i', 0)
    nav_x = mp.Value('d', 0.0)
    nav_y = mp.Value('d', 0.0)

    # 启动双核双进程：将 Camera 取流，和 推理计算 彻底隔离开，打破 GIL，跑满多核
    p_camera = mp.Process(target=camera_process, args=(frame_queue,))
    p_inference = mp.Process(target=inference_process,
                             args=(frame_queue, shared_yaw, shared_pitch, shared_cmd_id,
                                   cmd_yaw, cmd_pitch, cmd_dist, cmd_target, cmd_lock, cmd_buff,
                                   nav_detected, nav_x, nav_y))
    # 设置为守护模式
    p_camera.daemon = True
    p_inference.daemon = True

    p_camera.start()
    p_inference.start()


    def process_monitor_loop(p_cam, f_queue):
        # 看门狗守护线程：专门监视相机进程存活状态，如果进程死亡则重新派生新生进程
        while True:
            time.sleep(1.0)
            if not p_cam.is_alive():
                print("[看门狗_Monitor] 检测到相机取流进程已死亡/退出，正在操作系统层面彻底重启相机进程(重置句柄)...")
                p_cam = mp.Process(target=camera_process, args=(f_queue,))
                p_cam.daemon = True
                p_cam.start()


    # 启动看门狗线程
    watchdog_thread = threading.Thread(target=process_monitor_loop, args=(p_camera, frame_queue), daemon=True)
    watchdog_thread.start()


    def comms_sync_loop():
        # 微线程极速同步：负责将 UART 串口数据搬运进跨进程 Value
        while True:
            shared_yaw.value = vision.yaw
            shared_pitch.value = vision.pitch
            shared_cmd_id.value = vision.CmdID

            vision.set_data(cmd_yaw.value, cmd_pitch.value, cmd_dist.value,
                            cmd_target.value, cmd_lock.value, cmd_buff.value)

            if send_chase:
                sender.update_data(bool(nav_detected.value), nav_x.value, nav_y.value)

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
