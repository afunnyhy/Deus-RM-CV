import os
import sys
import time
import threading
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

# communication
vision = VisionData_t(baud_rate)
if send_chase:
    sender = EnemyVisionSender(target_ip=send_target_ip, target_port=send_target_port)


def run():
    # 初始化相机类
    print("Camera type:", cameraType, " , ID:", cameraID)
    camera = InitCamera(cameraType)
    print(cameraID, "finish init")
    if used_yolo:
        # 初始化模型推断类
        print("model:", model_name)
        armor_de = ArmorDetector(model_path, model_name, friend_color)
        print("Armor detector init success")
        print("Troop type:", my_TroopType, " , Friend color:", friend_color)
        print("Is show video:", is_show_video, " , Save video times:", save_video_time)
    else:
        # 初始化CV类
        armor_de = armor_getter(friend_color)
    # 初始化灯条解算类
    light_pos = LightDetector()
    # 初始化PnP解算类
    pnp_solver = PnPSolver()
    # 初始化目标选择类
    target_selector = TargetSelector()
    # #初始化预测类
    tra = Tracker()

    t = time.time()  # 初始化时间
    time1 = time.time()
    cnt = 0
    last_vision_yaw = 0

    if save_video_time > 0:
        output_file = time.strftime("%Y%m%d_%H%M%S") + "_output.mp4"  # 导出文件名为时间
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码器（MP4格式）
        fps = 30  # 帧率
        ret, orig_frame = camera.get_photo()
        frame_size = (orig_frame.shape[1], orig_frame.shape[0])  # 视频帧大小（宽度, 高度）
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

    start_time = time.time()
    armor = None
    predict_armor = None
    predicted_armor_yaw = 0

    print("Start working...")
    while True:
        # 读取视频流的一帧
        ret, orig_frame = camera.get_photo()
        if camera_flip:
            orig_frame = cv2.flip(orig_frame, -1)
        if not ret:
            continue

        detected_point = []  # 初始化装甲板中心点结果列表
        if used_yolo:
            all_detect_armor, out_img = armor_de.detect_armor(orig_frame)
        else:
            ret, all_detect_armor, out_img = armor_de.get_armors_by_img(orig_frame)
        # print(tra.state)
        is_find = False

        for detected_armor_box in all_detect_armor:  # 遍历所有检测到的装甲板
            if used_yolo:
                # 提取灯条角点
                ret_detected, detected_armor, out_img = light_pos.extract_light_points(orig_frame, detected_armor_box,
                                                                                       out_img)
            else:
                ret_detected = True
                detected_armor = detected_armor_box
            if ret_detected:  # 如果灯条角点提取成功
                # 计算装甲板中心3D坐标
                ret_pnp, armor, out_img = pnp_solver.get_armor_target(detected_armor, out_img, vision.pitch, vision.yaw)
                if ret_pnp:  # 如果PnP解算成功, 将装甲板中心点添加到结果列表
                    detected_point.append(armor)

        cv2.putText(out_img,
                    f"receive yaw:{vision.yaw * 180 / math.pi:<9.3f} pitch:{vision.pitch * 180 / math.pi:<9.3f} ",
                    (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

        if len(detected_point) > 0:
            # 选择最佳目标
            armor = target_selector.select_best_target(detected_point)
            if armor is None:
                continue
            # 标记显示识别到的装甲板
            if used_predict:
                found_pos2d = camera2xy(gimbal2camera(rotate_around_y(armor.gimbal_pos, -vision.yaw), vision.pitch))
            else:
                found_pos2d = camera2xy(gimbal2camera(armor.gimbal_pos, vision.pitch))
            cv2.circle(out_img, found_pos2d, 11, (0, 200, 200), 4)
            ax, ay, az = armor.gimbal_pos

            cv2.putText(out_img,
                        f"detecting x:{ax:<9.3f} y:{ay:<9.3f} z:{az:<9.3f} yaw:{armor.yaw * 180.0 / math.pi:<9.3f}",
                        (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

            is_find = True
            # print(armor)
            t_n = time.time()

            # 初始化跟踪器

            if tra.state == TracState.LOST:
                tra.initial(armor)
                t = t_n
                continue
            # ///////////////////////////
            predicted_pos2d = camera2xy(gimbal2camera(armor.gimbal_pos, vision.pitch))
            cv2.circle(out_img, predicted_pos2d, 14, (174, 29, 128), 4)
            # ///////////////////
            # 更新跟踪器
            dt = t_n - t
            predict_armor, out_img = tra.update(armor, dt, out_img)
            t = t_n

            # 使用预测结果
            if predict_armor is not None:
                predicted_armor_yaw = predict_armor.yaw
                # update_3d_fig(current)
            else:
                continue
        else:
            target_selector.add_empty_entry()  # 更新历史记录

        # 处理目标丢失的情况
        if not is_find and tra.state != TracState.LOST:
            t_n = time.time()
            dt = t_n - t
            predict_armor, out_img = tra.update(None, dt, out_img)
            t = t_n

        if tra.state == TracState.TRACKING:
            last_vision_yaw = vision.yaw

        # 处理跟踪状态下的目标
        if tra.state == TracState.TRACKING or tra.state == TracState.TEMP_LOST:
            angle_pitch = vision.pitch
            if used_predict:
                re_transform_pos = rotate_around_y(predict_armor.gimbal_pos, -vision.yaw)
                predict_armor.gimbal_pos = re_transform_pos
            else:
                predict_armor = armor
            # 用运动云台坐标系计算弹道
            change_angle = ballistic_compensation(predict_armor.gimbal_pos)
            ax, ay, az = predict_armor.gimbal_pos

            cv2.putText(out_img,
                        f"predicted x:{ax:<9.3f} y:{ay:<9.3f} z:{az:<9.3f} yaw:{predicted_armor_yaw * 180.0 / math.pi:<9.3f}",
                        (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 0), 2)

            if az < 0.1:  # 距离过近忽略
                continue

            # 计算发电控的弧度
            angle_xoz = math.atan2(ax, az) * yaw_buffer_factor
            angle_yoz = pitch_buffer_factor * (angle_pitch - change_angle)
            if not send_radian_diff:
                angle_xoz += vision.yaw
                angle_yoz += angle_pitch
            # 弧度归一化 [-pi, pi]
            angle_xoz = (angle_xoz + math.pi) % (2 * math.pi) - math.pi
            angle_yoz = (angle_yoz + math.pi) % (2 * math.pi) - math.pi

            if tra.state == TracState.TEMP_LOST:
                angle_xoz = angle_xoz - (vision.yaw - last_vision_yaw)

            # 是否锁上目标判断逻辑
            miss_yaw = miss_yaw_angle * math.pi / 180.0 / yaw_buffer_factor
            miss_pitch = miss_pitch_angle * math.pi / 180.0 / pitch_buffer_factor
            dy = angle_xoz if send_radian_diff else (angle_xoz - vision.yaw)
            dp = angle_yoz if send_radian_diff else (angle_yoz - vision.pitch)
            lock = 0 if abs(dy) > miss_yaw or abs(dp) > miss_pitch else 1

            vision.set_data(angle_xoz, angle_yoz, math.sqrt(az * az + ax * ax), 1, lock)  # 发给电控
            if send_chase:
                sender.update_data(is_detected=True, rel_x=az, rel_y=ax)  # 发给导航

            # 标记显示预测后的装甲板
            # predicted_pos2d = camera2xy(gimbal2camera(armor.gimbal_pos, vision.pitch))
            # cv2.circle(out_img, predicted_pos2d, 14, (174, 29, 128), 4)

            cv2.putText(out_img,
                        f"sending yaw:{angle_xoz * 180 / math.pi:<9.3f} pitch :{angle_yoz * 180 / math.pi:<9.3f} lock:{lock}",
                        (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            # print(f"yaw旋转到{angle_xoz * 180 / math.pi}°,pitch旋转{angle_yoz * 180 / math.pi}°")
            # vision.send()
        else:
            offset = (0, 0) if send_radian_diff else (vision.yaw, vision.pitch)
            vision.set_data(*offset, 0, 0, 0)  # 发给电控
            if send_chase:
                sender.update_data(is_detected=False, rel_x=0, rel_y=0)  # 发给导航

        # cv2.putText(out_img,
        #             f"received pitch:{(vision.pitch * 180 / math.pi) if vision.pitch is not None else 0:<9.3f} yaw:{(vision.yaw * 180 / math.pi) if vision.yaw is not None else 0:<9.3f} ",
        #             (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        cv2.putText(out_img, f"state:{tra.state},cmd_id:{vision.CmdID}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
        if save_video_time > 0:
            video_writer.write(out_img)

        # 显示图像和FPS计算
        if is_show_video:
            small_img = cv2.resize(out_img, (0, 0), fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
            cv2.imshow("vision output", small_img)
            cv2.waitKey(1)
        cnt += 1
        if cnt == 25:
            fps = 25 / (time.time() - time1)
            time1 = time.time()
            cnt = 0
            print("fps:", fps)

        if 0 < save_video_time < time.time() - start_time:
            video_writer.release()
            cv2.destroyAllWindows()
            print("video write to", output_file, "over")


if __name__ == "__main__":
    communication_control = threading.Thread(target=vision.start)
    detect = threading.Thread(target=run)
    communication_control.start()
    detect.start()
    if send_chase:
        sender.start(hz=50.0)
