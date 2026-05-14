#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
from lerobot.hand_eye_calibration.configs import load_intrinsics, load_extrinsics, CharucoBoardConfig
from lerobot.hand_eye_calibration.charuco_detector import CharucoDetector
from robot_interface import RobotInterface

def get_reprojection_error(frame, T_marker_to_cam, detector, K, D):
    """计算当前画面的像素级重投影误差 (兼容所有 OpenCV 版本)"""
    try:
        R = T_marker_to_cam[:3, :3]
        tvec = T_marker_to_cam[:3, 3].reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)

        cv_board = detector.board
        c_corners = None
        c_ids = None

        if hasattr(cv2.aruco, 'CharucoDetector'):
            charuco_detector = cv2.aruco.CharucoDetector(cv_board)
            c_corners, c_ids, _, _ = charuco_detector.detectBoard(frame)
        elif hasattr(cv2.aruco, 'interpolateCornersCharuco'):
            if hasattr(cv2.aruco, 'ArucoDetector'):
                aruco_det = cv2.aruco.ArucoDetector(cv_board.getDictionary(), cv2.aruco.DetectorParameters())
                m_corners, m_ids, _ = aruco_det.detectMarkers(frame)
            else:
                m_corners, m_ids, _ = cv2.aruco.detectMarkers(frame, cv_board.dictionary)

            if m_ids is not None and len(m_ids) > 0:
                _, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(m_corners, m_ids, frame, cv_board)
        else:
            print("  [DEBUG]: 你的 OpenCV 版本极其特殊，找不到 ChArUco 提取接口！")
            return None

        if c_corners is not None and c_ids is not None and len(c_corners) >= 4:
            if hasattr(cv_board, 'getChessboardCorners'):
                all_obj_pts = cv_board.getChessboardCorners()
            else:
                all_obj_pts = cv_board.chessboardCorners

            obj_points = all_obj_pts[c_ids.flatten()]
            proj_pts, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)

            errors = np.linalg.norm(c_corners.reshape(-1, 2) - proj_pts.reshape(-1, 2), axis=1)
            return float(np.mean(errors))
            
    except Exception as e:
        print(f"  [DEBUG-底层执行报错]: {e}")
        return None
    return None

def main():
    K, D, w, h = load_intrinsics(Path("./outputs/camera_intrinsics.json"))
    R_cam2ee, t_cam2ee, _ = load_extrinsics(Path("./outputs/hand_eye_extrinsics.json"))
    
    T_cam_to_ee = np.eye(4)
    T_cam_to_ee[:3, :3] = R_cam2ee
    T_cam_to_ee[:3, 3] = t_cam2ee.flatten()

    board = CharucoBoardConfig.standard_5x7_6x6()
    detector = CharucoDetector(board, K, D)
    robot = RobotInterface(robot_type="so100_follower", port="/dev/ttyACM0")
    robot.connect()
    robot.disable_torque() 
    
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    recorded_points = []
    recorded_errors = [] 
    font_scale = max(0.6, w / 800.0)

    # 🚀 提取标定板的物理 3D 属性，用于查字典算坐标
    if hasattr(detector.board, 'getChessboardCorners'):
        board_3d_corners = detector.board.getChessboardCorners()
    else:
        board_3d_corners = detector.board.chessboardCorners
    max_id = len(board_3d_corners) - 1

    # 🚀 交互状态变量
    input_mode = False
    target_id_str = ""

    print("\n" + "═"*50)
    print("🎯 手眼外参验证工具 (交互打靶版)")
    print("指令: [g] 输入目标ID | [c] 采集 | [s] 分析 | [r] 重置 | [q] 退出")
    print("═"*50)

    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        detection = detector.detect_and_pose(frame)
        display_raw = detector.draw_results(frame, detection) if detection else frame.copy()
        
        cur_x = cur_y = cur_z = 0.0
        px_err = None 

        if detection:
            T_marker_to_cam = detection["T_marker_to_cam"]
            T_base_to_ee = robot.get_ee_pose(robot.get_joint_positions())
            T_base_to_marker = T_base_to_ee @ T_cam_to_ee @ T_marker_to_cam
            cur_x, cur_y, cur_z = T_base_to_marker[:3, 3] * 1000
            px_err = get_reprojection_error(frame, T_marker_to_cam, detector, K, D)

        display_rotated = display_raw.copy()
        cv2.putText(display_rotated, f"Samples: {len(recorded_points)}", (20, 30), 1, font_scale, (0,255,255), 2)
        
        if detection:
            cv2.putText(display_rotated, f"LIVE (Origin): {cur_x:.1f}, {cur_y:.1f}, {cur_z:.1f}", (20, 60), 1, font_scale, (0,255,0), 2)
            if px_err is not None:
                color = (0, 255, 0) if px_err < 1.0 else (0, 0, 255)
                cv2.putText(display_rotated, f"ERR: {px_err:.2f} px", (20, 90), 1, font_scale, color, 2)

        # === 交互式 UI 绘制层 ===
        if input_mode:
            # 画一个黑底提示框
            cv2.rectangle(display_rotated, (10, h-80), (500, h-10), (0, 0, 0), -1)
            cv2.putText(display_rotated, f"Target Corner ID (0-{max_id}): {target_id_str}_", 
                        (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_rotated, "Press [Enter] to GO, [Esc] to Cancel", 
                        (20, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(display_rotated, "Press [g] to target specific ID", 
                        (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Hand-Eye Verification", display_rotated)
        
        # === 键盘状态机逻辑 ===
        key = cv2.waitKey(10) & 0xFF

        if input_mode:
            if ord('0') <= key <= ord('9'):
                target_id_str += chr(key)
            elif key == 8 or key == 127:   # Backspace 退格键
                target_id_str = target_id_str[:-1]
            elif key == 27:                # Esc 键取消
                input_mode = False
                target_id_str = ""
                print("🚫 已取消目标输入。")
            elif key == 13 or key == 10:   # Enter 确认键
                input_mode = False
                if not target_id_str.isdigit():
                    print("❌ 输入无效！")
                    continue
                    
                tid = int(target_id_str)
                if tid < 0 or tid > max_id:
                    print(f"❌ ID 超出范围！当前标定板 ID 范围: 0 ~ {max_id}")
                    continue
                    
                if not detection:
                    print("❌ 当前画面未检测到标定板，无法定位！")
                    continue

                print(f"🔍 正在计算角点 ID: {tid} 的空间物理坐标...")
                
                # 🚀 核心数学计算：求指定 ID 的绝对坐标
                # 1. 强制平铺张量，彻底屏蔽 OpenCV 4.6 与 4.7+ 版本的维度差异
                flat_corners = np.array(board_3d_corners).reshape(-1, 3)
                
                # 2. 拿到安全的一维数组 [X, Y, Z]
                local_pt = flat_corners[tid] 
                
                # 3. 补成齐次坐标 [X, Y, Z, 1.0]
                local_pt_h = np.array([local_pt[0], local_pt[1], local_pt[2], 1.0])
                
                # 4. 终极连乘：局部 -> 相机 -> 末端 -> 基座
                T_marker_to_cam_live = detection["T_marker_to_cam"]
                T_base_to_ee_live = robot.get_ee_pose(robot.get_joint_positions())
                target_base_pt = T_base_to_ee_live @ T_cam_to_ee @ T_marker_to_cam_live @ local_pt_h
                
                # 5. 提取 XYZ 并转换为毫米 (X和Y听视觉的，去找目标点)
                target_x, target_y, visual_z = target_base_pt[:3] * 1000
                
                # 🛡️ 物理桌面绝对基准 (你刚才实测出来的数据！)
                TABLE_Z = 5.67 
                
                # 悬停安全高度：永远基于真实的物理桌面向上悬空 40mm
                safe_z = TABLE_Z + 40.0 
                
                print(f"🎯 视觉锁定XY: X={target_x:.1f}, Y={target_y:.1f}")
                print(f"🛡️ 启用绝对安全高度: Z={safe_z:.1f} mm (物理桌面+40mm)")
                
                
                # 触发移动！
                if hasattr(robot, 'move_to_xyz'):
                    robot._bus.enable_torque()
                    success = robot.move_to_xyz(target_x, target_y, safe_z)
                    if success:
                        print(f"✅ 机械臂已精准悬停在角点 {tid} 正上方 40mm 处！")
                else:
                    print("❌ 警告：在 robot_interface.py 中未检测到 move_to_xyz 接口，无法移动！")
                
                target_id_str = "" # 清空备用

        else:
            # === 常规游离状态指令 ===
            if key == ord('g'):
                input_mode = True
                target_id_str = ""
            elif key == ord('c'):
                if detection:
                    recorded_points.append([cur_x, cur_y, cur_z])
                    if px_err is not None:
                        recorded_errors.append(px_err)
                        err_str = f"{px_err:.2f} px"
                    else:
                        err_str = "未能计算 (N/A)"
                    print(f"✅ 已记录第 {len(recorded_points)} 组坐标: [{cur_x:.1f}, {cur_y:.1f}, {cur_z:.1f}] (当前误差: {err_str})")
                else:
                    print("❌ 未检测到标定板，无法采集！")
            elif key == ord('s'):
                if len(recorded_points) < 2:
                    print("⚠️  数据太少（至少需要2个点），无法分析！")
                    continue
                pts = np.array(recorded_points)
                ranges = np.ptp(pts, axis=0) 
                stds = np.std(pts, axis=0)
                avg_pos = np.mean(pts, axis=0)
                max_range = np.max(ranges)

                print("\n" + "📊 验证分析报告 " + "═"*30)
                print("【1. 物理坐标稳定性 (跳动越小越好)】")
                print(f"   ► 估算中心 (平均): X={avg_pos[0]:.1f}, Y={avg_pos[1]:.1f}, Z={avg_pos[2]:.1f} mm")
                print(f"   ► 最大跳动 (Range): X:{ranges[0]:.2f}, Y:{ranges[1]:.2f}, Z:{ranges[2]:.2f} mm")
                print(f"   ► 离散程度 (STD):   X:{stds[0]:.2f}, Y:{stds[1]:.2f}, Z:{stds[2]:.2f}")
                
                print("\n【2. 视觉底层精度 (平均误差越小越好)】")
                if len(recorded_errors) > 0:
                    avg_px_err = sum(recorded_errors) / len(recorded_errors)
                    print(f"   ► 重投影平均误差: {avg_px_err:.2f} px")
                else:
                    print("   ► 重投影平均误差: [数据缺失] 底层算法兼容性报错。")

                print("\n【3. 综合评定】")
                if max_range < 5.0:
                    print("   🏆 完美！外参极度精准。")
                elif max_range < 15.0:
                    print("   ✅ 合格。符合普通抓取要求。")
                else:
                    print(f"   ❌ 不及格！最大坐标跳动达到了 {max_range:.2f} mm，远远超过抓取容忍度。")
                print("═"*45 + "\n")
            elif key == ord('r'):
                recorded_points.clear()
                recorded_errors.clear()
                print("🔄 数据已重置。")
            elif key == ord('h') or key == ord('H'):
                print("🏠 收到回零指令！正在平滑返回初始待机位...")
                
                # 刚才通过示教获取的绝对安全姿态
                HOME_ANGLES = {
                    'shoulder_pan': -10.725, 
                    'shoulder_lift': -98.197, 
                    'elbow_flex': 30.197, 
                    'wrist_flex': 107.340, 
                    'wrist_roll': -7.252, 
                    'gripper': -62.945
                }
                
                # 1. 唤醒电机发力
                robot._bus.enable_torque()
                
                # 2. 软件平滑插值 (把路程切成 40 份，耗时约 1.6 秒)
                import time
                steps = 40
                step_delay = 0.04
                start_angles = robot.get_joint_positions()
                
                for step in range(1, steps + 1):
                    intermediate_angles = {}
                    for name in HOME_ANGLES.keys():
                        start_angle = start_angles[name]
                        end_angle = HOME_ANGLES[name]
                        # 线性插值算中间角度
                        current_angle = start_angle + (end_angle - start_angle) * (step / steps)
                        intermediate_angles[name] = current_angle
                        
                    # 下发这一小碎步的指令
                    robot.set_joint_positions(intermediate_angles)
                    time.sleep(step_delay)
                    
                print("✅ 已安全回到初始位置！")
                
            elif key == ord('d'):
                robot.disable_torque()
                print("💤 扭矩已解除！机械臂现在是柔软状态，可以手动挪开了。")
            elif key == ord('d'):
                robot.disable_torque()
                print("💤 扭矩已解除！机械臂现在是柔软状态，可以手动挪开了。")
            elif key == ord('q'):
                break

    cap.release()
    robot.disconnect()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
