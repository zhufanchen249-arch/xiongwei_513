#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
from lerobot.hand_eye_calibration.configs import load_intrinsics, load_extrinsics, CharucoBoardConfig
from lerobot.hand_eye_calibration.charuco_detector import CharucoDetector
from lerobot.hand_eye_calibration.robot_interface import RobotInterface
def get_reprojection_error(frame, T_marker_to_cam, detector, K, D):
    """计算当前画面的像素级重投影误差 (兼容所有 OpenCV 版本)"""
    try:
        R = T_marker_to_cam[:3, :3]
        tvec = T_marker_to_cam[:3, 3].reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)

        cv_board = detector.board
        c_corners = None
        c_ids = None

        # === 核心修复：OpenCV 新老版本 API 智能适配 ===
        # 情况 1：OpenCV 4.7 及以上 (你的环境属于这种)
        if hasattr(cv2.aruco, 'CharucoDetector'):
            charuco_detector = cv2.aruco.CharucoDetector(cv_board)
            c_corners, c_ids, _, _ = charuco_detector.detectBoard(frame)
            
        # 情况 2：OpenCV 4.6 及以下 (旧版环境)
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
        # ==============================================

        # 如果成功找到了角点，就开始算误差
        if c_corners is not None and c_ids is not None and len(c_corners) >= 4:
            # 兼容不同版本获取 3D 物理角点的方法
            if hasattr(cv_board, 'getChessboardCorners'):
                all_obj_pts = cv_board.getChessboardCorners()
            else:
                all_obj_pts = cv_board.chessboardCorners

            obj_points = all_obj_pts[c_ids.flatten()]
            proj_pts, _ = cv2.projectPoints(obj_points, rvec, tvec, K, D)

            # 勾股定理计算每个像素点的距离偏差，并求平均值
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

    print("\n" + "═"*50)
    print("🎯 手眼外参验证工具 (数据透传版)")
    print("指令: [c] 采集当前点 | [s] 执行分析 | [r] 重置数据 | [q] 退出")
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

        display_rotated = cv2.rotate(display_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.putText(display_rotated, f"Samples: {len(recorded_points)}", (20, 30), 1, font_scale, (0,255,255), 2)
        
        if detection:
            cv2.putText(display_rotated, f"LIVE: {cur_x:.1f}, {cur_y:.1f}, {cur_z:.1f}", (20, 60), 1, font_scale, (0,255,0), 2)
            if px_err is not None:
                color = (0, 255, 0) if px_err < 1.0 else (0, 0, 255)
                cv2.putText(display_rotated, f"ERR: {px_err:.2f} px", (20, 90), 1, font_scale, color, 2)

        cv2.imshow("Hand-Eye Verification", display_rotated)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('c'):
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

        elif key == ord('q'):
            break

    cap.release()
    robot.disconnect()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
