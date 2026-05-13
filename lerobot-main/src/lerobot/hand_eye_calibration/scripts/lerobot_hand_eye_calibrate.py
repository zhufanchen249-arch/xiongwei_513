#!/usr/bin/env python3
"""
Interactive hand-eye calibration for LeRobot.

Supports:
  1. Camera intrinsic calibration (ChArUco board)
  2. Hand-eye calibration (eye-in-hand)
  3. Extensible interface for future AI applications
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MIN_PAIRS_FOR_CALIBRATION = 15
_MAX_PAIRS_AUTO_STOP = 30
_TARGET_PAIRS = 25

# ── helper ──────────────────────────────────────────────────────────

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera", type=str, default="/dev/video0", help="Camera device (default /dev/video0)")
    parser.add_argument("--camera_width", type=int, default=1280, help="Camera width")
    parser.add_argument("--camera_height", type=int, default=720, help="Camera height")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    # Board args
    parser.add_argument("--squares_x", type=int, default=5, help="ChArUco board columns")
    parser.add_argument("--squares_y", type=int, default=7, help="ChArUco board rows")
    parser.add_argument("--square_length", type=float, default=0.030, help="Square side length (metres)")
    parser.add_argument("--marker_length", type=float, default=0.024, help="Marker side length (metres)")
    parser.add_argument("--dictionary", type=str, default="DICT_6X6_250",
                        choices=["DICT_4X4_250", "DICT_6X6_250", "DICT_7X7_250", "DICT_5X5_250"])


def _build_board_config(args: argparse.Namespace):
    from lerobot.hand_eye_calibration.configs import CharucoBoardConfig
    return CharucoBoardConfig(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dictionary_name=args.dictionary,
    )


def setup_camera_and_window(args: argparse.Namespace, window_name: str):
    """集成 MJPG 和缓冲区优化的相机启动器 (完美适配 Linux/Jetson)"""
    # 🌟 强制使用 V4L2 后端 
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    
    # 🌟 必须先设格式，再设尺寸
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 🌟 设置缓冲区为 1，彻底解决延迟
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 450)
    
    return cap


# ── intrinsic calibration mode ──────────────────────────────────────

def run_intrinsic(args: argparse.Namespace) -> None:
    from lerobot.hand_eye_calibration.camera_calibrator import CameraCalibrator

    board = _build_board_config(args)
    calibrator = CameraCalibrator(board, args.camera_width, args.camera_height)

    # 调用通用配置
    cap = setup_camera_and_window(args, "Camera Intrinsic Calibration")

    logger.info("Camera intrinsic calibration started.")
    logger.info("Board: %dx%d, dict=%s, square=%.1f mm, marker=%.1f mm",
                 board.squares_x, board.squares_y, board.dictionary_name,
                 board.square_length * 1000, board.marker_length * 1000)
    logger.info("Resolution: %dx%d", args.camera_width, args.camera_height)
    logger.info("Keys: [c]apture  [s]ave intrinsics  [q]uit")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    font_scale = 1.0
    thickness = 2
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 1. 底层运算：使用最原始的横屏画面，保证数学严谨性
        result = calibrator.process_frame(frame, save=False)
        
        # 2. 拿到画好点的原始横屏画面
        display_raw = calibrator.draw_detection(frame, result)
        
        # 3. 视觉显示：不再旋转，直接使用原始画面
        display_img = display_raw

        info_lines = [
            f"Progress: {calibrator.frame_count} / 5 (Min required)",
            f"Detected: {'YES' if result else 'NO'}" + (f" ({result['num_corners']} corners)" if result else ""),
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = int(40 * font_scale) + i * int(35 * font_scale)
            # 4. 把字直接写在画面上
            cv2.putText(display_img, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        # 5. 显示画面
        cv2.imshow("Camera Intrinsic Calibration", display_img)
        key = cv2.waitKey(10) & 0xFF

        if key == ord("c"):
            if result:
                # 存数据时，依然用原始的横屏 frame！
                calibrator.process_frame(frame, save=True)
                logger.info("Captured frame %d (%d corners)", calibrator.frame_count, result["num_corners"])
            else:
                logger.warning("Board not detected — frame NOT added.")

        elif key == ord("s"):
            out = calibrator.calibrate()
            if out is None:
                logger.warning("Need >= 5 detected frames. Have %d.", calibrator.frame_count)
                continue
            K, D, err = out
            intrinsics_path = output_dir / "camera_intrinsics.json"
            calibrator.save(intrinsics_path, K, D)
            logger.info("Intrinsics saved to %s", intrinsics_path)
            logger.info("Camera matrix:\n%s", K)
            logger.info("Distortion:\n%s", D.flatten())
            logger.info("Reprojection error: %.4f", err)

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── hand-eye calibration mode ───────────────────────────────────────

def run_hand_eye(args: argparse.Namespace) -> None:
    from lerobot.hand_eye_calibration.charuco_detector import CharucoDetector
    from lerobot.hand_eye_calibration.configs import load_intrinsics, save_extrinsics
    from lerobot.hand_eye_calibration.data_collector import DataCollector
    from lerobot.hand_eye_calibration.hand_eye_calibrator import HandEyeCalibrator
    from lerobot.hand_eye_calibration.robot_interface import RobotInterface

    intrinsics_path = Path(args.intrinsics)
    if not intrinsics_path.exists():
        logger.error("Intrinsics file not found: %s", intrinsics_path)
        sys.exit(1)
    K, D, w, h = load_intrinsics(intrinsics_path)
    logger.info("Loaded intrinsics from %s", intrinsics_path)

    board = _build_board_config(args)
    detector = CharucoDetector(board, K, D)

    robot = RobotInterface(
        robot_type=args.robot_type,
        port=args.robot_port,
        urdf_path=getattr(args, "urdf_path", None),
    )
    robot.connect()
    robot.disable_torque()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    collector = DataCollector(output_dir)
    calibrator = HandEyeCalibrator(collector, method="tsai")

    cap = setup_camera_and_window(args, "Hand-Eye Calibration (eye-in-hand)")

    font_scale = 1.0
    thickness = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
    
        # 1. 底层运算：绝对不能旋转！保持和内参标定时一样的横屏视角
        detection = detector.detect_and_pose(frame)
        
       # 2. 拿到在横屏画面上画好绿框和坐标轴的原始图片
        display_raw = detector.draw_results(frame, detection)
        
        # 3. 视觉显示：不再旋转，直接使用原始画面
        display_img = display_raw
        
        info = [
            f"Progress: {collector.num_pairs} / {_MIN_PAIRS_FOR_CALIBRATION} (Target: 15~30)",
            f"Board: {'DETECTED' if detection else 'NOT DETECTED'}",
        ]
        if detection:
            info.append(f"Corners: {detection['num_corners']} | IDs: {detection['num_ids']}")
        if collector.num_pairs >= _MIN_PAIRS_FOR_CALIBRATION:
            info.append("READY for calibration [s]")
            
        for i, line in enumerate(info):
            y_pos = int(40 * font_scale) + i * int(35 * font_scale)
            # 4. 把提示文字直接写在画面上
            cv2.putText(display_img, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        # 5. 显示画面
        cv2.imshow("Hand-Eye Calibration (eye-in-hand)", display_img)
        key = cv2.waitKey(10) & 0xFF

        if key == ord("c"):
            if detection is None:
                logger.warning("Board not detected — pair NOT captured.")
                continue
            try:
                joint_positions = robot.get_joint_positions()
                T_base_to_ee = robot.get_ee_pose(joint_positions)
            except Exception as e:
                logger.error("Failed to read robot: %s", e)
                continue

            T_marker_to_cam = detection["T_marker_to_cam"]
            # 记录数据时，同样传入最原始的未旋转的 frame
            collector.add_pair(T_base_to_ee, T_marker_to_cam, joint_positions, frame)

            if collector.num_pairs >= _MAX_PAIRS_AUTO_STOP:
                logger.info("Reached max pairs. Auto-computing hand-eye calibration...")
                break

        elif key == ord("u"):
            collector.remove_last()

        elif key == ord("s"):
            if collector.num_pairs < _MIN_PAIRS_FOR_CALIBRATION:
                continue

            T, R, t = calibrator.calibrate()
            extrinsics_path = output_dir / "hand_eye_extrinsics.json"
            calibrator.save(extrinsics_path)
            errors = calibrator.get_reprojection_error()
            
            logger.info("--- 标定精度评估 ---")
            logger.info(f"平移误差: mean={errors.get('mean_translation_error_m', 0)*1000:.4f} mm")
            if "mean_rotation_error_deg" in errors:
                logger.info(f"旋转误差: mean={errors['mean_rotation_error_deg']:.4f} °")
            
            collector.save()
            break

        elif key == ord("q"):
            break

    robot.disconnect()
    cap.release()
    cv2.destroyAllWindows()


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Hand-eye calibration for LeRobot")
    parser.add_argument("--mode", type=str, required=True, choices=["intrinsic", "hand_eye"])
    _add_common_args(parser)
    parser.add_argument("--intrinsics", type=str, default="./outputs/camera_intrinsics.json")
    parser.add_argument("--robot_type", type=str, default="so100_follower")
    parser.add_argument("--robot_port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--urdf_path", type=str, default="./SO-ARM100-main/so100_new_calib.urdf")
    args = parser.parse_args()

    if args.mode == "intrinsic":
        run_intrinsic(args)
    elif args.mode == "hand_eye":
        run_hand_eye(args)

if __name__ == "__main__":
    main()
