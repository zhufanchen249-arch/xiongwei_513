#!/usr/bin/env python3
"""
Interactive hand-eye calibration for LeRobot.

Usage:
  python -m lerobot.scripts.lerobot_hand_eye_calibrate --mode intrinsic --camera /dev/video0
  python -m lerobot.scripts.lerobot_hand_eye_calibrate --mode hand_eye --camera /dev/video0 --robot_port /dev/ttyACM0 --intrinsics ./outputs/camera_intrinsics.json

Keyboard controls:
  Intrinsic mode:  [c]apture  [s]ave  [q]uit
  Hand-eye mode:   [c]apture  [u]ndo  [s]olve  [q]uit
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
_TARGET_INTRINSIC = 15


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera", type=str, default="/dev/video0", help="Camera device")
    parser.add_argument("--camera_width", type=int, default=640, help="Camera width")
    parser.add_argument("--camera_height", type=int, default=480, help="Camera height")
    parser.add_argument("--display_scale", type=float, default=1.0, help="Display scale factor")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--squares_x", type=int, default=5, help="ChArUco board columns")
    parser.add_argument("--squares_y", type=int, default=7, help="ChArUco board rows")
    parser.add_argument("--square_length", type=float, default=0.025, help="Square side length (metres)")
    parser.add_argument("--marker_length", type=float, default=0.018, help="Marker side length (metres)")
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


def _open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _display(image: np.ndarray, title: str, scale: float) -> int:
    if scale != 1.0:
        h, w = image.shape[:2]
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image)
    return cv2.waitKey(10) & 0xFF


# ── intrinsic calibration ────────────────────────────────────────────

def run_intrinsic(args: argparse.Namespace) -> None:
    from lerobot.hand_eye_calibration.camera_calibrator import CameraCalibrator

    board = _build_board_config(args)
    calibrator = CameraCalibrator(board, args.camera_width, args.camera_height)
    cap = _open_camera(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Camera intrinsic calibration started.")
    logger.info("Board: %dx%d, dict=%s, square=%.1fmm, marker=%.1fmm",
                 board.squares_x, board.squares_y, board.dictionary_name,
                 board.square_length * 1000, board.marker_length * 1000)
    logger.info("Resolution: %dx%d  Target: %d frames", args.camera_width, args.camera_height, _TARGET_INTRINSIC)
    logger.info("Keys: [c]apture  [s]ave intrinsics  [q]uit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        result = calibrator.process_frame(frame)
        display = calibrator.draw_detection(frame, result)

        info_lines = [
            f"Frames: {calibrator.frame_count}/{_TARGET_INTRINSIC}",
            f"Detected: {'YES' if result else 'NO'}" + (f" ({result['num_corners']} corners)" if result else ""),
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(display, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        key = _display(display, "Camera Intrinsic Calibration", args.display_scale)

        if key == ord("c"):
            if result:
                calibrator.process_frame(frame, save=True)
                logger.info("Captured frame %d/%d (%d corners)", calibrator.frame_count, _TARGET_INTRINSIC, result["num_corners"])
            else:
                logger.warning("Board not detected — frame NOT added.")

        if calibrator.frame_count >= _TARGET_INTRINSIC:
            out = calibrator.calibrate()
            if out is not None:
                K, D, err = out
                intrinsics_path = output_dir / "camera_intrinsics.json"
                calibrator.save(intrinsics_path, K, D)
                logger.info("Reached target %d frames. Intrinsics saved to %s", _TARGET_INTRINSIC, intrinsics_path)
                logger.info("Reprojection error: %.4f", err)
            break

        if key == ord("s"):
            out = calibrator.calibrate()
            if out is None:
                logger.warning("Need >= 5 detected frames. Have %d/%d.", calibrator.frame_count, _TARGET_INTRINSIC)
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


# ── hand-eye calibration ─────────────────────────────────────────────

def run_hand_eye(args: argparse.Namespace) -> None:
    from lerobot.hand_eye_calibration.charuco_detector import CharucoDetector
    from lerobot.hand_eye_calibration.configs import load_intrinsics
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

    robot = RobotInterface(robot_type=args.robot_type, port=args.robot_port, urdf_path=getattr(args, "urdf_path", None))
    robot.connect()
    robot.disable_torque()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    collector = DataCollector(output_dir)
    calibrator = HandEyeCalibrator(collector, method="tsai")

    cap = _open_camera(args)

    logger.info("Hand-eye calibration started (eye-in-hand).")
    logger.info("Board: %dx%d, dict=%s", board.squares_x, board.squares_y, board.dictionary_name)
    logger.info("Robot: %s @ %s", args.robot_type, args.robot_port)
    logger.info("Keys: [c]apture  [u]ndo  [s]olve  [q]uit")
    logger.info("Min pairs: %d, auto-stop at %d", _MIN_PAIRS_FOR_CALIBRATION, _MAX_PAIRS_AUTO_STOP)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            detection = detector.detect_and_pose(frame)
        except cv2.error as e:
            logger.warning("OpenCV error in detection: %s", e)
            detection = None
        display = detector.draw_results(frame, detection)

        ready = collector.num_pairs >= _MIN_PAIRS_FOR_CALIBRATION
        info = [
            f"Pairs: {collector.num_pairs}/{_MAX_PAIRS_AUTO_STOP} (min {_MIN_PAIRS_FOR_CALIBRATION})",
            f"Board: {'DETECTED' if detection else 'NOT DETECTED'}",
        ]
        if detection:
            info.append(f"Corners: {detection['num_corners']} | IDs: {detection['num_ids']}")
            t = detection["tvec"].flatten()
            info.append(f"Dist: {np.linalg.norm(t):.2f}m")
        if ready:
            info.append(">>> READY — press [s] to solve <<<")
        for i, line in enumerate(info):
            cv2.putText(display, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        key = _display(display, "Hand-Eye Calibration (eye-in-hand)", args.display_scale)

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
            collector.add_pair(T_base_to_ee, T_marker_to_cam, joint_positions, frame)
            logger.info("Captured pair %d/%d", collector.num_pairs, _MAX_PAIRS_AUTO_STOP)
            if collector.num_pairs >= _MAX_PAIRS_AUTO_STOP:
                logger.info("Reached max pairs (%d). Auto-computing...", _MAX_PAIRS_AUTO_STOP)
                break

        elif key == ord("u"):
            removed = collector.remove_last()
            logger.info("Removed pair %d. %d pairs remain." if removed else "No pairs to remove.",
                        removed.index if removed else 0, collector.num_pairs)

        elif key == ord("s"):
            if collector.num_pairs < _MIN_PAIRS_FOR_CALIBRATION:
                logger.warning("Need >= %d pairs. Have %d.", _MIN_PAIRS_FOR_CALIBRATION, collector.num_pairs)
                continue
            T, R, t = calibrator.calibrate()
            extrinsics_path = output_dir / "hand_eye_extrinsics.json"
            calibrator.save(extrinsics_path)
            logger.info("Extrinsics saved to %s", extrinsics_path)
            logger.info("T_cam_to_ee:\n%s", T)
            collector.save()
            break

        elif key == ord("q"):
            if collector.num_pairs >= _MIN_PAIRS_FOR_CALIBRATION:
                logger.info("Computing calibration with %d pairs before exit...", collector.num_pairs)
                T, R, t = calibrator.calibrate()
                extrinsics_path = output_dir / "hand_eye_extrinsics.json"
                calibrator.save(extrinsics_path)
                collector.save()
            else:
                logger.warning("Not enough pairs (%d < %d). Data NOT saved.", collector.num_pairs, _MIN_PAIRS_FOR_CALIBRATION)
            break

    robot.disconnect()
    cap.release()
    cv2.destroyAllWindows()


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Hand-eye calibration for LeRobot")
    parser.add_argument("--mode", type=str, required=True, choices=["intrinsic", "hand_eye"], help="Calibration mode")
    _add_common_args(parser)
    parser.add_argument("--intrinsics", type=str, default="./outputs/camera_intrinsics.json", help="Intrinsics JSON (hand_eye mode)")
    parser.add_argument("--robot_type", type=str, default="so100_follower", help="Robot type (hand_eye mode)")
    parser.add_argument("--robot_port", type=str, default="/dev/ttyACM0", help="Robot serial port (hand_eye mode)")
    parser.add_argument("--urdf_path", type=str, default=None, help="URDF path")
    args = parser.parse_args()

    if args.mode == "intrinsic":
        run_intrinsic(args)
    elif args.mode == "hand_eye":
        run_hand_eye(args)


if __name__ == "__main__":
    main()
