"""Hand-eye calibration module for LeRobot.

Provides:
  - Camera intrinsic calibration using ChArUco boards
  - Hand-eye (eye-in-hand) calibration
  - Extensible interface for robot-agnostic use

Usage:
  python -m lerobot.hand_eye_calibration.scripts.lerobot_hand_eye_calibrate --mode intrinsic --camera /dev/video0
  python -m lerobot.hand_eye_calibration.scripts.lerobot_hand_eye_calibrate --mode hand_eye --camera /dev/video0 --intrinsics ./outputs/camera_intrinsics.json
"""

from .hand_eye_calibrator import HandEyeCalibrator  # noqa: F401
from .camera_calibrator import CameraCalibrator  # noqa: F401
from .charuco_detector import CharucoDetector  # noqa: F401
from .configs import (  # noqa: F401
    CharucoBoardConfig,
    load_extrinsics,
    load_intrinsics,
    save_extrinsics,
    save_intrinsics,
)
from .data_collector import CalibrationPair, DataCollector  # noqa: F401
from .robot_interface import RobotInterface  # noqa: F401
