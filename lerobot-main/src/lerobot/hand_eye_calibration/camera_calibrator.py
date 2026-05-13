"""Camera intrinsic calibration using ChArUco board.

Uses OpenCV 4.7+ API (CharucoDetector + calibrateCamera).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import cv2.aruco as aruco
import numpy as np

from .charuco_detector import CharucoDetector
from .configs import CharucoBoardConfig, save_intrinsics

logger = logging.getLogger(__name__)


class CameraCalibrator:
    def __init__(self, board_config: CharucoBoardConfig, image_width: int, image_height: int):
        self.board_config = board_config
        self.image_width = image_width
        self.image_height = image_height

        aruco_dict = aruco.getPredefinedDictionary(self._cv_dict())
        self._board = aruco.CharucoBoard(
            (board_config.squares_x, board_config.squares_y),
            board_config.square_length,
            board_config.marker_length,
            aruco_dict,
        )

        charuco_params = aruco.CharucoParameters()
        detector_params = aruco.DetectorParameters()
        
        # ====== 强烈建议内参开启亚像素细化 ======
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        detector_params.cornerRefinementMinAccuracy = 0.05
        # ==========================================
        
        self._cv_detector = aruco.CharucoDetector(self._board, charuco_params, detector_params)

        self._all_obj_points: list[np.ndarray] = []
        self._all_img_points: list[np.ndarray] = []
        self._frame_count: int = 0

    def _cv_dict(self) -> int:
        mapping = {
            "DICT_4X4_250": aruco.DICT_4X4_250,
            "DICT_5X5_250": aruco.DICT_5X5_250,
            "DICT_6X6_250": aruco.DICT_6X6_250,
            "DICT_7X7_250": aruco.DICT_7X7_250,
        }
        return mapping[self.board_config.dictionary_name.upper()]

    def process_frame(self, image: np.ndarray, save: bool = False) -> dict[str, Any] | None:
        """核心修复：将画面检测与数据保存解耦。只有 save=True 时才存入数据集。"""
        charuco_corners, charuco_ids, _, _ = self._cv_detector.detectBoard(image)
        if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
            return None

        obj_points, img_points = self._board.matchImagePoints(charuco_corners, charuco_ids)
        if obj_points is None or img_points is None or len(obj_points) < 4:
            return None

        # 只有在主程序中明确下达了 save 指令（按下 C 键），才将其记为有效数据
        if save:
            self._all_obj_points.append(obj_points)
            self._all_img_points.append(img_points)
            self._frame_count += 1

        return {
            "frame_index": self._frame_count,
            "num_corners": len(charuco_corners),
            "num_ids": len(charuco_ids),
            "charuco_corners": charuco_corners,
            "charuco_ids": charuco_ids,
        }

    def calibrate(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        if len(self._all_obj_points) < 5:
            return None

        ret, K, D, _, _ = cv2.calibrateCamera(
            self._all_obj_points,
            self._all_img_points,
            (self.image_width, self.image_height),
            None,
            None,
        )
        if not ret:
            return None
        return K, D, float(ret)

    def save(self, path: Path, K: np.ndarray, D: np.ndarray) -> None:
        save_intrinsics(path, K, D, self.image_width, self.image_height)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def draw_detection(self, image: np.ndarray, result: dict[str, Any] | None) -> np.ndarray:
        display = image.copy()
        if result is not None:
            aruco.drawDetectedCornersCharuco(display, result["charuco_corners"], result["charuco_ids"])
        return display
