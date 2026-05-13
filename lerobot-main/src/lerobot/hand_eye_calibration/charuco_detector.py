"""ChArUco board detection and 6-DOF pose estimation.

Uses OpenCV 4.7+ API (CharucoDetector, CharucoParameters).
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import cv2.aruco as aruco
import numpy as np

from .configs import CharucoBoardConfig

logger = logging.getLogger(__name__)


class CharucoDetector:
    """Detect and estimate 6-DOF pose of a ChArUco board in an image."""

    def __init__(self, config: CharucoBoardConfig, K: np.ndarray, D: np.ndarray):
        self.config = config
        self.K = K
        self.D = D

        aruco_dict = aruco.getPredefinedDictionary(self._cv_dict())
        self._board = aruco.CharucoBoard(
            (config.squares_x, config.squares_y),
            config.square_length,
            config.marker_length,
            aruco_dict,
        )

        charuco_params = aruco.CharucoParameters()
        detector_params = aruco.DetectorParameters()
        detector_params.markerBorderBits = config.aruco_params_marker_border_bits
        
        # ====== 开启亚像素级细化 ======
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        detector_params.cornerRefinementMinAccuracy = 0.05
        # =========================================================
        
        self._cv_detector = aruco.CharucoDetector(self._board, charuco_params, detector_params)

    def _cv_dict(self) -> int:
        mapping = {
            "DICT_4X4_250": aruco.DICT_4X4_250,
            "DICT_5X5_250": aruco.DICT_5X5_250,
            "DICT_6X6_250": aruco.DICT_6X6_250,
            "DICT_7X7_250": aruco.DICT_7X7_250,
        }
        val = mapping.get(self.config.dictionary_name.upper())
        if val is None:
            raise ValueError(f"Unknown dictionary: {self.config.dictionary_name}. Choose from {list(mapping)}")
        return val

    def detect(self, image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Detect ChArUco corners using OpenCV 4.7+ CharucoDetector.

        Returns (charuco_corners, charuco_ids) or (None, None).
        """
        charuco_corners, charuco_ids, _, _ = self._cv_detector.detectBoard(image)
        if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
            return None, None
        return charuco_corners, charuco_ids

    def match_points(
        self, charuco_corners: np.ndarray, charuco_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match detected charuco corners to board 3D points.

        Returns (obj_points, img_points) — aligned 3D→2D pairs.
        """
        return self._board.matchImagePoints(charuco_corners, charuco_ids)

    def estimate_pose(
        self, obj_points: np.ndarray, img_points: np.ndarray
    ) -> tuple[bool, np.ndarray, np.ndarray]:
        """Estimate board pose from matched 3D-2D point pairs.

        Returns (success, rvec, tvec).
        """
        if obj_points is None or img_points is None or len(obj_points) < 6:
            return False, np.zeros(3), np.zeros(3)
        ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.K, self.D)
        return ret, rvec, tvec

    def detect_and_pose(self, image: np.ndarray) -> dict[str, Any] | None:
        """Full detection pipeline → corners → match → pose."""
        charuco_corners, charuco_ids = self.detect(image)
        if charuco_corners is None:
            return None

        obj_points, img_points = self.match_points(charuco_corners, charuco_ids)
        if obj_points is None or len(obj_points) < 6:
            return None

        ret, rvec, tvec = self.estimate_pose(obj_points, img_points)
        if not ret:
            return None

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()

        return {
            "charuco_corners": charuco_corners,
            "charuco_ids": charuco_ids,
            "rvec": rvec,
            "tvec": tvec,
            "T_marker_to_cam": T,
            "num_corners": len(charuco_corners),
            "num_ids": len(charuco_ids),
        }

    def draw_results(self, image: np.ndarray, result: dict[str, Any] | None) -> np.ndarray:
        display = image.copy()
        if result is None:
            return display
        if result.get("charuco_corners") is not None and result.get("charuco_ids") is not None:
            aruco.drawDetectedCornersCharuco(display, result["charuco_corners"], result["charuco_ids"])
        if result.get("rvec") is not None and result.get("tvec") is not None:
            cv2.drawFrameAxes(display, self.K, self.D, result["rvec"], result["tvec"], 0.04, 2)
        return display

    @property
    def board(self) -> aruco.CharucoBoard:
        return self._board

    @property
    def board_size(self) -> tuple[int, int]:
        return (self.config.squares_x, self.config.squares_y)

    @property
    def dictionary_name(self) -> str:
        return self.config.dictionary_name
