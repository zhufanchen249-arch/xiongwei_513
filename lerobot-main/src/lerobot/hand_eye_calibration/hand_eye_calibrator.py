"""Hand-eye calibration computation using OpenCV."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .configs import save_extrinsics
from .data_collector import DataCollector

logger = logging.getLogger(__name__)

# OpenCV hand-eye calibration methods
METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

DEFAULT_METHOD = "tsai"


class HandEyeCalibrator:
    """Compute hand-eye calibration from collected data pairs."""

    def __init__(self, collector: DataCollector, method: str = DEFAULT_METHOD):
        self.collector = collector
        self.method = method
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(METHODS)}")
        self._cv_method = METHODS[method]

        self.R_cam_to_ee: np.ndarray | None = None
        self.t_cam_to_ee: np.ndarray | None = None
        self._T: np.ndarray | None = None

    def calibrate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run hand-eye calibration.

        Returns (T_cam_to_ee, R_cam_to_ee, t_cam_to_ee).

        Raises ValueError if fewer than 3 pairs.
        """
        if self.collector.num_pairs < 3:
            raise ValueError(f"Need at least 3 pairs, got {self.collector.num_pairs}")

        R_gripper2base = self.collector.get_R_gripper2base()
        t_gripper2base = self.collector.get_t_gripper2base()
        R_target2cam = self.collector.get_R_target2cam()
        t_target2cam = self.collector.get_t_target2cam()

        R, t = cv2.calibrateHandEye(
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            method=self._cv_method,
        )

        self.R_cam_to_ee = R
        self.t_cam_to_ee = t
        self._T = np.eye(4)
        self._T[:3, :3] = R
        self._T[:3, 3] = t.flatten()

        logger.info(
            "Hand-eye calibration complete (%s, %d pairs).\nR:\n%s\nt:\n%s",
            self.method,
            self.collector.num_pairs,
            R,
            t.flatten(),
        )
        return self._T, R, t

    def save(self, path: Path) -> None:
        if self.R_cam_to_ee is None or self.t_cam_to_ee is None:
            raise RuntimeError("Must call calibrate() before save().")
        save_extrinsics(path, self.R_cam_to_ee, self.t_cam_to_ee, self.method, self.collector.num_pairs)

    def get_reprojection_error(self, pairs: list[int] | None = None) -> dict[str, float]:
        """Compute average reprojection errors.

        Uses the hand-eye equation: A * X = Z * B^{-1} where
        A = T_base_to_ee, X = T_cam_to_ee, B = T_marker_to_cam, Z = T_base_to_marker.
        """
        if self._T is None:
            raise RuntimeError("Must call calibrate() first.")

        if pairs is None:
            pairs = list(range(self.collector.num_pairs))

        errors: list[float] = []
        errors_trans: list[float] = []
        errors_rot_deg: list[float] = []  # 新增：记录旋转误差（度）
        for i in pairs:
            pair = self.collector.pairs[i]
            A = pair.T_base_to_ee
            B = pair.T_marker_to_cam

            X = self._T
            Z = A @ X @ np.linalg.inv(B)

            B_est = np.linalg.inv(Z) @ A @ X
            
            # 1. 计算平移误差 (米)
            err_t = np.linalg.norm(B[:3, 3] - B_est[:3, 3])
            errors_trans.append(err_t)
            
            # 2. 计算旋转误差 (度)
            R_diff = B[:3, :3] @ B_est[:3, :3].T
            rvec_diff, _ = cv2.Rodrigues(R_diff)
            err_r = np.linalg.norm(rvec_diff) * (180.0 / np.pi)
            errors_rot_deg.append(err_r)

        return {
            "mean_translation_error_m": float(np.mean(errors_trans)),
            "max_translation_error_m": float(np.max(errors_trans)),
            "std_translation_error_m": float(np.std(errors_trans)),
            "mean_rotation_error_deg": float(np.mean(errors_rot_deg)), # 新增返回
            "max_rotation_error_deg": float(np.max(errors_rot_deg)),   # 新增返回
        }

    @property
    def T_cam_to_ee(self) -> np.ndarray | None:
        return self._T
