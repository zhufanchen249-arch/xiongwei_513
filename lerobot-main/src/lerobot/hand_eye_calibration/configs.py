"""Hand-eye calibration types and configurations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CharucoBoardConfig:
    """ChArUco board physical configuration."""

    squares_x: int = 5
    squares_y: int = 7
    square_length: float = 0.030  # metres (25 mm)
    marker_length: float = 0.024  # metres (18 mm)
    dictionary_name: str = "DICT_6X6_250"
    aruco_params_marker_border_bits: int = 1
   
    _dictionary_options: tuple = field(
        default=("DICT_4X4_250", "DICT_6X6_250", "DICT_7X7_250", "DICT_5X5_250"),
        init=False,
        repr=False,
    )

    @classmethod
    def standard_5x7_6x6(cls) -> CharucoBoardConfig:
        """5×7 ChArUco board, DICT_6X6_250 dictionary, 25 mm squares, 18 mm markers."""
        return cls(squares_x=5, squares_y=7, square_length=0.025, marker_length=0.018, dictionary_name="DICT_6X6_250")

    @classmethod
    def standard_8x11_4x4(cls) -> CharucoBoardConfig:
        """8×11 ChArUco board, DICT_4X4_250 dictionary, 25 mm squares, 18 mm markers."""
        return cls(squares_x=8, squares_y=11, square_length=0.025, marker_length=0.018, dictionary_name="DICT_4X4_250")


def matrix_to_list(m: np.ndarray) -> list[list[float]]:
    return m.tolist()


INTRINSIC_KEYS = ("camera_matrix", "dist_coeffs", "image_width", "image_height")


def save_intrinsics(path: Path, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "camera_matrix": matrix_to_list(camera_matrix),
        "dist_coeffs": matrix_to_list(dist_coeffs),
        "image_width": width,
        "image_height": height,
    }
    content = json.dumps(data, indent=2, ensure_ascii=False)
    checksum = hashlib.md5(content.encode()).hexdigest()[:8]
    data["_checksum"] = checksum
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    with open(path) as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    D = np.array(data["dist_coeffs"], dtype=np.float64)
    w = int(data["image_width"])
    h = int(data["image_height"])
    return K, D, w, h


EXTRINSIC_KEYS = ("R_cam_to_ee", "t_cam_to_ee", "method", "num_pairs")


def save_extrinsics(path: Path, R: np.ndarray, t: np.ndarray, method: str, num_pairs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "R_cam_to_ee": matrix_to_list(R),
        "t_cam_to_ee": matrix_to_list(t),
        "method": method,
        "num_pairs": num_pairs,
    }
    content = json.dumps(data, indent=2, ensure_ascii=False)
    checksum = hashlib.md5(content.encode()).hexdigest()[:8]
    data["_checksum"] = checksum
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_extrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with open(path) as f:
        data = json.load(f)
    R = np.array(data["R_cam_to_ee"], dtype=np.float64)
    t = np.array(data["t_cam_to_ee"], dtype=np.float64)
    method = str(data.get("method", "unknown"))
    return R, t, method
