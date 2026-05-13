"""Hand-eye calibration data collector."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationPair:
    """A single hand-eye calibration data pair."""

    index: int
    T_base_to_ee: np.ndarray  # 4x4: EE pose relative to robot base
    T_marker_to_cam: np.ndarray  # 4x4: marker pose relative to camera
    joint_positions: dict[str, float] = field(default_factory=dict)
    image_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "T_base_to_ee": self.T_base_to_ee.tolist(),
            "T_marker_to_cam": self.T_marker_to_cam.tolist(),
            "joint_positions": self.joint_positions,
            "image_path": self.image_path,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibrationPair:
        return cls(
            index=int(d["index"]),
            T_base_to_ee=np.array(d["T_base_to_ee"]),
            T_marker_to_cam=np.array(d["T_marker_to_cam"]),
            joint_positions=d.get("joint_positions", {}),
            image_path=d.get("image_path"),
        )


class DataCollector:
    """Accumulates hand-eye calibration data pairs."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.pairs: list[CalibrationPair] = []
        self._images_dir = output_dir / "calibration_images"
        self._images_dir.mkdir(parents=True, exist_ok=True)

    def add_pair(
        self,
        T_base_to_ee: np.ndarray,
        T_marker_to_cam: np.ndarray,
        joint_positions: dict[str, float],
        image: np.ndarray | None = None,
    ) -> CalibrationPair | None:
        
        # —— 新增：有效运动校验逻辑 ——
        if self.pairs:
            last_T = self.pairs[-1].T_base_to_ee
            
            # 1. 计算平移距离（米）
            t_diff = np.linalg.norm(T_base_to_ee[:3, 3] - last_T[:3, 3])
            
            # 2. 计算旋转角度差（度）
            R_diff = T_base_to_ee[:3, :3] @ last_T[:3, :3].T
            rvec_diff, _ = cv2.Rodrigues(R_diff)
            angle_diff = np.linalg.norm(rvec_diff) * (180.0 / np.pi)
            
            # 阈值判断：平移 < 1mm 且 旋转 < 1度 则视为无效
            if t_diff < 0.001 and angle_diff < 1.0:
                logger.warning(
                    f"跳过采集：机械臂位姿变化太小 (平移:{t_diff*1000:.2f}mm, 旋转:{angle_diff:.2f}°) "
                    "会导致数学计算共线性问题，请变换姿态！"
                )
                return None
        # ————————————————————————

        index = len(self.pairs)
        image_path: str | None = None
        if image is not None:
            fname = f"pair_{index:03d}.jpg"
            image_path = str(self._images_dir / fname)
            cv2.imwrite(image_path, image)

        pair = CalibrationPair(
            index=index,
            T_base_to_ee=T_base_to_ee,
            T_marker_to_cam=T_marker_to_cam,
            joint_positions=joint_positions,
            image_path=image_path,
        )
        self.pairs.append(pair)
        logger.info("Captured pair %d — %d total pairs.", index + 1, len(self.pairs))
        return pair

    def save(self, path: Path | None = None) -> None:
        if path is None:
            path = self.output_dir / "hand_eye_data.json"
        data = {
            "num_pairs": len(self.pairs),
            "pairs": [p.to_dict() for p in self.pairs],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d calibration pairs to %s", len(self.pairs), path)

    @classmethod
    def load(cls, path: Path) -> DataCollector:
        with open(path) as f:
            data = json.load(f)
        collector = cls(output_dir=path.parent)
        for d in data.get("pairs", []):
            collector.pairs.append(CalibrationPair.from_dict(d))
        return collector

    def get_R_gripper2base(self) -> list[np.ndarray]:
        return [p.T_base_to_ee[:3, :3] for p in self.pairs]

    def get_t_gripper2base(self) -> list[np.ndarray]:
        return [p.T_base_to_ee[:3, 3:4] for p in self.pairs]

    def get_R_target2cam(self) -> list[np.ndarray]:
        return [p.T_marker_to_cam[:3, :3] for p in self.pairs]

    def get_t_target2cam(self) -> list[np.ndarray]:
        return [p.T_marker_to_cam[:3, 3:4] for p in self.pairs]

    def remove_last(self) -> CalibrationPair | None:
        if self.pairs:
            return self.pairs.pop()
        return None

    @property
    def num_pairs(self) -> int:
        return len(self.pairs)
