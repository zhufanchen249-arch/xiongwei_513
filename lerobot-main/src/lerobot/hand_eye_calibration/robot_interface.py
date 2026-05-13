"""Robot interface — connect, read joint positions, forward kinematics."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _parse_urdf_chain(urdf_path: str) -> list[dict[str, Any]]:
    """Parse URDF to extract the kinematic chain (from base_link to gripper_frame_link).

    Returns list of dicts: {'name': joint_name, 'origin': (xyz, rpy), 'axis': xyz, 'type': 'revolute'|'fixed'}
    in order from base to tip.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints: dict[str, dict] = {}
    for j in root.findall("joint"):
        name = j.attrib["name"]
        jtype = j.attrib["type"]
        parent = j.find("parent").attrib["link"]
        child = j.find("child").attrib["link"]
        origin = j.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
        axis = [0.0, 0.0, 1.0]
        axis_el = j.find("axis")
        if axis_el is not None:
            axis = [float(v) for v in axis_el.attrib.get("xyz", "0 0 1").split()]
        joints[name] = {"parent": parent, "child": child, "xyz": xyz, "rpy": rpy, "axis": axis, "type": jtype}

    chain: list[dict] = []
    current = "base_link"
    while current != "gripper_frame_link":
        found = False
        for name, j in joints.items():
            if j["parent"] == current:
                chain.append({
                    "name": name,
                    "xyz": j["xyz"],
                    "rpy": j["rpy"],
                    "axis": j["axis"],
                    "type": j["type"],
                })
                current = j["child"]
                found = True
                break
        if not found:
            break

    return chain


def _urdf_fk(chain: list[dict], joint_angles_deg: list[float]) -> np.ndarray:
    """Compute forward kinematics from URDF chain and joint angles (degrees)."""
    T = np.eye(4)
    for i, joint in enumerate(chain):
        xyz, rpy = joint["xyz"], joint["rpy"]
        T_origin = _xyz_rpy_to_matrix(xyz, rpy)
        if joint["type"] == "revolute" and i < len(joint_angles_deg):
            theta = np.deg2rad(joint_angles_deg[i])
            axis = joint["axis"]
            R = _rotation_matrix(axis, theta)
            T_joint = np.eye(4)
            T_joint[:3, :3] = R
            T = T @ T_origin @ T_joint
        else:
            T = T @ T_origin
    return T


def _xyz_rpy_to_matrix(xyz: list[float], rpy: list[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = xyz
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    T[:3, :3] = Rz @ Ry @ Rx
    return T


def _rotation_matrix(axis: list[float], theta: float) -> np.ndarray:
    """Rodrigues rotation matrix around given axis by theta radians."""
    ax = np.array(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    c = np.cos(theta)
    s = np.sin(theta)
    v = 1 - c
    return np.array([
        [c + ax[0]*ax[0]*v,       ax[0]*ax[1]*v - ax[2]*s,  ax[0]*ax[2]*v + ax[1]*s],
        [ax[1]*ax[0]*v + ax[2]*s, c + ax[1]*ax[1]*v,        ax[1]*ax[2]*v - ax[0]*s],
        [ax[2]*ax[0]*v - ax[1]*s, ax[2]*ax[1]*v + ax[0]*s,  c + ax[2]*ax[2]*v],
    ])


class SimpleKinematics:
    """Lightweight forward kinematics from URDF, no external dependency."""

    def __init__(self, urdf_path: str, target_frame_name: str = "gripper_frame_link"):
        self.chain = _parse_urdf_chain(urdf_path)

    def forward_kinematics(self, joint_angles_deg: list[float] | np.ndarray) -> np.ndarray:
        return _urdf_fk(self.chain, list(joint_angles_deg))


class RobotInterface:
    """Thin wrapper around a lerobot robot for hand-eye calibration data collection.

    Can operate in two modes:
    - **connected**: uses the real robot via FeetechMotorsBus
    - **standalone**: uses a URDF for FK only (no robot connection needed)

    The standalone mode is useful when the user wants to move the robot manually and
    compute EE pose from joint positions obtained from a calibration file.
    """

    def __init__(
        self,
        robot_type: str = "so100_follower",
        port: str = "/dev/ttyACM0",
        robot_id: str | None = None,
        urdf_path: str | None = None,
    ):
        self.robot_type = robot_type
        self.port = port
        self.robot_id = robot_id
        self._urdf_path = urdf_path

        self._robot: object | None = None
        self._kinematics: object | None = None
        self._connected = False

    @property
    def motor_names(self) -> list[str]:
        return ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    def connect(self) -> None:
        """Connect to the physical robot via FeetechMotorsBus."""
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import Motor, MotorNormMode

        motor_defs: dict[str, Motor] = {}
        norm = MotorNormMode.DEGREES
        for i, name in enumerate(self.motor_names):
            motor_defs[name] = Motor(i + 1, "sts3215", norm)

        self._bus = FeetechMotorsBus(port=self.port, motors=motor_defs)
        self._bus.connect()
        self._bus.calibration = self._bus.read_calibration()
        self._connected = True
        logger.info("Connected to robot on %s", self.port)

    def disconnect(self) -> None:
        if self._connected and self._bus is not None:
            self._bus.disconnect()
            self._connected = False
            logger.info("Disconnected from robot.")

    def disable_torque(self) -> None:
        if self._connected:
            self._bus.disable_torque()

    def get_joint_positions(self) -> dict[str, float]:
        """Read current joint positions in degrees.

        For manual mode (no connection), returns zeros.
        """
        if not self._connected:
            return {name: 0.0 for name in self.motor_names}

        values: dict[str, float] = {}
        for name in self.motor_names:
            values[name] = float(self._bus.read("Present_Position", name))
        return values

    def get_ee_pose(self, joint_positions: dict[str, float]) -> np.ndarray:
        """Compute end-effector pose (4×4 T_base_to_ee) from joint positions.

        Requires: URDF path and placo installed.
        """
        if self._kinematics is None:
            self._init_kinematics()

        joint_angles = [joint_positions[name] for name in self.motor_names[:-1]]
        T = self._kinematics.forward_kinematics(joint_angles)
        return np.array(T)

    def _init_kinematics(self) -> None:
        urdf_path = self._urdf_path
        if urdf_path is None:
            urdf_path = self._default_urdf_path()

        try:
            from lerobot.model.kinematics import RobotKinematics

            self._kinematics = RobotKinematics(
                urdf_path=str(urdf_path),
                target_frame_name="gripper_frame_link",
            )
        except ImportError:
            self._kinematics = SimpleKinematics(
                urdf_path=str(urdf_path),
                target_frame_name="gripper_frame_link",
            )
            logger.info("placo not available, using built-in FK")
        logger.info("Kinematics loaded from %s", urdf_path)

    @staticmethod
    def _default_urdf_path() -> str:
        """Try to find the SO-101 URDF."""
        # Path relative to this file's directory
        module_dir = Path(__file__).resolve().parent
        candidates = [
            module_dir / "SO-ARM100-main/so100_new_calib.urdf",
            Path("SO-ARM100-main/so100_new_calib.urdf"),
            Path("./SO-ARM100-main/so100_new_calib.urdf"),
        ]
        for p in candidates:
            if Path(p).exists():
                return p

        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id="lerobot/so-arm100",
                filename="SO101/so101_new_calib.urdf",
                repo_type="model",
            )
            return path
        except Exception:
            pass

        raise FileNotFoundError(
            "Cannot find SO-101 URDF. Download it from https://huggingface.co/lerobot/so-arm100 "
            "or pass --urdf_path to the script."
        )
