"""Pinocchio-based forward kinematics, Jacobians, and damped least-squares IK for G1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from .joint_defs import ACTIVE_ARM_TO_FRAME_CANDIDATES, ACTIVE_ARM_TO_JOINT_NAMES


@dataclass
class G1KinematicsConfig:
    """Configuration for building a fixed-base G1 kinematics model.

    Attributes:
        urdf_path: Absolute path to the G1 29-DoF URDF.
        active_arm: Which arm will be used for HIL-SERL training (``"left"`` or ``"right"``).
        ee_frame_name: Optional exact end-effector frame name in the URDF. If omitted, the
            loader will search common G1 wrist/hand frame candidates.
        active_joint_names: Optional explicit list of the 7 active-arm joint names. If
            omitted, defaults inferred from :mod:`g1_env.robot.joint_defs` are used.
    """

    urdf_path: str
    active_arm: str = "left"
    ee_frame_name: Optional[str] = None
    active_joint_names: Optional[Sequence[str]] = None


class G1Kinematics:
    """Thin Pinocchio wrapper used by the Flask server.

    The Unitree G1 arm-SDK interface operates in joint space, while the SpaceMouse in
    HIL-SERL naturally produces Cartesian end-effector deltas. This helper bridges the
    two by:

    1. computing end-effector pose from the current full-body state;
    2. computing the Jacobian of the selected arm end-effector; and
    3. solving a damped least-squares IK step over the selected 7 arm joints only.

    Notes:
        - The model is treated as *fixed base*. That matches the intended tabletop
          manipulation use case where the lower body remains under Unitree's motion
          controller and only the arms are mixed in via ``rt/arm_sdk``.
        - The IK solves in the arm joint subspace only. Waist and lower-body joints are
          frozen to the current measured configuration.
    """

    def __init__(self, config: G1KinematicsConfig) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:  # pragma: no cover - depends on user environment.
            raise ImportError(
                "Pinocchio is required for G1 kinematics. Install it with "
                "`conda install pinocchio -c conda-forge` or equivalent."
            ) from exc

        self.pin = pin
        self.config = config
        self.urdf_path = str(Path(config.urdf_path).expanduser().resolve())

        if config.active_arm not in ("left", "right"):
            raise ValueError(
                f"Unsupported active_arm={config.active_arm!r}; expected 'left' or 'right'."
            )
        self.active_arm = config.active_arm

        if not Path(self.urdf_path).exists():
            raise FileNotFoundError(
                f"G1 URDF not found at {self.urdf_path!r}. Update the YAML task config."
            )

        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        self.active_joint_names = list(
            config.active_joint_names or ACTIVE_ARM_TO_JOINT_NAMES[self.active_arm]
        )
        self.arm_joint_ids = [self.model.getJointId(name) for name in self.active_joint_names]
        missing = [
            name for name, jid in zip(self.active_joint_names, self.arm_joint_ids) if jid == 0
        ]
        if missing:
            raise ValueError(
                "The following active arm joints were not found in the supplied URDF: "
                + ", ".join(missing)
            )

        self.arm_q_indices = np.array(
            [self.model.joints[jid].idx_q for jid in self.arm_joint_ids], dtype=np.int64
        )
        self.arm_v_indices = np.array(
            [self.model.joints[jid].idx_v for jid in self.arm_joint_ids], dtype=np.int64
        )

        self.ee_frame_name = config.ee_frame_name or self._resolve_ee_frame_name()
        self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)
        if self.ee_frame_id == len(self.model.frames):
            raise ValueError(
                f"End-effector frame {self.ee_frame_name!r} not found in URDF. "
                "Pass an explicit ee_frame_name in the YAML task config."
            )

        self.lower_limits = self.model.lowerPositionLimit.copy()
        self.upper_limits = self.model.upperPositionLimit.copy()

    def _resolve_ee_frame_name(self) -> str:
        """Choose a reasonable end-effector frame from common G1 wrist/hand names."""
        for candidate in ACTIVE_ARM_TO_FRAME_CANDIDATES[self.active_arm]:
            frame_id = self.model.getFrameId(candidate)
            if frame_id != len(self.model.frames):
                return candidate
        raise ValueError(
            "Could not infer an end-effector frame. Tried: "
            + ", ".join(ACTIVE_ARM_TO_FRAME_CANDIDATES[self.active_arm])
        )

    def full_q_from_motor_state(self, motor_q: Sequence[float]) -> np.ndarray:
        """Convert the 35-element Unitree motor vector into the URDF's 29-DoF vector."""
        motor_q = np.asarray(motor_q, dtype=np.float64)
        if motor_q.shape[0] < self.model.nq:
            raise ValueError(
                f"Expected at least {self.model.nq} motor positions, got {motor_q.shape[0]}"
            )
        return motor_q[: self.model.nq].copy()

    def full_v_from_motor_state(self, motor_dq: Sequence[float]) -> np.ndarray:
        """Convert the 35-element Unitree motor velocity vector into the URDF velocity vector."""
        motor_dq = np.asarray(motor_dq, dtype=np.float64)
        if motor_dq.shape[0] < self.model.nv:
            raise ValueError(
                f"Expected at least {self.model.nv} motor velocities, got {motor_dq.shape[0]}"
            )
        return motor_dq[: self.model.nv].copy()

    def arm_subvector(self, q_full: Sequence[float]) -> np.ndarray:
        q_full = np.asarray(q_full, dtype=np.float64)
        return q_full[self.arm_q_indices].copy()

    def with_arm_subvector(self, q_full: Sequence[float], q_arm: Sequence[float]) -> np.ndarray:
        q_full = np.asarray(q_full, dtype=np.float64).copy()
        q_arm = np.asarray(q_arm, dtype=np.float64)
        if q_arm.shape != (7,):
            raise ValueError(f"Expected q_arm shape (7,), got {q_arm.shape}")
        q_full[self.arm_q_indices] = q_arm
        return q_full

    def fk(self, q_full: Sequence[float]) -> np.ndarray:
        """Return end-effector pose as ``[x, y, z, qx, qy, qz, qw]``."""
        q_full = np.asarray(q_full, dtype=np.float64)
        self.pin.forwardKinematics(self.model, self.data, q_full)
        self.pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.ee_frame_id]
        quat = R.from_matrix(np.asarray(placement.rotation)).as_quat()
        return np.concatenate([np.asarray(placement.translation).copy(), quat])

    def jacobian(self, q_full: Sequence[float]) -> np.ndarray:
        """Return the 6 x nv Jacobian in LOCAL_WORLD_ALIGNED coordinates."""
        q_full = np.asarray(q_full, dtype=np.float64)
        self.pin.forwardKinematics(self.model, self.data, q_full)
        self.pin.updateFramePlacements(self.model, self.data)
        return self.pin.computeFrameJacobian(
            self.model,
            self.data,
            q_full,
            self.ee_frame_id,
            self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

    def ee_twist(self, q_full: Sequence[float], dq_full: Sequence[float]) -> np.ndarray:
        """Compute end-effector spatial velocity from full-body ``dq``."""
        jac = self.jacobian(q_full)
        dq_full = np.asarray(dq_full, dtype=np.float64)
        return jac @ dq_full

    @staticmethod
    def pose_error(current_pose_xyzw: Sequence[float], target_pose_xyzw: Sequence[float]) -> np.ndarray:
        """Return a 6D pose error [dx, dy, dz, rx, ry, rz]."""
        current_pose_xyzw = np.asarray(current_pose_xyzw, dtype=np.float64)
        target_pose_xyzw = np.asarray(target_pose_xyzw, dtype=np.float64)

        pos_err = target_pose_xyzw[:3] - current_pose_xyzw[:3]
        rot_cur = R.from_quat(current_pose_xyzw[3:])
        rot_tgt = R.from_quat(target_pose_xyzw[3:])
        rot_err = (rot_tgt * rot_cur.inv()).as_rotvec()
        return np.concatenate([pos_err, rot_err])

    def solve_ik(
        self,
        target_pose_xyzw: Sequence[float],
        q_seed_full: Sequence[float],
        *,
        max_iters: int = 80,
        pos_tol: float = 1e-3,
        rot_tol: float = 1e-2,
        damping: float = 1e-4,
        step_size: float = 0.6,
    ) -> Tuple[np.ndarray, bool, Dict[str, float]]:
        """Solve single-arm IK while freezing all non-arm joints.

        Args:
            target_pose_xyzw: Desired end-effector pose ``[x, y, z, qx, qy, qz, qw]``.
            q_seed_full: Current *full-body* configuration (first 29 joints of G1).
            max_iters: Maximum DLS iterations.
            pos_tol: Position tolerance in meters.
            rot_tol: Orientation tolerance in radians (rotvec norm).
            damping: Levenberg-Marquardt damping coefficient.
            step_size: Multiplicative factor for each IK update.

        Returns:
            A tuple ``(q_arm, success, metrics)`` where ``q_arm`` is the commanded
            7-element active-arm joint vector, ``success`` indicates convergence, and
            ``metrics`` contains final error norms.
        """
        q_full = np.asarray(q_seed_full, dtype=np.float64).copy()
        target_pose_xyzw = np.asarray(target_pose_xyzw, dtype=np.float64)

        success = False
        err = np.zeros(6, dtype=np.float64)

        for _ in range(max_iters):
            current_pose = self.fk(q_full)
            err = self.pose_error(current_pose, target_pose_xyzw)

            pos_norm = float(np.linalg.norm(err[:3]))
            rot_norm = float(np.linalg.norm(err[3:]))
            if pos_norm <= pos_tol and rot_norm <= rot_tol:
                success = True
                break

            jac_full = self.jacobian(q_full)
            jac_arm = jac_full[:, self.arm_v_indices]  # 6 x 7

            # Damped least-squares step in the active-arm subspace only.
            lhs = jac_arm @ jac_arm.T + damping * np.eye(6, dtype=np.float64)
            dq_arm = jac_arm.T @ np.linalg.solve(lhs, err)

            q_arm_next = q_full[self.arm_q_indices] + step_size * dq_arm
            q_arm_next = np.clip(
                q_arm_next,
                self.lower_limits[self.arm_q_indices],
                self.upper_limits[self.arm_q_indices],
            )
            q_full[self.arm_q_indices] = q_arm_next

        return (
            q_full[self.arm_q_indices].copy(),
            success,
            {
                "pos_err_norm": float(np.linalg.norm(err[:3])),
                "rot_err_norm": float(np.linalg.norm(err[3:])),
            },
        )
