"""Joint definitions for the 29-DoF Unitree G1.

The numeric indices mirror Unitree's public XR teleoperation documentation for the G1
arm SDK interface, where the arm joints occupy indices 15-28 and the first "not used"
joint (index 29) is repurposed as the arm-mixing weight when publishing to ``rt/arm_sdk``.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List


class G1JointIndex(IntEnum):
    """Low-level motor indices for the 29-DoF Unitree G1 body."""

    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    # Waist
    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristYaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

    # Auxiliary / "not used" joints.
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34


class G1ArmJointIndex(IntEnum):
    """Convenience enum containing arm-only indices in the same order used by Unitree."""

    # Left arm
    kLeftShoulderPitch = G1JointIndex.kLeftShoulderPitch
    kLeftShoulderRoll = G1JointIndex.kLeftShoulderRoll
    kLeftShoulderYaw = G1JointIndex.kLeftShoulderYaw
    kLeftElbow = G1JointIndex.kLeftElbow
    kLeftWristRoll = G1JointIndex.kLeftWristRoll
    kLeftWristPitch = G1JointIndex.kLeftWristPitch
    kLeftWristYaw = G1JointIndex.kLeftWristYaw

    # Right arm
    kRightShoulderPitch = G1JointIndex.kRightShoulderPitch
    kRightShoulderRoll = G1JointIndex.kRightShoulderRoll
    kRightShoulderYaw = G1JointIndex.kRightShoulderYaw
    kRightElbow = G1JointIndex.kRightElbow
    kRightWristRoll = G1JointIndex.kRightWristRoll
    kRightWristPitch = G1JointIndex.kRightWristPitch
    kRightWristYaw = G1JointIndex.kRightWristYaw


LEFT_ARM_INDICES: List[int] = [
    G1JointIndex.kLeftShoulderPitch,
    G1JointIndex.kLeftShoulderRoll,
    G1JointIndex.kLeftShoulderYaw,
    G1JointIndex.kLeftElbow,
    G1JointIndex.kLeftWristRoll,
    G1JointIndex.kLeftWristPitch,
    G1JointIndex.kLeftWristYaw,
]

RIGHT_ARM_INDICES: List[int] = [
    G1JointIndex.kRightShoulderPitch,
    G1JointIndex.kRightShoulderRoll,
    G1JointIndex.kRightShoulderYaw,
    G1JointIndex.kRightElbow,
    G1JointIndex.kRightWristRoll,
    G1JointIndex.kRightWristPitch,
    G1JointIndex.kRightWristYaw,
]

ACTIVE_ARM_TO_INDICES: Dict[str, List[int]] = {
    "left": [int(idx) for idx in LEFT_ARM_INDICES],
    "right": [int(idx) for idx in RIGHT_ARM_INDICES],
}

ACTIVE_ARM_TO_JOINT_NAMES: Dict[str, List[str]] = {
    "left": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    "right": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
}

ACTIVE_ARM_TO_FRAME_CANDIDATES: Dict[str, List[str]] = {
    # Prefer the hand / final wrist link if available in the chosen URDF.
    "left": [
        "left_rubber_hand",
        "left_hand",
        "left_wrist_yaw_link",
        "left_wrist_pitch_link",
        "left_wrist_roll_link",
    ],
    "right": [
        "right_rubber_hand",
        "right_hand",
        "right_wrist_yaw_link",
        "right_wrist_pitch_link",
        "right_wrist_roll_link",
    ],
}

OPPOSITE_ARM = {"left": "right", "right": "left"}


def dual_arm_vector(
    active_arm: str,
    active_q: List[float],
    passive_q: List[float],
) -> List[float]:
    """Build a 14-element dual-arm command vector in Unitree order.

    Args:
        active_arm: ``"left"`` or ``"right"``.
        active_q: 7 commanded joints for the active arm.
        passive_q: 7 commanded joints for the passive arm.
    """
    if active_arm not in OPPOSITE_ARM:
        raise ValueError(f"Unsupported active_arm={active_arm!r}; expected 'left' or 'right'.")
    if len(active_q) != 7 or len(passive_q) != 7:
        raise ValueError("Both active_q and passive_q must contain exactly 7 joint values.")
    if active_arm == "left":
        return list(active_q) + list(passive_q)
    return list(passive_q) + list(active_q)
