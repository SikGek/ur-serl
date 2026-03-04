"""Arm-SDK publisher/subscriber wrapper for single-arm control on the 29-DoF Unitree G1.

This module adapts Unitree's public ``rt/arm_sdk`` interface to a more convenient
single-arm API suitable for HIL-SERL. Internally we still publish the full 14-joint
dual-arm vector because that is what the robot-side arm SDK expects.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .joint_defs import (
    ACTIVE_ARM_TO_INDICES,
    G1JointIndex,
    OPPOSITE_ARM,
    dual_arm_vector,
)


@dataclass
class G1ControllerConfig:
    """Configuration for the low-level arm SDK publisher.

    Attributes:
        active_arm: Which arm is under external control.
        motion_mode: If ``True``, publish to ``rt/arm_sdk`` and use joint index 29 as the
            motion/arm blending weight. This is the mode recommended for balancing while
            manipulating on the real robot.
        simulation_mode: If ``True``, initialize DDS in Unitree's local simulation mode.
        control_hz: Rate of the command publisher thread.
        arm_velocity_limit: Maximum commanded joint velocity used for clipping the
            difference between current and desired arm joint vectors.
        initial_weight: Initial arm-mixing weight in motion mode.
        weight_ramp_rate: Maximum absolute change in weight per second.
    """

    active_arm: str = "left"
    motion_mode: bool = True
    simulation_mode: bool = False
    control_hz: float = 250.0
    arm_velocity_limit: float = 20.0
    initial_weight: float = 1.0
    weight_ramp_rate: float = 1.0  # weight units / second

    # Gains mirrored from Unitree's XR teleoperation example.
    kp_body: float = 300.0
    kd_body: float = 3.0
    kp_arm: float = 80.0
    kd_arm: float = 3.0
    kp_wrist: float = 40.0
    kd_wrist: float = 1.5


class _MotorState:
    """Minimal copy of the data we need from Unitree lowstate."""

    __slots__ = ("q", "dq", "tau_est")

    def __init__(self) -> None:
        self.q = 0.0
        self.dq = 0.0
        self.tau_est = 0.0


class _LowStateBuffer:
    """Thread-safe low-state cache."""

    def __init__(self, num_motors: int) -> None:
        self._lock = threading.Lock()
        self._q = np.zeros(num_motors, dtype=np.float64)
        self._dq = np.zeros(num_motors, dtype=np.float64)
        self._tau_est = np.zeros(num_motors, dtype=np.float64)
        self._mode_machine = 0
        self._ready = False

    def update(self, q: np.ndarray, dq: np.ndarray, tau_est: np.ndarray, mode_machine: int) -> None:
        with self._lock:
            self._q[:] = q
            self._dq[:] = dq
            self._tau_est[:] = tau_est
            self._mode_machine = int(mode_machine)
            self._ready = True

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        with self._lock:
            return (
                self._q.copy(),
                self._dq.copy(),
                self._tau_est.copy(),
                self._mode_machine,
            )

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready


class G1DualArmSDKController:
    """DDS interface that continuously publishes a desired dual-arm joint target.

    The controller freezes all non-arm joints to their current measured position and only
    moves the arms. When ``motion_mode=True`` it publishes to ``rt/arm_sdk`` and manages
    the blending weight using joint index 29 exactly as in Unitree's public XR
    teleoperation example.
    """

    TOPIC_LOWCMD_DEBUG = "rt/lowcmd"
    TOPIC_LOWCMD_MOTION = "rt/arm_sdk"
    TOPIC_LOWSTATE = "rt/lowstate"

    NUM_MOTORS = 35

    def __init__(self, config: G1ControllerConfig) -> None:
        if config.active_arm not in ("left", "right"):
            raise ValueError(
                f"Unsupported active_arm={config.active_arm!r}; expected 'left' or 'right'."
            )

        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as exc:  # pragma: no cover - depends on user environment.
            raise ImportError(
                "unitree_sdk2_python is required. Install the official package from "
                "Unitree's repository before running the G1 server."
            ) from exc

        self.config = config
        self.active_arm = config.active_arm
        self.passive_arm = OPPOSITE_ARM[self.active_arm]

        self.ChannelFactoryInitialize = ChannelFactoryInitialize
        self.ChannelPublisher = ChannelPublisher
        self.ChannelSubscriber = ChannelSubscriber
        self.hg_LowCmd = hg_LowCmd
        self.hg_LowState = hg_LowState
        self.LowCmdMsgFactory = unitree_hg_msg_dds__LowCmd_
        self.CRC = CRC

        self._arm_velocity_limit = float(config.arm_velocity_limit)
        self._control_dt = 1.0 / float(config.control_hz)

        self._buffer = _LowStateBuffer(self.NUM_MOTORS)
        self._desired_dual_arm_q = np.zeros(14, dtype=np.float64)
        self._desired_dual_arm_tau = np.zeros(14, dtype=np.float64)
        self._active_arm_target_q = np.zeros(7, dtype=np.float64)
        self._passive_arm_hold_q = np.zeros(7, dtype=np.float64)

        self._weight_state = float(config.initial_weight if config.motion_mode else 0.0)
        self._weight_target = self._weight_state

        self._cmd_lock = threading.Lock()
        self._stop_event = threading.Event()

        if config.simulation_mode:
            self.ChannelFactoryInitialize(1)
        else:
            self.ChannelFactoryInitialize(0)

        topic = self.TOPIC_LOWCMD_MOTION if config.motion_mode else self.TOPIC_LOWCMD_DEBUG
        self._pub = self.ChannelPublisher(topic, self.hg_LowCmd)
        self._pub.Init()

        self._sub = self.ChannelSubscriber(self.TOPIC_LOWSTATE, self.hg_LowState)
        self._sub.Init()

        self._crc = self.CRC()
        self._msg = self.LowCmdMsgFactory()
        self._msg.mode_pr = 0

        self._subscribe_thread = threading.Thread(
            target=self._subscriber_loop, name="g1-lowstate-sub", daemon=True
        )
        self._subscribe_thread.start()

        t0 = time.time()
        while not self._buffer.ready:
            if time.time() - t0 > 10.0:
                raise TimeoutError(
                    "Timed out waiting for rt/lowstate. Check that the simulator/robot "
                    "is running and that CycloneDDS is configured."
                )
            time.sleep(0.05)

        q_all, _, _, mode_machine = self._buffer.snapshot()
        self._msg.mode_machine = int(mode_machine)

        # Use the current arm configuration as the initial target so arm takeover is smooth.
        dual_q = self.get_current_dual_arm_q()
        self._desired_dual_arm_q[:] = dual_q
        if self.active_arm == "left":
            self._active_arm_target_q[:] = dual_q[:7]
            self._passive_arm_hold_q[:] = dual_q[7:]
        else:
            self._active_arm_target_q[:] = dual_q[7:]
            self._passive_arm_hold_q[:] = dual_q[:7]

        # Initialize all joint commands to the current measured posture.
        for idx in range(self.NUM_MOTORS):
            self._msg.motor_cmd[idx].mode = 1
            self._msg.motor_cmd[idx].q = float(q_all[idx])
            self._msg.motor_cmd[idx].dq = 0.0
            self._msg.motor_cmd[idx].tau = 0.0
            if self._is_arm_motor(idx):
                if self._is_wrist_motor(idx):
                    self._msg.motor_cmd[idx].kp = self.config.kp_wrist
                    self._msg.motor_cmd[idx].kd = self.config.kd_wrist
                else:
                    self._msg.motor_cmd[idx].kp = self.config.kp_arm
                    self._msg.motor_cmd[idx].kd = self.config.kd_arm
            else:
                self._msg.motor_cmd[idx].kp = self.config.kp_body
                self._msg.motor_cmd[idx].kd = self.config.kd_body

        self._publish_thread = threading.Thread(
            target=self._publisher_loop, name="g1-lowcmd-pub", daemon=True
        )
        self._publish_thread.start()

    # ---------------------------------------------------------------------
    # Public getters/setters
    # ---------------------------------------------------------------------
    def get_current_full_q(self) -> np.ndarray:
        q, _, _, _ = self._buffer.snapshot()
        return q

    def get_current_full_dq(self) -> np.ndarray:
        _, dq, _, _ = self._buffer.snapshot()
        return dq

    def get_current_full_tau(self) -> np.ndarray:
        _, _, tau, _ = self._buffer.snapshot()
        return tau

    def get_current_dual_arm_q(self) -> np.ndarray:
        q = self.get_current_full_q()
        return np.concatenate([q[15:22], q[22:29]])

    def get_current_active_arm_q(self) -> np.ndarray:
        dual = self.get_current_dual_arm_q()
        return dual[:7].copy() if self.active_arm == "left" else dual[7:].copy()

    def get_current_passive_arm_q(self) -> np.ndarray:
        dual = self.get_current_dual_arm_q()
        return dual[7:].copy() if self.active_arm == "left" else dual[:7].copy()

    def set_weight_target(self, weight: float) -> None:
        with self._cmd_lock:
            self._weight_target = float(np.clip(weight, 0.0, 1.0))

    def set_passive_arm_hold_q(self, q_passive: Sequence[float]) -> None:
        q_passive = np.asarray(q_passive, dtype=np.float64)
        if q_passive.shape != (7,):
            raise ValueError(f"Expected passive arm hold q shape (7,), got {q_passive.shape}")
        with self._cmd_lock:
            self._passive_arm_hold_q[:] = q_passive
            self._desired_dual_arm_q[:] = np.asarray(
                dual_arm_vector(self.active_arm, self._active_arm_target_q, self._passive_arm_hold_q),
                dtype=np.float64,
            )

    def set_active_arm_target_q(self, q_active: Sequence[float], tau_active: Optional[Sequence[float]] = None) -> None:
        q_active = np.asarray(q_active, dtype=np.float64)
        if q_active.shape != (7,):
            raise ValueError(f"Expected active arm q shape (7,), got {q_active.shape}")

        if tau_active is None:
            tau_active_arr = np.zeros(7, dtype=np.float64)
        else:
            tau_active_arr = np.asarray(tau_active, dtype=np.float64)
            if tau_active_arr.shape != (7,):
                raise ValueError(f"Expected tau_active shape (7,), got {tau_active_arr.shape}")

        with self._cmd_lock:
            self._active_arm_target_q[:] = q_active
            self._desired_dual_arm_q[:] = np.asarray(
                dual_arm_vector(self.active_arm, self._active_arm_target_q, self._passive_arm_hold_q),
                dtype=np.float64,
            )

            tau_passive = np.zeros(7, dtype=np.float64)
            if self.active_arm == "left":
                self._desired_dual_arm_tau[:] = np.concatenate([tau_active_arr, tau_passive])
            else:
                self._desired_dual_arm_tau[:] = np.concatenate([tau_passive, tau_active_arr])

    def go_home(
        self,
        active_arm_home_q: Sequence[float],
        passive_arm_home_q: Optional[Sequence[float]] = None,
        *,
        timeout: float = 5.0,
        tolerance: float = 0.05,
        ramp_weight_down: bool = False,
    ) -> bool:
        """Drive the arm(s) to a joint-space home position.

        Returns:
            ``True`` if the active arm entered the tolerance band before ``timeout``.
        """
        active_arm_home_q = np.asarray(active_arm_home_q, dtype=np.float64)
        if active_arm_home_q.shape != (7,):
            raise ValueError("active_arm_home_q must have shape (7,)")

        if passive_arm_home_q is None:
            passive_arm_home_q = self.get_current_passive_arm_q()
        passive_arm_home_q = np.asarray(passive_arm_home_q, dtype=np.float64)
        if passive_arm_home_q.shape != (7,):
            raise ValueError("passive_arm_home_q must have shape (7,)")

        self.set_passive_arm_hold_q(passive_arm_home_q)
        self.set_active_arm_target_q(active_arm_home_q)

        t0 = time.time()
        while time.time() - t0 < timeout:
            q_err = np.abs(self.get_current_active_arm_q() - active_arm_home_q)
            if np.all(q_err < tolerance):
                if ramp_weight_down and self.config.motion_mode:
                    self.set_weight_target(0.0)
                return True
            time.sleep(0.05)
        return False

    def shutdown(self, *, home_q: Optional[Sequence[float]] = None) -> None:
        """Gracefully stop publishing and optionally return the active arm to home."""
        if home_q is not None:
            try:
                self.go_home(home_q, ramp_weight_down=True)
            except Exception:
                # Best-effort shutdown: never block program exit on cleanup.
                pass
        else:
            self.set_weight_target(0.0)
            time.sleep(1.0)

        self._stop_event.set()
        if self._publish_thread.is_alive():
            self._publish_thread.join(timeout=1.0)
        if self._subscribe_thread.is_alive():
            self._subscribe_thread.join(timeout=1.0)

    # ---------------------------------------------------------------------
    # Internal loops
    # ---------------------------------------------------------------------
    def _subscriber_loop(self) -> None:
        while not self._stop_event.is_set():
            msg = self._sub.Read()
            if msg is not None:
                q = np.array(
                    [float(msg.motor_state[i].q) for i in range(self.NUM_MOTORS)],
                    dtype=np.float64,
                )
                dq = np.array(
                    [float(msg.motor_state[i].dq) for i in range(self.NUM_MOTORS)],
                    dtype=np.float64,
                )
                tau = np.array(
                    [
                        float(getattr(msg.motor_state[i], "tau_est", 0.0))
                        for i in range(self.NUM_MOTORS)
                    ],
                    dtype=np.float64,
                )
                mode_machine = int(getattr(msg, "mode_machine", 0))
                self._buffer.update(q, dq, tau, mode_machine)
            time.sleep(0.002)

    def _publisher_loop(self) -> None:
        last_time = time.time()

        while not self._stop_event.is_set():
            start = time.time()
            q_all, _, _, mode_machine = self._buffer.snapshot()
            self._msg.mode_machine = int(mode_machine)

            # Freeze all joints to the measured configuration by default.
            for idx in range(self.NUM_MOTORS):
                self._msg.motor_cmd[idx].q = float(q_all[idx])
                self._msg.motor_cmd[idx].dq = 0.0
                self._msg.motor_cmd[idx].tau = 0.0

            with self._cmd_lock:
                desired_dual_q = self._desired_dual_arm_q.copy()
                desired_dual_tau = self._desired_dual_arm_tau.copy()
                target_weight = self._weight_target

            dual_q_cmd = self._clip_dual_arm_target(desired_dual_q)

            for offset, motor_idx in enumerate(range(15, 29)):
                self._msg.motor_cmd[motor_idx].q = float(dual_q_cmd[offset])
                self._msg.motor_cmd[motor_idx].dq = 0.0
                self._msg.motor_cmd[motor_idx].tau = float(desired_dual_tau[offset])

            now = time.time()
            dt = max(now - last_time, 1e-4)
            last_time = now
            self._weight_state = self._ramp_weight(
                self._weight_state,
                target_weight,
                dt,
                self.config.weight_ramp_rate,
            )

            if self.config.motion_mode:
                self._msg.motor_cmd[int(G1JointIndex.kNotUsedJoint0)].q = float(self._weight_state)
            else:
                self._msg.motor_cmd[int(G1JointIndex.kNotUsedJoint0)].q = 0.0

            self._msg.crc = self._crc.Crc(self._msg)
            self._pub.Write(self._msg)

            elapsed = time.time() - start
            time.sleep(max(0.0, self._control_dt - elapsed))

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _ramp_weight(current: float, target: float, dt: float, rate: float) -> float:
        max_delta = max(rate, 0.0) * dt
        delta = float(np.clip(target - current, -max_delta, max_delta))
        return float(np.clip(current + delta, 0.0, 1.0))

    def _clip_dual_arm_target(self, desired_dual_q: np.ndarray) -> np.ndarray:
        """Clip arm motion to a per-step joint velocity bound."""
        current_dual_q = self.get_current_dual_arm_q()
        delta = desired_dual_q - current_dual_q
        max_step = self._arm_velocity_limit * self._control_dt
        scale = max(1.0, float(np.max(np.abs(delta)) / max(max_step, 1e-6)))
        return current_dual_q + delta / scale

    @staticmethod
    def _is_arm_motor(motor_idx: int) -> bool:
        return 15 <= int(motor_idx) <= 28

    @staticmethod
    def _is_wrist_motor(motor_idx: int) -> bool:
        return int(motor_idx) in {19, 20, 21, 26, 27, 28}
