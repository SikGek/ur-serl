"""Flask server exposing a Unitree G1 single-arm Cartesian API for HIL-SERL.

Run this from inside ``serl_robot_infra`` after installing the package in editable mode:

    cd serl_robot_infra
    python robot_servers/g1_arm_server.py \
        --urdf_path /absolute/path/to/g1_29dof.urdf \
        --active_arm left \
        --flask_url 127.0.0.1 \
        --flask_port 5001

For ``unitree_mujoco`` / ``simulate_python`` dry-runs, add ``--simulation_mode``.
"""

from __future__ import annotations

import atexit
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from absl import app, flags
from flask import Flask, jsonify, request
from scipy.spatial.transform import Rotation as R

from g1_env.robot.arm_controller import G1ControllerConfig, G1DualArmSDKController
from g1_env.robot.kinematics import G1Kinematics, G1KinematicsConfig

FLAGS = flags.FLAGS

flags.DEFINE_string("urdf_path", None, "Absolute path to the G1 29-DoF URDF.")
flags.DEFINE_string("active_arm", "left", "Which arm to control: left or right.")
flags.DEFINE_string(
    "ee_frame_name",
    None,
    "Optional exact end-effector frame name. If unset, the server tries common G1 wrist/hand frames.",
)
flags.DEFINE_string("flask_url", "127.0.0.1", "Address the Flask server binds to.")
flags.DEFINE_integer("flask_port", 5001, "TCP port for the Flask server.")
flags.DEFINE_boolean(
    "motion_mode",
    True,
    "Publish to rt/arm_sdk and use Unitree's arm-weight blending in regular motion mode.",
)
flags.DEFINE_boolean(
    "simulation_mode",
    False,
    "Initialize DDS in local simulation mode for unitree_mujoco simulate_python.",
)
flags.DEFINE_list(
    "active_arm_home_q",
    None,
    "Optional 7-element home joint vector for the active arm.",
)
flags.DEFINE_list(
    "passive_arm_home_q",
    None,
    "Optional 7-element home joint vector for the passive arm.",
)
flags.DEFINE_float("control_hz", 250.0, "DDS publishing frequency.")
flags.DEFINE_float("ik_pos_tol", 1e-3, "Position tolerance for IK success.")
flags.DEFINE_float("ik_rot_tol", 1e-2, "Rotation tolerance for IK success.")
flags.DEFINE_integer("ik_max_iters", 80, "Maximum DLS iterations for IK.")


def _parse_optional_vector(raw: Optional[Sequence[str]], expected_len: int) -> Optional[np.ndarray]:
    if raw is None:
        return None
    if len(raw) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(raw)}")
    return np.asarray([float(v) for v in raw], dtype=np.float64)


class G1ArmServer:
    """Robot-side service wrapper exposing Cartesian end-effector control."""

    def __init__(self) -> None:
        if not FLAGS.urdf_path:
            raise ValueError("--urdf_path must point to a G1 29-DoF URDF.")
        self.active_arm = FLAGS.active_arm

        self.controller = G1DualArmSDKController(
            G1ControllerConfig(
                active_arm=FLAGS.active_arm,
                motion_mode=FLAGS.motion_mode,
                simulation_mode=FLAGS.simulation_mode,
                control_hz=FLAGS.control_hz,
                initial_weight=1.0 if FLAGS.motion_mode else 0.0,
            )
        )
        self.kin = G1Kinematics(
            G1KinematicsConfig(
                urdf_path=FLAGS.urdf_path,
                active_arm=FLAGS.active_arm,
                ee_frame_name=FLAGS.ee_frame_name,
            )
        )

        self.active_arm_home_q = _parse_optional_vector(FLAGS.active_arm_home_q, 7)
        if self.active_arm_home_q is None:
            self.active_arm_home_q = self.controller.get_current_active_arm_q()

        self.passive_arm_home_q = _parse_optional_vector(FLAGS.passive_arm_home_q, 7)
        if self.passive_arm_home_q is None:
            self.passive_arm_home_q = self.controller.get_current_passive_arm_q()

        # Make the passive arm stay in a consistent posture throughout training.
        self.controller.set_passive_arm_hold_q(self.passive_arm_home_q)
        if FLAGS.motion_mode:
            self.controller.set_weight_target(1.0)

    def state_dict(self) -> dict:
        q_all = self.controller.get_current_full_q()
        dq_all = self.controller.get_current_full_dq()

        q_full = self.kin.full_q_from_motor_state(q_all)
        dq_full = self.kin.full_v_from_motor_state(dq_all)

        pose = self.kin.fk(q_full)
        jac = self.kin.jacobian(q_full)
        vel = jac @ dq_full

        active_q = self.kin.arm_subvector(q_full)
        active_dq = dq_full[self.kin.arm_v_indices]

        # G1 arm SDK does not expose a force-torque estimate at the wrist through this
        # interface. Keep zeros so the observation shape is stable.
        return {
            "pose": pose.tolist(),
            "vel": np.asarray(vel, dtype=np.float64).tolist(),
            "force": np.zeros(3, dtype=np.float64).tolist(),
            "torque": np.zeros(3, dtype=np.float64).tolist(),
            "q": q_full.tolist(),
            "dq": dq_full.tolist(),
            "jacobian": np.asarray(jac, dtype=np.float64).tolist(),
            "active_q": np.asarray(active_q, dtype=np.float64).tolist(),
            "active_dq": np.asarray(active_dq, dtype=np.float64).tolist(),
            "passive_q": self.controller.get_current_passive_arm_q().tolist(),
            "active_arm": self.active_arm,
            "ee_frame_name": self.kin.ee_frame_name,
        }

    def move_pose(self, pose_xyzw: Sequence[float]) -> dict:
        pose_xyzw = np.asarray(pose_xyzw, dtype=np.float64)
        if pose_xyzw.shape != (7,):
            raise ValueError(f"Expected pose shape (7,), got {pose_xyzw.shape}")

        q_all = self.controller.get_current_full_q()
        q_full = self.kin.full_q_from_motor_state(q_all)
        q_arm, success, metrics = self.kin.solve_ik(
            pose_xyzw,
            q_full,
            max_iters=FLAGS.ik_max_iters,
            pos_tol=FLAGS.ik_pos_tol,
            rot_tol=FLAGS.ik_rot_tol,
        )
        self.controller.set_active_arm_target_q(q_arm)
        return {"success": bool(success), **metrics}

    def reset_joint_home(
        self,
        active_q: Optional[Sequence[float]] = None,
        passive_q: Optional[Sequence[float]] = None,
    ) -> dict:
        active_q = np.asarray(
            self.active_arm_home_q if active_q is None else active_q,
            dtype=np.float64,
        )
        passive_q = np.asarray(
            self.passive_arm_home_q if passive_q is None else passive_q,
            dtype=np.float64,
        )
        self.controller.set_passive_arm_hold_q(passive_q)
        ok = self.controller.go_home(active_q, passive_q, timeout=8.0, tolerance=0.05)
        self.active_arm_home_q = active_q
        self.passive_arm_home_q = passive_q
        return {"success": bool(ok)}

    def shutdown(self) -> None:
        self.controller.shutdown(home_q=self.active_arm_home_q)


def main(_argv) -> None:
    server = G1ArmServer()
    atexit.register(server.shutdown)

    webapp = Flask(__name__)

    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        return jsonify(server.state_dict())

    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        return jsonify({"pose": server.state_dict()["pose"]})

    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pos_euler():
        pose = np.asarray(server.state_dict()["pose"], dtype=np.float64)
        euler = R.from_quat(pose[3:]).as_euler("xyz")
        return jsonify({"pose": np.concatenate([pose[:3], euler]).tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": server.state_dict()["vel"]})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": server.state_dict()["force"]})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": server.state_dict()["torque"]})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": server.state_dict()["q"]})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": server.state_dict()["dq"]})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": server.state_dict()["jacobian"]})

    @webapp.route("/pose", methods=["POST"])
    def pose():
        payload = request.json or {}
        arr = payload.get("arr")
        if arr is None:
            return jsonify({"success": False, "error": "JSON body must contain key 'arr'."}), 400
        result = server.move_pose(arr)
        return jsonify(result)

    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        payload = request.json or {}
        active_q = payload.get("active_q")
        passive_q = payload.get("passive_q")
        result = server.reset_joint_home(active_q=active_q, passive_q=passive_q)
        return jsonify(result)

    @webapp.route("/set_weight", methods=["POST"])
    def set_weight():
        payload = request.json or {}
        if "weight" not in payload:
            return jsonify({"success": False, "error": "JSON body must contain key 'weight'."}), 400
        server.controller.set_weight_target(float(payload["weight"]))
        return jsonify({"success": True})

    @webapp.route("/clearerr", methods=["POST"])
    def clearerr():
        # Kept for API compatibility with HIL-SERL's Franka env.
        return jsonify({"success": True})

    @webapp.route("/shutdown", methods=["POST"])
    def shutdown():
        server.shutdown()
        return jsonify({"success": True})

    webapp.run(host=FLAGS.flask_url, port=FLAGS.flask_port)


if __name__ == "__main__":
    app.run(main)
