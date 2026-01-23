from __future__ import annotations

import time
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np
import zmq
from scipy.spatial.transform import Rotation as R


def _now_ms() -> int:
    return int(time.time() * 1000)


class MujocoUrImpedanceController(threading.Thread):
    """
    Drop-in controller for UR5Env that talks to a running MuJoCo ZMQ server.

    It implements the *same interface UR5Env expects*:
      - start(), stop(), is_ready(), is_reset(), is_truncated(), is_moving()
      - set_target_pose(target_pose=...)
      - set_reset_angles(reset_Q)
      - set_gripper_pos(gripper_pos)
      - get_state()

    It continuously streams target_ee_pose at CONTROLLER_HZ so it behaves like
    a real controller process (no "one-shot" commands).
    """

    def __init__(self, robot_ip: str = "127.0.0.1", config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daemon = True

        self.config = config
        self.robot_ip = robot_ip  # not used in sim (kept for compatibility)

        # Frequency
        self.frequency = int(getattr(config, "CONTROLLER_HZ", 100))
        self.dt = 1.0 / float(self.frequency)

        # ZMQ ports (Voxel-SERL style naming)
        self.cmd_port = int(getattr(config, "ZEROMQ_PUBLISHER_PORT", 5555))
        self.state_port = int(getattr(config, "ZEROMQ_SUBSCRIBER_PORT", 5556))

        # IPs / modes (defaults match typical “client binds cmd PUB, connects state SUB”)
        self.bind_ip = getattr(config, "ZMQ_BIND_IP", "127.0.0.1")
        self.server_ip = getattr(config, "ZMQ_SERVER_IP", "127.0.0.1")
        self.cmd_mode = getattr(config, "ZMQ_CMD_MODE", "bind")      # "bind" or "connect"
        self.state_mode = getattr(config, "ZMQ_STATE_MODE", "connect")  # "bind" or "connect"

        # Controller gains for optional smoothing (safe defaults)
        self.kp_pos = float(getattr(config, "SIM_KP_POS", 6.0))   # 1/s
        self.kd_pos = float(getattr(config, "SIM_KD_POS", 1.0))   # unitless-ish
        self.kp_rot = float(getattr(config, "SIM_KP_ROT", 6.0))   # 1/s
        self.kd_rot = float(getattr(config, "SIM_KD_ROT", 1.0))

        # Step clamps (prevents instability)
        self.max_step_xyz = float(getattr(config, "SIM_MAX_STEP_XYZ", 0.01))  # m per tick
        self.max_step_rot = float(getattr(config, "SIM_MAX_STEP_ROT", 0.15))  # rad per tick

        # Reset config
        self.reset_tol = float(getattr(config, "SIM_RESET_Q_TOL", 0.05))
        self.reset_timeout_s = float(getattr(config, "SIM_RESET_TIMEOUT_S", 2.5))

        # Threading / state
        self._stop = threading.Event()
        self._reset = threading.Event()
        self._ready = threading.Event()
        self._truncated = threading.Event()
        self._lock = threading.Lock()

        # Targets
        self._target_pose = np.zeros((7,), dtype=np.float64)  # xyz + quat xyzw
        self._target_pose_valid = False
        self._reset_q = np.zeros((6,), dtype=np.float64)

        self._gripper_cmd = 0.0  # 0 open, 1 close
        self._last_sent_gripper = None

        # Latest state cache (what UR5Env reads)
        self._state: Dict[str, Any] = {
            "pos": np.zeros((7,), dtype=np.float32),
            "vel": np.zeros((6,), dtype=np.float32),
            "Q": np.zeros((6,), dtype=np.float32),
            "Qd": np.zeros((6,), dtype=np.float32),
            "force": np.zeros((6,), dtype=np.float32),
            "timestamp_ms": 0,
            "is_truncated": 0,
            "gripper": np.zeros((2,), dtype=np.float32),
        }

        # ZMQ sockets (created in run() thread)
        self._ctx = None
        self._cmd_pub = None
        self._state_sub = None
        self._poller = None

    # ------------------- public API expected by UR5Env -------------------

    def stop(self):
        self._stop.set()

    def stopped(self) -> bool:
        return self._stop.is_set()

    def is_ready(self) -> bool:
        return self._ready.is_set()

    def is_reset(self) -> bool:
        return not self._reset.is_set()

    def is_truncated(self) -> bool:
        return self._truncated.is_set()

    def is_moving(self) -> bool:
        st = self.get_state()
        return float(np.linalg.norm(st["vel"])) > 1e-3

    def set_target_pose(self, target_pose: np.ndarray, **kwargs):
        pose = np.asarray(target_pose, dtype=np.float64).reshape(7)
        # normalize quat (xyzw)
        q = pose[3:]
        n = np.linalg.norm(q)
        if n > 1e-9:
            pose[3:] = q / n
        with self._lock:
            self._target_pose[:] = pose
            self._target_pose_valid = True

    def set_gripper_pos(self, gripper_pos: np.ndarray | float, **kwargs):
        # UR5Env sends scalar or np.array([..]) depending on neutral flag
        g = float(np.asarray(gripper_pos).reshape(-1)[0])
        # match Voxel convention: action in [-1,1] typically
        # map: >0.5 => close, < -0.5 => open, else hold
        if g > 0.5:
            cmd = 1.0
        elif g < -0.5:
            cmd = 0.0
        else:
            cmd = self._gripper_cmd
        with self._lock:
            self._gripper_cmd = cmd

    def set_reset_angles(self, reset_Q: np.ndarray, **kwargs):
        q = np.asarray(reset_Q, dtype=np.float64).reshape(6)
        with self._lock:
            self._reset_q[:] = q
        self._reset.set()

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            # return numpy arrays (UR5Env assigns into numpy buffers)
            return {
                "pos": self._state["pos"].copy(),
                "vel": self._state["vel"].copy(),
                "Q": self._state["Q"].copy(),
                "Qd": self._state["Qd"].copy(),
                "force": self._state["force"].copy(),  # 6D
                "timestamp_ms": int(self._state["timestamp_ms"]),
                "is_truncated": int(self._state["is_truncated"]),
                "gripper": self._state["gripper"].copy(),
            }

    # ------------------- internal ZMQ helpers -------------------

    def _zmq_setup(self):
        self._ctx = zmq.Context.instance()

        self._cmd_pub = self._ctx.socket(zmq.PUB)
        if self.cmd_mode == "bind":
            self._cmd_pub.bind(f"tcp://{self.bind_ip}:{self.cmd_port}")
        else:
            self._cmd_pub.connect(f"tcp://{self.server_ip}:{self.cmd_port}")

        self._state_sub = self._ctx.socket(zmq.SUB)
        self._state_sub.setsockopt(zmq.CONFLATE, 1)     # newest only
        self._state_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        if self.state_mode == "connect":
            self._state_sub.connect(f"tcp://{self.server_ip}:{self.state_port}")
        else:
            self._state_sub.bind(f"tcp://{self.bind_ip}:{self.state_port}")

        self._poller = zmq.Poller()
        self._poller.register(self._state_sub, zmq.POLLIN)

        # allow sockets to connect (PUB/SUB “slow joiner”)
        time.sleep(0.2)

    def _send(self, msg: Dict[str, Any]):
        assert self._cmd_pub is not None
        self._cmd_pub.send_json(msg)

    def _recv_latest_state(self, timeout_ms: int = 0) -> Optional[Dict[str, Any]]:
        assert self._poller is not None
        socks = dict(self._poller.poll(timeout_ms))
        if self._state_sub not in socks:
            return None
        assert self._state_sub is not None
        st = self._state_sub.recv_json()
        return st

    def _update_state_cache(self, st: Dict[str, Any]):
        # Fill missing keys safely
        pos = np.asarray(st.get("pos", self._state["pos"]), dtype=np.float32).reshape(7)
        vel = np.asarray(st.get("vel", self._state["vel"]), dtype=np.float32).reshape(6)
        Q = np.asarray(st.get("Q", self._state["Q"]), dtype=np.float32).reshape(6)
        Qd = np.asarray(st.get("Qd", self._state["Qd"]), dtype=np.float32).reshape(6)

        force = st.get("force", None)
        if force is None:
            force6 = np.zeros((6,), dtype=np.float32)
        else:
            force_arr = np.asarray(force, dtype=np.float32).reshape(-1)
            if force_arr.shape[0] == 6:
                force6 = force_arr
            elif force_arr.shape[0] == 3:
                force6 = np.concatenate([force_arr, np.zeros((3,), dtype=np.float32)], axis=0)
            else:
                force6 = np.zeros((6,), dtype=np.float32)

        ts = int(st.get("timestamp_ms", _now_ms()))
        is_trunc = int(st.get("is_truncated", 0))

        grip = st.get("gripper", None)
        if grip is None:
            # keep a reasonable sim gripper state
            grip2 = np.array([self._gripper_cmd, 1.0 if self._gripper_cmd > 0.5 else 0.0], dtype=np.float32)
        else:
            grip2 = np.asarray(grip, dtype=np.float32).reshape(2)

        with self._lock:
            self._state["pos"] = pos
            self._state["vel"] = vel
            self._state["Q"] = Q
            self._state["Qd"] = Qd
            self._state["force"] = force6
            self._state["timestamp_ms"] = ts
            self._state["is_truncated"] = is_trunc
            self._state["gripper"] = grip2

        if is_trunc:
            self._truncated.set()
        else:
            self._truncated.clear()

    # ------------------- reset behavior -------------------

    def _do_reset(self):
        with self._lock:
            q_des = self._reset_q.copy()

        # Ask server to reset if supported, then command joints
        self._send({"sim_reset": True})
        time.sleep(0.05)
        self._send({"target_q": q_des.tolist()})

        # wait until near q_des or timeout
        t0 = time.monotonic()
        while (time.monotonic() - t0) < self.reset_timeout_s and not self.stopped():
            st = self._recv_latest_state(timeout_ms=50)
            if st is not None:
                self._update_state_cache(st)
                q = self.get_state()["Q"]
                if float(np.max(np.abs(q - q_des))) < self.reset_tol:
                    break
            time.sleep(0.01)

        # after reset, latch target_pose to current pose to avoid jumps
        curr = self.get_state()["pos"]
        self.set_target_pose(curr)

        self._reset.clear()

    # ------------------- controller loop -------------------

    def _compute_stream_pose(self) -> np.ndarray:
        """
        Convert (target_pose, current state) -> streamed pose command.
        This is a stable “impedance-like” cartesian servo (bounded).
        """
        st = self.get_state()
        x = st["pos"].astype(np.float64)
        v = st["vel"].astype(np.float64)

        with self._lock:
            if not self._target_pose_valid:
                # no target yet: hold current
                self._target_pose[:] = x
                self._target_pose_valid = True
            x_des = self._target_pose.copy()

        # position error
        e_p = x_des[:3] - x[:3]
        v_p = v[:3]

        # rotation error as rotvec
        q = x[3:]
        qd = x_des[3:]
        e_r = (R.from_quat(qd) * R.from_quat(q).inv()).as_rotvec()
        w = v[3:]

        # cartesian PD -> desired delta pose per tick
        dp = (self.kp_pos * e_p - self.kd_pos * v_p) * self.dt
        dp = np.clip(dp, -self.max_step_xyz, self.max_step_xyz)

        drot = (self.kp_rot * e_r - self.kd_rot * w) * self.dt
        drot = np.clip(drot, -self.max_step_rot, self.max_step_rot)

        x_cmd = x.copy()
        x_cmd[:3] = x[:3] + dp
        x_cmd[3:] = (R.from_rotvec(drot) * R.from_quat(q)).as_quat()

        return x_cmd.astype(np.float64)

    def run(self):
        self._zmq_setup()

        # Wait for first state
        while not self.stopped():
            st = self._recv_latest_state(timeout_ms=500)
            if st is not None:
                self._update_state_cache(st)
                self._ready.set()
                break

        t_next = time.monotonic()

        while not self.stopped():
            t_next += self.dt

            # drain latest state quickly
            st = self._recv_latest_state(timeout_ms=0)
            if st is not None:
                self._update_state_cache(st)

            # handle reset
            if self._reset.is_set():
                self._do_reset()

            # stream pose command continuously
            pose_cmd = self._compute_stream_pose()
            self._send({"target_ee_pose": pose_cmd.tolist()})

            # stream gripper only when changed
            with self._lock:
                g = float(self._gripper_cmd)
            if self._last_sent_gripper is None or abs(g - self._last_sent_gripper) > 1e-6:
                self._send({"gripper_pos": g})
                self._last_sent_gripper = g

            # sleep to maintain rate
            to_sleep = t_next - time.monotonic()
            if to_sleep > 0:
                time.sleep(to_sleep)

        # On exit: keep holding last target pose once
        try:
            st = self.get_state()
            self._send({"target_ee_pose": st["pos"].astype(float).tolist()})
        except Exception:
            pass
