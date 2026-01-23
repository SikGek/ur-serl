#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import threading
from typing import Any, Dict, Optional

import numpy as np
import zmq

try:
    from scipy.spatial.transform import Rotation as R
except Exception as e:
    raise RuntimeError(
        "This script requires scipy. Install with: pip install scipy\n"
        f"Original import error: {e}"
    )

try:
    from pynput import keyboard
except Exception as e:
    raise RuntimeError(
        "This script requires pynput for keyboard capture. Install with: pip install pynput\n"
        f"Original import error: {e}"
    )


def _normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


class ZmqTeleopUR5:
    """
    Minimal ZMQ teleop client:
      - subscribes to state (expects at least {"pos":[x,y,z,qx,qy,qz,qw]})
      - publishes commands:
          {"target_ee_pose":[x,y,z,qx,qy,qz,qw]}
          {"gripper_pos":0.0|1.0}   (optional)
    """

    def __init__(
        self,
        server_ip: str,
        bind_ip: str,
        cmd_port: int,
        state_port: int,
        cmd_mode: str,
        state_mode: str,
        hz: float,
        step_xyz: float,
        step_rot: float,
        print_hz: float,
    ):
        self.server_ip = server_ip
        self.bind_ip = bind_ip
        self.cmd_port = int(cmd_port)
        self.state_port = int(state_port)
        self.cmd_mode = cmd_mode
        self.state_mode = state_mode

        self.hz = float(hz)
        self.dt = 1.0 / self.hz

        self.step_xyz = float(step_xyz)
        self.step_rot = float(step_rot)
        self.print_dt = 1.0 / float(print_hz) if print_hz > 0 else 0.0

        self._ctx = zmq.Context.instance()

        # PUB: commands out
        self._pub = self._ctx.socket(zmq.PUB)
        # Keep only last few messages; avoid queue buildup
        self._pub.setsockopt(zmq.SNDHWM, 2)

        # SUB: state in
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.CONFLATE, 1)  # newest only
        self._sub.setsockopt_string(zmq.SUBSCRIBE, "")

        if self.cmd_mode == "bind":
            self._pub.bind(f"tcp://{self.bind_ip}:{self.cmd_port}")
        else:
            self._pub.connect(f"tcp://{self.server_ip}:{self.cmd_port}")

        if self.state_mode == "connect":
            self._sub.connect(f"tcp://{self.server_ip}:{self.state_port}")
        else:
            self._sub.bind(f"tcp://{self.bind_ip}:{self.state_port}")

        self._poller = zmq.Poller()
        self._poller.register(self._sub, zmq.POLLIN)

        # Keyboard state
        self._pressed = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()

        # Teleop state
        self._curr_pose = None  # np.ndarray shape (7,)
        self._target_pose = None  # np.ndarray shape (7,)
        self._gripper = 0.0  # 0=open, 1=close

        # For one-shot toggles
        self._toggle_gripper_requested = False
        self._zero_target_requested = False

    # ---------------- keyboard handling ----------------

    def _key_to_token(self, key) -> Optional[str]:
        # returns a normalized string token
        if key == keyboard.Key.esc:
            return "esc"
        if key == keyboard.Key.space:
            return "space"
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            return "shift"
        try:
            ch = key.char
            if ch is None:
                return None
            return ch.lower()
        except AttributeError:
            return None

    def _on_press(self, key):
        tok = self._key_to_token(key)
        if tok is None:
            return

        if tok == "esc":
            self._stop.set()
            return False  # stop listener

        with self._lock:
            self._pressed.add(tok)

            # one-shot actions on press
            if tok == "space":
                self._toggle_gripper_requested = True
            if tok == "p":
                self._zero_target_requested = True

    def _on_release(self, key):
        tok = self._key_to_token(key)
        if tok is None:
            return
        with self._lock:
            if tok in self._pressed:
                self._pressed.remove(tok)

    # ---------------- ZMQ helpers ----------------

    def _recv_state_latest(self, timeout_ms: int = 0) -> Optional[Dict[str, Any]]:
        socks = dict(self._poller.poll(timeout_ms))
        if self._sub not in socks:
            return None
        return self._sub.recv_json()

    def _send(self, msg: Dict[str, Any]) -> None:
        self._pub.send_json(msg)

    # ---------------- main logic ----------------

    def _update_from_state(self, st: Dict[str, Any]) -> None:
        # We only require "pos": 7 floats
        pos = st.get("pos", None)
        if pos is None:
            return
        pose = np.asarray(pos, dtype=np.float64).reshape(7)
        pose[3:] = _normalize_quat_xyzw(pose[3:])
        self._curr_pose = pose

        if self._target_pose is None:
            # initialize target to current so robot holds immediately
            self._target_pose = pose.copy()

    def _apply_keyboard_to_target(self) -> None:
        assert self._target_pose is not None

        with self._lock:
            keys = set(self._pressed)
            toggle_grip = self._toggle_gripper_requested
            zero_target = self._zero_target_requested
            self._toggle_gripper_requested = False
            self._zero_target_requested = False

        if self._curr_pose is None:
            return

        # Reset target to current pose (panic “hold here”)
        if zero_target:
            self._target_pose = self._curr_pose.copy()

        # Toggle gripper
        if toggle_grip:
            self._gripper = 0.0 if self._gripper > 0.5 else 1.0

        speed = 5.0 if "shift" in keys else 1.0
        dxyz = np.zeros(3, dtype=np.float64)
        drot = np.zeros(3, dtype=np.float64)  # rotvec (rx, ry, rz)

        # Translation (world frame)
        # w/s: +x/-x, a/d: +y/-y, r/f: +z/-z
        if "w" in keys: dxyz[0] += 1
        if "s" in keys: dxyz[0] -= 1
        if "a" in keys: dxyz[1] += 1
        if "d" in keys: dxyz[1] -= 1
        if "r" in keys: dxyz[2] += 1
        if "f" in keys: dxyz[2] -= 1

        # Rotation (about world axes; simple and good enough for teleop)
        # u/o: roll +/- (x axis), i/k: pitch +/- (y axis), j/l: yaw +/- (z axis)
        if "u" in keys: drot[0] += 1
        if "o" in keys: drot[0] -= 1
        if "i" in keys: drot[1] += 1
        if "k" in keys: drot[1] -= 1
        if "j" in keys: drot[2] += 1
        if "l" in keys: drot[2] -= 1

        # Optional: direct gripper open/close
        if "c" in keys:  # close
            self._gripper = 1.0
        if "v" in keys:  # open
            self._gripper = 0.0

        if np.any(dxyz != 0):
            self._target_pose[:3] += dxyz * (self.step_xyz * speed)

        if np.any(drot != 0):
            q = self._target_pose[3:]
            dq = R.from_rotvec(drot * (self.step_rot * speed)).as_quat()
            q_new = (R.from_quat(dq) * R.from_quat(q)).as_quat()
            self._target_pose[3:] = _normalize_quat_xyzw(q_new)

    def run(self) -> None:
        print("\n=== ZMQ UR5 Teleop (keyboard) ===")
        print("Keys:")
        print("  Move:  W/S (+x/-x)   A/D (+y/-y)   R/F (+z/-z)")
        print("  Rot:   U/O (roll +/-)   I/K (pitch +/-)   J/L (yaw +/-)")
        print("  Gripper: SPACE toggle, C close, V open")
        print("  Hold/reset target to current pose: P")
        print("  Speed boost: hold SHIFT")
        print("  Quit: ESC\n")

        # PUB/SUB slow joiner: give time for connections
        time.sleep(0.25)

        # Wait for first state so we can initialize target pose
        t0 = time.monotonic()
        while self._curr_pose is None and not self._stop.is_set():
            st = self._recv_state_latest(timeout_ms=200)
            if st is not None:
                self._update_from_state(st)
            if (time.monotonic() - t0) > 5.0:
                raise RuntimeError(
                    "No state received from server within 5s. "
                    "Check state_port/mode and that server is publishing state."
                )

        # Start keyboard listener
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.start()

        last_print = time.monotonic()

        try:
            t_next = time.monotonic()
            while not self._stop.is_set():
                t_next += self.dt

                # Drain newest state quickly
                st = self._recv_state_latest(timeout_ms=0)
                print(st)
                if st is not None:
                    self._update_from_state(st)

                if self._target_pose is None:
                    # shouldn't happen after state received
                    continue

                # Apply key deltas
                self._apply_keyboard_to_target()

                # Continuously stream pose command (prevents “go limp” timeouts)
                self._send({"target_ee_pose": self._target_pose.astype(float).tolist()})

                # Stream gripper (safe to stream every tick)
                # self._send({"gripper_pos": float(self._gripper)})

                # Print status occasionally
                # if self.print_dt > 0 and (time.monotonic() - last_print) >= self.print_dt:
                #     last_print = time.monotonic()
                #     cp = self._curr_pose if self._curr_pose is not None else self._target_pose
                #     tp = self._target_pose
                #     print(
                #         f"curr xyz={cp[:3].round(3)}  "
                #         f"target xyz={tp[:3].round(3)}  "
                #         f"grip={self._gripper:.0f}"
                #     )

                # Sleep to maintain rate
                to_sleep = t_next - time.monotonic()
                if to_sleep > 0:
                    time.sleep(to_sleep)

        finally:
            self._stop.set()
            try:
                listener.stop()
            except Exception:
                pass

            # Send a final hold command
            if self._target_pose is not None:
                try:
                    self._send({"target_ee_pose": self._target_pose.astype(float).tolist()})
                except Exception:
                    pass

            print("\nTeleop stopped.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_ip", type=str, default="127.0.0.1")
    ap.add_argument("--bind_ip", type=str, default="127.0.0.1")

    ap.add_argument("--cmd_port", type=int, default=5555)
    ap.add_argument("--state_port", type=int, default=5556)

    # IMPORTANT: these must match your server’s expectation
    ap.add_argument("--cmd_mode", choices=["bind", "connect"], default="bind",
                    help="Where THIS client creates the cmd PUB socket.")
    ap.add_argument("--state_mode", choices=["bind", "connect"], default="connect",
                    help="Where THIS client creates the state SUB socket.")

    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--step_xyz", type=float, default=0.005, help="meters per tick")
    ap.add_argument("--step_rot", type=float, default=0.05, help="radians per tick")
    ap.add_argument("--print_hz", type=float, default=10.0)

    args = ap.parse_args()

    teleop = ZmqTeleopUR5(
        server_ip=args.server_ip,
        bind_ip=args.bind_ip,
        cmd_port=args.cmd_port,
        state_port=args.state_port,
        cmd_mode=args.cmd_mode,
        state_mode=args.state_mode,
        hz=args.hz,
        step_xyz=args.step_xyz,
        step_rot=args.step_rot,
        print_hz=args.print_hz,
    )
    teleop.run()


if __name__ == "__main__":
    main()
