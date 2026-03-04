"""Gymnasium environment for single-arm HIL-SERL on Unitree G1."""

from __future__ import annotations

import copy
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from g1_env.robot.camera_utils import G1CameraManager


class _ImageDisplayer(threading.Thread):
    """Best-effort image display thread; ignored if OpenCV windows are unavailable."""

    def __init__(self, img_queue: "queue.Queue[Optional[Dict[str, np.ndarray]]]", name: str):
        super().__init__(daemon=True)
        self.queue = img_queue
        self.name = name

    def run(self) -> None:  # pragma: no cover - requires GUI.
        while True:
            item = self.queue.get()
            if item is None:
                return
            try:
                tiles = [cv2.resize(img[..., ::-1], (128, 128)) for img in item.values()]
                if not tiles:
                    continue
                frame = np.concatenate(tiles, axis=1)
                cv2.imshow(self.name, frame)
                cv2.waitKey(1)
            except Exception:
                # Display should never crash training.
                pass


@dataclass
class G1EnvConfig:
    """Runtime configuration for the G1 arm-only HIL-SERL environment."""

    SERVER_URL: str = "http://127.0.0.1:5001/"
    ACTIVE_ARM: str = "left"
    ACTION_SCALE_TRANSLATION: float = 0.01
    ACTION_SCALE_ROTATION: float = 0.12

    # All poses below are Euler-format [x, y, z, roll, pitch, yaw].
    TARGET_POSE: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    RESET_POSE: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    REWARD_THRESHOLD: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.2, 0.2, 0.2], dtype=np.float64)
    )

    ABS_POSE_LIMIT_LOW: np.ndarray = field(
        default_factory=lambda: np.array([0.15, -0.20, 0.45, 2.2, -0.8, 0.5], dtype=np.float64)
    )
    ABS_POSE_LIMIT_HIGH: np.ndarray = field(
        default_factory=lambda: np.array([0.75, 0.55, 1.00, 3.9, 0.8, 2.6], dtype=np.float64)
    )

    RANDOM_RESET: bool = False
    RANDOM_XY_RANGE: float = 0.0
    RANDOM_RPY_RANGE: float = 0.0

    DISPLAY_IMAGE: bool = True
    MAX_EPISODE_LENGTH: int = 80
    RESET_INTERPOLATION_TIME: float = 2.0
    POST_RESET_SETTLE_SEC: float = 0.5
    IMAGE_SIZE: Tuple[int, int] = (128, 128)

    # Optional joint-space homes for the active and passive arms.
    RESET_Q_ARM: Optional[np.ndarray] = None
    PASSIVE_ARM_HOME_Q: Optional[np.ndarray] = None

    # Camera configuration dictionary passed into :class:`G1CameraManager`.
    CAMERAS: Mapping[str, Mapping] = field(default_factory=dict)


class G1ArmEnv(gym.Env):
    """Single-arm Cartesian control environment over a Flask-served G1 arm controller.

    Design choices:
        - The *robot server* owns DDS + IK + joint publishing.
        - The *gym environment* owns cameras + episode accounting + action/observation
          shaping. This mirrors HIL-SERL's Franka split and keeps robot-side latency low.
        - Actions are 6D Cartesian deltas in the pelvis/torso frame. The RelativeFrame
          wrapper from HIL-SERL can be stacked above this environment so that both the
          policy and the SpaceMouse operate in end-effector coordinates instead.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        hz: int = 10,
        fake_env: bool = False,
        save_video: bool = False,
        config: Optional[G1EnvConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or G1EnvConfig()
        self.hz = int(hz)
        self.fake_env = bool(fake_env)
        self.save_video = bool(save_video)

        self.url = self.config.SERVER_URL.rstrip("/") + "/"
        self.active_arm = self.config.ACTIVE_ARM
        self.action_scale = np.array(
            [
                self.config.ACTION_SCALE_TRANSLATION,
                self.config.ACTION_SCALE_ROTATION,
            ],
            dtype=np.float64,
        )

        self.max_episode_length = int(self.config.MAX_EPISODE_LENGTH)
        self.display_image = bool(self.config.DISPLAY_IMAGE)
        self.reset_interpolation_time = float(self.config.RESET_INTERPOLATION_TIME)
        self.post_reset_settle_sec = float(self.config.POST_RESET_SETTLE_SEC)

        self.curr_path_length = 0
        self.terminate = False
        self.currpos = np.zeros(7, dtype=np.float64)
        self.nextpos = np.zeros(7, dtype=np.float64)
        self.recording_frames = []

        self.xyz_bounding_box = gym.spaces.Box(
            self.config.ABS_POSE_LIMIT_LOW[:3],
            self.config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            self.config.ABS_POSE_LIMIT_LOW[3:],
            self.config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )

        self.action_space = gym.spaces.Box(
            low=np.full((6,), -1.0, dtype=np.float32),
            high=np.full((6,), 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                        "q_arm": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                        "dq_arm": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(
                            0,
                            255,
                            shape=(*self.config.IMAGE_SIZE, 3),
                            dtype=np.uint8,
                        )
                        for key in self.config.CAMERAS.keys()
                    }
                ),
            }
        )

        if self.fake_env:
            return

        self._verify_server()

        self.camera_manager: Optional[G1CameraManager] = None
        if self.config.CAMERAS:
            self.camera_manager = G1CameraManager(
                self.config.CAMERAS,
                output_hw=self.config.IMAGE_SIZE,
            )

        if self.display_image and self.camera_manager is not None:
            self.img_queue: "queue.Queue[Optional[Dict[str, np.ndarray]]]" = queue.Queue()
            self.displayer = _ImageDisplayer(self.img_queue, self.url)
            self.displayer.start()

        self._install_escape_listener()
        self._update_currpos()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.curr_path_length = 0
        self.terminate = False

        self.go_to_reset()
        time.sleep(self.post_reset_settle_sec)
        self._update_currpos()

        obs = self._get_obs()
        info = {"succeed": False}
        return obs, info

    def step(self, action: np.ndarray):
        start_time = time.time()
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        xyz_delta = action[:3] * self.action_scale[0]
        rot_delta = action[3:6] * self.action_scale[1]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] += xyz_delta
        self.nextpos[3:] = (
            Rotation.from_rotvec(rot_delta) * Rotation.from_quat(self.currpos[3:])
        ).as_quat()
        self.nextpos = self.clip_safety_box(self.nextpos)

        self._send_pose_command(self.nextpos)
        self.curr_path_length += 1

        elapsed = time.time() - start_time
        time.sleep(max(0.0, (1.0 / self.hz) - elapsed))

        self._update_currpos()
        obs = self._get_obs()
        reward = self.compute_reward(obs)
        done = bool(reward) or self.curr_path_length >= self.max_episode_length or self.terminate
        info = {"succeed": bool(reward)}
        return obs, int(reward), done, False, info

    def close(self) -> None:
        if getattr(self, "camera_manager", None) is not None:
            self.camera_manager.close()
        if getattr(self, "img_queue", None) is not None:
            try:
                self.img_queue.put(None)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Reset, reward, state
    # ------------------------------------------------------------------
    def go_to_reset(self) -> None:
        """Return to the configured arm home and optionally a randomized reset pose."""
        payload: Dict[str, object] = {}
        if self.config.RESET_Q_ARM is not None:
            payload["active_q"] = np.asarray(self.config.RESET_Q_ARM, dtype=np.float64).tolist()
        if self.config.PASSIVE_ARM_HOME_Q is not None:
            payload["passive_q"] = np.asarray(
                self.config.PASSIVE_ARM_HOME_Q, dtype=np.float64
            ).tolist()

        # Home the arm(s) in joint space first. This gives a reproducible IK seed.
        self._post_json("jointreset", payload=payload if payload else None, timeout=15.0)

        # Optionally move to a Cartesian reset pose around which demonstrations / RL start.
        reset_pose_euler = np.asarray(self.config.RESET_POSE, dtype=np.float64).copy()
        if self.config.RANDOM_RESET:
            reset_pose_euler[:2] += self.np_random.uniform(
                low=-self.config.RANDOM_XY_RANGE,
                high=self.config.RANDOM_XY_RANGE,
                size=(2,),
            )
            reset_pose_euler[3:] += self.np_random.uniform(
                low=-self.config.RANDOM_RPY_RANGE,
                high=self.config.RANDOM_RPY_RANGE,
                size=(3,),
            )

        if np.any(np.abs(reset_pose_euler) > 1e-9):
            self.interpolate_move(reset_pose_euler, timeout=self.reset_interpolation_time)

    def compute_reward(self, obs: dict) -> bool:
        """Simple geometric reward used for debugging before the classifier is trained."""
        target = np.asarray(self.config.TARGET_POSE, dtype=np.float64)
        if not np.any(np.abs(target) > 1e-9):
            return False

        current_pose_quat = np.asarray(obs["state"]["tcp_pose"], dtype=np.float64)
        current_euler = np.concatenate(
            [
                current_pose_quat[:3],
                Rotation.from_quat(current_pose_quat[3:]).as_euler("xyz"),
            ]
        )
        delta = np.abs(current_euler - target)
        return bool(np.all(delta < self.config.REWARD_THRESHOLD))

    def _get_obs(self) -> dict:
        state = self._post_json("getstate", timeout=5.0)

        images = {}
        full_res = {}
        if self.camera_manager is not None:
            images, full_res = self.camera_manager.read()
            if self.save_video:
                self.recording_frames.append(copy.deepcopy(full_res))
            if self.display_image:
                self.img_queue.put(copy.deepcopy(images))

        obs = {
            "state": {
                "tcp_pose": np.asarray(state["pose"], dtype=np.float32),
                "tcp_vel": np.asarray(state["vel"], dtype=np.float32),
                "tcp_force": np.asarray(state["force"], dtype=np.float32),
                "tcp_torque": np.asarray(state["torque"], dtype=np.float32),
                "q_arm": np.asarray(state["active_q"], dtype=np.float32),
                "dq_arm": np.asarray(state["active_dq"], dtype=np.float32),
            },
            "images": {key: np.asarray(img, dtype=np.uint8) for key, img in images.items()},
        }
        return obs

    def _update_currpos(self) -> None:
        data = self._post_json("getpos", timeout=5.0)
        self.currpos = np.asarray(data["pose"], dtype=np.float64)

    # ------------------------------------------------------------------
    # Motion helpers
    # ------------------------------------------------------------------
    def clip_safety_box(self, pose_xyzw: np.ndarray) -> np.ndarray:
        """Clip a quaternion pose into the configured Cartesian workspace bounds."""
        pose_xyzw = np.asarray(pose_xyzw, dtype=np.float64).copy()
        pose_xyzw[:3] = np.clip(
            pose_xyzw[:3],
            self.xyz_bounding_box.low,
            self.xyz_bounding_box.high,
        )

        euler = Rotation.from_quat(pose_xyzw[3:]).as_euler("xyz")
        euler = np.clip(euler, self.rpy_bounding_box.low, self.rpy_bounding_box.high)
        pose_xyzw[3:] = Rotation.from_euler("xyz", euler).as_quat()
        return pose_xyzw

    def interpolate_move(self, goal_pose_euler: np.ndarray, timeout: float) -> None:
        """Move to a Cartesian goal by linearly interpolating the current quaternion pose."""
        goal_pose_euler = np.asarray(goal_pose_euler, dtype=np.float64)
        if goal_pose_euler.shape != (6,):
            raise ValueError(f"Expected goal_pose_euler shape (6,), got {goal_pose_euler.shape}")

        goal_quat = np.concatenate(
            [goal_pose_euler[:3], Rotation.from_euler("xyz", goal_pose_euler[3:]).as_quat()]
        )

        self._update_currpos()
        steps = max(2, int(timeout * self.hz))
        path = np.linspace(self.currpos[:3], goal_quat[:3], steps)
        rot_start = Rotation.from_quat(self.currpos[3:])
        rot_goal = Rotation.from_quat(goal_quat[3:])

        # Approximate slerp by interpolating rotvec in the relative frame.
        relative = rot_goal * rot_start.inv()
        relative_rotvec = relative.as_rotvec()

        for alpha, xyz in zip(np.linspace(0.0, 1.0, steps), path):
            rot = Rotation.from_rotvec(relative_rotvec * alpha) * rot_start
            pose = np.concatenate([xyz, rot.as_quat()])
            self._send_pose_command(self.clip_safety_box(pose))
            time.sleep(1.0 / self.hz)

    def _send_pose_command(self, pose_xyzw: np.ndarray) -> None:
        self._post_json("pose", {"arr": np.asarray(pose_xyzw, dtype=np.float64).tolist()}, timeout=5.0)

    # ------------------------------------------------------------------
    # HTTP utilities
    # ------------------------------------------------------------------
    def _verify_server(self) -> None:
        try:
            self._post_json("getstate", timeout=5.0)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to contact the G1 server at {self.url!r}. "
                "Start `g1_arm_server.py` first and verify the SERVER_URL."
            ) from exc

    def _post_json(self, route: str, payload: Optional[dict] = None, timeout: float = 3.0) -> dict:
        resp = requests.post(
            self.url + route,
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        return {}

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def _install_escape_listener(self) -> None:
        try:
            from pynput import keyboard
        except ImportError:
            return

        def on_press(key):
            if key == keyboard.Key.esc:
                self.terminate = True

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.daemon = True
        self.listener.start()
