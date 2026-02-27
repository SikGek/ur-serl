from __future__ import annotations

import time
import copy
from typing import Optional

import numpy as np
import gymnasium as gym
import jax
import jax.numpy as jnp

from ur_env.envs.ur5_env import UR5Env
from ur_env.utils.rotations import quat_2_mrp
from ur_env.camera.video_capture import VideoCapture
from ur_env.camera.rs_capture import RSCapture

# -----------------------------
# Base task env (same idea)
# -----------------------------

class UR5EDoorPullEnv(UR5Env):
    """
    Door pulling task env:
    - reward is provided by a classifier wrapper
    - gripper is assumed to be closed on reset (object already held)
    """
    def __init__(
        self,
        *,
        config,
        hz=10,
        fake_env=False,
        max_episode_length=180,
        save_video=False,
        camera_mode="rgb",
        close_gripper_on_reset: bool = True,
        close_wait_s: float = 0.6,
    ):
        self.task_cfg = config
        self.close_gripper_on_reset = bool(close_gripper_on_reset)
        self.close_wait_s = float(close_wait_s)

        super().__init__(
            hz=hz,
            fake_env=fake_env,
            config=config,
            max_episode_length=max_episode_length,
            save_video=save_video,
            camera_mode=camera_mode,
        )

        # Ensure "action" exists in state space if your pipelines expect it
        if isinstance(self.observation_space, gym.spaces.Dict):
            st = self.observation_space.spaces.get("state", None)
            if isinstance(st, gym.spaces.Dict) and "action" not in st.spaces:
                st.spaces["action"] = gym.spaces.Box(
                    -1.0, 1.0, shape=self.action_space.shape, dtype=np.float32
                )
    def init_cameras(self, name_serial_dict=None):
        if self.cap is not None:
            self.close_cameras()

        self.cap = {}
        for cam_name, cam_spec in name_serial_dict.items():
            rgb = self.camera_mode in ["rgb", "both", "grey"]
            depth = self.camera_mode in ["depth", "both"]
            pointcloud = self.camera_mode in ["pointcloud"]

            if isinstance(cam_spec, str):
                kwargs = {"serial_number": cam_spec}
            else:
                kwargs = dict(cam_spec)
                # normalize key name
                if "serial" in kwargs and "serial_number" not in kwargs:
                    kwargs["serial_number"] = kwargs.pop("serial")

            cap = VideoCapture(
                RSCapture(name=cam_name, rgb=rgb, depth=depth, pointcloud=pointcloud, **kwargs)
            )
            self.cap[cam_name] = cap
            
    def crop_image(self, name: str, image: np.ndarray) -> np.ndarray:
        # Optional task-specific crop: config.IMAGE_CROP[name](img)
        if hasattr(self.task_cfg, "IMAGE_CROP") and name in self.task_cfg.IMAGE_CROP:
            return self.task_cfg.IMAGE_CROP[name](image)
        return super().crop_image(name, image)
    def _get_obs(self, action) -> dict:
        images = None
        if self.camera_mode is not None:
            images = self.get_image()

        self._update_currpos()

        state_observation = {
            "tcp_pose": self.curr_pos.astype(np.float32),     # quat (7) here
            "tcp_vel": self.curr_vel.astype(np.float32),
            "tcp_force": self.curr_force.astype(np.float32),
            "tcp_torque": self.curr_torque.astype(np.float32),
            "gripper_state": self.gripper_state.astype(np.float32),
            "gripper_pose": np.array([self.gripper_state[0]], dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
        }

        obs = {"state": state_observation}
        if images is not None:
            obs["images"] = images
        return copy.deepcopy(obs)

    def compute_reward(self, obs, action) -> float:
        return 0.0  # classifier wrapper handles reward

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        # IMPORTANT: base controller reset opens the gripper.
        # For pulling stage we want "already grasped" start.
        if (not self.close_gripper_on_reset) or getattr(self, "controller", None) is None:
            obs = self._get_obs(np.zeros_like(self.last_action))
            return obs, info

        # Issue a CLOSE command once at reset start.
        # This works because controller last_grip is generally old enough at reset.
        self._send_gripper_command(np.array([1.0], dtype=np.float32))
        time.sleep(self.close_wait_s)

        obs = self._get_obs(np.zeros_like(self.last_action))
        return obs, info


# -----------------------------
# Manual reset helper
# -----------------------------

class DoorPullManualResetWrapper(gym.Wrapper):
    """
    For pulling stage, you need the handle grasped.
    This is the RAM-insertion-style assumption: object already held.
    """
    def __init__(self, env: gym.Env, prompt_every_reset: bool = True):
        super().__init__(env)
        self.prompt_every_reset = bool(prompt_every_reset)

    def reset(self, **kwargs):
        self.env.unwrapped._send_gripper_command(np.array([-1.0]))
        time.sleep(0.4)
        if self.prompt_every_reset:
            input(
                "\n[RESET - PULL] Close the door fully.\n"
                "Make sure the robot is positioned over the handle (reset pose),\n"
                "then press Enter to let the env close the gripper and start the pull episode..."
            )
        return self.env.reset(**kwargs)


# -----------------------------
# Quat -> MRP wrapper
# -----------------------------

class ToMrpWrapper(gym.ObservationWrapper):
    """
    Convert tcp_pose orientation from quat -> MRP:
    tcp_pose becomes (x,y,z,mrp_x,mrp_y,mrp_z) length 6
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        tcp_pose_mrp = np.concatenate((tcp_pose[:3], quat_2_mrp(tcp_pose[3:])))
        observation["state"]["tcp_pose"] = tcp_pose_mrp.astype(np.float32)
        return observation


# -----------------------------
# Binary classifier terminate wrapper (reuse your version)
# -----------------------------

class RewardClassifierTerminateWrapper(gym.Wrapper):
    """
    prob_func(obs)->[0,1]
    reward=1 on success, terminate on success
    """
    def __init__(
        self,
        env,
        prob_func,
        threshold=0.85,
        consecutive=2,
        target_hz=None,
        trunc_penalty=0.0,
        pass_env_reward=False,
    ):
        super().__init__(env)
        self.prob_func = prob_func
        self.threshold = float(threshold)
        self.consecutive = int(consecutive)
        self.target_hz = float(target_hz) if target_hz is not None else None
        self.trunc_penalty = float(trunc_penalty)
        self.pass_env_reward = bool(pass_env_reward)
        self._streak = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._streak = 0
        info["succeed"] = False
        info["reward_clf_prob"] = 0.0
        return obs, info

    def _to_float(self, x) -> float:
        x_host = jax.device_get(x)
        return float(np.asarray(x_host).reshape(-1)[0])

    def step(self, action):
        t0 = time.time()
        obs, env_rew, terminated, truncated, info = self.env.step(action)

        p = float(np.clip(self._to_float(self.prob_func(obs)), 0.0, 1.0))
        is_pos = (p >= self.threshold)

        if truncated:
            self._streak = 0
            success = False
        else:
            self._streak = (self._streak + 1) if is_pos else 0
            success = (self._streak >= self.consecutive)

        reward = float(env_rew) if self.pass_env_reward else 0.0
        reward += 1.0 if success else 0.0
        if truncated:
            reward += self.trunc_penalty

        if success and not truncated:
            terminated = True
        if truncated:
            terminated = False

        info["succeed"] = bool(success)
        info["reward_clf_prob"] = p
        info["reward_clf_pos"] = bool(is_pos)
        info["reward_clf_streak"] = int(self._streak)

        if self.target_hz is not None:
            time.sleep(max(0.0, 1.0 / self.target_hz - (time.time() - t0)))

        return obs, float(reward), bool(terminated), bool(truncated), info
    
class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, penalty: float = -0.02):
        super().__init__(env)
        self.penalty = float(penalty)
        self.last_closed_norm = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_closed_norm = float(self.env.unwrapped.gripper_state[0])
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        effective_action = info.get("intervene_action", action)
        closed_norm = float(self.env.unwrapped.gripper_state[0])

        # Proper toggling/redundant-command detection for your convention:
        #  - action >  0.5 => CLOSE
        #  - action < -0.5 => OPEN
        toggling = (
            (effective_action[-1] >  0.5 and self.last_closed_norm > 0.9) or  # close while already closed
            (effective_action[-1] < -0.5 and self.last_closed_norm < 0.1)     # open while already open
        )

        info["grasp_penalty"] = self.penalty if toggling else 0.0
        self.last_closed_norm = closed_norm
        return obs, reward, term, trunc, info