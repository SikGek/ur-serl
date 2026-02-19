from __future__ import annotations

import time
import copy
from typing import Callable, Dict, Optional

import numpy as np
import gymnasium as gym
import cv2

import jax
import jax.numpy as jnp

from scipy.spatial.transform import Rotation as R

# Adjust these imports to your repo
from ur_env.envs.ur5_env import UR5Env
from ur_env.utils.rotations import quat_2_mrp


# ---------------------------------------------------------------------------
# 1) Base task env: Door opening (reward handled by classifier wrapper)
# ---------------------------------------------------------------------------

class UR5EDoorOpenEnv(UR5Env):
    """
    Minimal task env for door opening:
    - Uses UR5Env controller + camera stack
    - Returns obs with images + proprio + last action (useful for logging / SERL wrappers)
    - Does NOT implement reward shaping; reward comes from classifier wrapper.
    """

    def __init__(
        self,
        *,
        config,
        hz=10,
        fake_env=False,
        max_episode_length=120,
        save_video=False,
        camera_mode="rgb",
    ):
        self.task_cfg = config
        super().__init__(
            hz=hz,
            fake_env=fake_env,
            config=config,
            max_episode_length=max_episode_length,
            save_video=save_video,
            camera_mode=camera_mode,
        )

        # Make sure observation has an "action" entry if your pipelines expect it
        if isinstance(self.observation_space, gym.spaces.Dict):
            st = self.observation_space.spaces.get("state", None)
            if isinstance(st, gym.spaces.Dict) and "action" not in st.spaces:
                st.spaces["action"] = gym.spaces.Box(-1.0, 1.0, shape=self.action_space.shape, dtype=np.float32)

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
            "tcp_pose": self.curr_pos.astype(np.float32),      # (x,y,z,qx,qy,qz,qw) or your format
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
        # sparse reward handled by wrapper
        return 0.0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs = self._get_obs(np.zeros_like(self.last_action))
        return obs, info


# ---------------------------------------------------------------------------
# 2) Manual reset helper (close the door)
# ---------------------------------------------------------------------------

class DoorManualResetWrapper(gym.Wrapper):
    """
    Door tasks often require a manual reset (close the door).
    This wrapper simply prompts the user at reset().

    If you later add a scripted close-door motion, remove this wrapper.
    """
    def __init__(self, env: gym.Env, prompt_every_reset: bool = True):
        super().__init__(env)
        self.prompt_every_reset = bool(prompt_every_reset)

    def reset(self, **kwargs):
        if self.prompt_every_reset:
            input("\n[RESET] Close the door fully (0 deg), clear workspace, then press Enter...")
        return self.env.reset(**kwargs)


# ---------------------------------------------------------------------------
# 3) Quaternion->MRP observation wrapper (matches your existing ToMrpWrapper)
# ---------------------------------------------------------------------------

class ToMrpWrapper(gym.ObservationWrapper):
    """
    Convert tcp_pose orientation from quat -> MRP.
    Keeps position the same.

    Output tcp_pose becomes length-6: (x,y,z, mrp_x, mrp_y, mrp_z)
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


# ---------------------------------------------------------------------------
# 4) Sparse binary classifier reward + early termination
# ---------------------------------------------------------------------------

class RewardClassifierTerminateWrapper(gym.Wrapper):
    """
    - prob_func(obs) -> success probability in [0,1]
    - reward = 1.0 iff prob >= threshold for 'consecutive' steps
    - terminate episode on success (unless truncated)
    """

    def __init__(
        self,
        env: gym.Env,
        prob_func: Callable[[dict], jax.Array],
        threshold: float = 0.8,
        consecutive: int = 3,
        target_hz: Optional[float] = 10.0,
        trunc_penalty: float = -1.0,
        pass_env_reward: bool = False,
    ):
        super().__init__(env)
        self.prob_func = prob_func
        self.threshold = float(threshold)
        self.consecutive = int(consecutive)
        self.target_hz = float(target_hz) if target_hz is not None else None
        self.trunc_penalty = float(trunc_penalty)
        self.pass_env_reward = bool(pass_env_reward)

        self._streak = 0
        self._step_i = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._streak = 0
        self._step_i = 0
        info["succeed"] = False
        info["reward_clf_prob"] = 0.0
        return obs, info

    def _to_float(self, x) -> float:
        x_host = jax.device_get(x)
        return float(np.asarray(x_host).reshape(-1)[0])

    def step(self, action):
        t0 = time.time()
        self._step_i += 1

        obs, env_reward, terminated, truncated, info = self.env.step(action)

        # classifier probability
        p = self._to_float(self.prob_func(obs))
        p = float(np.clip(p, 0.0, 1.0))

        is_pos = (p >= self.threshold)

        # consecutive success logic
        if truncated:
            self._streak = 0
            success = False
        else:
            self._streak = (self._streak + 1) if is_pos else 0
            success = (self._streak >= self.consecutive)

        # sparse reward
        reward = float(env_reward) if self.pass_env_reward else 0.0
        reward += 1.0 if success else 0.0
        if truncated:
            reward += self.trunc_penalty

        # terminate on success (but don't override truncation)
        if success and not truncated:
            terminated = True
        if truncated:
            terminated = False

        info["succeed"] = bool(success)
        info["reward_clf_prob"] = p
        info["reward_clf_pos"] = bool(is_pos)
        info["reward_clf_streak"] = int(self._streak)

        # rate control
        if self.target_hz is not None:
            time.sleep(max(0.0, 1.0 / self.target_hz - (time.time() - t0)))

        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# 5) Optional gripper penalty (discourage unnecessary toggling)
# ---------------------------------------------------------------------------

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

        toggling = (
            (effective_action[-1] < -0.5 and self.last_closed_norm > 0.9) or
            (effective_action[-1] >  0.5 and self.last_closed_norm < 0.1)
        )

        info["grasp_penalty"] = self.penalty if toggling else 0.0
        self.last_closed_norm = closed_norm
        return obs, reward, term, trunc, info
