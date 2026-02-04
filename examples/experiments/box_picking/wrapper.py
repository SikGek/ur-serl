from __future__ import annotations

import time
import copy
from typing import Dict, Optional

import numpy as np
import gymnasium as gym
import cv2
from scipy.spatial.transform import Rotation as R

# --- UR5 base env + camera stack (adjust imports to your repo paths) ---
from ur_env.envs.ur5_env import UR5Env  # <- change if your UR5Env lives elsewhere
from ur_env.camera.video_capture import VideoCapture
from ur_env.camera.rs_capture import RSCapture

from ur_env.utils.aruco_pose import ArucoPoseEstimator
from ur_env.utils.transforms import (
    construct_homogeneous_matrix,
    construct_homogeneous_matrix_from_rvec_tvec,
    transform_point,
)
import jax

class UR5EArucoPickEnv(UR5Env):
    """
    Task env: detect ArUco-tagged cube using wrist cam, learn pick + lift.

    Important: the policy does NOT need access to box pose.
    We use ArUco for reward/success + debug info (like many real-robot systems).
    """
    def __init__(self, *, config, hz=10, fake_env=False, max_episode_length=150, save_video=False, camera_mode="rgb"):
        self.task_cfg = config
        self._raw_frames: Dict[str, np.ndarray] = {}
        self._aruco_visible = False
        self._box_center_base: Optional[np.ndarray] = None
        self._last_seen_ts = 0.0

        self.aruco = ArucoPoseEstimator(
            marker_length_m=config.ARUCO_MARKER_LENGTH_M,
            camera_matrix=config.WRIST_CAMERA_MATRIX,
            dist_coeffs=config.WRIST_DIST_COEFFS,
            dict_id=config.ARUCO_DICT_ID,
            target_ids=config.ARUCO_TARGET_IDS,
        )

        super().__init__(
            hz=hz,
            fake_env=fake_env,
            config=config,
            max_episode_length=max_episode_length,
            save_video=save_video,
            camera_mode=camera_mode,
        )

        # Patch observation_space to include gripper_pose/object if you want SERLObsWrapper keys.
        # (UR5Env base returns gripper_state; we expose both.)
        if isinstance(self.observation_space, gym.spaces.Dict):
            st = self.observation_space.spaces.get("state", None)
            if isinstance(st, gym.spaces.Dict):
                st.spaces["gripper_pose"] = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
                st.spaces["gripper_object"] = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

    # ---------- camera config (Franka-like dict support) ----------
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

    def crop_image(self, name, image) -> np.ndarray:
        # allow per-camera crop functions like Franka example
        if hasattr(self.task_cfg, "IMAGE_CROP") and name in self.task_cfg.IMAGE_CROP:
            return self.task_cfg.IMAGE_CROP[name](image)
        return super().crop_image(name, image)

    def get_image(self) -> Dict[str, np.ndarray]:
        """
        Override so we can store RAW frames for ArUco while still returning
        policy images exactly like UR5Env expects.
        """
        images = {}
        display_images = {}

        for key, cap in self.cap.items():
            frame = cap.read()
            bgr = frame[..., :3].astype(np.uint8)
            self._raw_frames[key] = bgr  # RAW for ArUco

            # policy image path: follow UR5Env behavior
            if self.camera_mode in ["rgb", "both", "grey"]:
                cropped = self.crop_image(key, bgr)
                resized = cv2.resize(cropped, self.observation_space["images"][key].shape[:2][::-1])

                if self.camera_mode == "grey":
                    grey = np.array([0.2989, 0.5870, 0.1140])
                    resized = np.dot(resized, grey)[..., None].astype(np.uint8)
                    display_images[key] = np.repeat(resized, 3, axis=-1)
                    images[key] = resized
                else:
                    display_images[key] = resized
                    images[key] = resized[..., ::-1]  # keep same channel convention as base env

                display_images[key + "_full"] = cropped

        if hasattr(self, "img_queue") and self.img_queue is not None:
            self.img_queue.put(display_images)

        return images

    # ---------- ArUco pose update ----------
    def _update_box_from_aruco(self) -> bool:
        cam = self.task_cfg.DETECT_CAMERA
        if cam not in self._raw_frames:
            self._aruco_visible = False
            return False

        det = self.aruco.detect(self._raw_frames[cam])
        if det is None:
            self._aruco_visible = False
            return False

        # base <- tcp from robot state (quat pose)
        tcp_pose = np.asarray(self.curr_pos, dtype=np.float64).reshape(-1)
        if tcp_pose.shape[0] != 7:
            self._aruco_visible = False
            return False

        T_base_tcp = construct_homogeneous_matrix(tcp_pose)
        T_tcp_cam = np.asarray(self.task_cfg.T_TCP_CAM, dtype=np.float64).reshape(4, 4)
        T_base_cam = T_base_tcp @ T_tcp_cam

        T_cam_marker = construct_homogeneous_matrix_from_rvec_tvec(det.rvec, det.tvec)
        T_base_marker = T_base_cam @ T_cam_marker

        box_center = transform_point(T_base_marker, np.asarray(self.task_cfg.MARKER_TO_BOX_CENTER, dtype=np.float64))
        self._box_center_base = box_center.astype(np.float32)
        self._aruco_visible = True
        self._last_seen_ts = time.time()

        # log info out
        self.cost_infos["aruco_visible"] = True
        self.cost_infos["aruco_marker_id"] = det.marker_id
        self.cost_infos["box_center_base"] = self._box_center_base.copy()
        return True

    # ---------- task reward/success ----------
    def _pregrasp_xyz(self) -> Optional[np.ndarray]:
        if self._box_center_base is None:
            return None
        p = self._box_center_base.copy()
        p[2] += float(self.task_cfg.PREGRASP_Z_OFFSET)
        return p

    def _grasp_xyz(self) -> Optional[np.ndarray]:
        if self._box_center_base is None:
            return None
        p = self._box_center_base.copy()
        p[2] += float(self.task_cfg.GRASP_Z_OFFSET)
        return p

    def reached_goal_state(self, obs) -> bool:
        if self._box_center_base is None:
            return False
        has_obj = bool(self.gripper_state[1] > 0.5)
        if not has_obj:
            return False
        return float(self.curr_pos[2]) > float(self._box_center_base[2] + self.task_cfg.LIFT_DELTA_Z)

    def compute_reward(self, obs, action) -> float:
        # Update from ArUco each step if possible
        if not self._update_box_from_aruco():
            return -0.01

        if self.reached_goal_state(obs):
            return 1.0  # IMPORTANT: your eval code treats reward as success

        tcp = np.asarray(self.curr_pos[:3], dtype=np.float32)
        pre = self._pregrasp_xyz()
        grasp = self._grasp_xyz()
        if pre is None or grasp is None:
            return -0.01

        d_pre = float(np.linalg.norm(tcp - pre))
        d_grasp = float(np.linalg.norm(tcp - grasp))

        shaped = -0.05 * d_pre - 0.10 * d_grasp

        # small bonus for "closing near grasp"
        if float(np.linalg.norm((tcp - grasp)[:2])) < float(self.task_cfg.GRASP_XY_TOL):
            if abs(float(tcp[2] - grasp[2])) < float(self.task_cfg.GRASP_Z_TOL):
                if action is not None and float(action[-1]) > 0.5:
                    shaped += 0.05

        if bool(self.gripper_state[1] > 0.5):
            shaped += 0.10

        return float(shaped)

    # ---------- observation ----------
    def _get_obs(self, action) -> dict:
        images = None
        if self.camera_mode is not None:
            images = self.get_image()

        self._update_currpos()
        self._update_box_from_aruco()

        # expose both:
        gripper_pose = np.array([self.gripper_state[0]], dtype=np.float32)
        gripper_object = np.array([self.gripper_state[1]], dtype=np.float32)

        state_observation = {
            "tcp_pose": self.curr_pos.astype(np.float32),
            "tcp_vel": self.curr_vel.astype(np.float32),
            "tcp_force": self.curr_force.astype(np.float32),
            "tcp_torque": self.curr_torque.astype(np.float32),
            "gripper_state": self.gripper_state.astype(np.float32),
            "gripper_pose": gripper_pose,
            "gripper_object": gripper_object,
            "action": np.asarray(action, dtype=np.float32),
        }

        obs = {"state": state_observation}
        if images is not None:
            obs["images"] = images

        return copy.deepcopy(obs)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        # optional: wait for marker on reset (helps stabilize early training)
        if (not getattr(self, "controller", None) is None) and getattr(self.task_cfg, "WAIT_FOR_MARKER_ON_RESET", True):
            t0 = time.time()
            while time.time() - t0 < float(self.task_cfg.MARKER_RESET_TIMEOUT_S):
                _ = self._get_obs(np.zeros((7,), dtype=np.float32))
                if self._aruco_visible:
                    break
                time.sleep(0.05)

        obs = self._get_obs(np.zeros((7,), dtype=np.float32))
        return obs, info


# -------------------- Wrappers --------------------

class Quat2EulerWrapper(gym.Wrapper):
    """
    Converts obs['state']['tcp_pose'] from (x,y,z,qx,qy,qz,qw) to (x,y,z,roll,pitch,yaw).
    Put this BEFORE SERLObsWrapper.
    """
    def __init__(self, env):
        super().__init__(env)
        self._patch_space()

    def _patch_space(self):
        if isinstance(self.observation_space, gym.spaces.Dict):
            st = self.observation_space.spaces.get("state", None)
            if isinstance(st, gym.spaces.Dict) and "tcp_pose" in st.spaces:
                st.spaces["tcp_pose"] = gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

    def observation(self, obs):
        pose = obs["state"]["tcp_pose"]
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.shape[0] == 7:
            xyz = pose[:3]
            eul = R.from_quat(pose[3:]).as_euler("xyz", degrees=False)
            obs["state"]["tcp_pose"] = np.concatenate([xyz, eul]).astype(np.float32)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return self.observation(obs), r, term, trunc, info


# class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
#     """
#     Optional HIL-SERL-style sparse reward from a learned binary classifier.
#     (HIL-SERL uses a binary reward classifier + demos + interventions). :contentReference[oaicite:5]{index=5}
#     """
#     def __init__(self, env, reward_func):
#         super().__init__(env)
#         self.reward_func = reward_func

#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)

#     def step(self, action):
#         obs, reward, term, trunc, info = self.env.step(action)
#         reward = float(self.reward_func(obs))
#         return obs, reward, term, trunc, info

class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz = None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or rew
        info['succeed'] = bool(rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info


class GripperPenaltyWrapper(gym.Wrapper):
    """
    Adds info['grasp_penalty'] to support your learned-gripper SAC variants.
    It uses env.unwrapped.gripper_state to avoid relying on SERLObsWrapper ordering.
    """
    def __init__(self, env, penalty=-0.02):
        super().__init__(env)
        self.penalty = float(penalty)
        self.last_closed_norm = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_closed_norm = float(self.env.unwrapped.gripper_state[0])
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        # If human intervention overrides action, use that for penalty accounting
        effective_action = info.get("intervene_action", action)

        closed_norm = float(self.env.unwrapped.gripper_state[0])

        toggling = (
            (effective_action[-1] < -0.5 and self.last_closed_norm > 0.9) or
            (effective_action[-1] >  0.5 and self.last_closed_norm < 0.1)
        )

        info["grasp_penalty"] = self.penalty if toggling else 0.0
        self.last_closed_norm = closed_norm
        return obs, reward, term, trunc, info


# class RewardClassifierTerminateWrapper(gym.Wrapper):
#     """
#     Turns a learned classifier into:
#       - binary reward (0/1)
#       - episode termination on success
#       - info["succeed"]=True on success

#     Also supports K-frame hysteresis to avoid one-frame false positives.
#     """
#     def __init__(self, env, prob_func, threshold=0.7, consecutive=3):
#         super().__init__(env)
#         self.prob_func = prob_func          # returns probability in [0,1]
#         self.threshold = float(threshold)
#         self.consecutive = int(consecutive)
#         self._streak = 0

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self._streak = 0
#         return obs, info

#     def step(self, action):
#         obs, _env_reward, terminated, truncated, info = self.env.step(action)

#         # Probability in [0,1]
#         p = float(self.prob_func(obs))
#         is_pos = (p >= self.threshold)

#         # K-frame hysteresis
#         if is_pos:
#             self._streak += 1
#         else:
#             self._streak = 0

#         success = (self._streak >= self.consecutive)

#         # Binary reward (HIL-SERL style)
#         reward = 1.0 if success else 0.0

#         # If success, end episode (unless already safety-truncated)
#         if success and not truncated:
#             terminated = True
#             info["succeed"] = True

#         # Useful debug signals
#         info["reward_clf_prob"] = p
#         info["reward_clf_pos"] = bool(is_pos)
#         info["reward_clf_success"] = bool(success)

#         return obs, reward, terminated, truncated, info
    
class RewardClassifierTerminateWrapper(gym.Wrapper):
    """
    Turns a learned classifier into:
      - binary reward (0/1)
      - episode termination on success
      - info["succeed"]=True on success

    Also supports K-frame hysteresis to avoid one-frame false positives.
    """
    def __init__(self, env, prob_func, threshold=0.7, consecutive=3):
        super().__init__(env)
        self.prob_func = prob_func          # returns probability in [0,1]
        self.threshold = float(threshold)
        self.consecutive = int(consecutive)
        self._streak = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._streak = 0
        info["succeed"] = False
        info["reward_clf_prob"] = 0.0
        info["reward_clf_pos"] = False
        info["reward_clf_success"] = False
        return obs, info

    def _to_float_scalar(self, x) -> float:
        """Robust scalar conversion for JAX/NumPy/python values."""
        x_host = jax.device_get(x)  # safe even if x is already numpy/python
        arr = np.asarray(x_host).reshape(-1)
        return float(arr[0])

    def step(self, action):
        obs, _env_reward, terminated, truncated, info = self.env.step(action)

        # Probability in [0,1]
        p_raw = self.prob_func(obs)
        p = self._to_float_scalar(p_raw)
        p = float(np.clip(p, 0.0, 1.0))  # just safety

        is_pos = (p >= self.threshold)

        # If safety-truncated, do NOT allow classifier to declare success
        if truncated:
            self._streak = 0
            success = False
            reward = 0.0
        else:
            # K-frame hysteresis
            if is_pos:
                self._streak += 1
            else:
                self._streak = 0

            success = (self._streak >= self.consecutive)
            reward = 1.0 if success else 0.0

            if success:
                terminated = True

        # Always populate succeed flag (prevents downstream KeyErrors)
        info["succeed"] = bool(success)

        # Useful debug signals
        info["reward_clf_prob"] = p
        info["reward_clf_pos"] = bool(is_pos)
        info["reward_clf_success"] = bool(success)

        return obs, reward, terminated, truncated, info