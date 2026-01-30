from __future__ import annotations

import os
import numpy as np
import jax
import jax.numpy as jnp

# --- UR5 base config (adjust import path) ---
from ur_env.envs.basic_env import DefaultEnvConfig  # <- change to your repo

# SERL/HIL-SERL style wrappers
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

# UR wrappers (Spacemouse, RelativeFrame)
from ur_env.envs.wrappers import SpacemouseIntervention  # should exist in your stack
from ur_env.envs.relative_env import RelativeFrame       # should exist in your stack

from experiments.config import DefaultTrainingConfig      # your training base
from experiments.box_picking.wrapper import (
    UR5EArucoPickEnv,
    Quat2EulerWrapper,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperPenaltyWrapper,
)


class EnvConfig(DefaultEnvConfig):
    # -------- Robot / safety --------
    ROBOT_IP: str = "192.168.0.10"      # <-- CHANGE
    CONTROLLER_HZ: int = 500

    # Workspace bounds (you must tune)
    ABS_POSE_LIMIT_LOW = np.array([0.25, -0.35, 0.05, -0.35, -0.35, -0.35], dtype=np.float32)
    ABS_POSE_LIMIT_HIGH = np.array([0.75,  0.35, 0.55,  0.35,  0.35,  0.35], dtype=np.float32)
    ABS_POSE_RANGE_LIMITS = np.array([-0.10, 0.10], dtype=np.float32)

    ACTION_SCALE = np.array([0.01, 0.10, 1.0], dtype=np.float32)

    # -------- Cameras (Franka-style dict) --------
    REALSENSE_CAMERAS = {
        "wrist": {
            "serial_number": "0123456789",   # <-- CHANGE
            "dim": (1280, 720),
            "exposure": 10500,
        },
    }

    # Optional: per-camera crop functions
    IMAGE_CROP = {
        "wrist": lambda img: img[:, 124:604, :],
    }

    # Which camera to use for ArUco detection
    DETECT_CAMERA = "wrist"

    # -------- ArUco parameters --------
    ARUCO_DICT_ID = 0  # cv2.aruco.DICT_4X4_50  (keep int here to avoid cv2 import in config)
    ARUCO_TARGET_IDS = [0]       # <-- marker ids you use
    ARUCO_MARKER_LENGTH_M = 0.04 # <-- meters

    # Wrist camera intrinsics at RAW resolution (must match capture)
    WRIST_CAMERA_MATRIX = np.array([
        [615.0, 0.0, 640.0],
        [0.0, 615.0, 360.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    WRIST_DIST_COEFFS = np.zeros((5,), dtype=np.float64)

    # Hand-eye: TCP -> camera
    T_TCP_CAM = np.array([
        [1, 0, 0, 0.00],
        [0, 1, 0, 0.05],
        [0, 0, 1, 0.10],
        [0, 0, 0, 1.00],
    ], dtype=np.float64)

    # Marker origin -> box center offset in marker frame
    MARKER_TO_BOX_CENTER = np.array([0.0, 0.0, -0.02], dtype=np.float64)

    # -------- Task thresholds --------
    PREGRASP_Z_OFFSET = 0.10
    GRASP_Z_OFFSET = 0.02
    LIFT_DELTA_Z = 0.08
    GRASP_XY_TOL = 0.02
    GRASP_Z_TOL = 0.02

    WAIT_FOR_MARKER_ON_RESET = True
    MARKER_RESET_TIMEOUT_S = 3.0

    MAX_EPISODE_LENGTH = 150


class TrainConfig(DefaultTrainingConfig):
    """
    TrainConfig compatible with your train_rlpd launcher:
    - image_keys used to build pixel encoders
    - setup_mode selects agent type (learned gripper => hybrid agent)
    - get_environment() composes wrappers like the Franka example
    """
    image_keys = ["wrist"]
    classifier_keys = ["wrist"]  # if you use reward classifier
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose", "gripper_object"]

    encoder_type = "resnet-pretrained"
    discount = 0.98
    cta_ratio = 2
    random_steps = 0
    buffer_period = 1000
    checkpoint_period = 2000

    # Choose learned-gripper mode if you want SAC hybrid + grasp_penalty
    setup_mode = "single-arm-learned-gripper"

    # Optional: enable a learned reward classifier (HIL-SERL style)
    use_reward_classifier = False
    classifier_ckpt_path = os.path.abspath("classifier_ckpt/")

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        # Base env
        env = UR5EArucoPickEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
            max_episode_length=EnvConfig.MAX_EPISODE_LENGTH,
            hz=10,
            camera_mode="rgb",
        )

        # Human-in-the-loop interventions (Spacemouse) like HIL-SERL workflow :contentReference[oaicite:6]{index=6}
        if not fake_env:
            env = SpacemouseIntervention(env)   # must output info["intervene_action"] for your train_rlpd actor loop

        # Relative observations (optional but requested)
        env = RelativeFrame(env)

        # Quaternion -> Euler (requested)
        env = Quat2EulerWrapper(env)

        # SERL obs formatting
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)

        # Chunking wrapper (requested)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # Optional reward classifier wrapper
        if classifier and self.use_reward_classifier:
            clf = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=self.classifier_ckpt_path,
            )

            def reward_func(obs):
                sigmoid = lambda x: 1.0 / (1.0 + jnp.exp(-x))
                return int(sigmoid(clf(obs)) > 0.7)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

        # Gripper penalty support for hybrid agent
        env = GripperPenaltyWrapper(env, penalty=-0.02)

        return env
