from __future__ import annotations

import os
import numpy as np
import jax
import jax.numpy as jnp

# --- UR5 base config (adjust import path) ---
from ur_env.envs.ur5_env import DefaultEnvConfig  # <- change to your repo

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
    Quat2RotvecWrapper,
    GripperPenaltyWrapper,
    RewardClassifierTerminateWrapper,
    ToMrpWrapper,
)
from scipy.spatial.transform import Rotation as R

import pyrealsense2 as rs

class EnvConfig(DefaultEnvConfig):
    # -------- Robot / safety --------
    ROBOT_IP: str = "192.168.56.2"      
    CONTROLLER_HZ: int = 100
    RESET_Q = np.array([
        [271.07, -93.98, -122.14, -166.376, 270.78, 180.0],
        # [-5, -78.62, 122.84, -134.22, -89.81, -13.03],
        # [10, -75.62, 122.84, -134.22, -89.81, -13.03],
    ])
    RESET_Q = np.deg2rad(RESET_Q)
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.06,)
    RANDOM_ROT_RANGE = (0.0,)
    p0 = [-0.1274, -0.4032, 0.2258]
    # Workspace bounds (you must tune)
    ABS_POSE_LIMIT_LOW = np.array([p0[0]-0.20, p0[1]-0.20, p0[2]-0.15, -0.08, -0.08, -0.15])
    ABS_POSE_LIMIT_HIGH = np.array([p0[0]+0.20, p0[1]+0.20, p0[2]+0.10, 0.08, 0.08, 0.15])
    ABS_POSE_RANGE_LIMITS = np.array([-0.10, 0.10], dtype=np.float32)
    # ACTION_SCALE = np.array([0.07, 0.1, 1.0], dtype=np.float32)
    ACTION_SCALE = np.array([0.02, 0.1, 1.0], dtype=np.float32)

    # -------- Cameras (Franka-style dict) --------
    REALSENSE_CAMERAS = {
        "wrist": {
            "serial_number": "218622274722",  
            "dim": (1280, 720),
        },
    }

    # Optional: per-camera crop functions
    # IMAGE_CROP = {
    #     "wrist": lambda img: img[:, 124:604, :],
    # }

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
    [-0.99987673, -0.01533774, -0.00336024, -0.19441016],
    [0.00317622, 0.01200614, -0.99992288, 0.46531429],
    [0.01537690, -0.99981028, -0.01195594, 0.72345337],
    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
])

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

    GRIPPER_TIMEOUT = 5000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.8  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])
    GRIPPER_USB_PORT = "/dev/ttyUSB0"
    GRIPPER_SLAVE_ID = 9
    GOAL_POSE = np.array([0.024, 0.02, 0.21, 0., 0., 3.14])     #box_1
    GOAL_POSE = np.array([0.03, 0.03, 0.21, 0., 0., 3.14])      #box_340
    GOAL_POSE = np.array([0.035, 0.03, 0.21, 0., 0., 3.14])     #box_330

    ROTATION_GENERALIZATION = R.from_euler("xyz", np.array([0, 0, 0])).as_matrix() # rotation applied to the box to bring it back to the training orientation
    BOX_ERROR = np.array([0.0, 0.0, 0.0])
    POSE_ESTIMATION = False
    POSE_ESTIMATION_IP = "ws://localhost:7777"
    WF_rot = np.array([[-1,  0,  0],
                        [ 0,  0, 1],
                        [ 0, 1,  0]], dtype=np.float32)
    LOW_PASS_FILTER = 0
    SUCCESS_COUNT = 0


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
    discount = 0.99
    cta_ratio = 2
    random_steps = 0
    buffer_period = 1000
    checkpoint_period = 2000

    # Choose learned-gripper mode if you want SAC hybrid + grasp_penalty
    setup_mode = 'single-arm-learned-gripper'

    # Optional: enable a learned reward classifier (HIL-SERL style)
    use_reward_classifier = True
    classifier_ckpt_path = os.path.abspath("classifier_ckpt/")

    def get_environment(self, fake_env=False, save_video=False, classifier=True):
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
        env = ToMrpWrapper(env)

        # SERL obs formatting
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)

        # Chunking wrapper (requested)
        # env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # Optional reward classifier wrapper
        if classifier:
            clf = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=self.classifier_ckpt_path,
            )

            def reward_func(obs):
                logits = clf(obs)
                logit0 = jnp.ravel(jnp.asarray(logits))[0]
                return jax.nn.sigmoid(logit0)
            # def reward_func(obs):
            #     sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
            #     return int(sigmoid(clf(obs)) > 0.7 and obs["state"][0, 0] > 0.4)

            env = RewardClassifierTerminateWrapper(env, reward_func, threshold=0.7, consecutive=3, target_hz=10)

        # Gripper penalty support for hybrid agent
        env = GripperPenaltyWrapper(env, penalty=-0.02)

        return env
