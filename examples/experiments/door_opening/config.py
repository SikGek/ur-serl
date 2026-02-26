from __future__ import annotations

import os
import numpy as np
import jax
import jax.numpy as jnp

from ur_env.envs.ur5_env import DefaultEnvConfig
from ur_env.envs.relative_env import RelativeFrame
from experiments.config import DefaultTrainingConfig

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from ur_env.envs.wrappers import SpacemouseIntervention, GripperCloseEnv

from experiments.door_opening.wrapper import (
    UR5EDoorPullEnv,
    DoorPullManualResetWrapper,
    RewardClassifierTerminateWrapper,
    ToMrpWrapper,
)


class DoorPullEnvConfig(DefaultEnvConfig):
    ROBOT_IP: str = "192.168.56.2"
    CONTROLLER_HZ: int = 100

    # IMPORTANT: set RESET_Q to a pose where the gripper is already aligned on the handle
    RESET_Q = np.deg2rad(np.array([
        [272.0, -77.0, -130.0, -150.0, 270.0, 180.0],
    ], dtype=np.float32))

    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.02,)
    RANDOM_ROT_RANGE = (np.tan(np.deg2rad(5)/4),)

    ABS_POSE_LIMIT_LOW  = np.array([-0.6, -0.9, 0.05, -0.5, -0.5, -0.5], dtype=np.float32)
    ABS_POSE_LIMIT_HIGH = np.array([ 0.5, -0.1, 0.60,  0.5,  0.5,  0.5], dtype=np.float32)

    ACTION_SCALE = np.array([0.08, 0.10, 1.0], dtype=np.float32)

    REALSENSE_CAMERAS = {
        "shoulder": {"serial_number": "239122070813", "dim": (1280, 720)},
        "wrist":    {"serial_number": "218622274722", "dim": (1280, 720)},
    }

    MAX_EPISODE_LENGTH = 150

    # I strongly recommend NOT using 5000ms here.
    # You want reset-close to always work, and you don't want grasp to "miss and wait 5s".
    GRIPPER_TIMEOUT = 500   # ms

    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.08
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])

    GRIPPER_USB_PORT = "/dev/ttyUSB0"
    GRIPPER_SLAVE_ID = 9


class TrainConfigDoorPull(DefaultTrainingConfig):
    # policy sees both
    image_keys = ["wrist", "shoulder"]

    # classifier should be shoulder for "door opened enough"
    classifier_keys = ["shoulder"]

    proprio_keys = [
        "tcp_pose",
        "tcp_vel",
        "tcp_force",
        "tcp_torque",
        "gripper_pose",
        "gripper_state",
    ]

    encoder_type = "resnet-pretrained"

    # Pulling is longer horizon; 0.99 is a sane starting point
    discount = 0.99
    cta_ratio = 2
    random_steps = 0

    # IMPORTANT: pulling stage has NO gripper action -> use fixed-gripper SAC
    setup_mode = "single-arm-fixed-gripper"

    classifier_ckpt_path = os.path.abspath("classifier_ckpt/door_pull_success/")
    clf_threshold = 0.85
    clf_consecutive = 2

    def get_environment(self, fake_env=False, save_video=False, classifier=True):
        env = UR5EDoorPullEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=DoorPullEnvConfig(),
            max_episode_length=DoorPullEnvConfig.MAX_EPISODE_LENGTH,
            hz=10,
            camera_mode="rgb",
            close_gripper_on_reset=True,
            close_wait_s=0.6,
        )

        env = GripperCloseEnv(env)
        # RAM-insertion-like assumption: start grasped
        env = DoorPullManualResetWrapper(env, prompt_every_reset=False)

        # Keep relative action convention consistent with your other UR tasks
        # env = RelativeFrame(env)

        # Convert quat->MRP for SERL
        env = ToMrpWrapper(env)

        # SERL formatting + chunking
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # Reward classifier
        if classifier:
            clf = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=self.classifier_ckpt_path,
            )

            def prob_func(obs):
                logits = clf(obs)
                logit0 = jnp.ravel(jnp.asarray(logits))[0]
                return jax.nn.sigmoid(logit0)

            env = RewardClassifierTerminateWrapper(
                env,
                prob_func,
                threshold=self.clf_threshold,
                consecutive=self.clf_consecutive,
                target_hz=10,
                trunc_penalty=0.0,
                pass_env_reward=False,
            )

        # ---- THIS is the RAM insertion move ----
        # Policy action space becomes 6D, gripper is held fixed (no commands) during episode.


        # Optional: still allow spacemouse intervention in 6D (no gripper buttons)
        if not fake_env:
            env = SpacemouseIntervention(env)

        return env