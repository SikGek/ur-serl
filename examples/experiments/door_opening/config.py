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


class EnvConfig(DefaultEnvConfig):
    ROBOT_IP: str = "192.168.56.2"
    CONTROLLER_HZ: int = 100

    # IMPORTANT: set RESET_Q to a pose where the gripper is already aligned on the handle
    RESET_Q = np.deg2rad(np.array([
        [283.03, -123.27, -52.70, -186.41, 277.63, 180.0],
    ], dtype=np.float32))

    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.02,)
    RANDOM_ROT_RANGE = (np.tan(np.deg2rad(5)/4),)

    ABS_POSE_LIMIT_LOW  = np.array([-0.6, -0.9, 0.05, -0.5, -0.5, -0.5], dtype=np.float32)
    ABS_POSE_LIMIT_HIGH = np.array([ 0.5, -0.1, 0.60,  0.5,  0.5,  0.5], dtype=np.float32)

    ACTION_SCALE = np.array([0.06, 0.10, 1.0], dtype=np.float32)

    REALSENSE_CAMERAS = {
        "shoulder": {
            "serial_number": "239122070813",  
            "dim": (1280, 720),
        },
        "wrist": {
            "serial_number": "218622274722",  
            "dim": (1280, 720),
        },
    }

    MAX_EPISODE_LENGTH = 120

    GRIPPER_TIMEOUT = 500   # ms

    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.07
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])

    GRIPPER_USB_PORT = "/dev/ttyUSB0"
    GRIPPER_SLAVE_ID = 9


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist", "shoulder"]

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

    discount = 0.985
    cta_ratio = 2
    random_steps = 0

    setup_mode = "single-arm-fixed-gripper"

    classifier_ckpt_path = os.path.abspath("classifier_ckpt/stage_2/")
    clf_threshold = 0.9
    clf_consecutive = 3
    buffer_period = 1000
    checkpoint_period = 2000
    
    def get_environment(self, fake_env=False, save_video=False, classifier=True):
        env = UR5EDoorPullEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
            max_episode_length=EnvConfig.MAX_EPISODE_LENGTH,
            hz=10,
            camera_mode="rgb",
            close_gripper_on_reset=True,
            close_wait_s=0.6,
        )

        if not fake_env:
            env = SpacemouseIntervention(env)

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
        env = DoorPullManualResetWrapper(env, prompt_every_reset=True)

        env = GripperCloseEnv(env)

        return env