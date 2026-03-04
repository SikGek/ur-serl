"""Task configuration for HIL-SERL on a single Unitree G1 arm with SpaceMouse intervention.

This config intentionally mirrors HIL-SERL's existing Franka examples so that the
standard workflow remains the same:

1. launch the robot server;
2. collect reward-classifier images;
3. train the reward classifier;
4. record demonstrations;
5. launch actor + learner for RLPD.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from experiments.config import DefaultTrainingConfig
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    MultiCameraBinaryRewardClassifierWrapper,
    Quat2EulerWrapper,
    SpacemouseIntervention,
)
from g1_env.envs.g1_arm_env import G1ArmEnv, G1EnvConfig
from serl_launcher.networks.reward_classifier import load_classifier_func
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper


_THIS_DIR = Path(__file__).resolve().parent
_TASK_CFG_PATH = _THIS_DIR / "task_config.yaml"
_TASK_CFG_EXAMPLE_PATH = _THIS_DIR / "task_config.example.yaml"


def _as_np(value, *, dtype=np.float64):
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def _load_yaml_config() -> Dict[str, Any]:
    source = _TASK_CFG_PATH if _TASK_CFG_PATH.exists() else _TASK_CFG_EXAMPLE_PATH
    with open(source, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    required = ["server_url", "active_arm", "cameras"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(
            f"Task config file {source} is missing keys: {', '.join(missing)}"
        )
    return cfg


class EnvConfig(G1EnvConfig):
    """Load runtime settings from the experiment-local YAML file."""

    _cfg = _load_yaml_config()

    SERVER_URL = _cfg["server_url"]
    ACTIVE_ARM = _cfg.get("active_arm", "left")

    ACTION_SCALE_TRANSLATION = float(_cfg.get("action_scale", {}).get("translation", 0.01))
    ACTION_SCALE_ROTATION = float(_cfg.get("action_scale", {}).get("rotation", 0.12))

    TARGET_POSE = _as_np(_cfg.get("target_pose", [0, 0, 0, 0, 0, 0]))
    RESET_POSE = _as_np(_cfg.get("reset_pose", [0, 0, 0, 0, 0, 0]))
    REWARD_THRESHOLD = _as_np(
        _cfg.get("reward_threshold", [0.02, 0.02, 0.02, 0.2, 0.2, 0.2])
    )

    ABS_POSE_LIMIT_LOW = _as_np(
        _cfg.get("abs_pose_limit_low", [0.15, -0.2, 0.45, 2.2, -0.8, 0.5])
    )
    ABS_POSE_LIMIT_HIGH = _as_np(
        _cfg.get("abs_pose_limit_high", [0.75, 0.55, 1.0, 3.9, 0.8, 2.6])
    )

    RANDOM_RESET = bool(_cfg.get("random_reset", False))
    RANDOM_XY_RANGE = float(_cfg.get("random_xy_range", 0.0))
    RANDOM_RPY_RANGE = float(_cfg.get("random_rpy_range", 0.0))

    DISPLAY_IMAGE = bool(_cfg.get("display_image", True))
    MAX_EPISODE_LENGTH = int(_cfg.get("episode_length", 80))
    RESET_INTERPOLATION_TIME = float(_cfg.get("reset_interpolation_time", 2.0))
    POST_RESET_SETTLE_SEC = float(_cfg.get("post_reset_settle_sec", 0.5))
    IMAGE_SIZE = tuple(_cfg.get("image_size", [128, 128]))

    RESET_Q_ARM = _as_np(_cfg.get("reset_q_arm"), dtype=np.float64)
    PASSIVE_ARM_HOME_Q = _as_np(_cfg.get("passive_arm_home_q"), dtype=np.float64)

    CAMERAS = _cfg.get("cameras", {})


class TrainConfig(DefaultTrainingConfig):
    """Training config consumed by the standard HIL-SERL scripts."""

    _cfg = _load_yaml_config()

    image_keys = list(_cfg.get("image_keys") or EnvConfig.CAMERAS.keys())
    classifier_keys = list(_cfg.get("classifier_keys") or image_keys)
    proprio_keys = list(
        _cfg.get(
            "proprio_keys",
            ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "q_arm", "dq_arm"],
        )
    )

    buffer_period = int(_cfg.get("buffer_period", 1000))
    checkpoint_period = int(_cfg.get("checkpoint_period", 5000))
    steps_per_update = int(_cfg.get("steps_per_update", 50))
    replay_buffer_capacity = int(_cfg.get("replay_buffer_capacity", 200000))
    batch_size = int(_cfg.get("batch_size", 256))
    discount = float(_cfg.get("discount", 0.97))
    max_steps = int(_cfg.get("max_steps", 1_000_000))
    training_starts = int(_cfg.get("training_starts", 100))
    log_period = int(_cfg.get("log_period", 10))
    encoder_type = _cfg.get("encoder_type", "resnet-pretrained")
    setup_mode = "single-arm-fixed-gripper"
    classifier_threshold = float(_cfg.get("classifier_threshold", 0.85))

    def process_demos(self, demo):
        return demo

    def get_environment(self, fake_env: bool = False, save_video: bool = False, classifier: bool = False):
        env = G1ArmEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )

        if not fake_env:
            env = SpacemouseIntervention(env)

        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        if classifier:
            classifier_fn = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            threshold = float(self.classifier_threshold)

            def reward_func(obs):
                sigmoid = lambda x: 1.0 / (1.0 + jnp.exp(-x))
                return int(sigmoid(classifier_fn(obs)) > threshold)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

        return env
