from __future__ import annotations

import os
import numpy as np
import jax
import jax.numpy as jnp

# --- UR5 base config (adjust import path for your repo) ---
from ur_env.envs.ur5_env import DefaultEnvConfig
from ur_env.envs.relative_env import RelativeFrame
from experiments.config import DefaultTrainingConfig

# --- SERL/HIL-SERL style wrappers ---
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

# --- Your UR wrappers ---
from ur_env.envs.wrappers import SpacemouseIntervention
from experiments.door_opening.wrapper import (
    UR5EDoorOpenEnv,
    DoorManualResetWrapper,
    RewardClassifierTerminateWrapper,
    GripperPenaltyWrapper,
    MultiStageRewardClassifierTerminateWrapper,
    # Optional:
    # TCPActionRelativeFrame,
    # RelativeFrame,
    ToMrpWrapper,
)


class EnvConfig(DefaultEnvConfig):
    # ---------------- Robot ----------------
    ROBOT_IP: str = "192.168.56.2"     # <-- set
    CONTROLLER_HZ: int = 100

    # A safe joint reset pose that starts near the handle (example placeholder)
    RESET_Q = np.deg2rad(np.array([
        # [271.07, -93.98, -122.14, -166.376, 270.78, 180.0],
        [272.0, -77.0, -130.0, -150.0, 270.0, 180.0],
        # [180.0, -80.0, -130.0, -166.0, 270.0, 180.0],
    ], dtype=np.float32))

    # Randomize initial EE pose slightly (helps generalization)
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.04,)        # ~4 cm
    RANDOM_ROT_RANGE = (np.tan(np.deg2rad(10)/4),)
  # ~10 deg about each axis in current reset logic

    # Safety workspace bounds (PLACEHOLDERS — tune!)
    # low/high are [x,y,z, mrp_x, mrp_y, mrp_z] in your UR5Env
    ABS_POSE_LIMIT_LOW  = np.array([-0.6, -0.9, 0.05, -0.5, -0.5, -0.5], dtype=np.float32)
    ABS_POSE_LIMIT_HIGH = np.array([ 0.5, -0.1, 0.60,  0.5,  0.5,  0.5], dtype=np.float32)

    # 2. Action Scales:
    # If the robot feels "sluggish" while opening, increase the translation scale.
    ACTION_SCALE = np.array([0.05, 0.1, 1.0], dtype=np.float32)

    # ---------------- Camera ----------------
    # IMPORTANT:
    # If your UR5Env only supports keys like "wrist", you can still name your side camera "wrist".
    # Otherwise, extend UR5Env image-space creation to accept arbitrary keys.
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
    # Optional: crop function to focus on the door region BEFORE resizing to 128x128.
    # You can implement this in UR5EDoorOpenEnv.crop_image() (see wrapper code below).
    # Example ROI numbers are placeholders.
    # IMAGE_CROP = {
    #     "wrist": lambda img: img[0:720, 200:1000, :],  # (y0:y1, x0:x1)
    # }

    MAX_EPISODE_LENGTH = 200

    GRIPPER_TIMEOUT = 5000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.08  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    # FORCEMODE_SELECTION_VECTOR = np.array([1,1,1,0,0,0], dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])
    GRIPPER_USB_PORT = "/dev/ttyUSB0"
    GRIPPER_SLAVE_ID = 9


class TrainConfig(DefaultTrainingConfig):
    # --- what the policy sees ---
    image_keys =["wrist", "shoulder"]           # “wrist” key is actually your side camera
    classifier_keys = ["shoulder"]      # use same camera for reward since you only have one

    proprio_keys = [
        "tcp_pose",
        "tcp_vel",
        "tcp_force",
        "tcp_torque",
        "gripper_pose",     # optional if you expose it
        "gripper_state",
    ]

    # --- RL hyperparams (match the paper’s typical settings) ---
    encoder_type = "resnet-pretrained"
    discount = 0.997         # good for ~100 step horizons:contentReference[oaicite:19]{index=19}
    cta_ratio = 2
    random_steps = 0

    setup_mode = "single-arm-learned-gripper"  # or fixed gripper if you don't want discrete gripper

    # Reward classifier checkpoint folder
    classifier_ckpt_path = os.path.abspath("classifier_ckpt/")

    # Reward classifier decision
    clf_threshold = 0.92
    clf_consecutive = 3       # require 3 consecutive frames above threshold

    def get_environment(self, fake_env=False, save_video=False, classifier=True):
        # ---- Base env ----
        env = UR5EDoorOpenEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
            max_episode_length=EnvConfig.MAX_EPISODE_LENGTH,
            hz=10,                      # 10 Hz like HIL-SERL:contentReference[oaicite:20]{index=20}
            camera_mode="rgb",
        )

        # ---- Manual reset for door tasks (recommended unless you have scripted closing) ----
        # You can remove this if you implement a scripted door-close reset.
        env = DoorManualResetWrapper(env, prompt_every_reset=False)

        # ---- Human interventions ----
        if not fake_env:
            env = SpacemouseIntervention(env)
        # env = RelativeFrame(env)
        # ---- (Optional) Make policy actions TCP-frame consistent ----
        # If your low-level controller expects base-frame deltas but you want TCP-frame actions:
        # env = TCPActionRelativeFrame(env)  # converts TCP-frame action -> base-frame action

        # ---- Orientation representation ----
        env = ToMrpWrapper(env)

        # ---- SERL formatting ----
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)

        # ---- Chunking (if your agent expects it) ----
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # ---- Binary reward classifier wrapper ----
        # if classifier:
        #     clf = load_classifier_func(
        #         key=jax.random.PRNGKey(0),
        #         sample=env.observation_space.sample(),
        #         image_keys=self.classifier_keys,
        #         checkpoint_path=self.classifier_ckpt_path,
        #     )

        #     def prob_func(obs):
        #         logits = clf(obs)
        #         logit0 = jnp.ravel(jnp.asarray(logits))[0]
        #         return jax.nn.sigmoid(logit0)

        #     env = RewardClassifierTerminateWrapper(
        #         env,
        #         prob_func,
        #         threshold=self.clf_threshold,
        #         consecutive=self.clf_consecutive,
        #         target_hz=10,
        #         trunc_penalty=00,
        #         pass_env_reward=False,
        #     )
        if classifier:
            # Stage 0: grasp classifier
            clf_grasp = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=["wrist"],  # choose best view for grasp
                checkpoint_path=os.path.abspath("classifier_ckpt/door_grasp/"),
            )

            # Stage 1: door-open classifier
            clf_open = load_classifier_func(
                key=jax.random.PRNGKey(1),
                sample=env.observation_space.sample(),
                image_keys=["shoulder"],  # choose best view for door-open
                checkpoint_path=os.path.abspath("classifier_ckpt/door_open_45deg/"),
            )

            def prob_from_clf(clf_fn):
                def _prob(obs):
                    logits = clf_fn(obs)
                    logit0 = jnp.ravel(jnp.asarray(logits))[0]
                    return jax.nn.sigmoid(logit0)
                return _prob

            prob_grasp = prob_from_clf(clf_grasp)
            prob_open  = prob_from_clf(clf_open)

            env = MultiStageRewardClassifierTerminateWrapper(
                env,
                prob_funcs=[prob_grasp, prob_open],
                thresholds=[0.8, 0.8],
                consecutive=[3, 3],
                stage_rewards=[0.2, 1.0],  # IMPORTANT: prevent “just grasp” local optimum
                in_order=True,
                target_hz=10,
                trunc_penalty=-1.0,
                pass_env_reward=False,
            )
        # ---- Optional gripper penalty (discourage spam) ----
        env = GripperPenaltyWrapper(env, penalty=-0.02)

        return env
