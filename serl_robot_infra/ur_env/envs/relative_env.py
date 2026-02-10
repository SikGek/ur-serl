from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
from gymnasium import Env
from franka_env.utils.transformations import (
    construct_homogeneous_matrix,
    construct_rotation_matrix,
)

class RelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base UR5 environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,)),
                "tcp_force": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "tcp_torque": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "gripper_state": spaces.Box(-np.inf, np.inf, shape=(2,)),
            }
        ),
        ......
    }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...)
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.rotation_matrix_reset = np.eye((3))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action_inv(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"]

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # obs['state']['tcp_pose'][:2] -= info['reset_shift']  # set rel pose to original reset pose (no random)

        self.rotation_matrix_reset = construct_rotation_matrix(obs["state"]["tcp_pose"])
        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the rotation and homogeneous matrix
        """
        obs["state"]["tcp_vel"][:3] = self.rotation_matrix_reset.T @ obs["state"]["tcp_vel"][:3]
        obs["state"]["tcp_vel"][3:6] = self.rotation_matrix_reset.T @ obs["state"]["tcp_vel"][3:6]
        obs["state"]["tcp_force"] = self.rotation_matrix_reset.T @ obs["state"]["tcp_force"]
        obs["state"]["tcp_torque"] = self.rotation_matrix_reset.T @ obs["state"]["tcp_torque"]
        obs["state"]["action"] = self.transform_action_inv(obs["state"]["action"])

        if "ema_tcp_vel" in obs["state"]:
            obs["state"]["ema_tcp_vel"][:3] = self.rotation_matrix_reset.T @ obs["state"]["ema_tcp_vel"][:3]
            obs["state"]["ema_tcp_vel"][3:6] = self.rotation_matrix_reset.T @ obs["state"]["ema_tcp_vel"][3:6]
        if "ema_force" in obs["state"]:
            obs["state"]["ema_force"][:3] = self.rotation_matrix_reset.T @ obs["state"]["ema_force"][:3]
            obs["state"]["ema_force"][3:6] = self.rotation_matrix_reset.T @ obs["state"]["ema_force"][3:6]

        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_b_r = self.T_r_o_inv @ T_b_o

            # Reconstruct transformed tcp_pose vector
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into spatial(base) frame
        using the rotation matrix
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:3] = self.rotation_matrix_reset @ action[:3]
        action[3:6] = self.rotation_matrix_reset @ action[3:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the rotation matrix.
        """
        action = np.array(action)
        action[:3] = self.rotation_matrix_reset.T @ action[:3]
        action[3:6] = self.rotation_matrix_reset.T @ action[3:6]
        return action


class DualRelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the DualUR5Env, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "left/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ...
                "right/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ...
            }
        ),
        ......
    }, and at least 12 DoF action space
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.rot_mat_left = np.eye((3))
        self.rot_mat_right = np.eye((3))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.left_T_r_o_inv = np.zeros((4, 4))
            self.right_T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"]

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Update rotation matrices
        self.rot_mat_left = construct_rotation_matrix(obs["state"]["left/tcp_pose"])
        self.rot_mat_right = construct_rotation_matrix(obs["state"]["right/tcp_pose"])

        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.left_T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["left/tcp_pose"])
            )
            self.right_T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["right/tcp_pose"])
            )
        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the rotation and homogeneous matrix
        """
        for both, rot_mat in zip(("left/", "right/"), (self.rot_mat_left, self.rot_mat_right)):
            # velocities (twist) rotate as vectors
            obs["state"][f"{both}tcp_vel"][:3] = rot_mat.T @ obs["state"][f"{both}tcp_vel"][:3]
            obs["state"][f"{both}tcp_vel"][3:6] = rot_mat.T @ obs["state"][f"{both}tcp_vel"][3:6]
            # forces/torques are vectors/pseudovectors
            obs["state"][f"{both}tcp_force"] = rot_mat.T @ obs["state"][f"{both}tcp_force"]
            obs["state"][f"{both}tcp_torque"] = rot_mat.T @ obs["state"][f"{both}tcp_torque"]
            # action in observation assumed to be a twist; rotate like velocities
            obs["state"][f"{both}action"][:3] = rot_mat.T @ obs["state"][f"{both}action"][:3]
            obs["state"][f"{both}action"][3:6] = rot_mat.T @ obs["state"][f"{both}action"][3:6]

            key_v = f"{both}ema_tcp_vel"
            key_f = f"{both}ema_force"
            if key_v in obs["state"]:
                obs["state"][key_v][:3] = rot_mat.T @ obs["state"][key_v][:3]
                obs["state"][key_v][3:6] = rot_mat.T @ obs["state"][key_v][3:6]
            if key_f in obs["state"]:
                obs["state"][key_f][:3] = rot_mat.T @ obs["state"][key_f][:3]
                obs["state"][key_f][3:6] = rot_mat.T @ obs["state"][key_f][3:6]


        if self.include_relative_pose:
            left_T_b_o = construct_homogeneous_matrix(obs["state"]["left/tcp_pose"])
            left_T_b_r = self.left_T_r_o_inv @ left_T_b_o

            left_p_b_r = left_T_b_r[:3, 3]
            left_theta_b_r = R.from_matrix(left_T_b_r[:3, :3]).as_quat()
            obs["state"]["left/tcp_pose"] = np.concatenate((left_p_b_r, left_theta_b_r))

            right_T_b_o = construct_homogeneous_matrix(obs["state"]["right/tcp_pose"])
            right_T_b_r = self.right_T_r_o_inv @ right_T_b_o

            right_p_b_r = right_T_b_r[:3, 3]
            right_theta_b_r = R.from_matrix(right_T_b_r[:3, :3]).as_quat()
            obs["state"]["right/tcp_pose"] = np.concatenate((right_p_b_r, right_theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action (12d) from body(end-effector) frame into spatial(base) frame
        using the rotation matrix
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:3] = self.rot_mat_left @ action[:3]
        action[3:6] = self.rot_mat_left @ action[3:6]
        action[7:10] = self.rot_mat_right @ action[7:10]
        action[10:13] = self.rot_mat_right @ action[10:13]
        return action

class BaseFrameRotation(gym.Wrapper):
    """
    Watch out, is legacy code, not used anywhere.
    """
    def __init__(self, env: Env, rx=0., ry=0., rz=0.):
        super().__init__(env)
        self.base_frame_rotation = R.from_euler("xyz", [rx, ry, rz]).as_matrix()

    def step(self, action: np.ndarray):
        transformed_action = self.base_transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)

        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"]

        transformed_obs = self.base_transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.base_transform_observation(obs), info

    def base_transform_observation(self, obs):
        """
        Transform observations from base frame to the rotated frame
        """
        obs["state"]["tcp_pose"][:3] = self.base_frame_rotation @ obs["state"]["tcp_pose"][:3]
        obs["state"]["tcp_pose"][3:] = (R.from_quat(obs["state"]["tcp_pose"][3:6]) * R.from_matrix(self.base_frame_rotation)).as_quat()
        obs["state"]["tcp_vel"][:3] = self.base_frame_rotation.T @ obs["state"]["tcp_vel"][:3]
        obs["state"]["tcp_vel"][3:6] = self.base_frame_rotation.T @ obs["state"]["tcp_vel"][3:6]
        obs["state"]["tcp_force"] = self.base_frame_rotation.T @ obs["state"]["tcp_force"]
        obs["state"]["tcp_torque"] = self.base_frame_rotation.T @ obs["state"]["tcp_torque"]
        return obs

    def base_transform_action(self, action: np.ndarray):
        action = np.array(action)  # in case action is a jax read-only array
        action[:3] = self.base_frame_rotation @ action[:3]
        action[3:6] = (R.from_mrp(action[3:6]) * R.from_matrix(self.base_frame_rotation)).as_mrp()
        return action


class TrueTCPRelativeFrame(gym.Wrapper):
    """
    True TCP/body-frame wrapper for UR-style envs that apply deltas in the base/space frame.

    Incoming action (agent-level):
        a_tcp = [dx, dy, dz, dσx, dσy, dσz, gripper]
    where translation is a direction in TCP frame, and rotation is an MRP increment in TCP frame.

    Underlying env expects (base/space-frame):
        a_base = [dx, dy, dz, dσx, dσy, dσz, gripper]
    but interprets rotation as:
        R_next = R_delta_base * R_current
        R_delta_base = from_mrp(action[3:6] * rot_scale)
    so we must conjugate to convert body-frame increments to space-frame increments.

    Also:
      - Optionally rotates tcp_vel/tcp_force/tcp_torque into current TCP frame.
      - Optionally replaces tcp_pose with pose relative to reset (in reset frame).
      - Converts info["intervene_action"] to TCP-frame so replay actions are consistent.

    Important: This wrapper assumes your env uses MRP for the rotational action part (like your UR5Env).
    """

    def __init__(
        self,
        env: gym.Env,
        include_relative_pose: bool = True,
        rotate_obs_to_tcp: bool = True,
        convert_intervene_action: bool = True,
        action_rot_is_mrp: bool = True,
    ):
        super().__init__(env)
        self.include_relative_pose = bool(include_relative_pose)
        self.rotate_obs_to_tcp = bool(rotate_obs_to_tcp)
        self.convert_intervene_action = bool(convert_intervene_action)
        self.action_rot_is_mrp = bool(action_rot_is_mrp)

        # Reset reference
        self._p0 = None
        self._R0 = None  # Rotation object at reset (base<-tcp)

        # Cache scales (read from env.unwrapped if possible)
        self._pos_scale, self._rot_scale = self._infer_scales()

    # -------------------- scale + state helpers --------------------

    def _infer_scales(self):
        # Defaults if not found
        pos_scale = 1.0
        rot_scale = 1.0
        try:
            base = self.env.unwrapped
            if hasattr(base, "action_scale"):
                pos_scale = float(base.action_scale[0])
                # matches your UR5Env.step: R.from_mrp(action[3:6] * action_scale[1] / 4.)
                rot_scale = float(base.action_scale[1]) / 4.0
        except Exception:
            pass
        return pos_scale, rot_scale

    def _get_base_pose_quat(self, obs=None):
        """
        Returns (p_base, q_base) for current TCP pose in BASE frame.
        Prefer env.unwrapped.curr_pos since other wrappers may modify obs tcp_pose.
        """
        base = self.env.unwrapped
        if hasattr(base, "curr_pos") and base.curr_pos is not None:
            cp = np.asarray(base.curr_pos, dtype=np.float64).reshape(-1)
            if cp.shape[0] >= 7:
                return cp[:3].copy(), cp[3:7].copy()

        # fallback to obs if needed
        if obs is not None:
            pose = np.asarray(obs["state"]["tcp_pose"], dtype=np.float64).reshape(-1)
            if pose.shape[0] >= 7:
                return pose[:3].copy(), pose[3:7].copy()

        raise RuntimeError("Could not retrieve base tcp pose (need curr_pos or obs['state']['tcp_pose']).")

    # -------------------- action transforms --------------------

    def tcp_action_to_base(self, a_tcp: np.ndarray, q_base: np.ndarray) -> np.ndarray:
        """
        Convert action expressed in CURRENT TCP frame to base/space-frame action for env.step.
        """
        a_tcp = np.asarray(a_tcp, dtype=np.float32).copy()
        a_base = a_tcp.copy()

        # Rotation base<-tcp (SciPy uses x,y,z,w)
        R_bt = R.from_quat(np.asarray(q_base, dtype=np.float64))

        # --- translation: just rotate normalized direction ---
        a_base[:3] = R_bt.apply(a_tcp[:3])

        # --- rotation: body increment -> space increment via conjugation ---
        if self.action_rot_is_mrp:
            sigma_tcp = a_tcp[3:6] * self._rot_scale
            R_delta_tcp = R.from_mrp(sigma_tcp)
            R_delta_base = R_bt * R_delta_tcp * R_bt.inv()
            sigma_base = R_delta_base.as_mrp()
            a_base[3:6] = sigma_base / self._rot_scale
        else:
            # If using small-angle rotvec increments, vector-rotate is a decent approximation.
            a_base[3:6] = R_bt.apply(a_tcp[3:6])

        return np.clip(a_base, -1.0, 1.0).astype(np.float32)

    def base_action_to_tcp(self, a_base: np.ndarray, q_base: np.ndarray) -> np.ndarray:
        """
        Convert base/space-frame action (what env executes) to CURRENT TCP-frame action.
        Used mainly for converting info["intervene_action"] into the policy frame.
        """
        a_base = np.asarray(a_base, dtype=np.float32).copy()
        a_tcp = a_base.copy()

        R_bt = R.from_quat(np.asarray(q_base, dtype=np.float64))
        R_tb = R_bt.inv()

        # translation
        a_tcp[:3] = R_tb.apply(a_base[:3])

        # rotation
        if self.action_rot_is_mrp:
            sigma_base = a_base[3:6] * self._rot_scale
            R_delta_base = R.from_mrp(sigma_base)
            R_delta_tcp = R_tb * R_delta_base * R_tb.inv()
            sigma_tcp = R_delta_tcp.as_mrp()
            a_tcp[3:6] = sigma_tcp / self._rot_scale
        else:
            a_tcp[3:6] = R_tb.apply(a_base[3:6])

        return np.clip(a_tcp, -1.0, 1.0).astype(np.float32)

    # -------------------- observation transforms --------------------

    def _transform_observation(self, obs, q_base_post: np.ndarray, action_tcp_executed: np.ndarray):
        """
        - Optionally rotate vel/force/torque to CURRENT TCP frame.
        - Optionally convert tcp_pose to reset-relative pose.
        - Optionally set obs["state"]["action"] to executed action in TCP frame.
        """
        if "state" not in obs:
            return obs

        obs = obs  # mutate in place (typical wrapper pattern)

        # Optionally overwrite action in observation with the TCP-frame executed action
        if "action" in obs["state"]:
            obs["state"]["action"] = np.asarray(action_tcp_executed, dtype=np.float32)

        # Rotate wrench / twist to TCP frame using CURRENT orientation
        if self.rotate_obs_to_tcp:
            R_bt = R.from_quat(np.asarray(q_base_post, dtype=np.float64))
            R_tb = R_bt.inv()

            def _rot3(x):
                x = np.asarray(x, dtype=np.float32).copy()
                return R_tb.apply(x).astype(np.float32)

            if "tcp_vel" in obs["state"]:
                v = np.asarray(obs["state"]["tcp_vel"], dtype=np.float32).copy()
                v[:3] = _rot3(v[:3])
                v[3:6] = _rot3(v[3:6])
                obs["state"]["tcp_vel"] = v

            if "tcp_force" in obs["state"]:
                obs["state"]["tcp_force"] = _rot3(obs["state"]["tcp_force"])

            if "tcp_torque" in obs["state"]:
                obs["state"]["tcp_torque"] = _rot3(obs["state"]["tcp_torque"])

            for k in ("ema_tcp_vel", "ema_force"):
                if k in obs["state"]:
                    vv = np.asarray(obs["state"][k], dtype=np.float32).copy()
                    vv[:3] = _rot3(vv[:3])
                    vv[3:6] = _rot3(vv[3:6])
                    obs["state"][k] = vv

        # Optionally convert tcp_pose to reset-relative pose (in reset frame)
        if self.include_relative_pose and self._R0 is not None and self._p0 is not None:
            # Need current base pose (p,q) — use env state, not obs (obs may be post-processed elsewhere)
            p, q = self._get_base_pose_quat(obs=None)
            R_bt = R.from_quat(q)
            R_0b = self._R0.inv()

            p_rel = R_0b.apply(p - self._p0)
            q_rel = (R_0b * R_bt).as_quat()  # reset^-1 * current
            obs["state"]["tcp_pose"] = np.concatenate([p_rel, q_rel]).astype(np.float32)

        return obs

    # -------------------- gym API --------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Store reset pose in BASE frame
        p0, q0 = self._get_base_pose_quat(obs)
        self._p0 = p0
        self._R0 = R.from_quat(q0)

        # For reset output transform, we use current pose (same as reset)
        action_dummy = np.zeros_like(self.env.action_space.sample(), dtype=np.float32)
        obs = self._transform_observation(obs, q_base_post=q0, action_tcp_executed=action_dummy)
        return obs, info

    def step(self, action):
        # q_pre used for action conversion and intervene_action conversion
        _, q_pre = self._get_base_pose_quat(obs=None)

        # Convert policy TCP-frame action -> base action
        a_base = self.tcp_action_to_base(action, q_pre)

        # Step underlying env (may override via SpacemouseIntervention)
        obs, reward, done, truncated, info = self.env.step(a_base)

        # Determine what action was actually executed, in TCP frame
        executed_tcp = np.asarray(action, dtype=np.float32).copy()

        if self.convert_intervene_action and isinstance(info, dict) and "intervene_action" in info:
            # Underlying env likely stored base-frame action here — convert to TCP frame for consistency
            a_exec_base = np.asarray(info["intervene_action"], dtype=np.float32)
            a_exec_tcp = self.base_action_to_tcp(a_exec_base, q_pre)
            info["intervene_action"] = a_exec_tcp
            executed_tcp = a_exec_tcp

        # q_post for rotating observations
        _, q_post = self._get_base_pose_quat(obs=None)

        obs = self._transform_observation(obs, q_base_post=q_post, action_tcp_executed=executed_tcp)
        return obs, reward, done, truncated, info
