from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
from gymnasium import Env
from franka_env.utils.transformations import (
    construct_homogeneous_matrix,
    construct_rotation_matrix,
)

# class RelativeFrame(gym.Wrapper):
#     """
#     Same wrapper as before, but now compensates for a fixed TCP/tool offset
#     between RTDE-reported pose (flange/tool0) and true end-effector tip.

#     tool_offset_tool: vector from reported frame F to desired tip frame E,
#                       expressed in the TOOL frame F (default +Z 0.16m).
#     """

#     def __init__(
#         self,
#         env: Env,
#         include_relative_pose=True,
#         tool_offset_tool=np.array([0.0, 0.0, 0.16], dtype=np.float64),
#         compensate_offset_in_actions=True,
#         compensate_offset_in_obs=True,
#     ):
#         super().__init__(env)
#         self.rotation_matrix_reset = np.eye(3)
#         self.include_relative_pose = include_relative_pose
#         self.compensate_offset_in_actions = bool(compensate_offset_in_actions)
#         self.compensate_offset_in_obs = bool(compensate_offset_in_obs)

#         self.tool_offset_tool = np.asarray(tool_offset_tool, dtype=np.float64).reshape(3)

#         if self.include_relative_pose:
#             self.T_r_o_inv = np.zeros((4, 4))

#         # infer env scaling for rotations (your UR5Env uses from_mrp(action*action_scale/4))
#         base = self.env.unwrapped
#         if hasattr(base, "action_scale"):
#             self._pos_scale = float(base.action_scale[0])
#             self._rot_scale = float(base.action_scale[1]) / 4.0
#         else:
#             self._pos_scale = 1.0
#             self._rot_scale = 1.0

#     # ---------------- helpers ----------------

#     def _get_curr_quat_base(self):
#         """
#         Use the *true* robot state (unwrapped.curr_pos) so wrapper stacking does not break it.
#         Expects [x,y,z,qx,qy,qz,qw].
#         """
#         cp = np.asarray(self.env.unwrapped.curr_pos, dtype=np.float64).reshape(-1)
#         if cp.shape[0] < 7:
#             raise RuntimeError("env.unwrapped.curr_pos must be xyz+quat (7).")
#         return cp[3:7]

#     def _R_BF(self):
#         """Rotation from tool/flange frame F to base frame B."""
#         return R.from_quat(self._get_curr_quat_base())

#     def _offset_base(self, R_BF: R):
#         """Offset vector expressed in base frame: R_BF * p_FE."""
#         return R_BF.apply(self.tool_offset_tool)

#     # ---------------- core: offset compensation on base-frame action ----------------

#     def _compensate_base_action_for_tip(self, action_base: np.ndarray) -> np.ndarray:
#         """
#         action_base is what will be sent to env.step().
#         The env applies:
#           p' = p + a[:3]*pos_scale
#           R' = R_delta_base * R
#           R_delta_base = from_mrp(a[3:6]*rot_scale)

#         We adjust translation so that rotations happen about the *tip* not about flange.
#         """
#         a = np.asarray(action_base, dtype=np.float32).copy()

#         # current orientation
#         R_BF_now = self._R_BF()

#         # rotation delta that env will apply
#         R_delta_base = R.from_mrp(a[3:6] * self._rot_scale)

#         # orientation after action
#         R_BF_next = R_delta_base * R_BF_now

#         # translation compensation in base coords
#         # dp_comp = R_now*p_FE - R_next*p_FE
#         dp_comp_base = self._offset_base(R_BF_now) - self._offset_base(R_BF_next)

#         # add dp_comp into env translation
#         dp_base = a[:3] * self._pos_scale
#         dp_base = dp_base + dp_comp_base.astype(np.float32)

#         a[:3] = (dp_base / self._pos_scale).astype(np.float32)

#         return np.clip(a, -1.0, 1.0).astype(np.float32)

#     # ---------------- existing wrapper API ----------------

#     def step(self, action: np.ndarray):
#         """
#         IMPORTANT: This wrapper assumes the incoming action is already in base-frame
#         (i.e., something like your old SpaceMouse 'true TCP' code produced).

#         If you want this wrapper to accept TCP-frame actions instead, say so and I’ll
#         swap the mapping accordingly.
#         """
#         action = np.asarray(action, dtype=np.float32)

#         # Your old code did transform_action_inv(action) here, but that mapping was reset-frame based.
#         # We keep your structure and treat incoming action as base-frame.
#         transformed_action = action.copy()

#         # Apply tool-offset compensation so rotations are about the true TCP (tip)
#         if self.compensate_offset_in_actions:
#             transformed_action = self._compensate_base_action_for_tip(transformed_action)

#         obs, reward, done, truncated, info = self.env.step(transformed_action)

#         # If a downstream wrapper injects intervene_action (base-frame), also compensate it for logging consistency
#         if "intervene_action" in info and self.compensate_offset_in_actions:
#             info["intervene_action"] = self._compensate_base_action_for_tip(info["intervene_action"])

#         transformed_obs = self.transform_observation(obs)
#         return transformed_obs, reward, done, truncated, info

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)

#         # keep your original reset reference
#         self.rotation_matrix_reset = construct_rotation_matrix(obs["state"]["tcp_pose"])

#         if self.include_relative_pose:
#             self.T_r_o_inv = np.linalg.inv(construct_homogeneous_matrix(obs["state"]["tcp_pose"]))

#         return self.transform_observation(obs), info

#     def transform_observation(self, obs):
#         """
#         Keep your existing observation transforms, but ALSO shift tcp_pose position by tool offset
#         so that obs tcp_pose is the *tip* pose rather than flange pose.
#         """
#         # Your existing transforms (reset-frame)
#         obs["state"]["tcp_vel"][:3] = self.rotation_matrix_reset.T @ obs["state"]["tcp_vel"][:3]
#         obs["state"]["tcp_vel"][3:6] = self.rotation_matrix_reset.T @ obs["state"]["tcp_vel"][3:6]
#         obs["state"]["tcp_force"] = self.rotation_matrix_reset.T @ obs["state"]["tcp_force"]
#         obs["state"]["tcp_torque"] = self.rotation_matrix_reset.T @ obs["state"]["tcp_torque"]

#         # keep action transform as before (note: this is just for the action stored in obs)
#         if "action" in obs["state"]:
#             obs["state"]["action"] = self.transform_action_inv(obs["state"]["action"])

#         if "ema_tcp_vel" in obs["state"]:
#             obs["state"]["ema_tcp_vel"][:3] = self.rotation_matrix_reset.T @ obs["state"]["ema_tcp_vel"][:3]
#             obs["state"]["ema_tcp_vel"][3:6] = self.rotation_matrix_reset.T @ obs["state"]["ema_tcp_vel"][3:6]
#         if "ema_force" in obs["state"]:
#             obs["state"]["ema_force"][:3] = self.rotation_matrix_reset.T @ obs["state"]["ema_force"][:3]
#             obs["state"]["ema_force"][3:6] = self.rotation_matrix_reset.T @ obs["state"]["ema_force"][3:6]

#         # Optional reset-relative pose
#         if self.include_relative_pose:
#             T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
#             T_b_r = self.T_r_o_inv @ T_b_o
#             p_b_r = T_b_r[:3, 3]
#             theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
#             obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

#         # ---- NEW: shift tcp_pose position by tool offset so it represents the tip ----
#         if self.compensate_offset_in_obs and "tcp_pose" in obs["state"]:
#             pose = np.asarray(obs["state"]["tcp_pose"], dtype=np.float64).reshape(-1)

#             # If tcp_pose is xyz+quat (7), shift xyz by R * offset
#             if pose.shape[0] >= 7:
#                 p = pose[:3]
#                 q = pose[3:7]
#                 R_BF = R.from_quat(q)
#                 p_tip = p + R_BF.apply(self.tool_offset_tool)
#                 obs["state"]["tcp_pose"] = np.concatenate([p_tip, q]).astype(np.float32)

#             # If tcp_pose is already 6D (xyz + something), we still shift xyz using true current quat
#             elif pose.shape[0] >= 3:
#                 p = pose[:3]
#                 R_BF = self._R_BF()
#                 p_tip = p + R_BF.apply(self.tool_offset_tool)
#                 pose = pose.astype(np.float32)
#                 pose[:3] = p_tip.astype(np.float32)
#                 obs["state"]["tcp_pose"] = pose

#         return obs

#     def transform_action(self, action: np.ndarray):
#         action = np.array(action, dtype=np.float32)
#         action[:3] = self.rotation_matrix_reset @ action[:3]
#         action[3:6] = self.rotation_matrix_reset @ action[3:6]
#         return action

#     def transform_action_inv(self, action: np.ndarray):
#         action = np.array(action, dtype=np.float32)
#         action[:3] = self.rotation_matrix_reset.T @ action[:3]
#         action[3:6] = self.rotation_matrix_reset.T @ action[3:6]
#         return action



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



class TCPActionRelativeFrame(gym.Wrapper):
    """
    Makes the *action space* be in TRUE TCP/body frame, while the underlying env still
    uses base/space-frame deltas (UR5Env.step uses left-multiplication by from_mrp).

    - Incoming action is interpreted as TCP-frame (body-frame):
        a_tcp = [dx, dy, dz, dσx, dσy, dσz, gripper]
      where dσ is an MRP increment in TCP frame.

    - Wrapper converts to base-frame action before calling env.step().

    - It also converts info["intervene_action"] to TCP frame (if present), so replay logs
      remain consistent (actions always stored in TCP frame).
    """

    def __init__(self, env, action_rot_is_mrp=True, convert_intervene_action=True):
        super().__init__(env)
        self.action_rot_is_mrp = bool(action_rot_is_mrp)
        self.convert_intervene_action = bool(convert_intervene_action)

        # Infer scaling used by UR5Env.step()
        base = self.env.unwrapped
        self.pos_scale = float(getattr(base, "action_scale", [1.0])[0])
        self.rot_scale = float(getattr(base, "action_scale", [1.0])[1]) / 4.0  # matches R.from_mrp(a*scale/4)

    def _get_R_bt(self):
        """
        R_bt: rotation TCP -> base.
        Uses env.unwrapped.curr_pos quaternion (xyz + quat).
        """
        q = np.asarray(self.env.unwrapped.curr_pos[3:7], dtype=np.float64)
        return R.from_quat(q)

    def tcp_to_base_action(self, a_tcp: np.ndarray) -> np.ndarray:
        a_tcp = np.asarray(a_tcp, dtype=np.float32).copy()
        R_bt = self._get_R_bt()

        # Translation: TCP -> base
        a_tcp[:3] = R_bt.apply(a_tcp[:3])

        # Rotation: TCP-body increment -> base-space increment
        if self.action_rot_is_mrp:
            sigma_tcp = a_tcp[3:6] * self.rot_scale
            R_delta_tcp = R.from_mrp(sigma_tcp)
            R_delta_base = R_bt * R_delta_tcp * R_bt.inv()
            sigma_base = R_delta_base.as_mrp()
            a_tcp[3:6] = sigma_base / self.rot_scale
        else:
            # fallback small-angle approx
            a_tcp[3:6] = R_bt.apply(a_tcp[3:6])

        return np.clip(a_tcp, -1.0, 1.0).astype(np.float32)

    def base_to_tcp_action(self, a_base: np.ndarray) -> np.ndarray:
        a_base = np.asarray(a_base, dtype=np.float32).copy()
        R_bt = self._get_R_bt()
        R_tb = R_bt.inv()

        # Translation: base -> TCP
        a_base[:3] = R_tb.apply(a_base[:3])

        if self.action_rot_is_mrp:
            sigma_base = a_base[3:6] * self.rot_scale
            R_delta_base = R.from_mrp(sigma_base)
            R_delta_tcp = R_tb * R_delta_base * R_tb.inv()
            sigma_tcp = R_delta_tcp.as_mrp()
            a_base[3:6] = sigma_tcp / self.rot_scale
        else:
            a_base[3:6] = R_tb.apply(a_base[3:6])

        return np.clip(a_base, -1.0, 1.0).astype(np.float32)

    def step(self, action):
        # policy action is interpreted as TCP-frame -> convert to base-frame for env
        a_base = self.tcp_to_base_action(action)

        obs, reward, done, truncated, info = self.env.step(a_base)

        # If an intervention action was injected downstream and is in base-frame,
        # convert it to TCP-frame so the replay stores consistent "TCP-frame actions"
        if self.convert_intervene_action and isinstance(info, dict) and "intervene_action" in info:
            info["intervene_action"] = self.base_to_tcp_action(info["intervene_action"])

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


