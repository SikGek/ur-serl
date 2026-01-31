import numpy as np
from typing import Tuple

from ur_env.envs.ur5_env import UR5Env
from ur_env.envs.basic_env.config import UR5PickingConfig

import time

# used for float value comparisons (pressure of vacuum-gripper)
def is_close(value, target):
    return abs(value - target) < 1e-4


class BoxPickingBasicEnv(UR5Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=UR5PickingConfig)

    def compute_reward(self, obs, action) -> float:
        # huge action gives negative reward (like in mountain car)
        action_cost = 0.1 * np.sum(np.power(action, 2))
        step_cost = 0.01

        gripper_state = obs["state"]['gripper_state']
        suck_cost = 0.1 * float(is_close(gripper_state[0], 0.99))

        pose = obs["state"]["tcp_pose"]
        # box_xy = np.array([0.009, -0.5437])     # TODO replace with camera / pointcloud info of box
        # xy_cost = 5 * np.sum(np.power(pose[:2] - box_xy, 2))        # TODO can be ignored

        # print(f"action_cost: {action_cost}, xy_cost: {xy_cost}")
        if self.reached_goal_state(obs):
            return 10. - action_cost - step_cost - suck_cost
        else:
            return 0.0 - action_cost - step_cost - suck_cost

    def reached_goal_state(self, obs) -> bool:
        # obs[0] == gripper pressure, obs[4] == force in Z-axis
        state = obs["state"]
        return 0.1 < state['gripper_state'][1] < 0.5 and state['tcp_pose'][2] > 0.25  # new min height with box
    
    # def step(self, action):
    #     start_time = time.time()
    #     gripper_action = action[6] * self.action_scale[2]
    #     self._send_pos_command(action)
    #     self._send_gripper_command(gripper_action)

    #     self.curr_path_length += 1

    #     obs = self._get_obs(action)

    #     reward = self.compute_reward(obs, action)
    #     truncated = self._is_truncated()
    #     reward = reward if not truncated else reward - 10.  # truncation penalty
    #     done = self.curr_path_length >= self.max_episode_length or self.reached_goal_state(obs) or truncated

    #     dt = time.time() - start_time
    #     to_sleep = max(0, (1.0 / self.hz) - dt)
    #     # if to_sleep == 0:
    #     #     warnings.warn(f"environment could not be within {self.hz} Hz, took {dt:.4f}s!")
    #     time.sleep(to_sleep)
    #     # print(self.get_cost_infos(done)["intervene_action"])
    #     print(action)
    #     return obs, reward, done, truncated, self.get_cost_infos(done)