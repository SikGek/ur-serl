import os
import datetime
import threading
import numpy as np
import copy
import pickle as pkl
from tqdm import tqdm
import gymnasium as gym
from pprint import pprint
from pynput import keyboard
import time

from ur_env.envs.wrappers import SpacemouseIntervention, Quat2MrpWrapper, KinestheticTeaching
from serl_launcher.wrappers.serl_obs_wrappers import SerlObsWrapperNoImages
from serl_launcher.wrappers.chunking import ChunkingWrapper

from gymnasium.wrappers import TransformReward
from ur_env.envs.relative_env import RelativeFrame

exit_program = threading.Event()
input_monitor = {"last_input": None, "last_input_time": None, "input_count": 0, "action_data": None}


def on_space(key, info_dict):
    if key == keyboard.Key.space:
        for key, item in info_dict.items():
            print(f'{key}:  {item}', end='   ')
        print()


def on_key_input(key):
    """Monitor all keyboard inputs."""
    try:
        key_name = key.char if hasattr(key, 'char') else str(key).split('.')[-1]
    except AttributeError:
        key_name = str(key).split('.')[-1]
    
    with threading.Lock():
        input_monitor["last_input"] = key_name
        input_monitor["last_input_time"] = time.time()
        input_monitor["input_count"] += 1
    print(f"[INPUT DETECTED] Key: {key_name} (Total inputs: {input_monitor['input_count']})")


def on_esc(key):
    if key == keyboard.Key.esc:
        exit_program.set()


if __name__ == "__main__":
    print("[STARTUP] Initializing environment...")
    env = gym.make("box_picking_basic_env")
    env = SpacemouseIntervention(env, verbose=True)  # Enable verbose input monitoring
    env = RelativeFrame(env)
    env = Quat2MrpWrapper(env)
    env = SerlObsWrapperNoImages(env)
    # env = TransformReward(env, lambda r: 10. * r)
    # env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    print("[STARTUP] Environment ready. Attempting reset...")
    obs, _ = env.reset()
    print("[STARTUP] Reset successful!")

    transitions = []
    success_count = 0
    success_needed = 20
    total_count = 0
    pbar = tqdm(total=success_needed)

    info_dict = {'state': env.unwrapped.curr_pos, 'gripper_state': env.unwrapped.gripper_state,
                 'force': env.unwrapped.curr_force}
    
    # Add input monitoring listener
    listener_input = keyboard.Listener(daemon=True, on_press=on_key_input)
    listener_input.start()
    print("[STARTUP] Input monitor started - press any key to test keyboard input!")
    
    listener_1 = keyboard.Listener(daemon=True, on_press=lambda event: on_space(event, info_dict=info_dict))
    listener_1.start()

    listener_2 = keyboard.Listener(on_press=on_esc, daemon=True)
    listener_2.start()

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"ur5_test_{success_needed}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    # Monitor for keyboard input activity
    def monitor_input_activity():
        """Periodic check for spacemouse expert input."""
        last_check = 0
        while not exit_program.is_set():
            time.sleep(1)
            try:
                # Get the spacemouse expert from the wrapper
                spacemouse_wrapper = env
                while hasattr(spacemouse_wrapper, 'env'):
                    spacemouse_wrapper = spacemouse_wrapper.env
                
                if hasattr(spacemouse_wrapper, 'expert'):
                    action, buttons = spacemouse_wrapper.expert.get_action()
                    is_active = np.any(action != 0) or buttons != [0, 1]
                    if is_active:
                        input_monitor["action_data"] = (action.copy(), buttons.copy())
                        print(f"[SPACEMOUSE] Action: {action}, Buttons: {buttons}")
            except Exception as e:
                pass  # Silent fail for monitoring

    monitor_thread = threading.Thread(target=monitor_input_activity, daemon=True)
    monitor_thread.start()
    print("[STARTUP] Spacemouse action monitor started!")
    print("[INFO] - Use arrow keys (↑↓←→) for movement, '0' and '1' for Z-axis, Ctrl+R for gripper")
    print("[INFO] - Press SPACE to print current state")
    print("[INFO] - Press ESC to exit\n")

    try:
        while success_count < success_needed:
            if exit_program.is_set():
                raise KeyboardInterrupt  # stop program, but clean up before

            next_obs, rew, done, truncated, info = env.step(action=np.zeros((7,)))
            actions = info["intervene_action"]

            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                )
            )
            transitions.append(transition)
            # pprint(transition)

            obs = next_obs

            if done:
                success_count += int(rew > 0.99)
                total_count += 1
                print(
                    f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed."
                )
                pbar.update(int(rew > 0.99))
                obs, _ = env.reset()

        with open(file_path, "wb") as f:
            pkl.dump(transitions, f)
            print(f"saved {success_needed} demos to {file_path}")

    except KeyboardInterrupt as e:
        print(f'\nProgram was interrupted, cleaning up...  ', e.__str__())

    finally:
        print(f"\n[STATS] Total keyboard inputs detected: {input_monitor['input_count']}")
        if input_monitor['last_input']:
            print(f"[STATS] Last input: {input_monitor['last_input']}")
        pbar.close()
        env.close()
        listener_input.stop()
        listener_1.stop()
        listener_2.stop()
