# import copy
# import os
# from tqdm import tqdm
# import numpy as np
# import pickle as pkl
# import datetime
# from absl import app, flags
# from pynput import keyboard

# from experiments.mappings import CONFIG_MAPPING

# FLAGS = flags.FLAGS
# flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
# flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")


# success_key = False
# def on_press(key):
#     global success_key
#     try:
#         if str(key) == 'Key.space':
#             success_key = True
#     except AttributeError:
#         pass

# def main(_):
#     global success_key
#     listener = keyboard.Listener(
#         on_press=on_press)
#     listener.start()
#     assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
#     config = CONFIG_MAPPING[FLAGS.exp_name]()
#     env = config.get_environment(fake_env=False, save_video=False, classifier=False)

#     obs, _ = env.reset()
#     successes = []
#     failures = []
#     success_needed = FLAGS.successes_needed
#     pbar = tqdm(total=success_needed)
    
#     while len(successes) < success_needed:
#         actions = np.zeros(env.action_space.sample().shape) 
#         next_obs, rew, done, truncated, info = env.step(actions)
#         if "intervene_action" in info:
#             actions = info["intervene_action"]

#         transition = copy.deepcopy(
#             dict(
#                 observations=obs,
#                 actions=actions,
#                 next_observations=next_obs,
#                 rewards=rew,
#                 masks=1.0 - done,
#                 dones=done,
#             )
#         )
#         obs = next_obs
#         if success_key:
#             successes.append(transition)
#             pbar.update(1)
#             success_key = False
#         else:
#             failures.append(transition)

#         if done or truncated:
#             obs, _ = env.reset()

#     if not os.path.exists("./classifier_data"):
#         os.makedirs("./classifier_data")
#     uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
#     with open(file_name, "wb") as f:
#         pkl.dump(successes, f)
#         print(f"saved {success_needed} successful transitions to {file_name}")

#     file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
#     with open(file_name, "wb") as f:
#         pkl.dump(failures, f)
#         print(f"saved {len(failures)} failure transitions to {file_name}")
        
# if __name__ == "__main__":
#     app.run(main)


import copy
import os
import time
import datetime
import pickle as pkl
from collections import deque

import numpy as np
from tqdm import tqdm
from absl import app, flags
from pynput import keyboard

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 200, "Number of successful transitions to collect.")

# --- labeling knobs ---
POS_FRAMES = 5          # label last 3-5 transitions as success (set to 3, 4, or 5)
HOLD_FRAMES = 15        # keep this many transitions "pending" before committing them to failures
WAIT_ON_DONE_S = 0.75   # if episode ends in success, wait this long for SPACE to be pressed
AUTO_LABEL_ON_ENV_SUCCESS = False  # if True: auto-label success when info["succeed"] is True

# keyboard flag
success_key = False
def on_press(key):
    global success_key
    if key == keyboard.Key.space:
        success_key = True


def commit_success_from_pending(pending, successes, failures, successes_needed):
    """
    Move the last POS_FRAMES transitions from pending -> successes,
    and move older pending transitions -> failures.
    Clears pending after committing.
    """
    if not pending:
        return

    pending_list = list(pending)
    k = min(POS_FRAMES, len(pending_list))

    neg = pending_list[:-k]
    pos = pending_list[-k:]

    # older transitions become negatives
    failures.extend(neg)

    # add positives up to target
    remaining = successes_needed - len(successes)
    if remaining > 0:
        successes.extend(pos[:remaining])

    pending.clear()


def flush_pending_to_failures(pending, failures):
    """Everything still pending becomes a negative."""
    if pending:
        failures.extend(list(pending))
        pending.clear()


def main(_):
    global success_key

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    obs, _ = env.reset()
    # print(env.unwrapped.curr_reset_pose)
    # print(env.unwrapped.curr_reset_pose[:3])
    # exit()
    successes = []
    failures = []
    pending = deque()  # transitions not yet committed to success/failure

    pbar = tqdm(total=FLAGS.successes_needed)

    while len(successes) < FLAGS.successes_needed:
        # Step with zeros: spacemouse wrapper (if present) may override and put executed action in info["intervene_action"]
        action_in = np.zeros(env.action_space.sample().shape, dtype=np.float32)
        
        next_obs, rew, done, truncated, info = env.step(action_in)

        # Store executed action if wrapper provides it
        executed_action = info.get("intervene_action", action_in)
        print(obs["state"][0].shape)
        assert(obs["state"][0].shape[1] == 25)
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=executed_action,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - float(done),
                dones=bool(done),
                infos=info,  # optional: keep info for debugging
            )
        )

        obs = next_obs
        pending.append(transition)

        # Keep pending bounded: old transitions become negative
        while len(pending) > HOLD_FRAMES:
            failures.append(pending.popleft())

        # --- Manual success key handling ---
        if success_key:
            commit_success_from_pending(pending, successes, failures, FLAGS.successes_needed)
            pbar.n = len(successes)
            pbar.refresh()
            success_key = False

        # --- Episode ended: give a short window to label success manually ---
        if done or truncated:
            # If the env itself signals success, you have two options:
            #  (A) auto label on env succeed
            #  (B) wait briefly so you can hit SPACE even after done
            env_succeed = bool(info.get("succeed", False))

            if env_succeed and not truncated:
                if AUTO_LABEL_ON_ENV_SUCCESS:
                    commit_success_from_pending(pending, successes, failures, FLAGS.successes_needed)
                    pbar.n = len(successes)
                    pbar.refresh()
                else:
                    # Wait for SPACE
                    t0 = time.time()
                    print("\n[record_success_fail] Episode ended with succeed=True. "
                          f"Press SPACE within {WAIT_ON_DONE_S:.2f}s to label the last frames as SUCCESS...")
                    while time.time() - t0 < WAIT_ON_DONE_S:
                        if success_key:
                            commit_success_from_pending(pending, successes, failures, FLAGS.successes_needed)
                            pbar.n = len(successes)
                            pbar.refresh()
                            success_key = False
                            break
                        time.sleep(0.01)

            # Whatever remains pending is negative
            flush_pending_to_failures(pending, failures)

            # Reset episode
            obs, _ = env.reset()

        pbar.n = len(successes)
        pbar.refresh()

    pbar.close()

    # Final flush: anything pending becomes negative
    flush_pending_to_failures(pending, failures)

    # Save
    os.makedirs("./classifier_data", exist_ok=True)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    succ_path = f"./classifier_data/{FLAGS.exp_name}_{FLAGS.successes_needed}_success_transitions_{uuid}.pkl"
    fail_path = f"./classifier_data/{FLAGS.exp_name}_failure_transitions_{uuid}.pkl"

    with open(succ_path, "wb") as f:
        pkl.dump(successes, f)
    print(f"Saved {len(successes)} SUCCESS transitions to {succ_path}")

    with open(fail_path, "wb") as f:
        pkl.dump(failures, f)
    print(f"Saved {len(failures)} FAILURE transitions to {fail_path}")


if __name__ == "__main__":
    app.run(main)
