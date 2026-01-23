#!/usr/bin/env python3
"""
Keyboard teleop for a MuJoCo UR5e Gymnasium environment using SpacemouseIntervention.

What it does:
- gym.make(env_id, render_mode="human")
- wraps with optional adapter if base env action dim is 4 (dx,dy,dz,grasp)
- wraps with SpacemouseIntervention (expects 6DoF + buttons => 7D action)
- replaces the expert with a keyboard-based fake spacemouse
- runs without any policy: always steps with zeros; teleop overrides when keys are pressed.

Key mapping (hold keys for continuous motion):
  Translation:
    W/S : +Y / -Y
    A/D : -X / +X
    R/F : +Z / -Z

  Rotation (ignored if your base env is 4D):
    J/L : +Roll / -Roll
    I/K : +Pitch / -Pitch
    U/O : +Yaw / -Yaw

  Gripper:
    Z : "left button"  (close / grip)
    X : "right button" (open / release)

  Quit:
    ESC
"""

import argparse
import time
from typing import Tuple, Set

import numpy as np
import gymnasium as gym
import ur_env
# You likely need one of these imports to trigger env registration:
# import franka_sim
# import ur_env

from pynput import keyboard


# -----------------------------
# 1) Keyboard -> "Fake SpaceMouse" expert
# -----------------------------
class KeyboardFakeSpaceMouseExpert:
    """
    Drop-in replacement for SpaceMouseExpert / FakeSpaceMouseExpert.

    Must implement:
      get_action() -> (np.ndarray shape (6,), buttons tuple(bool,bool))
    """

    def __init__(self, lin_gain: float = 1000.0, rot_gain: float = 1000.0):
        self.lin_gain = float(lin_gain)
        self.rot_gain = float(rot_gain)

        self._pressed: Set[str] = set()
        self._left = False   # emulate left spacemouse button (grip)
        self._right = False  # emulate right spacemouse button (release)
        self._quit = False

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            daemon=True,
        )
        self._listener.start()

    def _on_press(self, key):
        # Handle special keys
        if key == keyboard.Key.esc:
            self._quit = True
            return

        # Handle normal chars
        try:
            k = key.char.lower()
        except AttributeError:
            return

        self._pressed.add(k)

        # emulate spacemouse buttons
        if k == "z":
            self._left = True
        elif k == "x":
            self._right = True

    def _on_release(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            return

        if k in self._pressed:
            self._pressed.remove(k)

        if k == "z":
            self._left = False
        elif k == "x":
            self._right = False

    def get_action(self) -> Tuple[np.ndarray, Tuple[bool, bool]]:
        """
        Returns:
          expert_a: (6,) float32 in [-1, 1]
          buttons: (left, right) booleans
        """
        a = np.zeros((6,), dtype=np.float32)

        # Translation (X,Y,Z)
        if "d" in self._pressed:
            a[0] += self.lin_gain
        if "a" in self._pressed:
            a[0] -= self.lin_gain

        if "w" in self._pressed:
            a[1] += self.lin_gain
        if "s" in self._pressed:
            a[1] -= self.lin_gain

        if "r" in self._pressed:
            a[2] += self.lin_gain
        if "f" in self._pressed:
            a[2] -= self.lin_gain

        # Rotation (roll, pitch, yaw)
        if "j" in self._pressed:
            a[3] += self.rot_gain
        if "l" in self._pressed:
            a[3] -= self.rot_gain

        if "i" in self._pressed:
            a[4] += self.rot_gain
        if "k" in self._pressed:
            a[4] -= self.rot_gain

        if "u" in self._pressed:
            a[5] += self.rot_gain
        if "o" in self._pressed:
            a[5] -= self.rot_gain

        a = np.clip(a, -1.0, 1.0)
        buttons = (bool(self._left), bool(self._right))
        return a, buttons

    @property
    def quit(self) -> bool:
        return self._quit

    def close(self):
        try:
            self._listener.stop()
        except Exception:
            pass


# -----------------------------
# 2) Adapter: expose 7D "spacemouse" action to outer wrapper, map to env's 4D
# -----------------------------
class Spacemouse7DToXYZGWrapper(gym.ActionWrapper):
    """
    If your base env expects (dx,dy,dz,grasp) = 4D,
    but SpacemouseIntervention produces a 7D action:
      [dx, dy, dz, droll, dpitch, dyaw, grip]
    this wrapper converts 7D -> 4D by ignoring rotation and keeping grip.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"Expected 7D action into Spacemouse7DToXYZGWrapper, got {action.shape}")
        dx, dy, dz = action[0], action[1], action[2]
        grip = action[6]
        return np.array([dx, dy, dz, grip], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="box_picking_basic_env",
                        help="Gymnasium env id that runs the MuJoCo UR5e simulation.")
    parser.add_argument("--no_render", action="store_true",
                        help="Do not request render_mode='human'.")
    parser.add_argument("--lin_gain", type=float, default=1.0,
                        help="Keyboard linear axis magnitude (clipped to [-1,1]).")
    parser.add_argument("--rot_gain", type=float, default=1.0,
                        help="Keyboard rotational axis magnitude (clipped to [-1,1]).")
    parser.add_argument("--hz", type=float, default=50.0,
                        help="Client loop rate (best-effort).")
    args = parser.parse_args()

    # 1) Create env
    render_mode = None if args.no_render else "human"
    env = gym.make(args.env_id, camera_mode="none", max_episode_length=10_000_000) if render_mode else gym.make(args.env_id)

    # 2) Insert adapter if base env is 4D
    base_dim = int(np.prod(env.action_space.shape))
    if base_dim == 4:
        env = Spacemouse7DToXYZGWrapper(env)
        print("[teleop] Base env is 4D (dx,dy,dz,grasp). Inserted 7D->4D adapter wrapper.")
    else:
        print(f"[teleop] Base env action dim: {base_dim}. No adapter inserted.")

    # 3) Wrap with SpacemouseIntervention
    # NOTE: import path depends on your repo structure:
    # - In Voxel-SERL it's: from ur_env.envs.wrappers import SpacemouseIntervention
    # - In your pasted code, it was: from ur_env.envs.wrappers import SpacemouseIntervention
    from ur_env.envs.wrappers import SpacemouseIntervention

    sm = SpacemouseIntervention(env)
    # IMPORTANT: force our keyboard expert (so you don't depend on actual SpaceMouse availability)
    sm.expert = KeyboardFakeSpaceMouseExpert(lin_gain=args.lin_gain, rot_gain=args.rot_gain)

    # Optional: if you dislike the deadzone behavior (it’s harmless for +/-1 keyboard inputs)
    # sm.deadspace = 0.0

    env = sm

    print("\n=== Keyboard Teleop Started ===")
    print("W/S: +Y/-Y | A/D: -X/+X | R/F: +Z/-Z")
    print("I/K: +Pitch/-Pitch | J/L: +Roll/-Roll | U/O: +Yaw/-Yaw (ignored if env is 4D)")
    print("Z: close/grip | X: open/release")
    print("ESC: quit\n")

    obs, info = env.reset()
    # u = env.unwrapped
    # print("xyz low/high:", u.xyz_bounding_box.low, u.xyz_bounding_box.high)
    # print("mrp low/high:", u.mrp_bounding_box.low, u.mrp_bounding_box.high)


    dt = 1.0 / max(args.hz, 1e-6)
    try:
        while True:
            if env.expert.quit:  # our keyboard expert
                break

            # 4) No policy: we feed zeros; SpacemouseIntervention overrides if keys pressed.
            policy_action = np.zeros(env.action_space.shape, dtype=np.float32)
            sent_action = info.get("intervene_action", policy_action)
            print("sent_action:", sent_action)
            # if curr_path_length % 10 == 0:
            #     print("STEP action:", action)
            #     print("ACTION_SCALE:", self.action_scale)

            obs, rew, terminated, truncated, info = env.step(sent_action)

            # Some custom envs need explicit render calls even with render_mode="human"
            try:
                env.render()
            except Exception:
                pass

            # if terminated or truncated:
            #     obs, info = env.reset()

            time.sleep(dt)

    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            env.expert.close()
        except Exception:
            pass
        print("[teleop] Clean exit.")


if __name__ == "__main__":
    main()
