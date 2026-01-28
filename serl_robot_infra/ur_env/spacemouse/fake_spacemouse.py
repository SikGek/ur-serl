import threading
import numpy as np
from typing import Tuple
from pynput import keyboard
import time


class FakeSpaceMouseExpert:
    """
    This class emulates a SpaceMouse using keyboard input instead.
    Arrow keys control movement:
        - Up/Down: Forward/Backward (Y-axis)
        - Left/Right: Left/Right (X-axis)
        - '1': Up (Z-axis)
        - '0': Down (Z-axis)
    """

    def __init__(self, verbose=False):
        self.state_lock = threading.Lock()
        self.latest_data = {"action": np.zeros(6), "buttons": [0, 1]}
        self.verbose = verbose
        self.last_logged_action = None
        self.last_log_time = 0

        # Start a thread to listen for keyboard input
        self.thread = threading.Thread(target=self._listen_keyboard, daemon=True)
        self.thread.daemon = True
        self.thread.start()
        
        if self.verbose:
            print("[FakeSpaceMouseExpert] Initialized with keyboard input")

    def _on_press(self, key):
        key_name = None
        try:
            key_name = key.char if hasattr(key, 'char') else str(key).split('.')[-1]
        except AttributeError:
            key_name = str(key).split('.')[-1]
        
        with self.state_lock:
            if key == keyboard.Key.up:
                self.latest_data["action"][1] = 1  # Forward (Y+)
                if self.verbose:
                    print(f"[FakeSpaceMouseExpert] UP pressed")
            elif key == keyboard.Key.down:
                self.latest_data["action"][1] = -1  # Backward (Y-)
                if self.verbose:
                    print(f"[FakeSpaceMouseExpert] DOWN pressed")
            elif key == keyboard.Key.left:
                self.latest_data["action"][0] = -1  # Left (X-)
                if self.verbose:
                    print(f"[FakeSpaceMouseExpert] LEFT pressed")
            elif key == keyboard.Key.right:
                self.latest_data["action"][0] = 1  # Right (X+)
                if self.verbose:
                    print(f"[FakeSpaceMouseExpert] RIGHT pressed")
            elif key == keyboard.KeyCode.from_char('1'):
                self.latest_data["action"][2] = 1  # Up (Z+)
                if self.verbose:
                    print(f"[FakeSpaceMouseExpert] '1' pressed (Z+)")
            elif key == keyboard.KeyCode.from_char('0'):
                self.latest_data["action"][2] = -1  # Down (Z-)
                if self.verbose:
                    print(f"[FakeSpaceMouseExpert] '0' pressed (Z-)")
            elif key == keyboard.Key.ctrl_r:
                self.latest_data["buttons"] = [1, 0]
                if self.verbose:
                    print(f"[FakeSpaceMouseExpert] Ctrl+R pressed (gripper action)")

    def _on_release(self, key):
        with self.state_lock:
            if key in [keyboard.Key.up, keyboard.Key.down]:
                self.latest_data["action"][1] = 0
            elif key in [keyboard.Key.left, keyboard.Key.right]:
                self.latest_data["action"][0] = 0
            elif key in [keyboard.KeyCode.from_char('1'), keyboard.KeyCode.from_char('0')]:
                self.latest_data["action"][2] = 0
            elif key == keyboard.Key.ctrl_r:
                self.latest_data["buttons"] = [0, 1]

    def _listen_keyboard(self):
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state."""
        with self.state_lock:
            action = self.latest_data["action"].copy()
            buttons = self.latest_data["buttons"].copy()
        
        # Periodically log active actions
        if self.verbose and (np.any(action != 0) or buttons != [0, 1]):
            current_time = time.time()
            if current_time - self.last_log_time > 0.1:  # Log every 100ms
                action_str = f"Action: {action}, Buttons: {buttons}"
                if action_str != self.last_logged_action:
                    print(f"[FakeSpaceMouseExpert] {action_str}")
                    self.last_logged_action = action_str
                    self.last_log_time = current_time
        
        return action, buttons
