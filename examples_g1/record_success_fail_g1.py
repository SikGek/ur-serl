"""Run the stock HIL-SERL reward-image recorder with the G1 experiment registry.

Use SpaceMouse teleoperation to place the arm in successful terminal poses, and press
SPACE to mark the current transition as a positive example. All other transitions are
stored as negatives, matching the public HIL-SERL workflow.
"""

from __future__ import annotations

from absl import app

import record_success_fail as _record_success_fail
from experiments.mappings_g1 import CONFIG_MAPPING as _G1_CONFIG_MAPPING

_record_success_fail.CONFIG_MAPPING.update(_G1_CONFIG_MAPPING)


def main(argv):
    del argv
    _record_success_fail.main(None)


if __name__ == "__main__":
    app.run(main)
