"""Run the stock HIL-SERL RLPD trainer with the G1 experiment registry.

The public `examples/train_rlpd.py` already contains the actor/learner implementation.
This wrapper only swaps in the G1 experiment config so the rest of the stack remains
identical to upstream HIL-SERL.
"""

from __future__ import annotations

from absl import app

import train_rlpd as _train_rlpd
from experiments.mappings_g1 import CONFIG_MAPPING as _G1_CONFIG_MAPPING

_train_rlpd.CONFIG_MAPPING.update(_G1_CONFIG_MAPPING)


def main(argv):
    del argv
    _train_rlpd.main(None)


if __name__ == "__main__":
    app.run(main)
