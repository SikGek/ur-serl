"""Train the stock HIL-SERL reward classifier using the G1 experiment registry."""

from __future__ import annotations

from absl import app

import train_reward_classifier as _train_reward_classifier
from experiments.mappings_g1 import CONFIG_MAPPING as _G1_CONFIG_MAPPING

_train_reward_classifier.CONFIG_MAPPING.update(_G1_CONFIG_MAPPING)


def main(argv):
    del argv
    _train_reward_classifier.main(None)


if __name__ == "__main__":
    app.run(main)
