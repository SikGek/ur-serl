"""Run the stock HIL-SERL demo recorder with the G1 experiment registry.

This file intentionally stays thin: it reuses the upstream `examples/record_demos.py`
script and only injects the G1-specific CONFIG_MAPPING. That keeps future merges with
upstream HIL-SERL low-friction.
"""

from __future__ import annotations

from absl import app

import record_demos as _record_demos
from experiments.mappings_g1 import CONFIG_MAPPING as _G1_CONFIG_MAPPING

# Prefer the G1 configs on key collisions.
_record_demos.CONFIG_MAPPING.update(_G1_CONFIG_MAPPING)


def main(argv):
    del argv
    _record_demos.main(None)


if __name__ == "__main__":
    app.run(main)
