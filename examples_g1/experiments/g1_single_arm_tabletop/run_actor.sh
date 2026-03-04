#!/usr/bin/env bash
set -euo pipefail

IP="${IP:-localhost}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PWD/checkpoints}"
EXP_NAME="g1_single_arm_tabletop"

mkdir -p "${CHECKPOINT_PATH}"

python ../../train_rlpd_g1.py   --exp_name="${EXP_NAME}"   --actor   --ip="${IP}"   --checkpoint_path="${CHECKPOINT_PATH}"
