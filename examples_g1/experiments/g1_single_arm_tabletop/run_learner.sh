#!/usr/bin/env bash
set -euo pipefail

IP="${IP:-0.0.0.0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PWD/checkpoints}"
EXP_NAME="g1_single_arm_tabletop"

mkdir -p "${CHECKPOINT_PATH}"

if [[ -z "${DEMO_PATH:-}" ]]; then
  latest_demo="$(ls -1t "$PWD"/demo_data/${EXP_NAME}_*_demos_*.pkl 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_demo}" ]]; then
    echo "No demo file found under $PWD/demo_data. Set DEMO_PATH explicitly or collect demos first." >&2
    exit 1
  fi
  DEMO_PATH="${latest_demo}"
fi

python ../../train_rlpd_g1.py   --exp_name="${EXP_NAME}"   --learner   --ip="${IP}"   --checkpoint_path="${CHECKPOINT_PATH}"   --demo_path="${DEMO_PATH}"
