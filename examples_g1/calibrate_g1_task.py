"""Calibrate a G1 HIL-SERL task YAML file from the live arm state.

Typical usage:

    cd examples/experiments/g1_single_arm_tabletop
    python ../../calibrate_g1_task.py         --server_url http://127.0.0.1:5001         --task_config ./task_config.yaml         --set-reset-from-current         --set-target-from-current

This script is deliberately conservative: it only edits fields you explicitly request.
It is useful when you want to lock in a repeatable reset pose, set the success target
pose from the current end-effector pose, or snapshot the current arm joint vector as the
joint-space reset seed used by the IK-based controller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import requests
import yaml


def _post_json(server_url: str, route: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    resp = requests.post(server_url.rstrip('/') + '/' + route, json=payload, timeout=5.0)
    resp.raise_for_status()
    if not resp.content:
        return {}
    return resp.json()


def _round_list(arr, ndigits: int = 6) -> List[float]:
    return [round(float(x), ndigits) for x in np.asarray(arr).reshape(-1)]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server_url', required=True, help='Flask G1 server base URL, e.g. http://127.0.0.1:5001')
    parser.add_argument('--task_config', default='task_config.yaml', help='Path to the experiment-local YAML file to inspect/update.')
    parser.add_argument('--workspace_margin_xyz', type=float, default=0.12, help='Half-width margin used when writing abs_pose_limit_{low,high}.')
    parser.add_argument('--workspace_margin_rpy', type=float, default=0.40, help='Half-width margin used when writing rotational bounds around the current pose.')
    parser.add_argument('--set-reset-from-current', action='store_true', help='Write reset_pose from the current TCP pose (Euler).')
    parser.add_argument('--set-target-from-current', action='store_true', help='Write target_pose from the current TCP pose (Euler).')
    parser.add_argument('--set-reset-q-from-current', action='store_true', help='Write reset_q_arm from the current active-arm joint state.')
    parser.add_argument('--set-workspace-from-current', action='store_true', help='Write a conservative Cartesian safety box centered at the current TCP pose.')
    parser.add_argument('--print-only', action='store_true', help='Do not modify the YAML file; only print the sampled values.')
    args = parser.parse_args()

    task_path = Path(args.task_config)
    cfg = _load_yaml(task_path)

    state = _post_json(args.server_url, 'getstate')
    pose_euler = _post_json(args.server_url, 'getpos_euler')['pose']

    report = {
        'pose_quat': _round_list(state['pose']),
        'pose_euler': _round_list(pose_euler),
        'active_q': _round_list(state['active_q']),
        'active_dq': _round_list(state['active_dq']),
        'force': _round_list(state['force']),
        'torque': _round_list(state['torque']),
    }
    print(json.dumps(report, indent=2))

    if args.set_reset_from_current:
        cfg['reset_pose'] = _round_list(pose_euler)
    if args.set_target_from_current:
        cfg['target_pose'] = _round_list(pose_euler)
    if args.set_reset_q_from_current:
        cfg['reset_q_arm'] = _round_list(state['active_q'])
    if args.set_workspace_from_current:
        pose_euler_np = np.asarray(pose_euler, dtype=np.float64)
        low = pose_euler_np.copy()
        high = pose_euler_np.copy()
        low[:3] -= args.workspace_margin_xyz
        high[:3] += args.workspace_margin_xyz
        low[3:] -= args.workspace_margin_rpy
        high[3:] += args.workspace_margin_rpy
        cfg['abs_pose_limit_low'] = _round_list(low)
        cfg['abs_pose_limit_high'] = _round_list(high)

    if args.print_only:
        return

    if any([
        args.set_reset_from_current,
        args.set_target_from_current,
        args.set_reset_q_from_current,
        args.set_workspace_from_current,
    ]):
        task_path.parent.mkdir(parents=True, exist_ok=True)
        _save_yaml(task_path, cfg)
        print(f'Updated {task_path}')
    else:
        print('Nothing was written. Pass one or more --set-* flags to update the YAML.')


if __name__ == '__main__':
    main()
