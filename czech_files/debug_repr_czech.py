#!/usr/bin/env python3
"""
debug_repr_check.py

Usage:
  # 1) Check demo pkl only
  python debug_repr_check.py --demo_pkl /path/to/demos.pkl

  # 2) Live check (WARNING: will call env.reset() and env.step(); default steps use ZERO action)
  python debug_repr_check.py --exp_name ur5e_aruco_pick --live --n_steps 10 --action_mode zero

  # Optional: small random actions (still bounded) to see deltas
  python debug_repr_check.py --exp_name ur5e_aruco_pick --live --n_steps 20 --action_mode random_small
"""

import argparse
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

def mrp_from_quat_shadow(q_xyzw: np.ndarray) -> np.ndarray:
    """
    Robust quat -> MRP conversion with shadow-set handling.
    SciPy uses (x,y,z,w).
    """
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.zeros(3, dtype=np.float64)
    q = q / n

    qv = q[:3]
    qw = q[3]

    # If qw is near -1, 1+qw ~ 0 => huge MRPs. Flip sign (same rotation) to prefer qw >= 0.
    if qw < 0.0:
        q = -q
        qv = q[:3]
        qw = q[3]

    denom = 1.0 + qw
    if denom < 1e-9:
        # Extremely close to singular; return something finite
        return qv / 1e-9

    sigma = qv / denom

    # Shadow set: keep ||sigma|| <= 1
    n2 = float(np.dot(sigma, sigma))
    if n2 > 1.0:
        sigma = -sigma / n2

    return sigma.astype(np.float64)


def describe_transition_dict(tr):
    keys = list(tr.keys())
    print("Transition keys:", keys)
    for k in ["observations", "next_observations"]:
        if k in tr:
            obs = tr[k]
            if isinstance(obs, dict):
                print(f"  {k} keys:", list(obs.keys()))
                for ok, ov in obs.items():
                    if isinstance(ov, np.ndarray):
                        print(f"    {k}[{ok}]: shape={ov.shape} dtype={ov.dtype}")
                    else:
                        print(f"    {k}[{ok}]: type={type(ov)}")
            else:
                print(f"  {k}: type={type(obs)}")


def check_demo_pkl(pkl_path: str):
    print(f"\n=== Loading demo pkl: {pkl_path} ===")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    assert isinstance(data, (list, tuple)), f"Expected list of transitions, got {type(data)}"
    assert len(data) > 0, "Empty demo file"

    print(f"Loaded transitions: {len(data)}")
    describe_transition_dict(data[0])

    # Basic required keys
    required = {"observations", "actions", "next_observations", "rewards", "masks", "dones"}
    for i, tr in enumerate(data[:10]):
        missing = required - set(tr.keys())
        assert not missing, f"Transition {i} missing keys: {missing}"

    # Stack stats
    obs0 = data[0]["observations"]
    assert isinstance(obs0, dict) and "state" in obs0, "Expected observations to contain 'state'"
    state0 = obs0["state"]
    assert isinstance(state0, np.ndarray), "Expected observations['state'] to be a numpy array"

    state_dim = int(state0.reshape(-1).shape[0])
    print(f"\nState vector dim = {state_dim}")

    # Heuristic slice layout (matches your typical proprio_keys):
    # tcp_pose(6), tcp_vel(6), tcp_force(3), tcp_torque(3), gripper_pose(1), gripper_object(1) => 20
    if state_dim == 20:
        slices = {
            "tcp_pose_xyz": slice(0, 3),
            "tcp_pose_ori": slice(3, 6),
            "tcp_vel_lin":  slice(6, 9),
            "tcp_vel_ang":  slice(9, 12),
            "tcp_force":    slice(12, 15),
            "tcp_torque":   slice(15, 18),
            "gripper_pose": slice(18, 19),
            "gripper_obj":  slice(19, 20),
        }
    else:
        slices = {"state": slice(0, state_dim)}

    states = np.stack([np.asarray(tr["observations"]["state"], dtype=np.float32).reshape(-1) for tr in data], axis=0)
    next_states = np.stack([np.asarray(tr["next_observations"]["state"], dtype=np.float32).reshape(-1) for tr in data], axis=0)
    actions = np.stack([np.asarray(tr["actions"], dtype=np.float32).reshape(-1) for tr in data], axis=0)
    rewards = np.asarray([tr["rewards"] for tr in data], dtype=np.float32).reshape(-1)
    dones = np.asarray([tr["dones"] for tr in data], dtype=bool).reshape(-1)
    masks = np.asarray([tr["masks"] for tr in data], dtype=np.float32).reshape(-1)

    # Consistency checks
    bad_mask = np.where((dones & (masks != 0.0)) | (~dones & (masks != 1.0)))[0]
    print(f"\nMask consistency: {'OK' if len(bad_mask)==0 else 'BAD'} (bad={len(bad_mask)})")

    # Terminal reward pattern
    unique_rewards = np.unique(rewards)
    print("Unique rewards:", unique_rewards.tolist())
    print("Num terminal steps (dones=True):", int(dones.sum()))
    if dones.sum() > 0:
        print("Rewards on terminal steps:", np.unique(rewards[dones]).tolist())

    # Continuity check: next_obs(i) == obs(i+1) within episode
    cont_err = []
    for i in range(len(data) - 1):
        if dones[i]:
            continue
        cont_err.append(float(np.max(np.abs(next_states[i] - states[i+1]))))
    cont_err = np.asarray(cont_err, dtype=np.float32)
    print(f"State continuity max|next-obs_next|: {float(cont_err.max() if cont_err.size else 0.0)}")

    # Slice stats
    def stat(x):
        return dict(min=float(np.min(x)), max=float(np.max(x)), mean=float(np.mean(x)), std=float(np.std(x)))

    print("\nState slice statistics:")
    for name, sl in slices.items():
        s = states[:, sl]
        print(f"  {name:>14}: {stat(s)}")

    print("\nAction statistics:")
    print("  shape:", actions.shape, "min:", float(actions.min()), "max:", float(actions.max()))
    if actions.shape[1] >= 6:
        # crude notion of “how aggressive” demos are
        norms = np.linalg.norm(actions[:, :6], axis=1)
        print("  ||a[:6]|| percentiles:", np.percentile(norms, [0, 10, 50, 90, 99, 100]).tolist())

    print("\n✅ Demo pkl looks structurally valid (keys, masks/dones, continuity).")
    print("Next step: compare these stats to LIVE env output using --live.")


def extract_state_vector(obs) -> np.ndarray:
    """
    Handles typical SERLObsWrapper outputs:
      obs is dict with obs['state'] as a flat vector
    """
    assert isinstance(obs, dict) and "state" in obs, f"obs format unexpected: keys={list(obs.keys()) if isinstance(obs, dict) else type(obs)}"
    s = obs["state"]
    if isinstance(s, np.ndarray):
        return s.astype(np.float32).reshape(-1)
    # If your wrapper returns dict-of-proprio, flatten it deterministically:
    if isinstance(s, dict):
        flat = []
        for k in sorted(s.keys()):
            flat.append(np.asarray(s[k], dtype=np.float32).reshape(-1))
        return np.concatenate(flat, axis=0)
    raise TypeError(f"Unsupported obs['state'] type: {type(s)}")


def live_check(exp_name: str, n_steps: int, action_mode: str):
    print("\n=== LIVE CHECK MODE ===")
    print("This will construct your env via CONFIG_MAPPING and run reset/step.")
    print("Default action_mode=zero is safe (holds pose).")

    # Import here so the demo-only mode doesn’t require env deps.
    from experiments.mappings import CONFIG_MAPPING

    cfg = CONFIG_MAPPING[exp_name]()
    env = cfg.get_environment(fake_env=False, save_video=False, classifier=True)

    obs, info = env.reset()
    print("\nAfter reset():")
    print("  obs keys:", list(obs.keys()))
    s = extract_state_vector(obs)
    print("  state vec shape:", s.shape, "dtype:", s.dtype)
    print("  state[:12]:", s[:12])

    # Raw robot pose (xyz+quat) from base env
    raw = np.asarray(env.unwrapped.curr_pos, dtype=np.float64).reshape(-1)
    if raw.shape[0] == 7:
        xyz = raw[:3]
        quat = raw[3:]
        rotvec = R.from_quat(quat).as_rotvec()
        eul = R.from_quat(quat).as_euler("xyz", degrees=False)
        mrp = mrp_from_quat_shadow(quat)

        print("\n  RAW curr_pos (xyz+quat):", raw)
        print("  RAW rotvec:", rotvec, " | angle(deg)=", float(np.linalg.norm(rotvec) * 180 / np.pi))
        print("  RAW euler xyz:", eul)
        print("  RAW mrp (shadow):", mrp)

    # Action scales for physical interpretation
    pos_scale = float(env.unwrapped.action_scale[0])
    rot_scale = float(env.unwrapped.action_scale[1]) / 4.0

    # Safe action generator
    rng = np.random.default_rng(0)
    prev_filt = None

    for t in range(n_steps):
        if action_mode == "zero":
            a = np.zeros((7,), dtype=np.float32)
        elif action_mode == "random_small":
            a = rng.normal(size=(7,)).astype(np.float32)
            a[:6] *= 0.15  # small-ish
            a[6] = 0.0
            a = np.clip(a, -1.0, 1.0)
        else:
            raise ValueError("action_mode must be one of: zero, random_small")

        # Physical meaning if env interprets rot as MRP increment
        dp_m = a[:3] * pos_scale
        dR = R.from_mrp(a[3:6] * rot_scale)
        dtheta_deg = float(np.linalg.norm(dR.as_rotvec()) * 180 / np.pi)

        obs2, r, done, trunc, inf = env.step(a)
        s2 = extract_state_vector(obs2)

        print(f"\n[t={t:03d}] reward={r} done={done} trunc={trunc}")
        print("  action:", a)
        print(f"  Δpos(m): {dp_m}   Δrot(deg): {dtheta_deg:.3f}")
        print("  state[:12]:", s2[:12])

        # If the spacemouse wrapper intervened, it should show up here:
        if isinstance(inf, dict) and "intervene_action" in inf:
            ia = np.asarray(inf["intervene_action"], dtype=np.float32)
            print("  **HUMAN INTERVENTION** intervene_action:", ia)

        if done or trunc:
            obs2, info = env.reset()

    env.close()
    print("\n✅ Live check completed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_pkl", type=str, default=None)
    ap.add_argument("--exp_name", type=str, default=None)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--action_mode", type=str, default="zero", choices=["zero", "random_small"])
    args = ap.parse_args()

    if args.demo_pkl:
        check_demo_pkl(args.demo_pkl)

    if args.live:
        assert args.exp_name is not None, "--exp_name required for --live"
        live_check(args.exp_name, args.n_steps, args.action_mode)


if __name__ == "__main__":
    main()