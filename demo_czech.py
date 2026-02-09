import pickle as pkl
import numpy as np

DEMO_PATH = "../../../demo_data/ur5e_aruco_pick_20_demos_2026-02-06_17-13-23.pkl"

transitions = pkl.load(open(DEMO_PATH, "rb"))
rews = np.array([t["rewards"] for t in transitions], dtype=np.float32)
print("demo reward stats:", rews.min(), rews.max(), rews.mean(), "num>0:", (rews>0).sum())
dones = np.array([t["dones"] for t in transitions], dtype=np.bool_)
print("demo done count:", dones.sum(), "len:", len(dones))
