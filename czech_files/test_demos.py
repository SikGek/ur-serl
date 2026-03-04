import pickle

with open('demo_data/ur5e_aruco_pick_20_demos_2026-02-03_19-57-21.pkl', 'rb') as f:
	data = pickle.load(f)
print(data)
