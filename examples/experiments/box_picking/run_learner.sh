export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=ur5e_aruco_pick \
    --checkpoint_path=../../experiments/ur5e_aruco_pick/debug \
    --demo_path=../../../demo_data/ur5e_aruco_pick_20_demos_2026-02-03_19-57-21.pkl \
    --learner \