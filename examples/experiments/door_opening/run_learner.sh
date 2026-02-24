export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=door_opening \
    --checkpoint_path=../../experiments/door_opening/debug \
    --demo_path=../../../demo_data/door_opening_20_demos_2026-02-23_15-21-17.pkl \
    --learner \