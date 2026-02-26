export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python ../../train_rlpd.py "$@" \
    --exp_name=door_gripping \
    --checkpoint_path=../../experiments/door_gripping/gripping \
    --eval_checkpoint_step=212000\
    --eval_n_trajs=5\
    --actor \