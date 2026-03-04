# G1 single-arm tabletop template

This folder is a **drop-in HIL-SERL experiment** for running arm-only training on a Unitree G1.

## Before first run

1. Copy `task_config.example.yaml` to `task_config.yaml`.
2. Edit camera IDs, server URL, arm home, reset pose, and safety box.
3. Start the robot server:

   ```bash
   cd serl_robot_infra
   bash robot_servers/launch_g1_arm_server.sh      --urdf_path /absolute/path/to/g1_29dof.urdf      --active_arm left      --flask_url 127.0.0.1      --flask_port 5001
   ```

4. Calibrate current pose:

   ```bash
   cd examples/experiments/g1_single_arm_tabletop
   python ../../calibrate_g1_task.py --server_url http://127.0.0.1:5001
   ```

## Reward classifier data

```bash
cd examples/experiments/g1_single_arm_tabletop
python ../../record_success_fail_g1.py --exp_name g1_single_arm_tabletop --successes_needed 200
python ../../train_reward_classifier_g1.py --exp_name g1_single_arm_tabletop
```

## Demonstrations

```bash
cd examples/experiments/g1_single_arm_tabletop
python ../../record_demos_g1.py --exp_name g1_single_arm_tabletop --successes_needed 20
```

## RLPD actor / learner

Start the learner on the GPU machine:

```bash
cd examples/experiments/g1_single_arm_tabletop
bash run_learner.sh
```

Then start the actor on the robot-side / teleop machine:

```bash
cd examples/experiments/g1_single_arm_tabletop
bash run_actor.sh
```
