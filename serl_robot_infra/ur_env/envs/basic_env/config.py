from ur_env.envs.ur5_env import DefaultEnvConfig
import numpy as np
from scipy.spatial.transform import Rotation as R

# class UR5PickingConfig(DefaultEnvConfig):
#     # RESET_Q = np.array([[1.34231, -1.24585, 1.94961, -2.27267, -1.56428, -0.22641]])   # original one
#     # RESET_Q = np.array([[1.3463, -1.3584,  1.9014, -2.1243, -1.5758, -0.2312]])
#     # RESET_Q = np.array([
#     #     [-36.26, -82.24, 128.58, -136.35, -89.82, -46.5],
#     # ])
#     RESET_Q = np.array([
#         [272.0, -82.24, -105.75, -90.70, 90.27, 171.14],
#     ])
#     RESET_Q = np.deg2rad(RESET_Q)
#     RANDOM_RESET = False
#     RANDOM_XY_RANGE = (0.06,)
#     RANDOM_ROT_RANGE = (0.0,)
#     # ABS_POSE_LIMIT_HIGH = np.array([0.14, -0.4, 0.2, 3.2, 0.1, 3.2])            # TODO euler rotations suck :/
#     # ABS_POSE_LIMIT_LOW = np.array([-0.3, -0.7, -0.006, 3.0, -0.1, -3.2])
#     ABS_POSE_LIMIT_HIGH = np.array([-0.3, 0.5, 0.3, 0.05, 0.05, 0.2])
#     ABS_POSE_LIMIT_LOW = np.array([-0.6, 0., -0.1, -0.05, -0.05, -0.2])
#     ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

#     ROBOT_IP: str = "192.168.56.2"
#     CONTROLLER_HZ = 100
#     GRIPPER_TIMEOUT = 2000  # in milliseconds
#     ERROR_DELTA: float = 0.05
#     FORCEMODE_DAMPING: float = 0.0  # faster
#     FORCEMODE_TASK_FRAME = np.zeros(6)
#     FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
#     FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])
#     GRIPPER_USB_PORT = "/dev/ttyUSB0"
    # GRIPPER_SLAVE_ID = 9
    
class UR5PickingConfig(DefaultEnvConfig):
    # RESET_Q = np.array([[1.34231, -1.24585, 1.94961, -2.27267, -1.56428, -0.22641]])   # original one
    # RESET_Q = np.array([[1.3463, -1.3584,  1.9014, -2.1243, -1.5758, -0.2312]])
    RESET_Q = np.array([
        [272.26, -69.24, -112.58, -90.35, 89.82, 180.5],
        # [-5, -78.62, 122.84, -134.22, -89.81, -13.03],
        # [10, -75.62, 122.84, -134.22, -89.81, -13.03],
    ])
    RESET_Q = np.deg2rad(RESET_Q)
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.06,)
    RANDOM_ROT_RANGE = (0.0,)
    # ABS_POSE_LIMIT_HIGH = np.array([0.14, -0.4, 0.2, 3.2, 0.1, 3.2])            # TODO euler rotations suck :/
    # ABS_POSE_LIMIT_LOW = np.array([-0.3, -0.7, -0.006, 3.0, -0.1, -3.2])
    ABS_POSE_LIMIT_HIGH = np.array([-0.3, 0.2, 0.4, 0.05, 0.05, 0.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.65, -0.1, 0.08, -0.05, -0.05, -0.2])
    ACTION_SCALE = np.array([0.01, 0.05, 1.], dtype=np.float32)

    ROBOT_IP: str = "192.168.56.2"
    CONTROLLER_HZ = 10
    GRIPPER_TIMEOUT = 5000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.5  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])
    GRIPPER_USB_PORT = "/dev/ttyUSB0"
    GRIPPER_SLAVE_ID = 9
    GOAL_POSE = np.array([0.024, 0.02, 0.21, 0., 0., 3.14])     #box_1
    GOAL_POSE = np.array([0.03, 0.03, 0.21, 0., 0., 3.14])      #box_340
    GOAL_POSE = np.array([0.035, 0.03, 0.21, 0., 0., 3.14])     #box_330
    # GOAL_POSITION = np.array([-0.38, -0.01, -0.04])    #box_1 next to box_3
    # GOAL_POSITION = np.array([-0.38, -0.12, -0.04])    #box_5 next to box_1
    
    ROTATION_GENERALIZATION = R.from_euler("xyz", np.array([0, 0, 0])).as_matrix() # rotation applied to the box to bring it back to the training orientation
    BOX_ERROR = np.array([0.0, 0.0, 0.0])
    POSE_ESTIMATION = False
    POSE_ESTIMATION_IP = "ws://localhost:7777"
    WF_rot = np.array([[-1,  0,  0],
                        [ 0,  0, 1],
                        [ 0, 1,  0]], dtype=np.float32)
    LOW_PASS_FILTER = 0
    SUCCESS_COUNT = 0


# class UR5CornerConfigV1(DefaultEnvConfig):
#     """
#     Configuration for the first set of SAC training's (only 1 box)
#     """
#     # RESET_Q = np.array([[1.34231, -1.24585, 1.94961, -2.27267, -1.56428, -0.22641]])   # original one
#     RESET_Q = np.array([[1.3463, -1.3584, 1.9014, -2.1243, -1.5758, -0.2312]])
#     # TODO make multiple reset Q positions, one for each box to train on (randomize it)
#     RANDOM_RESET = False
#     RANDOM_XY_RANGE = (0.06,)
#     RANDOM_ROT_RANGE = (0.0,)
#     ABS_POSE_LIMIT_HIGH = np.array([0.14, -0.4, 0.2, 3.2, 0.1, 3.2])
#     ABS_POSE_LIMIT_LOW = np.array([-0.3, -0.7, -0.006, 3.0, -0.1, -3.2])
#     ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

#     ROBOT_IP: str = "172.22.22.2"
#     CONTROLLER_HZ = 100
#     GRIPPER_TIMEOUT = 2000  # in milliseconds
#     ERROR_DELTA: float = 0.05
#     FORCEMODE_DAMPING: float = 0.0  # faster
#     FORCEMODE_TASK_FRAME = np.zeros(6)
#     FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
#     FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])
