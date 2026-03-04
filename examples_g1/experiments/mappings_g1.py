"""Experiment registry for the G1 HIL-SERL integration pack."""

from experiments.g1_single_arm_tabletop.config import TrainConfig as G1SingleArmTabletopTrainConfig

CONFIG_MAPPING = {
    "g1_single_arm_tabletop": G1SingleArmTabletopTrainConfig,
}
