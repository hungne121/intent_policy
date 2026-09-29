"""Pre/post-processors: exactly LeRobot ACT's (rename -> batch -> device -> normalise)."""
from lerobot.policies.act.processor_act import make_act_pre_post_processors


def make_hri_act_pre_post_processors(config, dataset_stats=None):
    return make_act_pre_post_processors(config, dataset_stats)
