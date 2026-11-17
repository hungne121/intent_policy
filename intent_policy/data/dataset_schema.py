"""Fixed LeRobot schema of the HRC datasets (QĐ 53/QĐ-DL §6). Identical for `hrc_observed` and `hrc_forecast`.
Change anything here => bump `schema_version` in configs/hrc/collection.yaml.

Definitions
- `action`: joint position targets of the 6 UR3e joints [rad] + gripper command [m] exactly as sent to the servo
  at that tick (the only action; there is no separate safety filter: holding for the person is an expert action).
- `observation.state`: robot qpos (6) + qvel (6) + gripper opening (1) at the tick, before the action.
- human arrays are WORLD coordinates of the 6 ManiCast joints, order JOINTS (l/r shoulder, elbow, wrist).
- `observation.human_forecast`: the EFFECTIVE forecast at the tick, 10 horizons x 6 joints x 3 (0.1 ... 1.0 s): the
  ManiCast output, or inside an N5 fault window the injected one the expert really received (59/QĐ-DL). Recorded in
  BOTH datasets for one schema. `observation.forecast_valid` = 1 when that forecast is valid (zeros otherwise).
- `observation.forecast_provided` (hrc-1.1, 59/QĐ-DL): 1 when the forecast was handed to the expert at this tick.
  Always 0 in the observed dataset: its forecast tensor is recorded but was NOT used by the expert.
- Raw ManiCast output, fault strength, natural forecast error and the human keypoints go to the episode sidecar
  `meta/hrc_sidecar/episode_<index>.npz` (diagnostics, never a policy input).
- No ground-truth human future, target label, case label or task id is a frame feature.
"""
from intent_policy.shared.sim_env import JOINTS

N_JOINTS = len(JOINTS)
HORIZONS = 10
HISTORY = 9


def features(cameras, image_hw):
    h, w = image_hw
    f = {f'observation.images.{c}': {'dtype': 'video', 'shape': (h, w, 3), 'names': ['height', 'width', 'channels']}
         for c in cameras}
    f.update({
        'observation.state': {'dtype': 'float32', 'shape': (13,),
                              'names': [f'q{i}' for i in range(6)] + [f'qd{i}' for i in range(6)] + ['gripper']},
        'action': {'dtype': 'float32', 'shape': (7,), 'names': [f'q{i}_target' for i in range(6)] + ['gripper_cmd']},
        'observation.tcp_pose': {'dtype': 'float32', 'shape': (7,), 'names': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']},
        'observation.tcp_reference': {'dtype': 'float32', 'shape': (3,), 'names': ['x', 'y', 'z']},
        'observation.tcp_reference_valid': {'dtype': 'float32', 'shape': (1,), 'names': ['valid']},
        'observation.human_pose': {'dtype': 'float32', 'shape': (N_JOINTS * 3,), 'names': [f'{j}_{a}' for j in JOINTS for a in 'xyz']},
        'observation.human_history': {'dtype': 'float32', 'shape': (HISTORY * N_JOINTS * 3,), 'names': None},
        'observation.human_history_valid': {'dtype': 'float32', 'shape': (1,), 'names': ['valid']},
        'observation.human_forecast': {'dtype': 'float32', 'shape': (HORIZONS * N_JOINTS * 3,), 'names': None},
        'observation.forecast_valid': {'dtype': 'float32', 'shape': (1,), 'names': ['valid']},
        'observation.forecast_age': {'dtype': 'float32', 'shape': (1,), 'names': ['seconds']},
        'observation.forecast_provided': {'dtype': 'float32', 'shape': (1,), 'names': ['provided']},
    })
    return f


EPISODE_METADATA = ('episode_id', 'task_name', 'task_variant', 'expert_mode', 'seed', 'split', 'success', 'failure_reason',
                    'duration', 'physics_valid', 'forecast_validity_summary', 'expert_version', 'manicast_version',
                    'controller_version', 'schema_version', 'group', 'forecast_source', 'dataset_version',
                    'fault_type', 'fault_start', 'fault_end', 'fault_parameters', 'variation')
