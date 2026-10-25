"""Open-loop check (docs/sim/PIPELINE_v6.md §6): recorded training observations -> PolicyAgent.act (the evaluation path) vs the recorded action.
  ./run.sh scripts/check_openloop.py <checkpoint>/pretrained_model outputs/datasets/v6_early 0,46,96
"""
import json, sys
from types import SimpleNamespace
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from intent_policy.policies.policy_agent import PolicyAgent

ckpt, root, eps = sys.argv[1], sys.argv[2], [int(e) for e in sys.argv[3].split(',')]
meta = {json.loads(l)['episode_index']: json.loads(l) for l in open(f'{root}/meta/hri_episodes.jsonl')}
agent = PolicyAgent(ckpt, 'cuda')
keys = sorted(agent.expected_keys)
print('policy inputs', keys)
ds = LeRobotDataset('local/x', root=root, episodes=eps)
for e in eps:
    m = meta[e]
    sc = SimpleNamespace(cfg=SimpleNamespace(id=m['scenario_id'], task_code=m['task_code']), variation={'spec': m['spec']})
    agent.policy.reset(); agent.scenario = sc; agent.step_index = 0
    idx = [i for i in range(len(ds)) if int(ds.hf_dataset[i]['episode_index']) == e]
    err, step = [], []
    for k, i in enumerate(idx[:200]):
        fr = ds[i]
        obs = {}
        for key in keys:
            v = fr[key].numpy()
            if key.startswith('observation.images.'):
                v = np.round(np.transpose(v, (1, 2, 0)) * 255).astype(np.uint8)       # dataset CHW [0,1] -> scenario HWC uint8
            obs[key] = v
        a = agent.act(obs).joint_target
        rec = fr['action'].numpy()
        err.append(np.abs(a - rec)); step.append(np.abs(rec - fr['observation.state'].numpy()[:7]))
    err, step = np.array(err), np.array(step)
    print(f"episode {e} ({m['scenario_id']}): |policy - recorded action| mean {err[:, :6].mean():.4f} rad (joints), "
          f"gripper {err[:, 6].mean():.4f}; recorded |action - state| mean {step[:, :6].mean():.4f} rad")
