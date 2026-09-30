"""Offline, phase-resolved accuracy of the restricted head on held-out demonstration frames.

Frames of the validation episodes (every `--val-every`-th episode, as in training) are bucketed by
the information timeline: before the cue, cue -> evidence, the 1 s decision window after the
evidence point, and the rest. The benefit of intention information should concentrate in the
decision window. Also reports the mean probability assigned to the expert's action.

  ./run.sh -m scripts.offline_eval --checkpoint <ckpt>/pretrained_model --output outputs/eval/phase2/offline/oracle_s0
"""
import argparse
import json
from collections import defaultdict
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from intent_policy.policies.policy_agent import load_policy
from intent_policy.utils import resolve

WINDOW = 20      # frames (1 s) after the evidence point


def phase_of(m: dict, frame: int) -> str:
    ev = [e['frame'] for e in m.get('intention_evident', []) if e['kind'] in ('target_object', 'intrusion')]
    cue = int(round(m['human_cue_onset_t'] * 20)) if m.get('human_cue_onset_t') is not None else None
    if ev and ev[0] <= frame < ev[0] + WINDOW:
        return 'decision_window'
    if cue is not None and frame < cue:
        return 'before_cue'
    if cue is not None and ev and frame < ev[0]:
        return 'cue_to_evidence'
    if not ev and cue is None:
        return 'no_cue'
    return 'after_window' if ev and frame >= ev[0] + WINDOW else 'before_evidence'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--root', default='outputs/datasets/hri_phase2_oracle')
    p.add_argument('--repo-id')
    p.add_argument('--output', required=True)
    p.add_argument('--val-every', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    root = resolve(args.root)
    repo_id = args.repo_id or f'local/{root.name}'
    policy, pre, _ = load_policy(resolve(args.checkpoint), args.device)
    policy.eval()
    metas = {m['episode_index']: m for m in map(json.loads, (root / 'meta/hri_episodes.jsonl').read_text().splitlines())}
    val = [e for e in metas if e % args.val_every == args.val_every - 1]
    ds = LeRobotDataset(repo_id, root=root, episodes=val)
    keys = list(policy.config.input_features)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=4, shuffle=False)
    stats = defaultdict(lambda: dict(n=0, correct=0, p_label=0.0))
    for batch in loader:
        with torch.no_grad():
            logits = policy.predict_restricted_chunk(pre({k: batch[k].to(args.device) for k in keys}))[:, 0]
        probs = logits.softmax(-1).cpu()
        labels = batch['restricted_action'].reshape(-1).long()
        for i in range(len(labels)):
            m = metas[int(batch['episode_index'][i])]
            ph = phase_of(m, int(batch['frame_index'][i]))
            for key in (f"{m['scenario_id']}/{ph}", f'all/{ph}', f"{m['scenario_id']}/all", 'all/all'):
                s = stats[key]
                s['n'] += 1
                s['correct'] += int(probs[i].argmax() == labels[i])
                s['p_label'] += float(probs[i, labels[i]])
    result = {k: dict(n=v['n'], accuracy=v['correct'] / v['n'], mean_p_expert_action=v['p_label'] / v['n'])
              for k, v in sorted(stats.items())}
    out = resolve(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'offline.json').write_text(json.dumps(dict(checkpoint=args.checkpoint, intent_mask=policy.config.intent_mask,
                                                      use_intent=policy.config.use_intent, result=result), indent=2))
    lines = ['| scenario/phase | frames | step-0 accuracy | mean p(expert action) |', '|---|---|---|---|']
    lines += [f"| {k} | {v['n']} | {v['accuracy']:.3f} | {v['mean_p_expert_action']:.3f} |" for k, v in result.items()]
    (out / 'offline.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
