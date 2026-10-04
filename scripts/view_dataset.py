"""View recorded demonstrations (LeRobotDataset) in rerun.

Same content as LeRobot's `lerobot-dataset-viz` (camera streams, `action`, `observation.state`)
plus the HRI labels it does not show: the restricted-action id per frame, the current
expert/human stage, and the episode's ground-truth timeline from meta/hri_episodes.jsonl.

`--mp4` writes one mp4 per episode instead, cut from the dataset's own videos (no re-simulation; the evaluation
video layout of intent_policy/benchmark/video.py: `high` large, the other cameras beside it, the robot and human
stages on top, the protocol steps so far at the bottom), into `--save`, else outputs/videos/<date>/<time>_<dataset>.

  ./run.sh -m scripts.view_dataset --root outputs/datasets/intent_act_late_v5 --episode-index 0
  ./run.sh -m scripts.view_dataset --root outputs/datasets/intent_act_late_v5 --episode-index 0 5 --mp4
  ./run.sh -m scripts.view_dataset --root outputs/datasets/hri_phase1_preview --save outputs/viz/hri_phase1_preview --jpeg-quality 95
  ./run.sh -m rerun outputs/viz/hri_phase1_preview/*.rrd
(LeRobot's own viewer works on these datasets too: lerobot-dataset-viz --repo-id local/<name> --root <root> --episode-index 0)
"""
import argparse
import json
import rerun as rr
import rerun.blueprint as rrb
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import get_feature_names, to_hwc_uint8_numpy
from intent_policy.sim.restricted_action import RestrictedAction
from intent_policy.intent.representation import OBJECT_VOCAB, PREFIX, REGION_VOCAB
from intent_policy.benchmark.video import FPS, MAIN_CAMERA, VideoWriter, compose
from intent_policy.utils import resolve, run_dir

LABEL_TEXT = ', '.join(f'{a.value} {a.name}' for a in RestrictedAction)


def blueprint(ds: LeRobotDataset) -> rrb.Blueprint:
    cams = [rrb.Spatial2DView(origin=k, name=k.removeprefix('observation.images.')) for k in ds.meta.camera_keys]
    series = [rrb.TimeSeriesView(origin='action', name='action (joint targets)',
                                 overrides={'action': rr.SeriesLines(names=get_feature_names(ds, 'action'))}),
              rrb.TimeSeriesView(origin='state', name='observation.state',
                                 overrides={'state': rr.SeriesLines(names=get_feature_names(ds, 'observation.state'))}),
              rrb.TimeSeriesView(origin='restricted_action', name='restricted_action id')]
    if PREFIX + 'object_valid' in ds.meta.features:
        series.append(rrb.TimeSeriesView(origin='oracle', name='oracle (valid flags, future hand rel. 0.5 s)'))
    text = [rrb.TextDocumentView(origin='now', name='current frame'),
            rrb.TextLogView(origin='timeline', name='ground-truth timeline'),
            rrb.TextDocumentView(origin='episode', name='episode ground truth')]
    return rrb.Blueprint(rrb.Horizontal(rrb.Vertical(rrb.Horizontal(*cams), rrb.Horizontal(*series), row_shares=[3, 2]),
                                        rrb.Vertical(*text, row_shares=[1, 3, 3]), column_shares=[3, 1]))


GT_KEYS = ('selected_object', 'selected_target', 'target_object', 'target_region', 'human_selected_object')


def episode_markdown(m: dict) -> str:
    rows = '\n'.join(f"| {s['start']}-{s['end']} | {s['name']} | "
                     + ', '.join(f'{k} {v}' for k, v in s['actions'].items()) + ' |' for s in m['expert_stages'])
    proto = '\n'.join(f"- t={p['t']:.2f}s (frame {p['frame']}) **{p['step']}** by {p['by']}" for p in m['protocol_steps'])
    gt_start, gt_end = m['ground_truth_intention_at_start'], m.get('ground_truth_intention_at_end') or {}
    gt = '\n'.join(f'- {k}: {gt_start[k]}' + (f"  →  **{gt_end[k]}** (end)" if k in gt_end and gt_end[k] != gt_start[k] else '')
                   for k in GT_KEYS if k in gt_start)
    changes = m.get('intention_changes') or []
    change_md = '\n'.join(f"- t={c['t']:.2f}s (frame {c['frame']}) **{c['kind']}**: {c['previous']} → **{c['current']}**"
                          for c in changes) or '- none'
    skipped = (m.get('human_extras') or {}).get('intention_change_skipped_t')
    if skipped is not None:
        change_md += f'\n- change scheduled at t={skipped:.2f}s was skipped (robot already grasping)'
    header = f"**split:** {m['split']}  **expert trigger:** {(m.get('expert') or {}).get('trigger', 'cue_complete')}\n\n" \
        if 'split' in m else ''
    return (f"## {m['scenario_id']}  (episode {m['episode_index']}, seed {m['seed']})\n\n{header}"
            f"**task:** {m['task']}\n\n**frames:** {m['n_frames']}  **duration:** {m['duration_s']} s  "
            f"**min human-robot distance:** {m['min_human_robot_distance']} m\n\n"
            f"### Ground-truth intention (not a policy input)\n{gt}\n\n### Intention changes\n{change_md}\n\n"
            f"### Protocol steps\n{proto}\n\n"
            f"### Expert stages\n| frames | stage | restricted actions |\n|---|---|---|\n{rows}\n\n"
            f"### Restricted-action ids\n{LABEL_TEXT}\n")


def log_episode(ds: LeRobotDataset, m: dict, rec: rr.RecordingStream, jpeg_quality: int | None = None) -> None:
    starts = {}
    for kind, segs in (('expert', m['expert_stages']), ('human', m['human_stages'])):
        for s in segs:
            starts.setdefault(s['start'], []).append((kind, s['name']))
    for p in m['protocol_steps']:
        starts.setdefault(p['frame'], []).append(('protocol', f"{p['step']} (by {p['by']})"))
    for c in m.get('intention_changes') or []:
        starts.setdefault(c['frame'], []).append(('change', f"INTENTION CHANGE {c['kind']}: {c['previous']} -> {c['current']}"))
    for e in m.get('intention_evident') or []:
        what = e.get('object') or e.get('region') or 'intrusion'
        starts.setdefault(e['frame'], []).append(('cue', f"intention evident ({e['kind']}): {what}"))
    for c in m.get('expert_commits') or []:
        starts.setdefault(c['frame'], []).append(('protocol', f"expert commits to {c['target']}"))
    if m.get('human_cue_onset_t') is not None:
        starts.setdefault(int(round(m['human_cue_onset_t'] * 20)), []).append(('cue', 'human cue onset'))
    stage_kinds = [('expert', m['expert_stages']), ('human', m['human_stages'])]
    if m.get('human_target_stages'):
        stage_kinds.append(('target', m['human_target_stages']))
    stage_at = {kind: [s['name'] for s in segs for _ in range(s['end'] - s['start'] + 1)] for kind, segs in stage_kinds}
    rec.set_time('frame_index', sequence=0)
    rec.log('episode', rr.TextDocument(episode_markdown(m), media_type=rr.MediaType.MARKDOWN), static=True)
    colors = {'expert': [80, 160, 255], 'human': [255, 170, 60], 'protocol': [90, 220, 120], 'action': [170, 170, 170],
              'change': [255, 60, 60], 'cue': [255, 220, 60]}
    loader = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=0)
    i, previous = 0, None
    for batch in loader:
        for j in range(len(batch['index'])):
            t = float(batch['timestamp'][j])
            rec.set_time('frame_index', sequence=i)
            rec.set_time('time', duration=t)
            for key in ds.meta.camera_keys:
                img = rr.Image(to_hwc_uint8_numpy(batch[key][j]))
                rec.log(key, img.compress(jpeg_quality) if jpeg_quality else img)
            rec.log('action', rr.Scalars(batch['action'][j].numpy()))
            rec.log('state', rr.Scalars(batch['observation.state'][j].numpy()))
            a = RestrictedAction(int(batch['restricted_action'][j]))
            rec.log('restricted_action', rr.Scalars(float(a.value)))
            oracle_text = ''
            if PREFIX + 'object_valid' in batch:
                ov, rv = float(batch[PREFIX + 'object_valid'][j]), float(batch[PREFIX + 'region_valid'][j])
                rec.log('oracle/object_valid', rr.Scalars(ov))
                rec.log('oracle/region_valid', rr.Scalars(rv))
                rec.log('oracle/future_hand_rel_0p5', rr.Scalars(batch[PREFIX + 'future_hand_rel_0p5'][j].numpy()))
                obj = OBJECT_VOCAB[int(batch[PREFIX + 'object_id'][j].argmax())] if ov > 0.5 else '—'
                reg = REGION_VOCAB[int(batch[PREFIX + 'region_id'][j].argmax())] if rv > 0.5 else '—'
                oracle_text = f'\n\noracle (policy input in the Oracle condition): object **{obj}**, region **{reg}**'
            for kind, text in starts.get(i, []):
                level = 'WARN' if kind == 'change' else 'INFO'
                rec.log('timeline', rr.TextLog(f'{kind}: {text}', level=level, color=colors[kind]))
            if a != previous:
                rec.log('timeline', rr.TextLog(f'action: {a.name}', level='DEBUG', color=colors['action']))
                previous = a
            target = f"\n\nhuman's current target: **{stage_at['target'][i]}**" if 'target' in stage_at else ''
            rec.log('now', rr.TextDocument(f"t = {t:.2f} s   frame {i}\n\nrestricted action: **{a.name}** ({a.value})\n\n"
                                           f"expert stage: {stage_at['expert'][i]}\n\nhuman stage: {stage_at['human'][i]}{target}{oracle_text}",
                                           media_type=rr.MediaType.MARKDOWN))
            i += 1


def stages_per_frame(m: dict) -> dict[str, list[str]]:
    kinds = [('expert', m['expert_stages']), ('human', m['human_stages'])]
    return {kind: [s['name'] for s in segs for _ in range(s['end'] - s['start'] + 1)] for kind, segs in kinds}


def export_mp4(ds: LeRobotDataset, m: dict, path) -> None:
    """The episode as an mp4 from the dataset's own frames, in the evaluation video layout."""
    marks = sorted([(p['frame'], f"{p['t']:5.2f}s {p['step']} (by {p['by']})") for p in m['protocol_steps']]
                   + [(c['frame'], f"{c['t']:5.2f}s INTENTION CHANGE {c['kind']}: {c['previous']} -> {c['current']}")
                      for c in m.get('intention_changes') or []])
    stage = stages_per_frame(m)
    main_key = f'observation.images.{MAIN_CAMERA}'
    side_keys = [k for k in ds.meta.camera_keys if k != main_key]
    names = [k.removeprefix('observation.images.') for k in side_keys]
    writer, frame, i = VideoWriter(path), None, 0
    for batch in torch.utils.data.DataLoader(ds, batch_size=32, num_workers=0):
        for j in range(len(batch['index'])):
            at = lambda kind: stage[kind][min(i, len(stage[kind]) - 1)] if stage[kind] else '-'
            header = (f"ep {m['episode_index']} seed {m['seed']}  t={float(batch['timestamp'][j]):5.2f}s  "
                      f"robot: {at('expert')}  human: {at('human')}")
            frame = (to_hwc_uint8_numpy(batch[main_key][j]), [to_hwc_uint8_numpy(batch[k][j]) for k in side_keys],
                     header, [text for f, text in marks if f <= i])
            writer.add(compose(*frame, side_names=names))
            i += 1
    for _ in range(FPS):
        writer.add(compose(*frame, 'SUCCESS (stored demonstration)', side_names=names))
    writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default='outputs/datasets/hri_phase1_preview')
    p.add_argument('--repo-id', help='default: local/<root folder name>')
    p.add_argument('--episode-index', type=int, nargs='*', help='default: all episodes')
    p.add_argument('--save', help='write one .rrd (or with --mp4 one .mp4) per episode into this directory instead of opening the viewer')
    p.add_argument('--mp4', action='store_true', help='write mp4 videos (default folder: outputs/videos/<date>/<time>_<dataset>)')
    p.add_argument('--jpeg-quality', type=int, help='JPEG-compress logged frames (smaller .rrd files), e.g. 95')
    args = p.parse_args()
    root = resolve(args.root)
    repo_id = args.repo_id or f'local/{root.name}'
    metas = [json.loads(line) for line in (root / 'meta/hri_episodes.jsonl').read_text().splitlines()]
    episodes = args.episode_index if args.episode_index else [m['episode_index'] for m in metas]
    if args.mp4:
        out = run_dir('videos', f'dataset_{root.name}', args.save)
        out.mkdir(parents=True, exist_ok=True)
        for ep in episodes:
            m = metas[ep]
            path = out / f"episode_{ep:03d}_{m.get('list_task', m['scenario_id'])}_seed{m['seed']}.mp4"
            export_mp4(LeRobotDataset(repo_id, root=root, episodes=[ep]), m, path)
            print(f"episode {ep} ({m['scenario_id']}, {m['n_frames']} frames) -> {path}", flush=True)
        return
    for ep in episodes:
        m = metas[ep]
        ds = LeRobotDataset(repo_id, root=root, episodes=[ep])
        rec = rr.RecordingStream(repo_id, recording_id=f"episode_{ep:03d}_{m['scenario_id']}")
        if args.save:
            out = resolve(args.save)
            out.mkdir(parents=True, exist_ok=True)
            split = f"_{m['split']}" if m.get('split') not in (None, 'default', 'nominal') else ''
            path = out / f"episode_{ep:03d}_{m['scenario_id']}{split}_seed{m['seed']}.rrd"
            rec.save(path, default_blueprint=blueprint(ds))
        else:
            rec.spawn(default_blueprint=blueprint(ds))
        log_episode(ds, m, rec, args.jpeg_quality)
        rec.flush()
        rec.disconnect()
        print(f"episode {ep} ({m['scenario_id']}, seed {m['seed']}, {m['n_frames']} frames)"
              + (f' -> {path}' if args.save else ' -> viewer'), flush=True)


if __name__ == '__main__':
    main()
