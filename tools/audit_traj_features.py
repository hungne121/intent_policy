"""Phase 1.2-1.3 helper: per-episode kinematic features from trajectory npz (all 1290 episodes) -> outputs/diag/phase1_1120/features.jsonl"""
import json, glob, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DIRS = ['outputs/eval_full_pod/eval_full_2026-11-19', 'outputs/eval_full_pod2/eval_full_b']
out = open(ROOT / 'outputs/diag/phase1_1120/features.jsonl', 'w')
for d in DIRS:
    for f in sorted(glob.glob(str(ROOT / d / '**/episodes.jsonl'), recursive=True)):
        base = Path(f).parent
        for l in open(f):
            r = json.loads(l)
            z = np.load(base / 'trajectories' / f"{r['task']}_{r['variant']}_{r['seed']}_true.npz", allow_pickle=True)
            t, tcp, a = z['t'], z['tcp'], z['action_exec']
            sp = np.linalg.norm(np.diff(tcp, axis=0), axis=1) / np.maximum(np.diff(t), 1e-6)
            mov = np.where(sp > 0.01)[0]
            fe = dict(model=r['model'], task=r['task'], variant=r['variant'], seed=r['seed'], success=r['success'], ct=r['ct_s'], dur=r['duration_s'],
                      grasp=r['first_robot_grasp_s'], n_ev=len(r['events']), ev=[e['event'] for e in r['events']][:12], cues=[(c['t'], c['cue']) for c in r['human_cues']],
                      first_move=float(t[mov[0]]) if len(mov) else None, path_m=float(np.linalg.norm(np.diff(tcp, axis=0), axis=1).sum()),
                      tcp_end=tcp[-1].round(3).tolist(), tcp_zmin=float(tcp[:, 2].min()), tcp_zmax=float(tcp[:, 2].max()),
                      idle_frac=float((sp < 0.005).mean()), last5s_path=float(np.linalg.norm(np.diff(tcp[t > t[-1] - 5], axis=0), axis=1).sum()),
                      grip_end=float(a[-1, 6]), grip_min=float(a[:, 6].min()), grip_max=float(a[:, 6].max()),
                      mind=float(np.nanmin(z['robot_person_m'])) if np.isfinite(z['robot_person_m']).any() else None,
                      grasp_drops=r['grasp_drops'], phys=r['physics_valid'], grade=r['grade'], contact=r['contact_violations'], obs=r['observable_violations'],
                      steps_done=r['metrics']['steps_done'], steps_req=r['metrics']['steps_required'], disrupted=r['metrics'].get('disrupted'),
                      safety=r['metrics'].get('safety_distance_violation'))
            out.write(json.dumps(fe) + '\n')
