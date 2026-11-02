"""Summary of scripts/diag/oracle_bound.py output: totals, paired gain over observed with a 95 % t interval, human
held-up time. Recovered 02/11/2026 from the session log.
  ./run.sh scripts/diag/oracle_analyse.py out.jsonl"""
import json, sys, collections, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1])]
R = ('observed', 'cv', 'early', 'oracle1', 'oracle_onset', 'oracle2')
def yield_s(r):
    s, t0 = 0.0, None
    for e in r['events']:
        if e['event_type'] == 'human_yield_to_robot': t0 = e['timestamp']
        elif e['event_type'] == 'human_yield_end' and t0 is not None: s += e['timestamp'] - t0; t0 = None
    return s
def i1_lat(r):
    put, out = {}, []
    for e in r['events']:
        if e['event_type'] != 'protocol_step_complete': continue
        s = e['payload'].get('step', '')
        if s.startswith('coaster_placed_'): put[len(put)] = e['timestamp']
        if s.startswith('cup_on_coaster_') and len(out) < len(put): out.append(e['timestamp'] - put[len(out)])
    return out
by = collections.defaultdict(dict)
for r in rows: by[(r['task'], r['seed'])][r['rule']] = r
print('task seed | total time (s): ' + ' / '.join(R) + ' | human wait for robot (s) | mismatch (m) or1/or2')
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for (t, s), rr in by.items():
    if not all(k in rr for k in R): continue
    ok = all(rr[k]['success'] for k in R)
    d = [rr[k]['dur'] for k in R]; y = [yield_s(rr[k]) for k in R]
    print(f"{t} {s} | " + ' / '.join(f'{x:.2f}' for x in d) + ' | ' + ' / '.join(f'{x:.2f}' for x in y)
          + ' | ' + ' '.join(str(rr[k].get('mismatch_m')) for k in R[3:]) + ('' if ok else '  (some failed)'))
    if ok:
        for k in R: agg[t][k].append(rr[k]['dur'])
        if t == 'I1':
            for k in R: agg['I1_lat'][k] += i1_lat(rr[k])
print('\nmean gain over observed (s), successful seeds only:')
for t, a in agg.items():
    if t == 'I1_lat': continue
    o = st.mean(a['observed'])
    print(f"  {t} n={len(a['observed'])}: " + '  '.join(f"{k} {o - st.mean(a[k]):.2f}" for k in R[1:]))
a = agg['I1_lat']
if a: print('  I1 per-round latency coaster down -> cup on coaster (s): ' + '  '.join(f'{k} {st.mean(a[k]):.2f}' for k in R))

# paired gain over observed with a 95 % t interval (all seeds where both rules succeeded)
try:
    from scipy.stats import t as _t
    tq = lambda n: float(_t.ppf(0.975, n - 1))
except Exception:
    tq = lambda n: {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}.get(n, 2.0)
print('\npaired gain over observed, 95 % CI (seeds where both succeeded):')
for task in dict.fromkeys(t for t, _ in by):
    for k in R[1:]:
        d = [rr['observed']['dur'] - rr[k]['dur'] for (t, s), rr in by.items()
             if t == task and k in rr and 'observed' in rr and rr['observed']['success'] and rr[k]['success']]
        if len(d) >= 2:
            m, sd = st.mean(d), st.stdev(d)
            h = tq(len(d)) * sd / len(d) ** 0.5
            print(f'  {task} {k:8s} n={len(d):2d} mean {m:+.2f} s  CI [{m - h:+.2f}, {m + h:+.2f}]')
    n_ok = {k: sum(1 for (t, s), rr in by.items() if t == task and k in rr and rr[k]['success']) for k in R}
    print(f'  {task} success: ' + ', '.join(f'{k} {v}' for k, v in n_ok.items()))

# Cowork: how often / how long the human was held up (pre-start waits + mid-motion pauses)
print('\nhuman held up by the robot: mean count / mean seconds per episode (successful episodes):')
for task in dict.fromkeys(t for t, _ in by):
    line = []
    for k in R:
        eps = [rr[k] for (t, s), rr in by.items() if t == task and k in rr and rr[k]['success']]
        if eps:
            n = st.mean(sum(1 for e in r['events'] if e['event_type'] == 'human_yield_to_robot') for r in eps)
            line.append(f'{k} {n:.1f}/{st.mean(yield_s(r) for r in eps):.2f}s')
    print(f'  {task}: ' + '  '.join(line))