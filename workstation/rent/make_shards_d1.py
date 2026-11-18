"""68/QĐ-DL đợt 1 (18/11/2026): shard list for the pilot collection of 9 tasks (W3 later). One line = one shard =
`group task split seed [fault_type]`; order = round-robin over seeds so a partial run stays balanced across tasks.
Targets (A pairs): N1 40 train + 10 val per case, N5 8 + 2 per task, N6 12 + 3 per case. Seeds are drawn with a margin
for non-A episodes (A rate 0.85-0.95 measured in BC-DL-19 / BC-DL-20); extra seeds are topped up later if a case is short.
    python3 workstation/rent/make_shards_d1.py I1,I2,I3,C1,C2,C3,C4 > workstation/rent/shards_d1.txt   (69/QĐ-DL)"""
import sys
TASKS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['I1', 'I2', 'I3', 'C1', 'C2', 'C3', 'C4', 'W1', 'W2']   # 69/QĐ-DL: 7 tasks first, W1 / W2 later
N6 = [t for t in ('I3', 'C4') if t in TASKS]
FAULTS = ['spatial_fault', 'temporal_fault', 'direction_fault']
TRAIN0, VAL0 = 300000, 300800
N1_TRAIN, N1_VAL = 46, 12            # -> >= 40 / 10 A pairs at A rate >= 0.87
N5_TRAIN, N5_VAL = 10, 3             # -> >= 8 / 2
N6_TRAIN, N6_VAL = 15, 4             # -> >= 12 / 3
lines = []
for i in range(N1_TRAIN):
    s = TRAIN0 + i
    lines += [f'N1 {t} train {s}' for t in TASKS]
    if i < N5_TRAIN:
        lines += [f'N5 {t} train {s} {FAULTS[(s + k) % 3]}' for k, t in enumerate(TASKS)]
    if i < N6_TRAIN:
        lines += [f'N6 {t} train {s}' for t in N6]
    if i % 4 == 0 and i // 4 < N1_VAL:
        v = VAL0 + i // 4
        lines += [f'N1 {t} val {v}' for t in TASKS]
        if i // 4 < N5_VAL:
            lines += [f'N5 {t} val {v} {FAULTS[(v + k) % 3]}' for k, t in enumerate(TASKS)]
        if i // 4 < N6_VAL:
            lines += [f'N6 {t} val {v}' for t in N6]
print('\n'.join(lines))
