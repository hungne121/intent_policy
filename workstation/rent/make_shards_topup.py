"""70/QĐ-DL thu bù đợt 1 (18/11/2026): oversampled seeds for the short cases, sized from the đợt 1 A-pair rates
(BC-DL-21); only A pairs are kept at train-set build. W1 = the version before agent D's fixes (both variants per
shard). W2 deferred (0/6 A pairs). Seeds continue after the ones used in đợt 1 (no overlap).
    python3 workstation/rent/make_shards_topup.py > workstation/rent/shards_topup.txt"""
ORDER = ['I1', 'I2', 'I3', 'C1', 'C2', 'C3', 'C4']
FAULTS = ['spatial_fault', 'temporal_fault', 'direction_fault']
plan = [  # (group, task, split, first_seed, n)          rate in đợt 1 (train / val)  -> need
    ('N1', 'W1', 'train', 300000, 90), ('N1', 'W1', 'val', 300800, 24),        # nominal 3/6, linger 2/6 (ds_before)
    ('N1', 'C2', 'train', 300046, 14), ('N1', 'C2', 'val', 300812, 10),        # 33/46, 6/12  -> 7 + 4
    ('N1', 'C3', 'train', 300046, 18), ('N1', 'C3', 'val', 300812, 18),        # 30/46, 4/12  -> 10 + 6
    ('N6', 'I3', 'train', 300015, 10), ('N6', 'I3', 'val', 300804, 10),        # before 8/15, 0/4 -> 4 + 3
    ('N5', 'C3', 'train', 300010, 40), ('N5', 'C3', 'val', 300803, 14),        # 2/10, 0/3 -> 6 + 2
    ('N5', 'C4', 'train', 300010, 40), ('N5', 'C4', 'val', 300803, 14),        # 2/10, 0/3 -> 6 + 2
    ('N5', 'C2', 'train', 300010, 3), ('N5', 'I1', 'val', 300803, 3),          # 7/10 -> 1; I1 val 1/3 -> 1
]
rows = []
for g, t, sp, s0, n in plan:
    for i in range(n):
        s = s0 + i
        f = f' {FAULTS[(s + ORDER.index(t)) % 3]}' if g == 'N5' else ''
        rows.append((i, f'{g} {t} {sp} {s}{f}'))
rows.sort(key=lambda r: r[0])          # round-robin over cases
print('\n'.join(r[1] for r in rows))
# shards_tu.txt (71/QĐ-DL, 18/11 10:4x): W2 test 6 seeds + topup minus 31 laptop-done shards (outputs/datasets_hrc_pod/2026-11-18/tu_laptop) + W1 90+24, interleaved
