# Scenario list `demo_v1`

280 episodes; generator seed 20260929.

| task | episodes | main combination counts (min-max) | identical pair | other |
|---|---|---|---|---|
| T1 | 60 | 12 combos, 5-5 | 20 (33%) |  |
| T2 | 50 | 12 combos, 4-5 | 17 (34%) | timing {'on_time': 17, 'late': 17, 'early': 16}; cup targets 25 |
| T3 | 60 | 12 combos, 5-5 | 0 (0%) | cube order in U {'B2-B1': 30, 'B1-B2': 30} |
| T4 | 40 | 9 combos, 4-5 | 0 (0%) | hold {1.0: 14, 2.0: 14, 4.0: 12}; slots {'S1': 7, 'S2': 6, 'S3': 6, 'S4': 7, 'S5': 7, 'S6': 7} |
| T4neg | 10 | 6 combos, 1-2 | 0 (0%) | hold {2.0: 4, 4.0: 3, 1.0: 3}; slots {'S1': 2, 'S2': 1, 'S3': 2, 'S4': 2, 'S5': 1, 'S6': 2} |
| T5 | 60 | 6 combos, 10-10 | 14 (23%) | base {'T1': 20, 'T3': 20, 'T2': 20} |

Consecutive episodes with the same target slot: 0.
Holdout combinations (never in the demo list): none.
