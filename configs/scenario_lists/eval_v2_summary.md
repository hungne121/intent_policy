# Scenario list `eval_v2`

120 episodes; generator seed 20261004.

| task | episodes | main combination counts (min-max) | identical pair | other |
|---|---|---|---|---|
| T1 | 20 | 12 combos, 1-2 | 20 (100%) |  |
| T2 | 20 | 12 combos, 1-2 | 20 (100%) | timing {'early': 7, 'on_time': 7, 'late': 6}; cup targets 10 |
| T3 | 20 | 12 combos, 1-2 | 0 (0%) | cube order in U {'B1-B2': 10, 'B2-B1': 10} |
| T4 | 20 | 9 combos, 2-3 | 0 (0%) | hold {2.0: 7, 4.0: 6, 1.0: 7}; slots {'S1': 3, 'S2': 3, 'S3': 4, 'S4': 4, 'S5': 3, 'S6': 3} |
| T4neg | 20 | 6 combos, 3-4 | 0 (0%) | hold {1.0: 7, 2.0: 7, 4.0: 6}; slots {'S1': 4, 'S2': 3, 'S3': 4, 'S4': 3, 'S5': 3, 'S6': 3} |
| T5 | 20 | 6 combos, 3-4 | 13 (65%) | base {'T2': 7, 'T1': 6, 'T3': 7} |

Consecutive episodes with the same target slot: 0.
Holdout combinations (never in the demo list): none.
