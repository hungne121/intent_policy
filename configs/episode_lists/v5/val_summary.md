# Episode lists v5 (docs/specs/EPISODE_PLAN_v5.md)

## val: 55 episodes

| task | episodes | DART | details |
|---|---|---|---|
| T1 | 10 | 0 | target slot {'S1': 2, 'S2': 1, 'S3': 2, 'S4': 2, 'S5': 1, 'S6': 2}; place {'P1': 5, 'P2': 5}; pair colour {'blue': 4, 'red': 3, 'yellow': 3} |
| T2 | 10 | 0 | pair {'cube': 5, 'cup': 5}; hand {'H1': 5, 'H2': 5}; stratum {0: 3, 1: 4, 2: 3}; lowered {False: 5, True: 5} |
| T3 | 10 | 0 | target slot {'S1': 2, 'S2': 1, 'S3': 2, 'S4': 2, 'S5': 1, 'S6': 2}; block order {'B1-B2': 5, 'B2-B1': 5} |
| T4 | 10 | 0 | stratum {0: 2, 1: 1, 2: 2, 3: 2, 4: 1, 5: 2}; slot {'S1': 2, 'S2': 2, 'S3': 1, 'S4': 1, 'S5': 2, 'S6': 2}; colour {'blue': 4, 'red': 3, 'yellow': 3}; extra 0 |
| T4neg | 5 | 0 | goal {'U': 2, 'edge': 3}; stratum {0: 2, 1: 1, 2: 1, 3: 1} |
| T5 | 10 | 0 | base {'T1': 3, 'T2': 4, 'T3': 3}; change stratum {0: 4, 1: 3, 2: 3} |

Failed cells: 0

T4 (selected seeds): entered the robot zone / phase at intrusion, per stratum:

| stratum | late entered | early entered | late phases | early phases |
|---|---|---|---|---|
| 0 | 2/2 | 2/2 | {'approach': 2} | {'approach': 2} |
| 1 | 1/1 | 1/1 | {'approach': 1} | {'approach': 1} |
| 2 | 2/2 | 2/2 | {'carry': 2} | {'carry': 2} |
| 3 | 2/2 | 2/2 | {'carry': 1, 'approach': 1} | {'carry': 1, 'approach': 1} |
| 4 | 1/1 | 1/1 | {'approach': 1} | {'approach': 1} |
| 5 | 2/2 | 2/2 | {'approach': 1, 'place': 1} | {'approach': 1, 'place': 1} |

T5 (selected seeds): robot state at the change of mind, per base and stratum:

| base | stratum | late | early |
|---|---|---|---|
| T1 | 0 | {'approach': 1} | {'approach': 1} |
| T1 | 1 | {'holding': 1} | {'holding': 1} |
| T1 | 2 | {'holding': 1} | {'holding': 1} |
| T2 | 0 | {'approach': 2} | {'grasp': 1, 'approach': 1} |
| T2 | 1 | {'grasp': 1} | {'holding': 1} |
| T2 | 2 | {'holding': 1} | {'holding': 1} |
| T3 | 0 | {'approach': 1} | {'grasp': 1} |
| T3 | 1 | {'grasp': 1} | {'holding': 1} |
| T3 | 2 | {'holding': 1} | {'holding': 1} |

