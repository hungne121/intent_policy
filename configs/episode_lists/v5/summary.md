# Episode lists v5 (docs/specs/EPISODE_PLAN_v5.md)

## demo: 559 episodes

| task | episodes | DART | details |
|---|---|---|---|
| T1 | 120 | 60 | target slot {'S1': 20, 'S2': 20, 'S3': 20, 'S4': 20, 'S5': 20, 'S6': 20}; place {'P1': 60, 'P2': 60}; pair colour {'blue': 40, 'red': 40, 'yellow': 40} |
| T2 | 141 | 70 | pair {'cube': 70, 'cup': 71}; hand {'H1': 71, 'H2': 70}; stratum {0: 46, 1: 48, 2: 47}; lowered {False: 70, True: 71} |
| T3 | 83 | 39 | target slot {'S1': 14, 'S2': 11, 'S3': 15, 'S4': 15, 'S5': 12, 'S6': 16}; block order {'B1-B2': 44, 'B2-B1': 39} |
| T4 | 72 | 36 | stratum {0: 12, 1: 12, 2: 12, 3: 12, 4: 12, 5: 12}; slot {'S1': 12, 'S2': 12, 'S3': 12, 'S4': 12, 'S5': 12, 'S6': 12}; colour {'blue': 24, 'red': 24, 'yellow': 24}; extra 0 |
| T4neg | 46 | 23 | goal {'U': 24, 'edge': 22}; stratum {0: 11, 1: 11, 2: 12, 3: 12} |
| T5 | 97 | 45 | base {'T1': 34, 'T2': 35, 'T3': 28}; change stratum {0: 33, 1: 32, 2: 32} |

Failed cells: 21

T4 (selected seeds): entered the robot zone / phase at intrusion, per stratum:

| stratum | late entered | early entered | late phases | early phases |
|---|---|---|---|---|
| 0 | 12/12 | 11/12 | {'approach': 12} | {'approach': 12} |
| 1 | 10/12 | 11/12 | {'approach': 12} | {'approach': 12} |
| 2 | 10/12 | 12/12 | {'approach': 10, 'carry': 2} | {'approach': 10, 'carry': 2} |
| 3 | 11/12 | 11/12 | {'approach': 5, 'carry': 6, 'place': 1} | {'approach': 5, 'carry': 6, 'place': 1} |
| 4 | 12/12 | 12/12 | {'carry': 4, 'place': 4, 'approach': 4} | {'carry': 4, 'place': 4, 'approach': 4} |
| 5 | 12/12 | 12/12 | {'place': 3, 'approach': 8, 'carry': 1} | {'place': 3, 'approach': 8, 'carry': 1} |

T5 (selected seeds): robot state at the change of mind, per base and stratum:

| base | stratum | late | early |
|---|---|---|---|
| T1 | 0 | {'approach': 12} | {'approach': 11, 'grasp': 1} |
| T1 | 1 | {'approach': 8, 'grasp': 4} | {'grasp': 3, 'holding': 6, 'approach': 3} |
| T1 | 2 | {'grasp': 4, 'holding': 6} | {'holding': 10} |
| T2 | 0 | {'approach': 12} | {'approach': 10, 'grasp': 2} |
| T2 | 1 | {'approach': 3, 'grasp': 8} | {'grasp': 3, 'holding': 6, 'approach': 2} |
| T2 | 2 | {'approach': 2, 'holding': 8, 'grasp': 2} | {'grasp': 1, 'holding': 10, 'approach': 1} |
| T3 | 0 | {'approach': 9} | {'approach': 2, 'grasp': 7} |
| T3 | 1 | {'grasp': 5, 'approach': 3, 'holding': 1} | {'holding': 6, 'delivered': 1, 'approach': 1, 'grasp': 1} |
| T3 | 2 | {'holding': 8, 'grasp': 2} | {'holding': 10} |

## eval: 263 episodes

| task | episodes | DART | details |
|---|---|---|---|
| T1 | 48 | 0 | target slot {'S1': 8, 'S2': 8, 'S3': 8, 'S4': 8, 'S5': 8, 'S6': 8}; place {'P1': 24, 'P2': 24}; pair colour {'blue': 16, 'red': 16, 'yellow': 16} |
| T2 | 48 | 0 | pair {'cube': 24, 'cup': 24}; hand {'H1': 24, 'H2': 24}; stratum {0: 16, 1: 16, 2: 16}; lowered {False: 24, True: 24} |
| T3 | 48 | 0 | target slot {'S1': 8, 'S2': 8, 'S3': 8, 'S4': 8, 'S5': 8, 'S6': 8}; block order {'B1-B2': 24, 'B2-B1': 24} |
| T4 | 48 | 0 | stratum {0: 8, 1: 8, 2: 8, 3: 8, 4: 8, 5: 8}; slot {'S1': 8, 'S2': 8, 'S3': 8, 'S4': 8, 'S5': 8, 'S6': 8}; colour {'blue': 16, 'red': 16, 'yellow': 16}; extra 0 |
| T4neg | 23 | 0 | goal {'U': 12, 'edge': 11}; stratum {0: 6, 1: 5, 2: 6, 3: 6} |
| T5 | 48 | 0 | base {'T1': 16, 'T2': 16, 'T3': 16}; change stratum {0: 18, 1: 15, 2: 15} |

Failed cells: 1

T4 (selected seeds): entered the robot zone / phase at intrusion, per stratum:

| stratum | late entered | early entered | late phases | early phases |
|---|---|---|---|---|
| 0 | 8/8 | 8/8 | {'approach': 8} | {'approach': 8} |
| 1 | 7/8 | 8/8 | {'approach': 8} | {'approach': 8} |
| 2 | 6/8 | 6/8 | {'approach': 7, 'carry': 1} | {'approach': 7, 'carry': 1} |
| 3 | 8/8 | 8/8 | {'carry': 7, 'approach': 1} | {'carry': 7, 'approach': 1} |
| 4 | 8/8 | 8/8 | {'carry': 1, 'approach': 3, 'place': 4} | {'carry': 1, 'approach': 3, 'place': 4} |
| 5 | 8/8 | 8/8 | {'place': 3, 'approach': 5} | {'place': 3, 'approach': 5} |

T5 (selected seeds): robot state at the change of mind, per base and stratum:

| base | stratum | late | early |
|---|---|---|---|
| T1 | 0 | {'approach': 6} | {'approach': 6} |
| T1 | 1 | {'grasp': 3, 'holding': 1, 'approach': 1} | {'holding': 4, 'grasp': 1} |
| T1 | 2 | {'grasp': 3, 'holding': 2} | {'holding': 5} |
| T2 | 0 | {'approach': 6} | {'grasp': 2, 'approach': 4} |
| T2 | 1 | {'holding': 1, 'grasp': 3, 'approach': 1} | {'holding': 3, 'grasp': 2} |
| T2 | 2 | {'holding': 5} | {'holding': 5} |
| T3 | 0 | {'approach': 6} | {'grasp': 4, 'approach': 2} |
| T3 | 1 | {'approach': 1, 'grasp': 4} | {'grasp': 1, 'holding': 4} |
| T3 | 2 | {'holding': 5} | {'holding': 5} |

