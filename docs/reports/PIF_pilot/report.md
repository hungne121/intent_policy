# PIF v3 pilot, stage B (HABIT S8789)

Episodes: 178, 5-fold CV by episode. 95% bootstrap intervals. Criteria pre-registered in docs/PIF.md §6 (commit 7ad563e).

**Verdict: partial** (G1 True, G2 False, G3 False)


## t_leave -1.0 s

| model | accuracy | ECE |
|---|---|---|
| v3 | 0.607 [0.534, 0.680] | 0.176 |
| III_s0 | 0.949 [0.916, 0.978] | 0.023 |
| III_s1 | 0.854 [0.803, 0.904] | 0.059 |
| III_s2 | 0.933 [0.893, 0.966] | 0.026 |
| pointing rule | 0.298 [0.230, 0.365] (n = 178) | |

| vs | v3 - III accuracy | ECE(III) - ECE(v3) |
|---|---|---|
| III_s0 | -0.343 [-0.421, -0.264] | -0.154 [-0.223, -0.085] |
| III_s1 | -0.247 [-0.331, -0.163] | -0.117 [-0.186, -0.049] |
| III_s2 | -0.326 [-0.399, -0.253] | -0.150 [-0.218, -0.088] |

## t_leave -0.5 s

| model | accuracy | ECE |
|---|---|---|
| v3 | 0.596 [0.522, 0.669] | 0.188 |
| III_s0 | 0.949 [0.916, 0.978] | 0.029 |
| III_s1 | 0.865 [0.815, 0.910] | 0.047 |
| III_s2 | 0.938 [0.899, 0.972] | 0.023 |
| pointing rule | 0.275 [0.213, 0.343] (n = 178) | |

| vs | v3 - III accuracy | ECE(III) - ECE(v3) |
|---|---|---|
| III_s0 | -0.354 [-0.427, -0.281] | -0.160 [-0.225, -0.094] |
| III_s1 | -0.270 [-0.354, -0.185] | -0.142 [-0.204, -0.064] |
| III_s2 | -0.343 [-0.416, -0.270] | -0.166 [-0.226, -0.101] |

## t_leave +0.0 s

| model | accuracy | ECE |
|---|---|---|
| v3 | 0.584 [0.511, 0.657] | 0.167 |
| III_s0 | 0.944 [0.910, 0.978] | 0.027 |
| III_s1 | 0.843 [0.787, 0.893] | 0.067 |
| III_s2 | 0.933 [0.893, 0.966] | 0.031 |
| pointing rule | 0.270 [0.208, 0.337] (n = 178) | |

| vs | v3 - III accuracy | ECE(III) - ECE(v3) |
|---|---|---|
| III_s0 | -0.360 [-0.438, -0.281] | -0.140 [-0.219, -0.086] |
| III_s1 | -0.258 [-0.348, -0.174] | -0.100 [-0.177, -0.041] |
| III_s2 | -0.348 [-0.421, -0.275] | -0.136 [-0.215, -0.086] |

## P4 self-correction (report only)

Belief forced onto a wrong container with b = 0.9, 2.0 s before t_leave: recovered in 177/178 episodes, 98/178 by t_leave; frames to recover median 19.0, p90 45.400000000000006 (10 Hz).

## Fitted v3 parameters per fold

- fold 0: sigma_m 0.01314, beta 19.35, kappa 472.1, a 0.8281, hazard 0.02837, w_none 0.3805, act_prob 0.2988, outlier 0.1604, sigma_out_m 0.2911, outlier_dir 0.1675 (NLL -12.110 -> -17.107)
- fold 1: sigma_m 0.01289, beta 18.72, kappa 465, a 0.827, hazard 0.03064, w_none 0.3936, act_prob 0.3066, outlier 0.1612, sigma_out_m 0.2706, outlier_dir 0.167 (NLL -12.244 -> -17.237)
- fold 2: sigma_m 0.01302, beta 19.26, kappa 469.2, a 0.829, hazard 0.03019, w_none 0.3967, act_prob 0.3041, outlier 0.1607, sigma_out_m 0.2859, outlier_dir 0.1668 (NLL -12.160 -> -17.163)
- fold 3: sigma_m 0.01312, beta 19.18, kappa 469.1, a 0.8299, hazard 0.03113, w_none 0.392, act_prob 0.3039, outlier 0.1576, sigma_out_m 0.2911, outlier_dir 0.1659 (NLL -12.178 -> -17.182)
- fold 4: sigma_m 0.01307, beta 18.84, kappa 456.7, a 0.8256, hazard 0.03093, w_none 0.4043, act_prob 0.3077, outlier 0.1586, sigma_out_m 0.2847, outlier_dir 0.1647 (NLL -12.188 -> -17.176)
