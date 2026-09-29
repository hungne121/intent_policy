"""intent_policy: does human-intention information improve an imitation-learned robot policy in HRI?

Subpackages:
  sim         MuJoCo UR3e environment, restricted-action controller, scripted human
  scenarios   config-driven HRI scenarios (instructor / collaborator / intruder)
  experts     privileged scripted demonstrators (data generation only)
  intent      oracle intention information: provider, feature representation, corruption
  policies    LeRobot ACT with restricted / continuous heads, intention fusion, evaluation agent
  benchmark   episode runner, event logging, HRIBench-style metrics
  utils       config loading and path helpers shared by the scripts
"""
