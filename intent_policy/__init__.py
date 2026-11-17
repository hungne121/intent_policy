"""intent_policy: ACT policy with ManiCast human-motion forecasts for table-top HRC (UR3e, MuJoCo).

  sim         MuJoCo scene builder, robot env, articulated human, contact audit
  scenarios   the 10 HRC task scenarios: scene layout + scripted human (53/QĐ-DL)
  experts     one FSM expert per task, observed / forecast modes (53/QĐ-DL, 54/QĐ-DL)
  shared      ManiCast adapter, forecast planner, Cartesian controller, physics validator, episode runner
  data        fixed LeRobot schema, recorder, episode validator
  intent      ManiCast forecaster and intent-model code
  policies    ACT policy variants
  benchmark   video writer, transition index (training)
"""
