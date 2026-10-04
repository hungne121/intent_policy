"""Online intent for closed-loop rollouts: the per-frame contract (intent/contract.py) computed while an episode runs.

  OnlineHindsight  the hindsight source's labels from the simulator's plan as it unfolds: segments from the human
                   motions and stages recorded so far (labels.segments_from_sim), a running motion's end taken from
                   its planned duration, so p_* / phase / tte / c_* equal the offline hindsight labels of the same
                   episode. Only xi differs: the true future is unknown online, it is the constant-velocity forecast.

Frames follow scripts/collect_demos.py: frame k is observed before the update at k / fps, an event at time t shows
from frame labels.event_frame(t) on.
"""
from __future__ import annotations
import numpy as np
from intent_policy.intent.labels import event_frame, motions_from_events, segments_from_sim
from intent_policy.intent.perception import CVMForecaster
from intent_policy.intent.sources import HindsightSource, HumanObs
from intent_policy.intent.tracker import HoldingDetector, RobotState


def run_lengths(names: list[str]) -> list[dict]:
    out = []
    for i, n in enumerate(names):
        if out and out[-1]['name'] == n:
            out[-1]['end'] = i
        else:
            out.append({'name': n, 'start': i, 'end': i})
    return out


class OnlineHindsight(HindsightSource):
    name = 'hindsight_online'

    def __init__(self, schema: dict):
        super().__init__(schema)
        self.fps = float(schema['fps'])
        self.forecaster = CVMForecaster(schema['horizon_s'], schema['n_waypoints'], self.fps)

    def reset(self, episode_meta: dict | None = None) -> None:
        self.tracker.reset()
        self.holding = HoldingDetector(**self.schema['holding'])
        self.segments, self.stages, self.history = [], [], []

    def observe(self, scenario, t: int, gripper: float) -> dict:
        """One frame of the running episode (call once per decision, before acting)."""
        kp = scenario.human_keypoints(self.schema['keypoints']).astype(np.float32)
        self.stages.append(scenario.human.stage)
        self.history.append(kp)
        events = scenario.get_events()
        motions = motions_from_events(events, self.fps)
        running = motions[-1] if motions and motions[-1]['end'] is None else None
        segs = segments_from_sim(motions, run_lengths(self.stages), scenario.variation['spec'], self.schema, t + 1)
        if running is not None:                 # its goal pose lies ahead: planned end = start + duration
            e = [e for e in events if e['event_type'] == 'human_motion_start'][-1]
            planned = event_frame(e['timestamp'] + float(e['payload'].get('duration', 0.0)), self.fps)
            frac = float(self.schema.get('evidence_fraction', 0.4))
            for s in segs:
                if s['source'].split('+')[-1] == running['label'] and s['t_end'] >= t and planned > s['t_clear']:
                    s['t_clear'] = int(planned)
                    s['t_evident'] = int(min(s['t_onset'] + round(frac * (s['t_clear'] - s['t_onset'])), s['t_clear']))
        self.segments = segs
        robot = RobotState(tcp=np.asarray(scenario.tcp(), float), gripper=float(gripper), holding=self.holding.step(float(gripper)))
        self._xi_now = self.forecaster.forecast(self.history[-3:], self.fps)
        return self.step(HumanObs(t, kp), robot)

    def step(self, obs: HumanObs, robot: RobotState) -> dict:
        self.xi = {obs.t: self._xi_now}
        return super().step(obs, robot)
