"""I3 expert (Instructor, FOLLOW reading) — per-tick port of scripts/i3_collect.py::rollout (14/11 rules).

Both modes first stage at the shared neutral point between the trays. Observed: choose the cake after a stable
pointing ray (5 frames) with a still hand (3 frames), wait until the hand is 22 cm away, then pick and place on
the receive tray. Forecast: once the hand moves, the TCP reference follows the 1 s forecast end point projected
0.27 m forward (pointing hand -> cake row), clipped to the cake area at 0.82 m (54/QĐ-DL); the cake nearest the
projected end point starts the approach at human pace before the ray is confirmed (13/11 refinement); a forecast
withdrawal lifts the gripper high above the selected cake early. Change of mind (identical in both modes, from
observation only): after the pointing hand withdrew and reaches out again, hold above an ungrasped cake or return a
grasped one to its spot, then follow the new confirmed ray.
"""
import numpy as np

from intent_policy.experts.base_expert import BaseExpert
from intent_policy.scenarios.i3 import CAKES, PLATE_XY, MIDPOINT, SWITCH_WITHDRAW_M
from intent_policy.shared.sim_env import DT, GRIP_OPEN, GRIP_CLOSED
from intent_policy.shared.forecast_planner import TARGET_RADIUS
from intent_policy.shared.episode import select_pointed

NORMAL = ['approach', 'descend', 'close', 'lift', 'move_plate', 'lower_plate', 'open', 'retreat', 'settle']
UNDO = ['return_above', 'return_lower', 'return_open', 'return_retreat']
COUNT_PHASES = {'open': 10, 'return_open': 10}
PROJECT = np.array([0., .27, 0.])            # pointing hand -> cake row (i3_collect projection)


class Expert(BaseExpert):
    task, reading = 'I3', 'follow'

    def set_phase(self, phase, **kw):
        self.phase = phase; self.stage_start = self.ep.tick
        self.ep.event('phase', phase=phase, item=self.selected, **kw)

    def targets(self, current):
        xy = current[self.selected][:2]; src = self.source_pos[:2]
        return {'approach': ([*xy, .78], GRIP_OPEN), 'descend': ([*xy, .667], GRIP_OPEN), 'close': ([*xy, .667], GRIP_CLOSED),
                'lift': ([*xy, .78], GRIP_CLOSED), 'move_plate': ([*PLATE_XY, .78], GRIP_CLOSED),
                'lower_plate': ([*PLATE_XY, .668], GRIP_CLOSED), 'open': ([*PLATE_XY, .668], GRIP_OPEN),
                'retreat': ([*PLATE_XY, .78], GRIP_OPEN), 'return_above': ([*src, .78], GRIP_CLOSED),
                'return_lower': ([*src, .667], GRIP_CLOSED), 'return_open': ([*src, .667], GRIP_OPEN),
                'return_retreat': ([*src, .78], GRIP_OPEN)}

    def run(self):
        ep = self.ep
        self.phase, self.selected, self.source_pos, self.stage_start = 'observe', None, None, 0
        pending, stable, clear_samples, settled = None, [], 0, 0
        prep_target, switched, switch_pending = None, False, False
        visible_still, last_confirmation, confirm_t, t_switch = 0, None, {}, -1.
        closest, last_sel_cue, withdrawn, reextend, prev_wrist, hold_goal = None, None, False, 0, None, None
        early_candidate, prep_goal, at_midpoint = None, None, False
        while self.phase != 'done':
            current = {k: ep.pos(k) for k in CAKES}
            wrist = ep.wrist()
            if not at_midpoint and np.linalg.norm(ep.tcp() - MIDPOINT) < .025:
                at_midpoint = True
            # ---- FOLLOW reading: continuous reversible prep towards the projected 1 s end point
            if self.forecast_mode and self.phase == 'observe' and at_midpoint:
                f = self.plan.follow_point(offset=PROJECT)
                if f is not None:
                    bounds = np.asarray([v[:2] for v in current.values()])
                    proposed = np.r_[np.clip(f[:2], bounds.min(axis=0), bounds.max(axis=0)), .82]
                    if prep_goal is None:
                        prep_goal = ep.tcp(); ep.event('forecast_continuous_follow')
                    prep_goal = prep_goal + (1 - np.exp(-DT / .18)) * (proposed - prep_goal)
                    d = {k: float(np.linalg.norm(f[:2] - v[:2])) for k, v in current.items()}
                    cand = min(d, key=d.get)
                    cand = cand if d[cand] < TARGET_RADIUS else None
                    if cand != early_candidate:
                        early_candidate = cand; ep.event('forecast_route_update', item=cand)
            # ---- observed pointing
            arm = ep.human.active
            ray = ep.human.keypoints(ep.pose, [arm + '_index_base', arm + '_index_tip'])
            stable = (stable + [select_pointed(ray[0], ray[1], current, max_distance_m=.065, margin_m=.01)])[-5:]
            decided = stable[0] if len(stable) == 5 and stable[0] is not None and all(x == stable[0] for x in stable) else None
            visible_still = visible_still + 1 if ep.hand_speed < .045 and decided is not None else 0
            if decided is not None and decided != last_confirmation:
                ep.event('pointing_confirmed', item=decided); last_confirmation = decided; confirm_t[decided] = ep.t
            if self.phase == 'observe' and self.forecast_mode and early_candidate is not None and ep.hand_speed > .025:
                self.selected = early_candidate; self.source_pos = current[self.selected].copy()
                self.set_phase('approach', basis='forecast')
            # ---- change of mind (same in both modes, observation only)
            if self.selected is not None and not switched and not switch_pending and self.phase not in ('settle', 'done'):
                dist_sel = float(np.linalg.norm(wrist[:2] - current[self.selected][:2]))
                if last_sel_cue != self.selected:
                    last_sel_cue = self.selected; closest = dist_sel; withdrawn = False; reextend = 0
                closest = min(closest, dist_sel)
                if dist_sel > closest + SWITCH_WITHDRAW_M: withdrawn = True
                outward = prev_wrist is not None and (wrist[1] - prev_wrist[1]) > .002
                reextend = reextend + 1 if (withdrawn and ep.hand_speed > .05 and outward) else 0
                if reextend >= 3:
                    switch_pending = switched = True; withdrawn = False; reextend = 0; t_switch = ep.t
                    ep.event('switch_observed', item=self.selected, attached=ep.attached_name() is not None)
                    if ep.attached_name() is not None and self.phase in ('lift', 'move_plate', 'lower_plate'):
                        pending = None; self.set_phase('return_above')
                    elif ep.attached_name() is None and self.phase in ('wait_hand_clear', 'approach', 'descend', 'close'):
                        hold_goal = np.r_[ep.tcp()[:2], .78]; self.set_phase('hold_switch')
                    else:
                        switch_pending = False
            prev_wrist = wrist.copy()
            new_cue = None
            if switch_pending and last_confirmation not in (None, self.selected) and confirm_t.get(last_confirmation, -1.) >= t_switch:
                new_cue = last_confirmation
            if switch_pending and new_cue is not None and self.phase in ('hold_switch', 'wait_new_cue'):
                old = self.selected; self.selected = new_cue; self.source_pos = current[self.selected].copy()
                clear_samples = 0; switch_pending = False; self.set_phase('wait_hand_clear', switched_from=old)
            if self.phase == 'observe' and decided is not None and (self.forecast_mode or visible_still >= 3):
                self.selected = decided; self.source_pos = current[self.selected].copy(); self.set_phase('wait_hand_clear')
            elif (decided is not None and self.selected is not None and decided != self.selected
                  and self.phase in ('wait_hand_clear', 'approach', 'descend', 'close', 'lift', 'move_plate', 'lower_plate', 'open')):
                switched = True
                if ep.attached_name() is not None:
                    pending = decided; self.set_phase('return_above', new=pending)
                else:
                    old = self.selected; self.selected = decided; self.source_pos = current[self.selected].copy()
                    clear_samples = 0; self.set_phase('wait_hand_clear', switched_from=old)
            if self.phase == 'wait_hand_clear':
                clear_samples = clear_samples + 1 if np.linalg.norm(wrist - current[self.selected]) > .22 else 0
                if clear_samples >= 3:
                    self.set_phase('approach')
                elif self.forecast_mode and not switched and prep_target != self.selected:
                    f = self.plan.follow_point()
                    if f is not None and np.linalg.norm(f - current[self.selected]) > np.linalg.norm(wrist - current[self.selected]) + .03:
                        prep_target = self.selected; ep.event('forecast_high_prep', item=self.selected)
            # ---- command
            advance = False
            if self.phase in NORMAL[:-1] or self.phase in UNDO:
                goal, grip = self.targets(current)[self.phase]
                following = self.forecast_mode and self.phase == 'approach' and decided is None and not switched
                cmd = ep.command_to(goal, self.phase, grip, following=following)
                if self.phase in COUNT_PHASES:
                    advance = ep.tick - self.stage_start + 1 >= COUNT_PHASES[self.phase]
                elif self.phase != 'close':
                    advance = ep.reached(goal)
            elif self.phase == 'hold_switch':
                cmd = ep.command_to(hold_goal, 'hold_switch', GRIP_OPEN)
            elif self.phase == 'wait_new_cue':
                cmd = ep.hold()
            elif self.phase == 'wait_hand_clear' and self.forecast_mode and prep_target == self.selected:
                cmd = ep.command_to(np.r_[current[self.selected][:2], .91], 'wait_hand_clear', GRIP_OPEN)
            elif self.phase == 'observe':
                follow = self.forecast_mode and prep_goal is not None
                cmd = ep.command_to(prep_goal if follow else MIDPOINT, 'observe', GRIP_OPEN, following=follow)
            elif self.phase == 'settle':
                cmd = ep.hold()
                dof = int(ep.m.joint(self.selected + '_free').dofadr[0]); vel = ep.d.qvel[dof:dof + 6]
                settled = settled + 1 if np.linalg.norm(vel[:3]) < .01 and np.linalg.norm(vel[3:]) < .1 else 0
                if settled >= 10:
                    self.set_phase('done')
            else:
                cmd = ep.hold()
            ep.phase = self.phase
            ep.robot_selection = self.selected          # allowed finger contact while closing / opening on it
            yield cmd
            if self.phase == 'close':
                if ep.attached_name() == self.selected:
                    ep.event('robot_grasp', object=self.selected); advance = True
                elif ep.tick - self.stage_start > 60:
                    ep.event('grasp_failed', item=self.selected); return
            if advance:
                if self.phase == 'return_retreat':
                    if pending is not None:
                        self.selected, pending = pending, None; self.source_pos = current[self.selected].copy()
                        clear_samples = 0; self.set_phase('wait_hand_clear')
                    else:
                        self.set_phase('wait_new_cue')
                elif self.phase in UNDO:
                    self.set_phase(UNDO[UNDO.index(self.phase) + 1])
                else:
                    self.set_phase(NORMAL[NORMAL.index(self.phase) + 1])
        ep.phase = 'done'

    def success(self):
        ep = self.ep
        ok = self.phase == 'done' and ep.attached_name() is None
        return ok, None if ok else f'last_phase_{self.phase}'

    def extra(self):
        return {'expert_selected': self.selected, 'human_motion_source': getattr(self.ep, 'human_motion_source', None)}
