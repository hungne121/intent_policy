"""C4 expert (Collaborator, FOLLOW reading). Observed: when the upturned palm stops at a socket, pick the matching
part and put it on the palm. Forecast (56/QĐ-DL): while the hand is reaching, hover over the part row at the forecast
x; commit to a part as soon as the forecast palm (1 s wrist end point + observed palm-wrist offset) is clearly at
one socket for a few ticks, then bring it to the palm; an observed stop at another socket corrects the choice.
58/QĐ-DL: both modes drop the choice when the hand leaves the chosen socket (change of mind), putting a held part back."""
import numpy as np

from intent_policy.experts.base_expert import BaseExpert
from intent_policy.scenarios.c4 import SOCKETS, PART_ON_PALM
from intent_policy.shared.sim_env import TRAVEL_Z, GRIP_CLOSED, GRASP_DZ, STATION

TUNE = STATION['task_tuning']['C4']


class Expert(BaseExpert):
    task, reading = 'C4', 'follow'

    def observed_socket(self, done):
        ep = self.ep
        if ep.hand_speed > .03 or ep.pose is None:
            return None
        w = ep.wrist()[:2]
        d = {i: float(np.linalg.norm(w - s)) for i, s in SOCKETS.items() if i not in done}
        if not d:
            return None
        i = min(d, key=d.get)
        return i if d[i] < .09 and ep.human_phase.startswith(('reach', 'wait_part')) and ep.wrist()[2] > .66 else None

    def forecast_socket(self, done, peak):
        """FOLLOW reading as a commit cue (56/QĐ-DL): socket nearest the forecast palm along the row, if clearly
        ahead, while the observed hand decelerates into it."""
        ep = self.ep
        f = self.plan.follow_point()
        if f is None or ep.pose is None or ep.hand_speed > TUNE['commit_decel_ratio'] * peak:
            return None
        palm_now = np.asarray(ep.pose['palm'])
        palm = f + (palm_now - ep.wrist())          # the forecast is of the wrist; the palm is what is offered
        d = sorted((abs(float(palm[0] - SOCKETS[i][0])), i) for i in SOCKETS if i not in done)
        if not d or d[0][0] > TUNE['commit_radius_m']:
            return None
        if len(d) > 1 and d[1][0] - d[0][0] < TUNE['commit_margin_m']:
            return None
        i = d[0][1]
        return i if abs(palm_now[0] - SOCKETS[i][0]) < TUNE['commit_palm_m'] else None

    def leaving(self, target):
        """58/QĐ-DL change of mind (observable): the hand has come to socket `target` and is now moving away from it
        (palm >= switch_leave_m farther than its closest approach since the choice; the wrist sits 6 cm in front of the
        socket row, so a move along the row barely changed its distance)."""
        d = float(np.linalg.norm(np.asarray(self.ep.pose['palm'])[:2] - SOCKETS[target]))
        self.closest = min(self.closest, d)
        return d > self.closest + TUNE['switch_leave_m']

    def choose(self, done):
        ep = self.ep
        if self.forecast_mode:
            # Follow continuously over the source row; do not grasp an unconfirmed colour.
            streak, last, peak = 0, None, 0.
            while True:
                target = self.observed_socket(done)
                if target is not None:
                    return target
                peak = max(peak, ep.hand_speed)
                guess = self.forecast_socket(done, peak)
                streak = streak + 1 if guess is not None and guess == last else int(guess is not None)
                last = guess
                if streak >= int(TUNE['commit_ticks']):
                    ep.event('forecast_commit', socket=guess)
                    return guess
                f = self.plan.follow_point()
                if f is not None:
                    xs = [ep.initial[f'part{i}'][0] for i in SOCKETS if i not in done]
                    yield self.plan.follow_step(np.array([np.clip(f[0], min(xs), max(xs)), .12, TRAVEL_Z]), 'prepare')
                else:
                    yield ep.hold()
        yield from ep.wait(lambda: self.observed_socket(done) is not None, 'observe')
        target = self.observed_socket(done); ep.event('target_observed', socket=target)
        return target

    def drop_choice(self, target):
        """The person left the chosen socket: give up the part (put it back where it was) and choose again."""
        ep = self.ep
        part = f'part{target}'
        ep.event('robot_follows_intention_change', socket=target, holding=ep.attached_name())
        if ep.attached_name() == part:
            yield from ep.place(np.r_[ep.initial[part][:2], ep.initial[part][2] + GRASP_DZ])
            ep.event('part_put_back', part=part)
        elif ep.tcp()[2] < TRAVEL_Z - .01:
            yield from ep.goto(np.r_[ep.tcp()[:2], TRAVEL_Z], 'lift')
        return False

    def serve(self, target, done):
        """Take the part of `target` and put it on the palm; False if the person changed its mind on the way."""
        ep = self.ep
        self.closest = np.inf
        lost = lambda: self.leaving(target)
        part = f'part{target}'
        xy = ep.pos(part)[:2]
        yield from ep.goto(np.r_[xy, TRAVEL_Z], 'approach', until=lost)
        if lost(): return (yield from self.drop_choice(target))
        yield from ep.goto(np.r_[ep.pos(part)[:2], ep.pos(part)[2] + GRASP_DZ], 'descend', until=lost)
        if lost(): return (yield from self.drop_choice(target))
        ok = yield from ep.close_on(part)
        if not ok:
            ep.event('pick_failed', socket=target); return None
        yield from ep.goto(np.r_[xy, TRAVEL_Z], 'lift', GRIP_CLOSED)
        if self.forecast_mode:
            while self.observed_socket(done) is None:
                if lost(): return (yield from self.drop_choice(target))
                f = self.plan.follow_point()
                g = f[:2] if f is not None else SOCKETS[target]
                yield self.plan.follow_step(np.r_[g, TRAVEL_Z], 'bring', GRIP_CLOSED)
            confirmed = self.observed_socket(done)
            if confirmed != target:                     # the forecast pointed to another socket: put back, choose again
                ep.event('forecast_correction', predicted=target, observed=confirmed)
                yield from ep.place(np.r_[ep.initial[part][:2], ep.initial[part][2] + GRASP_DZ])
                return False
            ep.event('target_observed', socket=target)
        if lost(): return (yield from self.drop_choice(target))
        ep.event('robot_ready', socket=target)
        palm = lambda: ep.pose['palm']
        # handing the part onto the open palm is a deliberate close approach (as C2): no wait-for-hand rule
        yield from ep.goto(lambda: np.r_[palm()[:2], palm()[2] + PART_ON_PALM + GRASP_DZ + .04], 'move', GRIP_CLOSED, keep_clear=0., until=lost)
        if lost(): return (yield from self.drop_choice(target))
        taken = lambda: part in ep.held_by_human
        yield from ep.goto(lambda: np.r_[palm()[:2], palm()[2] + PART_ON_PALM + GRASP_DZ], 'lower', GRIP_CLOSED, keep_clear=0.,
                           until=lambda: taken() or lost())
        if not taken() and lost(): return (yield from self.drop_choice(target))
        # 09/QĐ-KT: the person closes its hand on the part itself (events.py C4 part_on_palm); open once it holds it
        # (observable: the hand closed on the part), as c2_expert waits for the person's release (timeout 40 ticks)
        yield from ep.wait(taken, 'wait_human_grasp', GRIP_CLOSED, timeout=40)
        if not taken():
            ep.event('human_grasp_timeout', socket=target)
        yield from ep.open_gripper(); ep.event('handover_complete', socket=target)
        retreat = np.r_[ep.tcp()[:2], STATION['task_tuning']['C4']['handover_z_m']]
        yield from ep.goto(retreat, 'retreat', keep_clear=0.)
        return True

    def run(self):
        ep = self.ep
        done = set()
        yield from ep.goto(np.array([.28, .12, TRAVEL_Z]), 'prepare')
        while len(done) < 3:
            target = yield from self.choose(done)
            ok = yield from self.serve(target, done)
            if ok is None:
                return
            if ok:
                done.add(target)
                yield from ep.goto(np.array([.28, .12, TRAVEL_Z]), 'prepare')
        ep.phase = 'done'

    def success(self):
        ep = self.ep
        ok = all(np.linalg.norm(ep.pos(f'part{i}')[:2] - SOCKETS[i]) < .03 and ep.pos(f'part{i}')[2] < .67 for i in range(3))
        ok = ok and ep.attached_name() is None
        return ok, None if ok else 'parts_not_installed'
