"""C4 nominal paired preview: the human holds an upturned palm at one of three sockets, the robot
brings the matching part into the hand, the human installs it. Scripted expert, not ACT."""
import numpy as np
from v2_preview_common import Preview, move, hold, main_pair, TRAVEL_Z, GRIP_OPEN, GRIP_CLOSED, GRASP_DZ

SOCKETS = {i: np.array([x, -.12]) for i, x in enumerate((.16, .28, .40))}
PALM_DZ = .09               # upturned palm above the socket
PART_ON_PALM = .054         # part centre above the palm centre when resting on the palm support
IN_HAND = np.array([0., .03, 0.])   # part held in the closed fingers


def hook(scene, positions, xml):
    scene['human']['stand_xy'] = [.30, -.56]
    from c_preview_visual import connected_human
    return scene, positions, connected_human(scene, xml)


def human(pv, order):
    home = pv.human.hanging_palm(pv.human.active)
    cur = home
    for i in order:
        s = SOCKETS[i]
        recv = np.array([s[0], s[1] - .02, .62 + PALM_DZ])
        for f in move(cur, recv, 1.4, lift=.05, shape='offer', shape_from='relaxed', phase=f'reach_{i}'):
            yield {**f, 'palm_up': f['progress']}
        pv.event('palm_up', socket=i)
        yield from hold(recv, until=lambda: pv.delivered == i, shape='offer', palm_up=1., phase=f'wait_part_{i}')
        # close the hand on the delivered part, then carry it to the socket and let it go on the socket
        part = f'part{i}'
        for f in move(recv, recv, .4, shape='grasp', shape_from='offer', phase=f'grip_{i}'):
            yield {**f, 'palm_up': 1.}
        pv.place_kinematic(part, recv + IN_HAND); pv.kinematic.pop(part, None)
        pv.human_grasp(part); pv.event('human_holds_part', part=part)
        yield from hold(recv, until=lambda: pv.attached_name() is None and pv.tcp()[2] > recv[2] + .12, shape='grasp', palm_up=1., phase=f'wait_robot_clear_{i}')
        # Keep the observed hand-to-part offset when positioning the held part over its socket.
        # Using a fixed palm offset left the part outside the original success radius.
        offset = pv.pos(part) - np.asarray(pv.pose['palm'])
        down = np.array([s[0], s[1], .653]) - offset
        above = down + [0, 0, .08]
        for f in move(recv, above, .9, shape='grasp', phase=f'to_socket_{i}'):
            yield {**f, 'palm_up': 1. - f['progress']}
        yield from move(above, down, .6, shape='grasp', phase=f'insert_{i}')
        pv.human_release(part); pv.event('install_complete', socket=i)
        yield from move(down, above + [0, 0, .04], .5, shape='relaxed', shape_from='grasp', phase=f'release_{i}')
        cur = above + [0, 0, .04]
    yield from move(cur, home, .9, lift=.04, phase='rest')
    yield from hold(home, 1.0, phase='done')


def robot(pv):
    pv.delivered = None
    done = set()
    def observed_socket():
        if pv.hand_speed > .03 or pv.pose is None: return None
        w = pv.wrist()[:2]
        d = {i: float(np.linalg.norm(w - s)) for i, s in SOCKETS.items() if i not in done}
        i = min(d, key=d.get)
        return i if d[i] < .09 and pv.human_phase.startswith(('reach', 'wait_part')) and pv.wrist()[2] > .66 else None
    def predicted_socket():
        if pv.forecast is None or pv.hand_speed < .03: return None
        f = pv.wrist_future()[-3:].mean(axis=0)[:2]
        d = {i: float(np.linalg.norm(f - s)) for i, s in SOCKETS.items() if i not in done}
        i = min(d, key=d.get)
        return i if d[i] < .10 and (len(d) == 1 or sorted(d.values())[1] - d[i] > .04) else None
    yield from pv.goto(np.array([.28, .12, TRAVEL_Z]), 'prepare')       # neutral above the part source
    for _ in range(3):
        target = None
        if pv.mode == 'intent':
            while True:
                o = observed_socket()
                if o is not None: target = o; break
                p = predicted_socket()
                if p is not None:
                    target = p; pv.event('forecast_pick', socket=p); break
                yield pv.hold()
            part = f'part{target}'
            # forecast-based pick approach at human pace; the rest is independent work
            yield from pv.goto(lambda: np.r_[pv.pos(part)[:2], TRAVEL_Z], 'approach', following=True)
        else:
            yield from pv.wait(lambda: observed_socket() is not None, 'observe')
            target = observed_socket(); pv.event('target_observed', socket=target)
            part = f'part{target}'
        ok = yield from pv.pick(part)
        if not ok:
            pv.event('pick_failed', socket=target)
            return
        # A forecast can point to the wrong socket. Observe the palm and physically return
        # a wrongly picked part before taking the matching one; no access to the human's order.
        yield from pv.wait(lambda: observed_socket() is not None, 'wait_palm', GRIP_CLOSED)
        confirmed = observed_socket()
        if confirmed != target:
            pv.event('forecast_correction', predicted=target, observed=confirmed)
            yield from pv.place(np.r_[pv.initial[part][:2], pv.initial[part][2] + GRASP_DZ])
            target = confirmed
            part = f'part{target}'
            ok = yield from pv.pick(part)
            if not ok:
                pv.event('pick_failed', socket=target)
                return
        pv.event('robot_ready', socket=target)
        palm = lambda: pv.pose['palm']
        yield from pv.goto(lambda: np.r_[palm()[:2], palm()[2] + PART_ON_PALM + GRASP_DZ + .06], 'move', GRIP_CLOSED)
        yield from pv.goto(lambda: np.r_[palm()[:2], palm()[2] + PART_ON_PALM + GRASP_DZ], 'lower', GRIP_CLOSED)
        # the part rests on the palm support; hand the ownership to the human (weld) and release
        pv.human_grasp(part); pv.delivered = target
        yield from pv.open_gripper(); pv.event('handover_complete', socket=target)
        yield from pv.goto(lambda: np.r_[pv.tcp()[:2], palm()[2] + .22], 'retreat')
        done.add(target)
        yield from pv.goto(np.array([.28, .12, TRAVEL_Z]), 'prepare')
    pv.phase = 'done'


def run(mode, seed, predictor, out):
    pv = Preview('C4', mode, seed, out, predictor, scene_hook=hook, max_ticks=1400)
    order = [int(i) for i in np.random.default_rng(seed).permutation(3)]
    pv.event('task_active', order_evaluator_only=order)
    pv.run(human(pv, order), robot(pv), done=lambda: pv.human_phase == 'done')
    ok = all(np.linalg.norm(pv.pos(f'part{i}')[:2] - SOCKETS[i]) < .03 and pv.pos(f'part{i}')[2] < .67 for i in range(3))
    return pv.finish(ok and pv.attached_name() is None,
                     {'human_order_evaluator_only': order,
                      'limitations': ['part ownership passes by weld at release, then re-seated in the fist (kinematic)',
                                      'scripted human, not HY-Motion', 'expert preview, not trained ACT']})


if __name__ == '__main__':
    main_pair('C4', run, 300100)
