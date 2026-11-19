"""C4 detector hrc-events-1.2 (15/QĐ-KT A2-A6, 19/11/2026): positive / negative tests on the REAL simulated state.

Every case builds the real C4 episode (scene, human model v4, physics, the scripted person of scenarios/c4.py), runs the
person until it holds its upturned palm at the first socket (`wait_part_<i>`), then sets the part's free-joint qpos /
qvel or drives the robot with the episode controller, and keeps stepping physics. The detector is never mocked: the
person itself decides (scenarios/c4.py `det.until('part_on_palm', ...)`) and grasps with the shared `ep.human_grasp`
(weld at the current hand-object pose, only when the hand is within HUMAN_GRASP_GAP_M of the part).
  (i)  receiver grasps before the robot releases: robot holds part i still on the palm            -> cue, then grasp
  (ii) robot released, part resting on the supported palm for `rest_ticks` consecutive ticks     -> cue, then grasp
Negatives: released outside the palm, falling part (in flight / onto the table), unstable part, robot still holding and
moving, robot still holding (path ii never counts a held part), person never grasped (C4 check fails).
"""
import mujoco
import numpy as np
import pytest

from intent_policy.shared.events import CFG, DETECTOR_VERSION, detector

P = CFG['C4']['part_on_palm']
SEED = 300800                     # val split (configs/hrc/collection.yaml); never a test seed (300900+)


@pytest.fixture(scope='module')
def predictor():
    from intent_policy.shared.manicast_adapter import load_predictor
    return load_predictor()


def start(predictor, seed=SEED):
    """Real C4 episode with the person waiting, palm up, at its first socket; the robot holds its start pose."""
    from intent_policy.scenarios import scenario
    from intent_policy.shared.episode import Episode
    sc = scenario('C4')
    scene, positions, xml, info = sc.build(seed, 'nominal')
    ep = Episode('C4', 'observed', seed, scene, positions, xml, predictor, out=None, max_ticks=sc.MAX_TICKS)
    ep.event('task_active', **{k + '_evaluator_only': v for k, v in info.items() if k != 'variant'})
    human = sc.human(ep, info)
    i = info['order'][0]
    while not (ep.human_phase or '').startswith('wait_part_'):
        tick(ep, human)
    for _ in range(10):                                  # palm settled at the receive point
        tick(ep, human)
    assert ep.human_phase == f'wait_part_{i}'
    return ep, human, info, i


def tick(ep, human, command=None):
    """One tick as Episode.run: the person's frame (it evaluates its wait predicate first), then the robot command."""
    ep.apply_human(next(human))
    ep.step(ep.hold() if command is None else command)


def put(ep, part, xyz, vel=None):
    """Place the part's free body (real qpos / qvel), identity orientation."""
    j = ep.m.joint(part + '_free')
    q, v = int(j.qposadr[0]), int(j.dofadr[0])
    ep.d.qpos[q:q + 3] = xyz
    ep.d.qpos[q + 3:q + 7] = [1, 0, 0, 0]
    ep.d.qvel[v:v + 6] = 0 if vel is None else np.r_[vel, 0, 0, 0]
    mujoco.mj_forward(ep.m, ep.d)


def palm_rest(ep, dxy=(0., 0.), dz=.002):
    """Part centre on the upturned palm (scenarios/c4.py PART_ON_PALM above the palm centre) + an offset."""
    from intent_policy.scenarios.c4 import PART_ON_PALM
    palm = np.asarray(ep.pose['palm'], float)
    return palm + [dxy[0], dxy[1], PART_ON_PALM + dz]


def cues(ep, name='part_on_palm'):
    return [e for e in ep.events if e['event'] == 'human_cue' and e['cue'] == name]


def run_until_cue(ep, human, n, each=None):
    for k in range(n):
        if each is not None:
            each(k)
        tick(ep, human)
        if cues(ep):
            return k
    return None


def test_version_locked():
    assert DETECTOR_VERSION == 'hrc-events-1.2'
    assert CFG['version'] == DETECTOR_VERSION and 'changelog' in CFG
    assert P['rest_ticks'] >= 2


# ------------------------------------------------------------------------------------------- positives
@pytest.mark.parametrize('dxy', [(0., 0.), (.012, -.01)], ids=['centre', 'offset'])
def test_released_on_palm_person_grasps(predictor, dxy):
    """(ii) released part resting on the palm (centred, or offset inside the palm support where the hand still
    carries it; probe_offsets.log): the person sees it at rest for rest_ticks ticks, grasps it with the shared grasp,
    and carries it away in its hand."""
    ep, human, info, i = start(predictor)
    part = f'part{i}'
    det = detector(ep)
    assert det._over_support(palm_rest(ep, dxy), np.asarray(ep.pose['palm']), P)
    put(ep, part, palm_rest(ep, dxy))
    k = run_until_cue(ep, human, 40)
    assert k is not None and k + 1 >= P['rest_ticks']               # never on the first ticks: stability window
    assert ep.attached_name() is None
    tick(ep, human)                                                  # the person's grasp happens on its next frame
    assert part in ep.held_by_human
    assert any(e['event'] == 'human_grasp' and e['object'] == part for e in ep.events)
    for _ in range(12):                                              # grip, then the robot is already clear: it lifts
        tick(ep, human)
    assert part in ep.held_by_human
    assert np.linalg.norm(ep.pos(part) - np.asarray(ep.pose['palm'])) < .1   # moves with the hand (weld)


def test_robot_holds_still_on_palm_person_grasps(predictor):
    """(i) unchanged 1.1 path: the robot still grips the part, holds it still on the palm -> cue while it holds it."""
    from intent_policy.scenarios.c4 import PART_ON_PALM
    from intent_policy.shared.sim_env import GRASP_DZ, GRIP_CLOSED
    ep, human, info, i = start(predictor)
    part = f'part{i}'
    held_at_cue = []
    grip_robot(ep, part)
    palm = lambda: np.asarray(ep.pose['palm'], float)
    gen = ep.goto(lambda: np.r_[palm()[:2], palm()[2] + PART_ON_PALM + GRASP_DZ], 'lower', GRIP_CLOSED, keep_clear=0.)
    for _ in range(200):
        cmd = next(gen, None)
        tick(ep, human, ep.hold(GRIP_CLOSED) if cmd is None else cmd)
        if cues(ep):
            held_at_cue.append(ep.attached_name()); break
    assert held_at_cue == [part]
    tick(ep, human, ep.hold(GRIP_CLOSED))
    assert part in ep.held_by_human


def grip_robot(ep, part):
    """Real gripper weld (assisted grasp, sim/base_env._attach) on the part placed between the fingers."""
    from intent_policy.shared.sim_env import GRASP_DZ
    put(ep, part, ep.tcp() - [0, 0, GRASP_DZ])
    ep.env._attach(ep.m.body(part).id)
    assert ep.attached_name() == part


# ------------------------------------------------------------------------------------------- negatives
@pytest.mark.parametrize('dxy', [(0., .02), (.01, 0.)], ids=['offset_fingers_slides', 'offset_side_slides'])
def test_released_offset_slides_off(predictor, dxy):
    """Released inside the palm-support footprint but where the hand does not carry it (measured 19/11,
    outputs/diag/c4_detector12_1119/probe_offsets.log: it slides off within 3-5 ticks): never at rest -> no cue, the part
    ends below the palm."""
    ep, human, info, i = start(predictor)
    part = f'part{i}'
    assert detector(ep)._over_support(palm_rest(ep, dxy), np.asarray(ep.pose['palm']), P)
    put(ep, part, palm_rest(ep, dxy))
    assert run_until_cue(ep, human, 40) is None
    assert part not in ep.held_by_human and ep.pos(part)[2] < np.asarray(ep.pose['palm'])[2]


def test_released_outside_palm(predictor):
    """Released beside the palm support (outside the footprint): it falls to the table; the person never reacts."""
    ep, human, info, i = start(predictor)
    part = f'part{i}'
    x = palm_rest(ep, (.09, 0.))
    assert not detector(ep)._over_support(x, np.asarray(ep.pose['palm']), P)
    put(ep, part, x)
    assert run_until_cue(ep, human, 40) is None
    assert part not in ep.held_by_human


def test_falling_part(predictor):
    """Dropped from 6 cm above the palm: while it falls (any tick before it comes to rest) no cue; a part dropped next
    to the hand onto the table: never."""
    ep, human, info, i = start(predictor)
    part = f'part{i}'
    det = detector(ep)
    put(ep, part, palm_rest(ep, dz=.06))
    k = run_until_cue(ep, human, 40)
    # falling 6 cm takes sqrt(2 * 0.06 / 9.81) = 0.11 s = 2.2 ticks; the cue needs rest_ticks at rest after that
    assert k is None or k >= 2 + P['rest_ticks'] - 1
    ep, human, info, i = start(predictor)
    part = f'part{i}'
    put(ep, part, palm_rest(ep, (0., -.08), dz=.03))                  # behind the wrist: falls past the hand
    assert run_until_cue(ep, human, 40) is None


def test_unstable_part(predictor):
    """On the palm but not at rest: the part rocks in place (+-0.1 m/s re-imposed every tick, alternating; friction
    damps it within the tick, so its speed after the tick is measured). Whenever its speed after a tick exceeds the
    bound the rest count restarts; if that happens at least once every rest_ticks ticks the person never reacts."""
    ep, human, info, i = start(predictor)
    part = f'part{i}'
    det = detector(ep)
    put(ep, part, palm_rest(ep))
    v = int(ep.m.joint(part + '_free').dofadr[0])
    fast = []

    def kick(k):
        ep.d.qvel[v:v + 3] = [.1 * (-1) ** k, 0., 0.]
    for k in range(24):
        kick(k)
        tick(ep, human)
        fast.append(det.body_speed(part) > P['grasp_gap_m'] / P['close_s'])
        assert det._rest.get(part, 0) < P['rest_ticks']
    assert all(any(fast[k:k + P['rest_ticks']]) for k in range(len(fast) - P['rest_ticks']))
    assert not cues(ep) and part not in ep.held_by_human
    # it is still on the palm (a fall would make the case trivial)
    assert det._over_support(ep.pos(part), np.asarray(ep.pose['palm']), P)
    assert ep.hand_object_distance(part) <= P['grasp_gap_m']


def test_robot_holding_moving_no_cue(predictor):
    """Robot not released and still moving the part over / onto the palm (TCP speed > gap / close time): no cue while
    it moves, and path (ii) never counts a held part even when the part itself is still."""
    from intent_policy.scenarios.c4 import PART_ON_PALM
    from intent_policy.shared.sim_env import GRASP_DZ, GRIP_CLOSED
    ep, human, info, i = start(predictor)
    part = f'part{i}'
    det = detector(ep)
    grip_robot(ep, part)
    palm = lambda: np.asarray(ep.pose['palm'], float)
    gen = ep.goto(lambda: np.r_[palm()[:2], palm()[2] + PART_ON_PALM + GRASP_DZ + .04], 'move', GRIP_CLOSED, keep_clear=0.)
    fast = 0
    for _ in range(200):
        cmd = next(gen, None)
        if cmd is None:
            break
        tick(ep, human, cmd)
        if det.tcp_speed() > P['grasp_gap_m'] / P['close_s']:
            fast += 1
            assert not cues(ep)
        assert det._rest.get(part, 0) == 0                            # held: never "resting after release"
    assert fast > 10 and not cues(ep)                                 # 4 cm above the palm: out of the hand's reach


def test_robot_holding_no_release_path(predictor):
    """A held part never satisfies path (ii): with the robot holding the part on the palm but the hand farther than the
    grasp gap, nothing fires, whatever the part speed."""
    ep, human, info, i = start(predictor)
    part = f'part{i}'
    det = detector(ep)
    grip_robot(ep, part)
    for _ in range(10):
        tick(ep, human)
    assert det._rest.get(part, 0) == 0 and not cues(ep)


def test_person_not_grasped_no_completion(predictor):
    """A5: completion needs the person to have held every part and the robot to have released it. Parts put into
    their sockets without the person (e.g. a policy seating them itself) pass the socket test of c4_expert.success()
    but not the C4 check."""
    from intent_policy.experts import expert_class
    from intent_policy.scenarios import scenario
    from intent_policy.scenarios.c4 import SOCKETS, SEATED_Z
    ep, human, info, i = start(predictor)
    for k in range(3):
        put(ep, f'part{k}', [SOCKETS[k][0], SOCKETS[k][1], SEATED_Z])
    for _ in range(3):
        tick(ep, human)
    ok, _ = expert_class('C4')(ep).success()
    assert ok                                                        # the parts are seated, the robot holds nothing
    ok, reason = scenario('C4').check(ep, info)
    assert not ok and reason == 'part_not_taken_by_person'
    # the person's grasp is recorded by the shared mechanism; with it the check passes only once the hand let go
    ep.events += [{'t': ep.t, 'event': 'human_grasp', 'object': f'part{k}'} for k in range(3)]
    ep.held_by_human.add(f'part{i}')
    assert scenario('C4').check(ep, info) == (False, 'part_still_in_hand')
    ep.held_by_human.discard(f'part{i}')
    assert scenario('C4').check(ep, info) == (True, None)
