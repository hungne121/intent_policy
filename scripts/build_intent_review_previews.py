"""Build 10 x (workflow + C1-C4) review storyboards; not a demo/policy generator."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import random

import yaml

from intent_policy.sim.base_env import ROOT

CASES = {
    'BASE': 'workflow đầy đủ', 'C1': 'dự báo đúng', 'C2': 'mơ hồ',
    'C3': 'dự báo sai', 'C4': 'người đổi ý',
}
HOME = [0.15, 0.0, 0.85]
REST = [0.49, -0.40, 0.70]
LEFT_REST = [0.13, -0.40, 0.70]


def plus(p, d):
    return [round(float(a + b), 5) for a, b in zip(p, d)]


class Story:
    def __init__(self, code, name, case, role):
        self.code, self.case = code, case
        self.d = dict(title=f'{code}: {name} | {case}: {CASES[case]}', role=role, case=case,
                      objects={}, decor=[], human=[], human_l=[], robot=[], events=[], marks=[], captions=[])

    def obj(self, name, shape, pos, **kw):
        self.d['objects'][name] = dict(shape=shape, pos=pos, **kw)

    def decor(self, name, typ, pos, color='grey', **kw):
        self.d['decor'].append(dict(name=name, type=typ, pos=pos, color=color, **kw))

    def h(self, a, b, p, shape='grasp', left=False, **kw):
        self.d['human_l' if left else 'human'].append(dict(t=[a, b], to=p, shape=shape, **kw))

    def r(self, a, b, p=None, grip=None):
        s = dict(t=[a, b])
        if p is not None:
            s['to'] = p
        if grip is not None:
            s['grip'] = grip
        self.d['robot'].append(s)

    def grab(self, t, obj, who):
        self.d['events'].append(dict(t=t, grab=obj, by=who))

    def drop(self, t, obj, pos):
        self.d['events'].append(dict(t=t, drop=obj, at=pos))

    def cap(self, a, b, text):
        self.d['captions'].append([a, b, text])

    def decision(self, action, reveal):
        if 'decision' not in self.d:
            self.d['decision'] = dict(action_start=action, reveal_t=reveal,
                                      forecast_is_scripted=True, human_changes_goal=self.case == 'C4')
            self.d['marks'] += [dict(t=action, who='robot', label='robot đã đi'),
                                dict(t=reveal, who='human', label='thông tin rõ/đổi')]

    def finish(self, end):
        self.d['duration'] = round(end, 3)
        self.d['events'].sort(key=lambda e: e['t'])
        for arm in ('human', 'human_l', 'robot'):
            self.d[arm].sort(key=lambda x: x['t'][0])
            if not self.d[arm] and arm == 'human_l':
                self.d.pop(arm)
        self.d['captions'].sort(key=lambda c: c[0])
        dc = self.d.get('decision', {'action_start': end * .2, 'reveal_t': end * .4})
        self.d['review_times'] = [round(dc['action_start'] + .65, 2),
                                  round(min(end - .8, dc['reveal_t'] + .25), 2), round(end - 1.0, 2)]
        self.d['marks'].append(dict(t=end - .5, who='robot', label='xong lượt/chuỗi'))
        return self.d


def lead(s, start, final, wrong, *, left=False, shape='grasp', robot_z=.85, both=False, hand_half=.065):
    """Hand-carried target + robot staging; error cases turn only after robot moves."""
    v = s.case
    target = lambda p: [p[0], p[1] + .06, robot_z]
    hh = lambda a, b, p: s.h(a, b, p, shape, left=left)
    if both:
        # The supplied final/wrong point is the centre; grasp the two ends.
        def hh(a, b, p):
            s.h(a, b, plus(p, [hand_half, 0, 0]), shape)
            s.h(a, b, plus(p, [-hand_half, 0, 0]), shape, left=True)
    s.cap(0, start, 'Storyboard dựng tay: robot và người chuẩn bị công việc; chưa có ManiCast hoặc policy.')
    if v in ('BASE', 'C1'):
        dt = 1.5 if v == 'BASE' else 1.8
        hh(start, start + dt, final)
        s.r(start + .2, start + dt + .05, target(final))
        s.cap(start, start + dt + .1, 'Cánh tay đang hướng tới đích; minh họa robot chuẩn bị đúng trước khi động tác người kết thúc.')
        s.decision(start + .2, start + dt)
        return start + dt + .1
    if v == 'C2':
        mid = [(final[0] + wrong[0]) / 2, min(final[1], wrong[1]) - .04, max(final[2], wrong[2]) + .05]
        hh(start, start + .7, mid)
        hh(start + 1.8, start + 2.8, final)
        s.r(start + .2, start + 1.0, [.22, .07, robot_z])
        s.r(start + 1.9, start + 2.85, target(final))
        s.cap(start, start + 1.8, 'Mơ hồ: tay dừng giữa hai hướng. Robot giữ vị trí chuẩn bị, chưa thao tác tại đích.')
        s.cap(start + 1.8, start + 2.9, 'Chuyển động tiếp theo làm rõ đích; robot mới chuyển tới vị trí phù hợp.')
        s.decision(start + .2, start + 1.8)
        return start + 2.9
    reveal = start + (1.35 if v == 'C3' else 1.65)
    first = plus(wrong, [0, -.08, .10]) if v == 'C3' else plus(wrong, [0, 0, .10])
    hh(start, start + 1.0, first)
    hh(reveal, reveal + 1.25, final)
    s.r(start + .2, start + 1.15, target(wrong))
    s.r(reveal + .1, reveal + .65, [.22, .08, robot_z + .03])
    s.r(reveal + .65, reveal + 1.4, target(final))
    s.cap(start, reveal, 'Dự báo minh họa nhầm hướng: robot đã tiến về A; người vẫn đang đi tới B.' if v == 'C3'
          else 'Người ban đầu hướng tới A. Robot đã tiến về A trước khi người đổi lựa chọn.')
    s.cap(reveal, reveal + 1.5, 'Lộ sai: robot ngừng đi A, chuyển sang B; người không đổi mục tiêu.' if v == 'C3'
          else 'Người đổi A sang B: robot rút khỏi hướng cũ rồi chuyển sang đích mới.')
    s.decision(start + .2, reveal)
    return reveal + 1.5


def pick(s, obj, pos, start, dest, *, dz=.035, release=True):
    """Pick a small object, transport high, place and release. Returns completion time."""
    s.r(start, start + .55, plus(pos, [0, 0, .14]), grip=.085)
    s.r(start + .55, start + .90, plus(pos, [0, 0, dz]))
    s.r(start + .90, start + 1.08, grip=.040)
    s.grab(start + 1.08, obj, 'robot')
    s.r(start + 1.08, start + 1.50, plus(pos, [0, 0, .17]))
    s.r(start + 1.50, start + 2.10, plus(dest, [0, 0, .17]))
    s.r(start + 2.10, start + 2.45, plus(dest, [0, 0, dz]))
    if release:
        s.r(start + 2.45, start + 2.60, grip=.085)
        s.drop(start + 2.60, obj, dest)
    s.r(start + 2.65, start + 3.0, plus(dest, [0, 0, .18]))
    return start + 3.0


def i1(v):
    s = Story('I1', 'Nhiều đế lót, vị trí liên tục', v, 'instructor')
    mugs = [[.16, .26, .62], [.27, .24, .62], [.38, .18, .62]]
    rng = random.Random(20261028 + list(CASES).index(v))
    positions = []
    for _ in range(20000):
        p = [round(rng.uniform(.14, .38), 4), round(rng.uniform(-.19, .06), 4), .624]
        if not .22 <= math.hypot(p[0], p[1]) <= .40:
            continue
        if any(math.dist(p[:2], q[:2]) < .115 for q in positions + mugs):
            continue
        if not positions and math.dist(p[:2], [.34, -.02]) < .14:
            continue
        positions.append(p)
        if len(positions) == 3:
            break
    assert len(positions) == 3, v
    s.d['sampled_coaster_positions'] = positions
    s.d['layout_seed'] = 20261028 + list(CASES).index(v)
    for j in range(3):
        s.obj(f'COASTER{j}', 'cylinder', [.49, -.27, .624 + j * .005], size=[.035, .002], color='cork')
        s.obj(f'CUP{j}', 'mesh', mugs[j], mesh='Cole_Hardware_Mug_Classic_Blue', scale=.50)
    t = 0
    for j, p in enumerate(positions):
        s.h(t + .15, t + .7, [.49, -.27, .66])
        s.grab(t + .7, f'COASTER{j}', 'human')
        s.r(t + .1, t + .6, plus(mugs[j], [0, 0, .12]), grip=.085)
        s.r(t + .6, t + .95, plus(mugs[j], [0, 0, .055]))
        s.r(t + .95, t + 1.12, grip=.045)
        s.grab(t + 1.12, f'CUP{j}', 'robot')
        s.r(t + 1.12, t + 1.6, plus(mugs[j], [0, 0, .24]))
        if j == 0:
            end = lead(s, t + 1.7, plus(p, [0, 0, .035]), [.34, -.02, .66])
        else:
            end = t + 3.2
            s.h(t + 1.7, end, plus(p, [0, 0, .035]))
            s.r(t + 1.8, end, plus(p, [0, .06, .23]))
            s.cap(t, end + .1, f'Lượt {j + 1}: người mang đế mới đến vị trí khác; robot xử lý đế còn trống theo thứ tự.')
        s.drop(end, f'COASTER{j}', p)
        s.h(end + .1, end + .8, REST, 'relaxed')
        s.r(end + .85, end + 1.3, plus(p, [0, 0, .12]))
        s.r(end + 1.3, end + 1.7, plus(p, [0, 0, .059]))
        s.r(end + 1.7, end + 1.9, grip=.085)
        s.drop(end + 1.9, f'CUP{j}', plus(p, [0, 0, .002]))
        s.r(end + 2.0, end + 2.45, plus(p, [0, 0, .20]))
        s.cap(end + .1, end + 2.5, 'Đế đã đặt, tay rời: robot hạ cốc lên đúng đế. Các đế trước đã có cốc vẫn giữ nguyên.')
        t = end + 2.6
    s.r(t, t + .65, HOME)
    return s.finish(t + 1)


def i2(v):
    s = Story('I2', 'Tay trái/phải chọn phía vật', v, 'instructor')
    for j, p in enumerate([[.11, .22, .644], [.36, .16, .644], [.17, .22, .644], [.40, .10, .644]]):
        s.obj(f'ITEM{j}', 'box', p, size=[.019, .019, .023], color='blue')
    s.obj('PERSON_TOOL', 'box', [.11, -.29, .64], size=[.025, .012, .015], color='orange')
    s.decor('receiving', 'open_box', [.26, -.20, .62], color='box', half=[.075, .045], height=.025, wall=.003)
    first_right = v in ('C3', 'C4', 'C1')
    final = [.47, -.15, .86] if first_right else [.12, -.15, .86]
    t = 1.0
    if v in ('BASE', 'C1'):
        s.h(t, t + 1.5, final, 'relaxed', left=not first_right)
        s.r(t + .2, t + 1.5, [.36 if first_right else .12, .12, .85])
        clear = t + 1.6
        s.decision(t + .2, t + 1.5)
        s.cap(0, clear, 'Người nâng đúng tay phía muốn lấy. Robot chuẩn bị cùng phía; các vật không có mẫu chỉ định.')
    elif v == 'C2':
        s.h(1, 1.7, [.40, -.28, .77], 'relaxed')
        s.h(1, 1.7, [.20, -.28, .77], 'relaxed', left=True)
        s.h(3.0, 4.0, final, 'relaxed', left=True)
        s.h(3.0, 3.6, REST, 'relaxed')
        s.r(1.3, 2.0, [.22, .10, .88])
        s.r(3.1, 4.05, [.12, .14, .85])
        clear = 4.1
        s.decision(1.3, 3.0)
        s.cap(0, 3.0, 'Hai tay cùng nhúc nhích rồi dừng: robot chưa chọn phía.')
        s.cap(3.0, clear, 'Chỉ tay trái tiếp tục đưa lên; robot chuyển sang cụm bên trái của người.')
    else:
        reveal = 2.8
        s.h(1, 2.1, [.11, -.28, .72] if v == 'C3' else [.12, -.15, .86], 'relaxed', left=True)
        if v == 'C3':
            s.grab(2.2, 'PERSON_TOOL', 'human_l')
            s.h(2.8, 3.5, [.16, -.32, .78], 'grasp', left=True)
        else:
            s.h(2.8, 3.5, LEFT_REST, 'relaxed', left=True)
        s.h(2.8, 4.0, final, 'relaxed')
        s.r(1.2, 2.3, [.12, .14, .85])
        s.r(2.9, 3.4, [.22, .10, .90])
        s.r(3.4, 4.1, [.36, .12, .85])
        clear = 4.2
        s.decision(1.2, reveal)
        s.cap(0, reveal, 'Robot nhầm tay trái lấy đồ riêng là hướng dẫn, đã đi trái.' if v == 'C3' else 'Người ra hiệu trái; robot đã đi. Sau đó người đổi sang tay phải.')
        s.cap(reveal, clear, 'Tay phải ra hiệu rõ: robot bỏ hướng trái và đi phía phải; đây là sửa sau khi đã hành động.')
    idx = 1 if first_right else 0
    p = s.d['objects'][f'ITEM{idx}']['pos']
    end = pick(s, f'ITEM{idx}', p, clear, [.23, -.20, .648])
    if v == 'C3':
        s.h(clear + .2, clear + .8, [.11, -.29, .675], left=True)
        s.drop(clear + .8, 'PERSON_TOOL', [.11, -.29, .64])
        s.h(clear + .8, clear + 1.3, LEFT_REST, 'relaxed', left=True)
    s.h(clear + .2, clear + .8, REST if first_right else LEFT_REST, 'relaxed', left=not first_right)
    s.cap(clear, end, 'Robot lấy một vật cùng loại ở phía đã được người chọn, đặt vào khay chung.')
    # A second independent instruction makes this a task, not a single pose demonstration.
    t = end + .3
    second_right = not first_right
    s.h(t, t + 1.2, [.47 if second_right else .12, -.15, .86], 'relaxed', left=not second_right)
    idx = 3 if second_right else 2
    end = pick(s, f'ITEM{idx}', s.d['objects'][f'ITEM{idx}']['pos'], t + .35, [.29, -.20, .648])
    s.h(t + 1.5, t + 2.1, REST if second_right else LEFT_REST, 'relaxed', left=not second_right)
    s.cap(t, end + .5, 'Lượt tiếp theo chọn tay còn lại. Đo đúng phía; forecast không nhận biết ngón tay hay tên vật.')
    s.r(end + .05, end + .7, HOME)
    return s.finish(end + 1.0)


def i3(v):
    base = yaml.safe_load((ROOT / 'configs/scenarios/intent_roles_revision_preview.yaml').read_text())['tasks']['I3']
    s = Story('I3', 'Ra hiệu dừng khay', v, 'instructor')
    for key in ('objects', 'decor'):
        s.d[key] = copy.deepcopy(base[key])
    s.d['events'] = [copy.deepcopy(e) for e in base['events'] if e['t'] <= 1.8]
    s.r(.4, 1, [.10, -.18, .82], .085)
    s.r(1, 1.5, [.10, -.18, .72])
    s.r(1.5, 1.8, grip=.035)
    if v in ('BASE', 'C1'):
        stop = 4.4 if v == 'BASE' else 4.8
        s.h(3.0, stop - .3, [.48, -.29, .92], 'relaxed')
        s.r(2.2, stop, [.36, -.18, .72])
        s.decision(3.0, stop - .3)
        s.cap(0, stop, 'Robot đưa khay dọc ray. Người đưa cả tay vào vùng ra hiệu; minh họa giảm tốc để dừng đúng lúc.')
    elif v == 'C2':
        stop = 5.6
        s.h(3.0, 3.5, [.48, -.34, .79], 'relaxed')
        s.h(4.4, 5.3, [.48, -.29, .92], 'relaxed')
        s.r(2.2, 3.8, [.23, -.18, .72])
        s.r(4.5, stop, [.36, -.18, .72])
        s.decision(3.0, 4.4)
        s.cap(0, 4.4, 'Tay mới nâng một phần rồi ngập ngừng. Robot giảm tốc, giữ giữa ray, chưa hiểu là dừng cuối.')
        s.cap(4.4, stop, 'Tín hiệu rõ hơn; robot tiếp tục một đoạn ngắn rồi dừng ở vị trí yêu cầu.')
    else:
        stop = 7.0
        s.h(2.8, 3.6, [.48, -.29, .92], 'relaxed')
        s.h(3.8, 4.5, [.44, -.35, .94] if v == 'C3' else REST, 'relaxed')
        s.h(5.5, 6.7, [.48, -.29, .92], 'relaxed')
        s.r(2.2, 3.7, [.23, -.18, .72])
        s.r(4.7, stop, [.36, -.18, .72])
        s.decision(2.8, 3.8)
        s.cap(0, 3.8, 'Robot dừng nhầm: người đưa tay chỉnh tay áo, không yêu cầu dừng.' if v == 'C3' else 'Người ra hiệu dừng; robot đã giảm tốc/dừng, rồi người hủy yêu cầu để đi tiếp.')
        s.cap(3.8, stop, 'Robot nhận ra tín hiệu bị bác bỏ và đưa khay đi tiếp; lần ra hiệu sau mới là điểm dừng cuối.')
    # Reuse the validated geometry, but shift the pickup after this case's final stop.
    for step in base['human']:
        if step['t'][0] >= 4.6:
            x = copy.deepcopy(step)
            x['t'] = [round(t + stop - 4.4, 3) for t in x['t']]
            s.d['human'].append(x)
    for step in base['robot']:
        if step['t'][0] >= 8.2:
            x = copy.deepcopy(step)
            x['t'] = [round(t + stop - 4.4, 3) for t in x['t']]
            s.d['robot'].append(x)
    for e in base['events']:
        if e['t'] > 1.8:
            x = copy.deepcopy(e)
            x['t'] = round(x['t'] + stop - 4.4, 3)
            s.d['events'].append(x)
    s.cap(stop, stop + 4.7, 'Sau khi khay đứng yên người mới lấy món; robot giữ khay. Timeline viết tay, chưa kiểm khả năng dừng bằng policy.')
    return s.finish(stop + 4.6)


def c1(v):
    s = Story('C1', 'T2: xong việc, nhận lon', v, 'collaborator')
    s.obj('CAN', 'can', [.20, .20, .655], color='red')
    s.obj('WP', 'pcb', [.32, -.26, .63])
    s.obj('SD', 'mesh', [.49, -.22, .62], mesh='Craftsman_Grip_Screwdriver_Phillips_Cushion', scale=.7, yaw=80)
    s.obj('OWN_CUP', 'mesh', [.40, -.19, .62], mesh='Room_Essentials_Mug_White_Yellow', scale=.45)
    s.decor('finished', 'open_box', [.12, -.26, .62], 'box', half=[.075, .055], height=.018, wall=.003)
    s.h(.1, .6, [.49, -.22, .66])
    s.grab(.6, 'SD', 'human')
    s.h(.6, 1.2, [.35, -.26, .74])
    s.h(.1, .6, [.27, -.26, .67], left=True)
    for j in range(3):
        s.h(1.2 + j * .5, 1.7 + j * .5, [.35 + (j % 2) * .02, -.26, .74])
    s.h(2.7, 3.2, [.49, -.22, .66])
    s.drop(3.2, 'SD', [.49, -.22, .62])
    s.h(3.2, 3.6, [.37, -.26, .67])
    s.grab(3.6, 'WP', 'human')
    s.h(3.6, 4.6, [.17, -.26, .71])
    s.h(3.6, 4.6, [.07, -.26, .71], left=True)
    s.drop(4.6, 'WP', [.12, -.26, .643])
    s.h(4.65, 5.2, LEFT_REST, 'relaxed', left=True)
    s.cap(0, 3.6, 'Người làm việc trên tấm. Chậm thao tác chưa được coi là bằng chứng sắp xong.')
    s.cap(3.6, 4.7, 'Hai tay chuyển tấm vào khay thành phẩm: chuyển pha quan sát được. Robot mới bắt đầu chuẩn bị lon.')
    s.r(3.8, 4.3, [.20, .20, .80], .085)
    s.r(4.3, 4.65, [.20, .20, .69])
    s.r(4.65, 4.85, grip=.043)
    s.grab(4.85, 'CAN', 'robot')
    s.r(4.85, 5.3, [.20, .12, .88])
    t = 5.3
    if v in ('BASE', 'C1'):
        ready = 6.7 if v == 'BASE' else 7.0
        s.h(t, ready, [.30, -.15, .78], 'offer', palm_up=1.)
        s.r(t + .2, ready + .15, [.30, -.135, .85])
        s.decision(t + .2, ready)
        s.cap(t, ready + .2, 'Tay rời khay thành phẩm và đưa ra nhận: robot tiến về điểm gặp dự kiến trong cửa sổ ngắn hạn.')
    elif v == 'C2':
        ready = 8.4
        s.h(t, 6.2, [.27, -.23, .75], 'relaxed')
        s.h(7.2, ready, [.30, -.15, .78], 'offer', palm_up=1.)
        s.r(5.5, 6.3, [.22, .03, .90])
        s.r(7.4, ready + .15, [.30, -.135, .85])
        s.decision(5.5, 7.2)
        s.cap(t, 7.2, 'Tay đưa ra một phần rồi dừng; robot giữ lon ở chỗ chờ, chưa áp vào tay người.')
        s.cap(7.2, ready + .2, 'Tay tiếp tục đưa ra nhận, robot mới tiếp cận điểm gặp rõ hơn.')
    else:
        ready = 11.3
        s.r(5.5, 6.5, [.30, -.135, .85])
        if v == 'C3':
            s.h(t, 6.6, [.40, -.19, .69], 'offer', palm_up=.5)
            s.grab(6.7, 'OWN_CUP', 'human')
            s.h(6.7, 7.4, [.41, -.31, .94])
            s.h(7.8, 8.5, [.40, -.19, .69])
            s.drop(8.5, 'OWN_CUP', [.40, -.19, .62])
        else:
            s.h(t, 6.6, [.30, -.15, .78], 'offer', palm_up=1.)
            s.h(6.8, 7.5, [.35, -.26, .72], 'relaxed')
            s.h(7.5, 8.4, [.32, -.26, .72], 'grasp')
        s.r(6.9, 7.6, [.20, .06, .94])
        s.h(9.7, ready, [.30, -.15, .78], 'offer', palm_up=1.)
        s.r(9.9, ready + .15, [.30, -.135, .85])
        s.decision(5.5, 6.8)
        s.cap(t, 6.8, 'Robot nhầm tay lấy cốc riêng là xin lon, đã bắt đầu đưa lon.' if v == 'C3' else 'Người đưa tay xin; robot đã đưa lon tới, người rút tay để làm tiếp.')
        s.cap(6.8, 9.7, 'Robot hủy tiếp cận, nâng lon về vùng chờ; không ép trao khi người chưa nhận.')
        s.cap(9.7, ready + .2, 'Người thật sự đưa tay nhận lại; robot tiếp cận điểm gặp mới.')
    s.h(ready + .2, ready + .5, [.30, -.15, .78], 'grasp', palm_up=1.)
    s.grab(ready + .5, 'CAN', 'human')
    s.r(ready + .5, ready + .7, grip=.085)
    s.h(ready + .5, ready + 1.2, [.33, -.23, .78])
    s.h(ready + 1.2, ready + 1.9, [.40, -.33, .90])
    s.r(ready + .8, ready + 1.7, HOME)
    s.cap(ready + .2, ready + 2.3, 'Người cầm và kéo lon, robot nhả. Giữ phạm vi T2 chỉ đưa lon; đây là storyboard, không đo claim.')
    return s.finish(ready + 2.3)


def c2(v):
    s = Story('C2', 'Nhấc giá, robot lau vùng dưới', v, 'collaborator')
    centers = [[.20, -.16, .623], [.36, -.035, .623]]
    for j, p in enumerate(centers):
        s.obj(f'RACK{j}', 'rack', p, half=[.085, .055])
        s.obj(f'DIRT{j}', 'cylinder', plus(p, [0, 0, -.001]), size=[.035, .0015], color='tan')
    s.obj('ROLLER', 'roller', [.12, .20, .635])
    s.obj('PERSON_TOWEL', 'box', [.09, -.27, .632], size=[.035, .025, .005], color='cloth')
    s.r(.1, .6, [.09, .20, .75], .085)
    s.r(.6, .9, [.09, .20, .69])
    s.r(.9, 1.1, grip=.035)
    s.grab(1.1, 'ROLLER', 'robot')
    s.r(1.1, 1.55, [.18, .15, .86])
    selected = 1 if v in ('C3', 'C4', 'C1') else 0
    p = centers[selected]
    end = lead(s, 1.7, plus(p, [0, 0, .07]), plus(centers[1 - selected], [0, 0, .07]), both=True, hand_half=.085)
    if v == 'C3':
        # A private towel reach beside A is not a request to lift A.
        for step in s.d['human']:
            if step['t'] == [1.7, 2.7]:
                step['to'] = [.09, -.27, .665]
        for step in s.d['human_l']:
            if step['t'] == [1.7, 2.7]:
                step['to'] = LEFT_REST
        s.grab(2.72, 'PERSON_TOWEL', 'human')
        s.drop(3.03, 'PERSON_TOWEL', [.09, -.27, .632])
    s.grab(end, f'RACK{selected}', 'human')
    held_center = [.40 if selected == 0 else .18, -.20, .94]
    s.h(end, end + .9, plus(held_center, [.085, 0, 0]))
    s.h(end, end + .9, plus(held_center, [-.085, 0, 0]), left=True)
    a = end + 1.0
    s.r(a, a + .5, [p[0] - .04, p[1], .678])
    for j, (dx, dy) in enumerate([(.02, 0), (-.04, .025), (.02, .025)]):
        s.r(a + .5 + j * .45, a + .95 + j * .45, [p[0] + dx, p[1] + dy, .678])
    s.drop(a + 1.8, f'DIRT{selected}', [0, 0, -1])
    s.r(a + 1.9, a + 2.5, [.22, .16, .87])
    s.h(a + 2.6, a + 3.5, plus(p, [.085, 0, .07]))
    s.h(a + 2.6, a + 3.5, plus(p, [-.085, 0, .07]), left=True)
    s.drop(a + 3.5, f'RACK{selected}', p)
    s.h(a + 3.6, a + 4.2, REST, 'relaxed')
    s.h(a + 3.6, a + 4.2, LEFT_REST, 'relaxed', left=True)
    s.cap(end, a + .05, 'Người thực sự nhấc giá bằng hai tay. Robot chỉ vào lau khi giá và tay đã rời vùng.')
    s.cap(a + .05, a + 2.5, 'Robot lau vùng vừa được giải phóng; người giữ giá. Phần dự báo sai chỉ chuẩn bị nhầm, chưa lau dưới giá.')
    s.cap(a + 2.5, a + 4.7, 'Robot rời vùng trước, người mới đặt giá lại. Hai giá tạo lựa chọn; bố cục 1–3 giá sẽ chọn sau pilot.')
    result = s.finish(a + 4.7)
    result['review_times'].insert(2, round(a + 1.1, 2))
    return result


def c3(v):
    s = Story('C3', 'Nhấc bát để người trải khăn', v, 'collaborator')
    centers = [[.15, -.02, .628], [.37, -.02, .628]]
    for j, p in enumerate(centers):
        s.obj(f'BOWL{j}', 'mesh', p, mesh='Cole_Hardware_Bowl_Scirocco_YellowBlue', scale=.60)
        s.decor(f'tray{j}', 'mesh', [p[0], -.03, .62], mesh='Threshold_Tray_Rectangle_Porcelain', scale=.66)
    s.obj('TOWEL', 'mesh', [.14, -.29, .62], mesh='Cole_Hardware_Dishtowel_Blue', scale=.4)
    s.obj('MAT', 'box', [0, 0, -1], size=[.068, .052, .002], color='cloth')
    s.h(.2, .8, [.20, -.29, .66])
    s.h(.2, .8, [.08, -.29, .66], left=True)
    s.grab(.8, 'TOWEL', 'human')
    selected = 1 if v in ('C1', 'C3', 'C4') else 0
    p = centers[selected]
    end = lead(s, 1.0, [p[0], -.16, .77], [centers[1 - selected][0], -.16, .77], both=True)
    a = end + .15
    s.r(a, a + .45, plus(p, [0, 0, .13]))
    s.r(a + .45, a + .8, plus(p, [0, 0, .035]))
    s.r(a + .8, a + 1, grip=.060)
    s.grab(a + 1, f'BOWL{selected}', 'robot')
    s.r(a + 1, a + 1.6, [p[0], .12, .85])
    # Replace the folded visual with a flat mat at the person's hands, then lay it down.
    s.drop(a + 1.6, 'TOWEL', [0, 0, -1])
    s.drop(a + 1.6, 'MAT', [p[0], -.16, .735])
    s.grab(a + 1.6, 'MAT', 'human')
    s.h(a + 1.6, a + 2.5, [p[0] + .06, -.03, .665])
    s.h(a + 1.6, a + 2.5, [p[0] - .06, -.03, .665], left=True)
    s.drop(a + 2.5, 'MAT', [p[0], -.03, .627])
    s.h(a + 2.8, a + 3.6, REST, 'relaxed')
    s.h(a + 2.8, a + 3.6, LEFT_REST, 'relaxed', left=True)
    s.r(a + 3.7, a + 4.25, [p[0], -.02, .82])
    s.r(a + 4.25, a + 4.7, [p[0], -.02, .666])
    s.r(a + 4.7, a + 4.9, grip=.085)
    s.drop(a + 4.9, f'BOWL{selected}', [p[0], -.02, .631])
    s.r(a + 5.0, a + 5.7, HOME)
    s.cap(end, a + 1.6, 'Khay đã rõ: robot nhấc bát một tay và giữ xa vùng người trải khăn.')
    s.cap(a + 1.6, a + 3.7, 'Người trải khăn rồi rút hai tay; robot tiếp tục giữ bát, không hạ khi còn tay.')
    s.cap(a + 3.7, a + 6, 'Robot đặt bát lại trên khăn. C3/C4 đã thể hiện đi nhầm và sửa trước khi gắp.')
    return s.finish(a + 6)


def c4(v):
    s = Story('C4', 'Mở ngăn kéo để người cất đồ', v, 'collaborator')
    s.d['cameras'] = dict(side=dict(pos=[.95, -.65, 1.25], target=[.27, .015, .76], fovy=52))
    pos = [[.17, .13, .632], [.37, .13, .632]]
    for j, p in enumerate(pos):
        s.obj(f'DRAWER{j}', 'drawer', p, half=[.072, .085], height=.07, color='wood')
        # A cabinet frame open toward -Y: side walls, rear, top and base.
        for side, dx, dy, dz, size in [('L', -.082, 0, .06, [.007, .105, .06]),
                                       ('R', .082, 0, .06, [.007, .105, .06]),
                                       ('back', 0, .104, .06, [.082, .006, .06]),
                                       ('top', 0, 0, .12, [.089, .11, .006]),
                                       ('base', 0, 0, -.006, [.089, .11, .006])]:
            s.decor(f'cab{j}_{side}', 'box', plus(p, [dx, dy, dz]), 'wood', size=size)
    s.obj('ITEM', 'pcb', [.43, -.29, .634])
    s.obj('OWN_PEN', 'box', [.17, -.11, .64], size=[.03, .007, .008], color='orange')
    s.h(.15, .8, [.43, -.29, .67])
    s.grab(.8, 'ITEM', 'human')
    selected = 1 if v in ('C1', 'C3', 'C4') else 0
    wrong = 1 - selected

    def open_drawer(idx, t):
        p = pos[idx]
        handle = [p[0], .031, .735]
        s.r(t, t + .55, plus(handle, [0, -.03, .10]), .085)
        s.r(t + .55, t + .9, handle)
        s.r(t + .9, t + 1.1, grip=.037)
        s.grab(t + 1.1, f'DRAWER{idx}', 'robot')
        s.r(t + 1.1, t + 2.0, plus(handle, [0, -.14, 0]))
        return t + 2.0

    if v in ('C3', 'C4'):
        s.h(1, 2.1, [.17 + .04, -.12, .77])
        if v == 'C3':
            s.h(1.0, 2.05, [.17, -.11, .68], left=True)
            s.grab(2.1, 'OWN_PEN', 'human_l')
            s.h(2.2, 3.1, LEFT_REST, 'grasp', left=True)
        opened = open_drawer(wrong, 1.25)
        reveal = opened + .2
        s.h(reveal, reveal + .9, [.37, -.11, .78])
        s.r(reveal + .1, reveal + .9, [pos[wrong][0], .031, .735])
        s.drop(reveal + .9, f'DRAWER{wrong}', pos[wrong])
        s.r(reveal + .9, reveal + 1.1, grip=.085)
        opened = open_drawer(selected, reveal + 1.2)
        s.decision(2.35, reveal)
        s.cap(0, reveal, 'Robot nhầm hướng tay đi qua A là cần ngăn A, đã kéo A ra.' if v == 'C3' else 'Người hướng tới ngăn A, robot đã mở; người đổi ý sang ngăn B.')
        s.cap(reveal, opened, 'Robot đóng ngăn sai, chuyển sang tay nắm khác và mở ngăn đúng: chi phí sửa sai nhìn thấy được.')
    else:
        if v == 'C2':
            s.h(1, 1.7, [.27, -.16, .77])
            s.h(3, 4, [pos[selected][0], -.10, .78])
            s.r(1.2, 2, [.25, -.04, .87])
            t = 3.1
            s.decision(1.2, 3)
            s.cap(0, 3.1, 'Người giữ đồ giữa hai ngăn, còn lưỡng lự. Robot chờ ở cao, chưa kéo ngăn.')
        else:
            s.h(1, 2.5, [pos[selected][0], -.10, .78])
            t = 1.25 if v == 'BASE' else 1.5
            s.decision(t, 2.5)
            s.cap(0, 2.5, 'Tay mang đồ hướng về một ngăn: robot tới tay nắm và bắt đầu chuẩn bị mở đúng ngăn.')
        opened = open_drawer(selected, t)
    p = pos[selected]
    s.r(opened, opened + .2, grip=.085)
    s.drop(opened + .2, f'DRAWER{selected}', plus(p, [0, -.14, 0]))
    s.r(opened + .2, opened + .75, [p[0], -.15, .88])
    s.h(opened + .85, opened + 1.6, [p[0], -.01, .75])
    s.drop(opened + 1.6, 'ITEM', [p[0], -.01, .645])
    s.grab(opened + 1.6, 'ITEM', f'DRAWER{selected}')
    s.h(opened + 1.8, opened + 2.5, REST, 'relaxed')
    s.r(opened + 2.6, opened + 3.1, [p[0], -.109, .735])
    s.r(opened + 3.1, opened + 3.3, grip=.037)
    s.grab(opened + 3.3, f'DRAWER{selected}', 'robot')
    s.r(opened + 3.3, opened + 4.2, [p[0], .031, .735])
    s.drop(opened + 4.2, f'DRAWER{selected}', p)
    s.r(opened + 4.2, opened + 4.4, grip=.085)
    s.r(opened + 4.4, opened + 5, HOME)
    s.cap(opened, opened + 2.6, 'Ngăn mở, robot rời miệng ngăn; người đặt đồ vào rồi rút tay.')
    s.cap(opened + 2.6, opened + 5.4, 'Tay đã rời: robot đóng ngăn chứa đồ. Ray, gắp và tiếp xúc là minh họa động học.')
    result = s.finish(opened + 5.4)
    result['review_times'].insert(2, round(opened + 1.3, 2))
    return result


def human_pick(s, obj, p, a, dest, *, left=False):
    s.h(a, a + .65, plus(p, [0, 0, .035]), left=left)
    s.grab(a + .65, obj, 'human_l' if left else 'human')
    s.h(a + .65, a + 1.1, plus(p, [0, 0, .16]), left=left)
    s.h(a + 1.1, a + 1.9, plus(dest, [0, 0, .08]), left=left)
    s.drop(a + 1.9, obj, dest)
    s.h(a + 2.0, a + 2.5, LEFT_REST if left else REST, 'relaxed', left=left)


def cowork_prefix(s, v, *, start=1.0, long=False):
    """Robot moves into an initially plausible path, then yields before the human arrives."""
    if v in ('BASE', 'C1'):
        s.r(start, start + .7, [.12, .18, .86])
        clear = start + 2.6
        s.decision(start, start + .9)
        s.cap(0, clear, 'Người làm việc riêng; robot chuẩn bị ở phía không giao cắt với quỹ đạo tay đang hình thành.')
    elif v == 'C2':
        s.r(start, start + .7, [.20, .10, .90])
        clear = start + 3.8
        s.decision(start, start + 2.0)
        s.cap(0, clear, 'Tay người ngập ngừng ở vùng chung. Robot giảm tốc/giữ vị trí, không ép đi qua khi hướng chưa rõ.')
    else:
        s.r(start, start + .9, [.30, -.03, .86])
        reveal = start + 1.1
        s.r(reveal + .05, reveal + .85, [.12, .16, .95])
        clear = start + (4.0 if long else 3.3)
        s.decision(start, reveal)
        s.cap(0, reveal, 'Robot đã tiến vào đường dự kiến trống; dự báo minh họa sai vùng tay/vật sẽ đi qua.' if v == 'C3'
              else 'Robot đã đi theo hướng người đang để trống; người đổi đường và đi vào vùng đó.')
        s.cap(reveal, clear, 'Robot dừng tiếp cận và nhường trước khi tay/vật tới. Người tiếp tục việc riêng, không phải né robot.')
    return clear


def w1(v):
    s = Story('W1', 'Hai hộp, đồ lộn xộn và nhường vùng', v, 'cowork')
    robot_pos = [[.20, -.08, .644], [.34, -.06, .644], [.12, .17, .644], [.38, .13, .644]]
    person_pos = [[.29, -.12, .644], [.18, .045, .644], [.35, .035, .644], [.28, .12, .644]]
    for j, p in enumerate(robot_pos):
        s.obj(f'R{j}', 'box', p, size=[.017, .017, .023], color='blue')
    for j, p in enumerate(person_pos):
        s.obj(f'H{j}', 'cylinder', p, size=[.018, .021], color='orange')
    s.obj('RBOX', 'tray', [.09, .28, .62], half=[.065, .045], height=.055, color='box')
    s.obj('HBOX', 'tray', [.47, -.25, .62], half=[.065, .045], height=.055, color='box2')
    s.obj('RLID', 'box', [.08, .37, .626], size=[.069, .049, .004], color='grey')
    s.obj('HLID', 'box', [.47, -.35, .626], size=[.069, .049, .004], color='grey')
    clear = cowork_prefix(s, v)
    if v == 'C2':
        s.h(.5, 1.3, [.29, -.20, .78], 'relaxed')
        first = 2.7
    elif v in ('C3', 'C4'):
        s.h(.4, 1.2, [.32, -.19, .76] if v == 'C3' else [.18, .045, .75], 'relaxed')
        s.h(2.1, 3.0, [.28, -.07, .78], 'relaxed')
        first = 3.1
    else:
        first = .6
    for j, p in enumerate(person_pos):
        human_pick(s, f'H{j}', p, first + j * 3.0, [.45 + (j % 2) * .035, -.26 + (j // 2) * .03, .65])
    # Two safe alternatives first, then a genuine wait when the remaining cluster is occupied.
    a = clear
    for k, j in enumerate([2, 3]):
        a = pick(s, f'R{j}', robot_pos[j], a, [.07 + k * .035, .27, .65]) + .1
    wait = a
    s.r(wait, wait + .6, [.14, .12, .91])
    # The left arm holds/works over the last cluster while the right arm finishes its own packing.
    s.h(wait - .4, wait + .4, [.24, -.055, .77], 'grasp', left=True)
    s.h(wait + 2.2, wait + 3, LEFT_REST, 'relaxed', left=True)
    s.cap(clear, wait, 'Robot lấy trước hai món xanh ở vùng khác; người độc lập xếp đồ cam vào hộp riêng.')
    s.cap(wait, wait + 3.1, 'Các món xanh còn lại ở vùng tay người đang chiếm: robot chờ ổn định, không cố lách hoặc tạo vật thay thế.')
    a = wait + 3.15
    for k, j in enumerate([0, 1]):
        a = pick(s, f'R{j}', robot_pos[j], a, [.07 + k * .035, .30, .65]) + .1
    # Close both boxes to show completion of both independent workflows.
    a = pick(s, 'RLID', [.08, .37, .626], a, [.09, .28, .678], dz=.014)
    h_close = max(first + 12.3, wait + 3.3)
    human_pick(s, 'HLID', [.47, -.35, .626], h_close, [.47, -.25, .678])
    end = max(a, h_close + 2.5)
    s.r(end, end + .6, HOME)
    s.cap(wait + 3.1, end + 1, 'Tay đã rút, robot tiếp tục các món còn lại rồi đóng hộp. Đánh giá an toàn, mượt và mức cản người; không ép nhanh bằng mọi giá.')
    result = s.finish(end + 1)
    result['review_times'].insert(2, round(wait + 1.2, 2))
    return result


def w2(v):
    s = Story('W2', 'Nhường người mang khay lớn', v, 'cowork')
    s.obj('BIG_TRAY', 'tray', [.18, -.26, .633], half=[.125, .065], height=.035, color='box2')
    for j, dx in enumerate([-.06, 0, .06]):
        s.obj(f'H_ITEM{j}', 'box', [.18 + dx, -.26, .658], size=[.021, .021, .020], color='orange')
        s.grab(.05, f'H_ITEM{j}', 'BIG_TRAY')
    robot_pos = [[.12, .20, .642], [.23, .13, .642], [.35, .08, .642]]
    s.decor('r_box', 'open_box', [.10, .32, .62], 'box', half=[.07, .045], height=.04, wall=.003)
    for j, p in enumerate(robot_pos):
        s.obj(f'PART{j}', 'box', p, size=[.018, .018, .021], color='blue')
    s.h(.2, .8, [.30, -.26, .67])
    s.h(.2, .8, [.06, -.26, .67], left=True)
    s.grab(.8, 'BIG_TRAY', 'human')
    s.h(.8, 1.5, [.30, -.20, .87])
    s.h(.8, 1.5, [.06, -.20, .87], left=True)
    clear = cowork_prefix(s, v, start=1.6, long=True)
    if v == 'C2':
        points = [(1.5, 2.3, [.25, -.17, .87]), (3.8, 4.8, [.28, .015, .87])]
    elif v == 'C3':
        points = [(1.5, 2.5, [.22, -.15, .87]), (2.8, 3.7, [.28, .015, .87])]
    elif v == 'C4':
        points = [(1.5, 2.5, [.34, -.17, .87]), (2.8, 3.7, [.28, .015, .87])]
    else:
        points = [(1.5, 3.0, [.28, .015, .87])]
    for a, b, p in points:
        s.h(a, b, plus(p, [.12, 0, 0]))
        s.h(a, b, plus(p, [-.12, 0, 0]), left=True)
    last = points[-1][1]
    s.h(last + .2, last + 1.4, [.53, -.25, .67])
    s.h(last + .2, last + 1.4, [.29, -.25, .67], left=True)
    s.drop(last + 1.4, 'BIG_TRAY', [.41, -.25, .633])
    s.h(last + 1.5, last + 2.2, REST, 'relaxed')
    s.h(last + 1.5, last + 2.2, LEFT_REST, 'relaxed', left=True)
    # Separate sorting by the human continues after carrying the tray.
    human_pick(s, 'H_ITEM0', [.35, -.25, .658], last + 2.4, [.47, -.34, .65])
    a = max(clear, last + 1.5)
    for j, p in enumerate(robot_pos):
        a = pick(s, f'PART{j}', p, a, [.075 + j * .025, .32, .65]) + .1
    s.r(a, a + .6, HOME)
    s.cap(max(clear, last + 1.5), a + 1, 'Robot tiếp tục việc riêng sau khi cả tay và khay lớn đã rời đường. Ảnh cung cấp hình học vật; forecast tay không tự cho biên khay.')
    return s.finish(a + 1)


def w3(v):
    s = Story('W3', 'Người lắp ráp lâu, robot làm việc cạnh bên', v, 'cowork')
    s.obj('PCB', 'pcb', [.31, -.18, .634])
    s.obj('SD', 'mesh', [.48, -.24, .62], mesh='Craftsman_Grip_Screwdriver_Phillips_Cushion', scale=.7, yaw=80)
    pos = [[.12, .22, .643], [.34, .17, .643], [.21, -.035, .643]]
    for j, p in enumerate(pos):
        s.obj(f'R_PART{j}', 'box', p, size=[.019, .019, .022], color='blue')
    s.decor('bin', 'open_box', [.10, .33, .62], 'box', half=[.07, .045], height=.04, wall=.003)
    s.decor('screw_bin', 'open_box', [.27, -.055, .62], 'box2', half=[.025, .025], height=.018, wall=.002)
    s.decor('finished_board', 'open_box', [.47, -.29, .62], 'box2', half=[.065, .045], height=.02, wall=.002)
    s.h(.2, .8, [.48, -.24, .66])
    s.grab(.8, 'SD', 'human')
    s.h(.8, 1.4, [.33, -.18, .75])
    s.h(.2, .8, [.26, -.18, .68], left=True)
    if v == 'C2':
        s.h(1.0, 1.7, [.23, -.22, .77], 'relaxed', left=True)
        s.h(3.0, 3.8, [.27, -.055, .73], 'grasp', left=True)
        s.h(4.1, 4.8, [.26, -.18, .68], 'grasp', left=True)
    elif v == 'C3':
        # A reach for fasteners is mistaken for withdrawing from the shared zone.
        s.h(1.0, 1.8, [.23, -.16, .77], 'relaxed', left=True)
        s.h(2.1, 3.0, [.27, -.055, .73], 'grasp', left=True)
        s.h(3.3, 4.0, [.26, -.18, .68], 'grasp', left=True)
    elif v == 'C4':
        # The person really starts clearing the area, then returns after the robot has moved.
        s.h(1.0, 1.8, [.20, -.29, .77], 'relaxed', left=True)
        s.h(2.1, 3.0, [.27, -.055, .73], 'grasp', left=True)
        s.h(3.3, 4.0, [.26, -.18, .68], 'grasp', left=True)
    clear = cowork_prefix(s, v, long=True)
    a = clear
    for j in [0, 1]:
        a = pick(s, f'R_PART{j}', pos[j], a, [.075 + j * .035, .32, .65]) + .1
    # Long occupancy: human assembly and a sustained reach for fasteners.
    for j in range(4):
        t = 1.4 + j * 1.0
        s.h(t, t + .65, [.33 + (j % 2) * .015, -.18, .75])
    s.h(5.5, 6.2, [.48, -.24, .66])
    s.drop(6.2, 'SD', [.48, -.24, .62])
    wait = max(a, 7.0)
    s.h(6.3, 7.1, [.27, -.055, .73])
    s.h(wait + 2.0, wait + 2.8, [.33, -.18, .74])
    s.r(wait, wait + .6, [.14, .12, .91])
    s.cap(clear, wait, 'Người tiếp tục lắp tấm; robot phân loại hai chi tiết ở vùng bên cạnh, không tiến-lùi quanh tay người.')
    s.cap(wait, wait + 2.9, 'Món cuối nằm sát vùng người đang lấy linh kiện: robot giữ chỗ chờ ổn định, không lặp tiếp cận rồi giật lùi.')
    # The eventual withdrawal has distinct interpretations in C2/C3/C4.
    t = wait + 2.9
    s.h(t, t + .7, [.36, -.18, .674])
    s.h(t, t + .7, [.26, -.18, .674], left=True)
    s.grab(t + .7, 'PCB', 'human')
    s.h(t + .7, t + 1.2, [.36, -.18, .82])
    s.h(t + .7, t + 1.2, [.26, -.18, .82], left=True)
    s.h(t + 1.2, t + 2.2, [.52, -.29, .68])
    s.h(t + 1.2, t + 2.2, [.42, -.29, .68], left=True)
    s.drop(t + 2.2, 'PCB', [.47, -.29, .64])
    s.h(t + 2.3, t + 3.0, REST, 'relaxed')
    s.h(t + 2.3, t + 3.0, LEFT_REST, 'relaxed', left=True)
    end = pick(s, 'R_PART2', pos[2], t + 3.1, [.10, .35, .65])
    s.r(end, end + .6, HOME)
    s.cap(t, end + 1, 'Người hoàn tất và chuyển tấm đi. Robot tiếp tục món còn lại sau khi tay rút; hai phần việc kết thúc độc lập.')
    result = s.finish(end + 1)
    result['review_times'].insert(2, round(wait + 1.2, 2))
    return result


BUILDERS = [i1, i2, i3, c1, c2, c3, c4, w1, w2, w3]


def vary_correct_episode(task):
    """A separate successful episode, with slower human/robot timing than the workflow overview."""
    warp = lambda t: round(float(t) * 1.08 + .15, 4)
    for key in ('human', 'human_l', 'robot'):
        for step in task.get(key, []):
            step['t'] = [warp(t) for t in step['t']]
    for key in ('events', 'marks'):
        for item in task[key]:
            item['t'] = warp(item['t'])
    for caption in task['captions']:
        caption[:2] = [warp(t) for t in caption[:2]]
    for key in ('action_start', 'reveal_t'):
        if key in task.get('decision', {}):
            task['decision'][key] = warp(task['decision'][key])
    task['duration'] = warp(task['duration'])
    task['review_times'] = [warp(t) for t in task['review_times']]


def main():
    base = yaml.safe_load((ROOT / 'configs/scenarios/habit_motion_candidates_preview.yaml').read_text())
    base['colors'].update(cork=[.63, .45, .28, 1], rail=[.24, .28, .32, 1])
    rail = yaml.safe_load((ROOT / 'configs/scenarios/intent_roles_revision_preview.yaml').read_text())
    base['colors'].update(rail['colors'])
    base['fps'] = 15
    base['out_dir'] = 'outputs/intent_review_50'
    base['tasks'] = {}
    for builder in BUILDERS:
        for case in CASES:
            task = builder(case)
            if case == 'C1':
                vary_correct_episode(task)
            key = f'{builder.__name__.upper()}_{case}'
            base['tasks'][key] = task
    path = ROOT / 'configs/scenarios/intent_review_50.yaml'
    path.write_text('# 10 workflows + 40 C1-C4 cases. Kinematic review only; docs/PIF.md §8.6.\n'
                    + yaml.safe_dump(base, allow_unicode=True, sort_keys=False, width=125))
    print(f'{path}: {len(base["tasks"])} storyboards')


if __name__ == '__main__':
    main()
