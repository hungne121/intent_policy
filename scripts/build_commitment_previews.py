"""Ten review videos, with explicit preparation and matched commitment chapters.

Kinematic storyboards only. The chapter branches are not policy or forecast results.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from intent_policy.sim.base_env import ROOT
from scripts.build_intent_review_previews import Story, HOME, REST, LEFT_REST, plus, pick, human_pick, i1, i3, c1, c2, c3, c4, w1, w2
from scripts.build_natural_case_previews import rack_cases

LEGEND = 'Duyệt kịch bản: robot chạy timeline; chưa phải ACT, chưa kiểm tiếp xúc/va chạm hoặc lợi thế thời gian.'


def scene_story(code, name, role, template):
    s = Story(code, name, 'BASE', role)
    for k in ('objects', 'decor', 'cameras'):
        if k in template:
            s.d[k] = copy.deepcopy(template[k])
    return s


def done(s, end, title, chapter, question, times, *, cue=None, probes=()):
    d = s.finish(end)
    d['title'] = title
    d['chapter'] = chapter
    d['review_question'] = question
    d['annotation_legend'] = LEGEND
    d['review_times'] = times
    d.pop('decision', None)
    d['marks'] = []
    d['state_probes'] = [dict(t=t, label=label) for t, label in probes]
    if cue is not None:
        d['cost_review'] = dict(cue_t=cue, scope='scripted_path_and_object_state_only')
        d['marks'].append(dict(t=cue, who='human', label='đổi sang B'))
    d['evaluation_only'] = dict(initial_robot_gripper_empty=True, annotations_are_policy_inputs=False)
    if s.code in ('C3', 'C4'):
        d['evaluation_only']['human_goal_events'] = [dict(t=0, goal='A')]
        if cue is not None:
            d['evaluation_only']['human_goal_events'].append(dict(t=cue, goal='B'))
        d['evaluation_only']['policy_observes_goal_log'] = False
    return d


def coasters():
    base = i1('BASE')
    s = scene_story('I1', 'Đặt lần lượt ba đế lót', 'instructor', base)
    for j, p in enumerate(base['sampled_coaster_positions']):
        t = j * 7.0
        cup = base['objects'][f'CUP{j}']['pos']
        s.h(t + .2, t + .9, [.49, -.27, .66])
        s.grab(t + .9, f'COASTER{j}', 'human')
        s.h(t + 1.3, t + 3.9, plus(p, [0, 0, .035]))
        s.drop(t + 3.9, f'COASTER{j}', p)
        s.h(t + 4.0, t + 4.8, REST, 'relaxed')
        s.r(t + 1.55, t + 2.1, plus(cup, [0, 0, .14]), .085)
        s.r(t + 2.1, t + 2.5, plus(cup, [0, 0, .055]))
        s.r(t + 2.5, t + 2.7, grip=.045)
        s.grab(t + 2.7, f'CUP{j}', 'robot')
        s.r(t + 2.7, t + 3.2, plus(cup, [0, 0, .23]))
        s.r(t + 3.2, t + 4.25, plus(p, [0, .04, .23]))
        s.r(t + 4.9, t + 5.3, plus(p, [0, 0, .13]))
        s.r(t + 5.3, t + 5.7, plus(p, [0, 0, .059]))
        s.r(t + 5.7, t + 5.9, grip=.085)
        s.drop(t + 5.9, f'CUP{j}', plus(p, [0, 0, .002]))
        s.r(t + 6.0, t + 6.6, HOME)
        s.cap(t, t + 1.3, f'Lượt {j + 1}: người lấy đế mới. Robot tay trống; cốc còn ở kho phía robot.')
        s.cap(t + 1.3, t + 3.9, 'Tay mang đế đang hình thành hướng đặt. Robot bắt đầu tiếp cận, gắp và vận chuyển cốc.')
        s.cap(t + 3.9, t + 7, 'Đế đã nằm ổn định, tay rút rồi robot mới đặt cốc. Đích liên tục, không có ô đích đánh dấu sẵn.')
    return done(s, 21.0, 'I1 | Ba đế liên tiếp, robot tự lấy từng cốc', 'Ba vị trí đặt',
                'Có thể bắt đầu vận chuyển cốc theo hướng tay trước khi đế được đặt xong không?', [0.5, 2.5, 4.3, 12.7, 20.0])


def sides():
    s = Story('I2', 'Tay trái/phải chọn phía lấy vật', 'BASE', 'instructor')
    pp = [[.11, .22, .644], [.36, .16, .644], [.17, .22, .644], [.40, .10, .644]]
    for j, p in enumerate(pp):
        s.obj(f'ITEM{j}', 'box', p, size=[.019, .019, .023], color='blue')
    s.decor('out', 'open_box', [.26, -.20, .62], 'box', half=[.075, .045], height=.025, wall=.003)
    for turn, idx in enumerate([0, 1]):
        t = turn * 5.4
        left = idx == 0
        s.h(t + .7, t + 2.3, [.12 if left else .47, -.15, .86], 'relaxed', left=left)
        end = pick(s, f'ITEM{idx}', pp[idx], t + 1.0, [.23 + turn * .06, -.20, .648])
        s.h(t + 2.5, t + 3.2, LEFT_REST if left else REST, 'relaxed', left=left)
        s.r(end + .1, end + .7, HOME)
        s.cap(t, t + 1.0, f'Lượt {turn + 1}: robot tay trống. Người bắt đầu đưa tay {"trái" if left else "phải"}; không có vật mẫu.')
        s.cap(t + 1.0, t + 2.5, 'Robot bắt đầu đi tới cụm tương ứng rồi mới gắp. Vật chưa được cầm sẵn trước yêu cầu.')
        s.cap(t + 2.5, t + 5.4, 'Vật được đưa vào khay chung. Đây là đối chứng đơn giản: ảnh hiện tại có thể đã đủ nhận bên hoạt động.')
    return done(s, 10.8, 'I2 | Chọn phía bằng tay đang đưa ra', 'Hai yêu cầu liên tiếp',
                'Forecast có thêm lợi ích so với nhận bên tay đang hoạt động từ ảnh không?', [.4, 1.7, 3.7, 7.1, 9.4])


def receiving_tray(x):
    base = i3('BASE')
    s = scene_story('I3', 'Khay trên ray tới tay đón', 'instructor', base)
    s.d['decor'] = [d for d in s.d['decor'] if d['name'] != 'signal_zone']
    s.d['events'] = [copy.deepcopy(e) for e in base['events'] if e['t'] < .1]
    s.h(.8, 2.3, [.29, -.32, .85], 'relaxed')
    s.h(2.3, 3.8, [x, -.30, .84], 'relaxed')
    s.r(1.05, 1.65, [.10, -.18, .82], .085)
    s.r(1.65, 2.15, [.10, -.18, .72])
    s.r(2.15, 2.4, grip=.035)
    s.grab(2.4, 'HANDLE', 'robot')
    s.r(2.7, 4.3, [x, -.18, .72])
    s.h(4.6, 5.2, [x - .015, -.285, .71])
    s.grab(5.2, 'DONUT', 'human')
    s.h(5.2, 5.8, [x - .015, -.285, .82])
    s.h(5.8, 6.6, [.50, -.27, .69])
    s.drop(6.6, 'DONUT', [.50, -.27, .633])
    s.h(6.7, 7.3, REST, 'relaxed')
    s.r(7.4, 7.7, grip=.085)
    s.drop(7.7, 'HANDLE', [x, -.18, .702])
    s.r(7.8, 8.5, HOME)
    s.cap(0, 2.3, 'Người đưa tay đón tới vùng lấy đồ; robot mới tiếp cận và nắm tay cầm khay ở đầu ray.')
    s.cap(2.3, 4.3, 'Tay đón dịch tới vị trí muốn lấy. Robot kéo khay theo ray tới đó; không dùng cử chỉ chỉ ngón hoặc ra hiệu dừng.')
    s.cap(4.3, 6.7, 'Khay đã dừng, người mới đưa tay xuống lấy món. Vị trí tay dừng là đại lượng cần dự báo và kiểm.')
    s.cap(6.7, 8.8, 'Người cất món riêng, robot nhả tay cầm. Chương khác dùng cùng scene nhưng vị trí đón xa hơn.')
    return done(s, 8.8, 'I3 | Đưa khay trên ray tới vị trí tay đón', 'Đón gần' if x < .3 else 'Đón xa',
                'Có thể chuẩn bị vận chuyển và dừng khay theo vị trí tay sắp ổn định không?', [.5, 1.9, 3.4, 5.4, 8.0])


def handover():
    t = c1('BASE')
    t.pop('decision', None)
    t['title'] = 'C1 / T2 | Tự lấy lon khi người chuyển pha công việc'
    t['chapter'] = 'Từ lắp tấm tới nhận lon'
    t['review_question'] = 'Chuyển động chuyển tấm và tay rời khay có báo sẵn sàng nhận đủ sớm để bắt đầu lấy lon không?'
    t['annotation_legend'] = LEGEND
    t['review_times'] = [.7, 3.0, 4.2, 5.0, 6.2, 8.4]
    t['marks'] = [dict(t=3.6, who='human', label='chuyển tấm'), dict(t=3.8, who='robot', label='bắt đầu lấy lon')]
    t['captions'] = [[0, 3.6, 'Robot tay trống; lon ở kho. Người lấy tua-vít, làm trên tấm rồi cất dụng cụ.'],
                     [3.6, 4.85, 'Hai tay chuyển tấm vào khay thành phẩm: robot mới đi tới lon, hạ tay và gắp.'],
                     [4.85, 6.9, 'Robot nâng và mang lon; tay người rời khay rồi đưa ra nhận. Cơ hội hỗ trợ sớm nằm trên cả đoạn chuẩn bị này.'],
                     [6.9, 9.1, 'Người nắm lon, robot nhả và rút tay. T2 kiểm timing/handover, không gán chi phí cao cho việc chờ với lon.']]
    return t


def rack(adjust):
    t = rack_cases()['RACK_GRIP_B' if adjust else 'RACK_CLEAR']
    # The human starts approaching the rack at 1.7; picking the tool starts afterwards.
    for st in t['robot']:
        if st['t'][0] < 1.6:
            st['t'] = [round(x + 1.75, 3) for x in st['t']]
    t['robot'] = [st for st in t['robot'] if st['t'][0] != 1.9]
    t['robot'].append(dict(t=[3.35, 4.15], to=[.20, -.10, .85]))
    for e in t['events']:
        if e.get('grab') == 'ROLLER': e['t'] += 1.75
    t['robot'].sort(key=lambda st: st['t'][0])
    lift = 6.5 if adjust else 3.3
    t['title'] = 'C2 | Tự lấy con lăn, chuẩn bị lau dưới giá'
    t['chapter'] = 'Chỉnh tay nắm rồi mới nâng' if adjust else 'Nâng ngay sau tiếp cận'
    t['review_question'] = 'Chuẩn bị dụng cụ/vùng lau có giảm thời gian người phải giữ giá mà không vào vùng quá sớm không?'
    t['annotation_legend'] = LEGEND
    t.pop('decision', None)
    t.pop('ambiguity_pair', None)
    t['marks'] = [dict(t=1.7, who='human', label='hai tay tới giá'), dict(t=1.85, who='robot', label='lấy con lăn'), dict(t=lift, who='human', label='bắt đầu nâng')]
    t['captions'] = [[0, 1.7, 'Con lăn còn nằm ở kho; robot tay trống. Hai giá tạo lựa chọn vùng làm việc.'],
                     [1.7, 3.3, 'Hai tay người tới giá A; robot bắt đầu lấy và nâng con lăn. Chưa được lau khi giá vẫn trên bàn.'],
                     [3.3, lift + .9, 'Người chỉnh tay nắm, giá chưa rời bàn; robot đã chuẩn bị nhưng chưa vào lau.' if adjust else 'Người nhấc và giữ giá sang bên; robot hoàn tất lấy dụng cụ và tiếp cận vùng lau.'],
                     [lift + .9, lift + 3.5, 'Vùng dưới giá đã trống: robot lau, người giữ giá. Kiểm thời gian giữ giá và điều kiện vùng trống.'],
                     [lift + 3.5, t['duration'] + .1, 'Robot rời vùng trước, người mới đặt giá lại. Dụng cụ chung giúp timing, không tự tạo chi phí chọn sai cao.']]
    t['review_times'] = [.6, 2.4, 3.1, lift + 1.6, t['duration'] - .7]
    return t


def bowl_pick(s, idx, start, hold):
    p = s.d['objects'][f'BOWL{idx}']['pos']
    s.r(start, start + .7, plus(p, [0, 0, .14]), .085)
    s.r(start + .7, start + 1.2, plus(p, [0, 0, .035]))
    s.r(start + 1.2, start + 1.45, grip=.060)
    s.grab(start + 1.45, f'BOWL{idx}', 'robot')
    s.r(start + 1.45, start + 2.0, plus(p, [0, 0, .22]))
    s.r(start + 2.0, start + 2.8, hold)
    return start + 2.8


def bowl_place(s, idx, start, on_mat=False):
    p = s.d['objects'][f'BOWL{idx}']['pos']
    s.r(start, start + .7, plus(p, [0, 0, .17]))
    s.r(start + .7, start + 1.2, plus(p, [0, 0, .038 if on_mat else .035]))
    s.r(start + 1.2, start + 1.4, grip=.085)
    s.drop(start + 1.4, f'BOWL{idx}', plus(p, [0, 0, .003 if on_mat else 0]))
    return start + 1.4


def mats(kind):
    s = scene_story('C3', 'Nhấc bát để trải khăn', 'collaborator', c3('BASE'))
    s.h(.2, .8, [.20, -.29, .66])
    s.h(.2, .8, [.08, -.29, .66], left=True)
    s.grab(.8, 'TOWEL', 'human')
    # Split at the early cue so all branches have an identical motion prefix.
    for left, x in [(False, .215), (True, .085)]:
        s.h(1.0, 2.2, [x, -.20, .77], left=left)
    s.r(1.5, 2.2, [.15, -.02, .768], .085)
    target = 0 if kind == 'KEEP' else 1
    cue = None if kind == 'KEEP' else (2.2 if kind == 'EARLY' else 4.7)
    if kind == 'EARLY':
        for left, x in [(False, .435), (True, .305)]:
            s.h(2.2, 3.2, [x, -.16, .77], left=left)
        held = bowl_pick(s, 1, 2.35, [.38, .20, .86])
        lay = max(5.2, held + .2)
    else:
        for left, x in [(False, .215), (True, .085)]:
            s.h(2.2, 2.8, [x, -.16, .77], left=left)
        s.r(2.2, 2.7, [.15, -.02, .663])
        s.r(2.7, 2.95, grip=.06)
        s.grab(2.95, 'BOWL0', 'robot')
        s.r(2.95, 3.5, [.15, -.02, .848])
        s.r(3.5, 4.3, [.10, .22, .86])
        if kind == 'LATE':
            for left, x in [(False, .435), (True, .305)]:
                s.h(4.7, 5.7, [x, -.16, .77], left=left)
            bowl_place(s, 0, 4.9)
            held = bowl_pick(s, 1, 6.4, [.38, .20, .86])
            lay = held + .2
        else:
            lay = 5.2
    x = [.15, .37][target]
    s.drop(lay, 'TOWEL', [0, 0, -1])
    s.drop(lay, 'MAT', [x, -.16, .735])
    s.grab(lay, 'MAT', 'human')
    s.h(lay, lay + .9, [x + .06, -.03, .665])
    s.h(lay, lay + .9, [x - .06, -.03, .665], left=True)
    s.drop(lay + .9, 'MAT', [x, -.03, .627])
    s.h(lay + 1.2, lay + 2.0, REST, 'relaxed')
    s.h(lay + 1.2, lay + 2.0, LEFT_REST, 'relaxed', left=True)
    placed = bowl_place(s, target, lay + 2.1, True)
    s.r(placed + .1, placed + .8, HOME)
    end = placed + 1.1
    if kind == 'KEEP':
        s.cap(0, 2.2, 'Cùng trạng thái đầu: robot tay trống, hai bát trên hai khay. Người mang khăn về A; robot bắt đầu tiếp cận A.')
        s.cap(2.2, 4.7, 'Giữ yêu cầu A: robot hạ tay, gắp bát A, nâng và mang ra ngoài vùng trải khăn.')
    elif kind == 'EARLY':
        s.cap(0, 2.2, 'Cùng đoạn đầu với lượt giữ A. Robot đang tiếp cận bát A nhưng chưa gắp vật nào.')
        s.cap(2.2, lay, 'Người đổi sang B trước khi gắp A. Robot đổi đường, gắp B; không phải đặt lại A để giải phóng tay.')
    else:
        s.cap(0, 2.95, 'Cùng đoạn đầu với lượt giữ A. Người thật sự hướng khăn tới A; robot tiếp cận và gắp A.')
        s.cap(2.95, 4.7, 'Robot đã nhấc A và mang ra xa khay. Đến đây người mới bắt đầu chuyển khăn sang B.')
        s.cap(4.7, 6.4, 'Đổi muộn: robot đang giữ A, phải đặt A xuống để có tay lấy B. Người giữ khăn chờ bát B được nhấc.')
        s.cap(6.4, lay, 'Sau khi đặt A lại, robot đi tới B, gắp và mang B ra khỏi vùng trải. Đây là phần việc sửa phát sinh.')
    s.cap(lay, lay + 2.1, 'Bát của khay được chọn đã rời: người trải khăn rồi rút hai tay. Robot giữ bát ngoài vùng người làm.')
    s.cap(lay + 2.1, end + .1, 'Robot đặt bát lên khăn; bát còn lại vẫn về đúng khay. Thời gian trong clip là kịch bản, không phải số đo policy.')
    chapter = {'KEEP': 'Giữ A', 'EARLY': 'Đổi trước khi gắp A', 'LATE': 'Đổi sau khi mang bát A ra xa'}[kind]
    probes = [(0, 'start'), (2.2, 'before_grasp'), (4.7, 'late_cue_state'), (end, 'final')]
    return done(s, end, f'C3 | {chapter}', chapter,
                'Khác biệt giữa đổi đường tay trống và phải đặt lại vật đang giữ rồi mới phục vụ B là gì?',
                [.5, 2.2, 4.5, 5.6, lay + .5, placed + .1], cue=cue, probes=probes)


def drawer_open(s, idx, t, *, approach=True):
    x = [.17, .37][idx]
    if approach:
        s.r(t, t + .7, [x, .001, .835], .085)
    s.r(t + .7, t + 1.1, [x, .031, .735])
    s.r(t + 1.1, t + 1.35, grip=.037)
    s.grab(t + 1.35, f'DRAWER{idx}', 'robot')
    s.r(t + 1.35, t + 2.25, [x, -.109, .735])
    s.r(t + 2.25, t + 2.45, grip=.085)
    s.drop(t + 2.45, f'DRAWER{idx}', [x, -.01, .632])
    s.r(t + 2.45, t + 2.9, [x, -.15, .88])
    return t + 2.25


def drawer_close(s, idx, t):
    x = [.17, .37][idx]
    s.r(t, t + .55, [x, -.109, .735], .085)
    s.r(t + .55, t + .75, grip=.037)
    s.grab(t + .75, f'DRAWER{idx}', 'robot')
    s.r(t + .75, t + 1.65, [x, .031, .735])
    s.drop(t + 1.65, f'DRAWER{idx}', [x, .13, .632])
    s.r(t + 1.65, t + 1.85, grip=.085)
    s.r(t + 1.85, t + 2.2, [x, -.15, .88])
    return t + 2.2


def drawers(kind):
    s = scene_story('C4', 'Mở ngăn để người cất đồ', 'collaborator', c4('BASE'))
    s.d['objects'].pop('OWN_PEN', None)
    s.h(.15, .8, [.43, -.29, .67])
    s.grab(.8, 'ITEM', 'human')
    s.h(1.0, 2.0, [.23, -.15, .78])
    s.r(1.3, 2.0, [.17, .001, .835], .085)
    if kind == 'EARLY':
        s.h(2.0, 3.0, [.37, -.10, .78])
        opened = drawer_open(s, 1, 2.15)
        target, cue = 1, 2.0
    else:
        s.h(2.0, 2.8, [.17, -.10, .78])
        opened = drawer_open(s, 0, 1.3, approach=False)
        target, cue = 0, None
        if kind == 'LATE':
            cue = 4.0
            s.h(4.0, 5.0, [.37, -.10, .78])
            opened = drawer_open(s, 1, 4.3)
            target = 1
    x = [.17, .37][target]
    s.h(opened + .85, opened + 1.6, [x, -.01, .75])
    s.drop(opened + 1.6, 'ITEM', [x, -.01, .645])
    s.grab(opened + 1.6, 'ITEM', f'DRAWER{target}')
    s.h(opened + 1.8, opened + 2.5, REST, 'relaxed')
    close = opened + 2.6
    if kind == 'LATE':
        close = drawer_close(s, 0, close) + .15
    end_close = drawer_close(s, target, close)
    s.r(end_close + .1, end_close + .7, HOME)
    end = end_close + 1
    s.cap(0, 2.0, 'Robot tay trống, cả hai ngăn đóng. Người cầm vật hướng tới A; robot mới tới tay nắm A.')
    if kind == 'EARLY':
        s.cap(2.0, opened, 'Người đổi sang B trước khi robot nắm và kéo A. Robot đổi đường tới B; ngăn A vẫn đóng.')
    elif kind == 'LATE':
        s.cap(2.0, 4.0, 'Người vẫn hướng tới A. Robot nắm, kéo A mở hết và nhả tay; sau đó người mới chuyển sang B.')
        s.cap(4.0, opened, 'A đã mở. Robot đi mở B trước để phục vụ yêu cầu mới; không ép đóng A trước khi hình học không yêu cầu.')
    else:
        s.cap(2.0, opened, 'Giữ yêu cầu A: robot nắm tay cầm và kéo ngăn ra. Toàn bộ công đoạn chuẩn bị nằm trong video.')
    s.cap(opened, opened + 2.6, 'Robot rời miệng ngăn đúng; người cất vật rồi rút tay. Ngăn A còn mở trong lượt đổi muộn.')
    s.cap(opened + 2.6, end + .1, 'Đổi muộn phát sinh phần đóng A, rồi đóng B. Kết thúc: vật ở B, cả hai ngăn đóng.' if kind == 'LATE'
          else 'Robot đóng ngăn đã dùng. Kết thúc: vật cất đúng ngăn và cả hai ngăn đóng.')
    chapter = {'KEEP': 'Giữ A', 'EARLY': 'Đổi trước khi mở A', 'LATE': 'Đổi sau khi A đã mở hết'}[kind]
    return done(s, end, f'C4 | {chapter}', chapter,
                'Mở sớm đúng tiết kiệm gì, và mở A rồi đổi B phát sinh thao tác nào?',
                [.5, 2.0, 3.8, opened + 1.3, opened + 3.3, end - .5], cue=cue,
                probes=[(0, 'start'), (2.0, 'before_open'), (4.0, 'late_cue_state'), (opened + .7, 'correct_drawer_open'), (end, 'final')])


def mixed_boxes():
    s = scene_story('W1', 'Hai hộp trong vùng đồ lộn xộn', 'cowork', w1('BASE'))
    for j in range(3):
        p = s.d['objects'][f'H{j}']['pos']
        human_pick(s, f'H{j}', p, .4 + j * 3.0, [.45 + (j % 2) * .035, -.26 + (j // 2) * .03, .65])
    p = s.d['objects']['H3']['pos']
    s.h(9.0, 9.7, plus(p, [0, 0, .035]), left=True)
    s.grab(9.7, 'H3', 'human_l')
    s.h(9.7, 10.3, [.24, -.055, .76], left=True)
    s.h(12.8, 14.0, [.485, -.23, .70], left=True)
    s.drop(14.0, 'H3', [.485, -.23, .65])
    s.h(14.1, 14.7, LEFT_REST, 'relaxed', left=True)
    human_pick(s, 'HLID', s.d['objects']['HLID']['pos'], 15, [.47, -.25, .678])
    t = 3.0
    for j, dest in [(2, [.07, .27, .65]), (3, [.105, .27, .65])]:
        t = pick(s, f'R{j}', s.d['objects'][f'R{j}']['pos'], t, dest) + .1
    s.r(t, t + .6, [.14, .16, .91])
    t = 14.4
    for j, dest in [(0, [.07, .30, .65]), (1, [.105, .30, .65])]:
        t = pick(s, f'R{j}', s.d['objects'][f'R{j}']['pos'], t, dest) + .1
    t = pick(s, 'RLID', s.d['objects']['RLID']['pos'], t, [.09, .28, .678], dz=.014)
    s.r(t, t + .6, HOME)
    s.cap(0, 3, 'Đồ hai bên xen lẫn. Người xếp các món cam, robot tay trống quan sát chuyển động rồi chọn món xanh ở vùng khác.')
    s.cap(3, 9.2, 'Robot tự gắp và xếp hai món xanh phía xa trước; người tiếp tục việc riêng, không chờ robot.')
    s.cap(9.2, 14.4, 'Người đang cầm/kiểm món cam trên cụm còn lại. Robot hết lựa chọn gắp thuận tiện trong kịch bản nên chờ ở ngoài.')
    s.cap(14.4, t + .7, 'Người đưa món cam về hộp; robot lấy các món còn lại rồi đóng hộp. Cần kiểm mọi đường tránh bằng physics sau.')
    return done(s, t + .8, 'W1 | Đổi thứ tự lấy đồ, rồi chờ khi vùng còn lại bị chiếm', 'Hai bên hoàn thành hộp riêng',
                'Robot có duy trì tiến độ bằng đổi thứ tự, rồi chờ ổn định khi cần không?', [1.8, 4.5, 10.8, 15.6, t - .5])


def large_tray():
    t = w2('BASE')
    s = scene_story('W2', 'Nhường khay lớn trong vùng chung', 'cowork', t)
    for k in ('human', 'human_l', 'events'):
        s.d[k] = copy.deepcopy(t[k])
    # Keep only human and object-to-object events; rebuild the independent robot workflow.
    s.d['events'] = [e for e in s.d['events'] if not e.get('grab', e.get('drop', '')).startswith('PART')]
    end = pick(s, 'PART0', s.d['objects']['PART0']['pos'], .25, [.075, .32, .65])
    s.r(end + .05, end + .55, [.12, .22, .91])
    a = 4.8
    for j in (2, 1):
        a = pick(s, f'PART{j}', s.d['objects'][f'PART{j}']['pos'], a, [.075 + j * .025, .32, .65]) + .1
    s.r(a, a + .6, HOME)
    s.cap(0, 1.5, 'Robot bắt đầu tự lấy món xanh phía ngoài. Người nắm và nâng khay lớn bằng hai tay, có đồ thật trên khay.')
    s.cap(1.5, 3.4, 'Người mang khay qua vùng chung; robot hoàn tất món ngoài vùng quét. Phải xét cả mép khay, không chỉ hai cổ tay.')
    s.cap(3.4, 4.8, 'Robot chờ ngoài đường mang khay. Hai cổ tay ở hai mép không có nghĩa phần giữa đã an toàn.')
    s.cap(4.8, a + .7, 'Khay được đặt xuống và người tiếp tục xử lý đồ riêng. Robot lấy nốt hai món trong vùng vừa được giải phóng.')
    return done(s, a + .8, 'W2 | Nhường cả khay lớn, tiếp tục việc riêng', 'Hình học vật người mang',
                'Ảnh hình học vật kết hợp forecast tay có giúp nhường đúng vùng quét không?', [.6, 2.5, 3.8, 5.6, a - .4])


def repeated_inspection():
    s = Story('W3', 'Vận chuyển qua vùng kiểm tra theo chu kỳ', 'BASE', 'cowork')
    # Reuse the validated I2 source/destination envelopes and vertical pickup path.
    # The discarded layout used an arbitrary carrying waypoint that failed IK.
    sources = [[.36, .16, .644], [.40, .10, .644], [.17, .22, .644]]
    for j, p in enumerate(sources):
        s.obj(f'R{j}', 'box', p, size=[.018, .018, .023], color='blue')
        s.obj(f'H{j}', 'cylinder', [.44 + (j % 2) * .04, -.29 + (j // 2) * .055, .644], size=[.018, .022], color='orange')
    s.decor('inspection', 'box', [.25, .055, .624], 'grey', size=[.048, .045, .003])
    s.decor('robot_box', 'open_box', [.26, -.20, .62], 'box', half=[.075, .045], height=.025, wall=.003)
    s.decor('human_box', 'open_box', [.48, -.17, .62], 'box2', half=[.05, .04], height=.025, wall=.003)
    # Human works independently: inspect, sometimes look again, then store each part.
    for j, (start, arrival, departure, stored) in enumerate([(.3, 2.0, 4.5, 5.4), (5.9, 7.5, 9.3, 10.2), (10.7, 12.3, 14.0, 14.9)]):
        p = s.d['objects'][f'H{j}']['pos']
        s.h(start, start + .65, plus(p, [0, 0, .035]))
        s.grab(start + .65, f'H{j}', 'human')
        s.h(start + .65, arrival, [.25, .055, .72])
        if j == 0:
            s.h(3.0, 3.6, [.32, -.08, .78])
            s.h(3.6, 4.2, [.25, .055, .72])
        s.h(departure, stored, [.46 + .025 * (j % 2), -.18 + .025 * (j // 2), .70])
        s.drop(stored, f'H{j}', [.46 + .025 * (j % 2), -.18 + .025 * (j // 2), .65])
        s.h(stored + .1, stored + .45, REST, 'relaxed')
    for j, (start, cross) in enumerate([(1.0, 5.1), (7.1, 9.8), (12.0, 14.5)]):
        p = sources[j]
        dest = [[.23, -.20, .648], [.29, -.20, .648], [.26, -.175, .648]][j]
        n_steps, n_events = len(s.d['robot']), len(s.d['events'])
        pick(s, f'R{j}', p, start, dest)
        delta = cross - (start + 1.5)
        for step in s.d['robot'][n_steps:]:
            if step['t'][0] >= start + 1.5 - 1e-7:
                step['t'] = [round(v + delta, 6) for v in step['t']]
        for event in s.d['events'][n_events:]:
            if event['t'] >= start + 1.5:
                event['t'] = round(event['t'] + delta, 6)
        s.r(cross + 1.55, cross + 1.9, HOME)
    s.r(16.6, 17.2, HOME)
    s.cap(0, 2.5, 'Hai việc riêng: người kiểm món cam trên đệm giữa bàn; robot tự gắp món xanh từ kho để chuyển vào hộp của mình.')
    s.cap(2.5, 4.5, 'Robot đang mang vật. Người rút tay ngắn rồi quay lại kiểm tiếp: chưa đủ khoảng trống để kết thúc lượt vận chuyển.')
    s.cap(4.5, 7.1, 'Người rời đệm để cất món; robot đi qua vùng chung và đặt vật, rồi rời trước lượt kiểm kế tiếp.')
    s.cap(7.1, 14.5, 'Nhịp kiểm tiếp theo thay đổi. Robot gắp, giữ ngoài vùng chung và đi khi tay rời; kiểm chờ thừa, giật và cản người.')
    s.cap(14.5, 17.5, 'Hai bên hoàn thành ba món riêng. Có thể chọn đường vòng nếu hợp lệ; video không khẳng định đường đang vẽ là duy nhất.')
    return done(s, 17.5, 'W3 | Mang vật qua khoảng trống giữa các lượt người kiểm tra', 'Rút tay ngắn và rời vùng thật',
                'Forecast giúp phân biệt khoảng rút tay ngắn với cửa sổ đủ vận chuyển mà không dừng–đi liên tục không?',
                [.6, 2.6, 3.8, 5.8, 8.7, 15.5], probes=[(0, 'start'), (3.8, 'held_during_human_return'), (17.5, 'final')])


def main():
    cfg = yaml.safe_load((ROOT / 'configs/scenarios/intent_review_50.yaml').read_text())
    cfg['out_dir'] = 'outputs/commitment_review_10'
    cfg['video']['caption_h'] = 128
    cfg['tasks'] = {}
    groups = {
        'I1': ('Instructor', 'Ba đế lót, ba vị trí liên tục', [coasters()]),
        'I2': ('Instructor', 'Tay trái/phải chọn phía lấy vật', [sides()]),
        'I3': ('Instructor', 'Khay trên ray tới vị trí tay đón', [receiving_tray(.23), receiving_tray(.36)]),
        'C1': ('Collaborator', 'T2 — làm xong tấm rồi nhận lon', [handover()]),
        'C2': ('Collaborator', 'Lấy dụng cụ và lau dưới giá được nhấc', [rack(False), rack(True)]),
        'C3': ('Collaborator', 'Nhấc bát để trải khăn — đổi sớm/muộn', [mats(k) for k in ('KEEP', 'EARLY', 'LATE')]),
        'C4': ('Collaborator', 'Mở ngăn kéo — đổi sớm/muộn', [drawers(k) for k in ('KEEP', 'EARLY', 'LATE')]),
        'W1': ('Cowork', 'Đồ lộn xộn: đổi thứ tự và chờ vùng còn lại', [mixed_boxes()]),
        'W2': ('Cowork', 'Nhường vùng quét của khay lớn', [large_tray()]),
        'W3': ('Cowork', 'Vận chuyển giữa các lượt người kiểm tra', [repeated_inspection()]),
    }
    cfg['review_groups'] = {}
    for group, (role, name, chapters) in groups.items():
        keys = []
        for i, task in enumerate(chapters):
            key = f'{group}_{i + 1:02d}'
            task.pop('forecast_overlay', None)
            task.pop('forecast_witness', None)
            task.pop('forecast_intervention', None)
            if not task.get('state_probes'):
                task['state_probes'] = [dict(t=0, label='start'), dict(t=task['duration'], label='final')]
            task.setdefault('evaluation_only', {}).update(initial_robot_gripper_empty=True, annotations_are_policy_inputs=False)
            cfg['tasks'][key] = task
            keys.append(key)
        cfg['review_groups'][group] = dict(role=role, name=name, keys=keys, question=chapters[0]['review_question'])
    cfg['review_protocol'] = dict(main_videos=10, roles=dict(instructor=3, collaborator=4, cowork=3),
                                  chapter_resets_are_explicit=True, actual_policy_rollout=False,
                                  forecasts_displayed=False, initial_gripper_empty=True,
                                  scope='workflow_and_commitment_review_not_benchmark_results')
    path = ROOT / 'configs/scenarios/commitment_review_10.yaml'
    path.write_text('# Review only. Decisions: docs/PIF.md §8.10–8.11.\n' + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    print(f'{path}: {len(groups)} tasks / {len(cfg["tasks"])} chapters')


if __name__ == '__main__':
    main()
