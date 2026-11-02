"""Selected, task-specific review cases after the alphaXiv/Consensus audit.

Human/robot motion is scripted. Goal logs and branch labels are evaluation-only.
The paired ambiguity cases share observable prefixes; no policy reads the logs.
"""
from __future__ import annotations

import copy
import math
import random

import yaml

from intent_policy.sim.base_env import ROOT
from scripts.build_intent_review_previews import c1, c2, c3, c4, i1, w1


def point(t0, t1, p, shape='grasp', **kw):
    return dict(t=[t0, t1], to=p, shape=shape, **kw)


def robot(t0, t1, p=None, grip=None):
    d = dict(t=[t0, t1])
    if p is not None:
        d['to'] = p
    if grip is not None:
        d['grip'] = grip
    return d


def shift_after(task, threshold, delta, *, arms=('human', 'human_l', 'robot'), events=True):
    for arm in arms:
        for step in task.get(arm, []):
            if step['t'][0] >= threshold - 1e-7:
                step['t'] = [round(t + delta, 5) for t in step['t']]
    if events:
        for e in task['events']:
            if e['t'] >= threshold - 1e-7:
                e['t'] = round(e['t'] + delta, 5)
    task['duration'] += delta


def finalize(t, key, title, family, goals, captions, *, action=None, reveal=None, review=None, pair=None):
    t['title'] = title
    t['case_family'] = family
    t['case'] = family
    t['review_key'] = key
    t['evaluation_only'] = dict(human_goal_events=goals, policy_observes_goal_log=False)
    t['captions'] = captions
    t['annotation_legend'] = 'Chấm xanh: cổ tay ManiCast thật (8 mốc/1 s). Robot: chuyển động minh họa; chưa chạy ACT/physics.'
    t['marks'] = []
    if action is not None:
        t['decision'] = dict(action_start=action, reveal_t=reveal, robot_response_scripted=True)
        t['marks'] += [dict(t=action, who='robot', label='robot đã đi'),
                       dict(t=reveal, who='human', label='cue mới/đích rõ')]
    else:
        t.pop('decision', None)
    t['review_times'] = review or [round(t['duration'] * x, 2) for x in (.25, .5, .85)]
    if pair:
        t['ambiguity_pair'] = dict(group=pair[0], branch_t=pair[1])
    for arm in ('human', 'human_l', 'robot'):
        if arm in t:
            t[arm].sort(key=lambda p: p['t'][0])
    t['events'].sort(key=lambda e: e['t'])
    return t


def retarget_first_coaster(t, new, branch=3.5):
    old = t['sampled_coaster_positions'][0]
    dx, dy = new[0] - old[0], new[1] - old[1]
    for arm in ('human', 'robot'):
        for st in t[arm]:
            p = st.get('to')
            if st['t'][0] >= branch and st['t'][0] < 7.15 and isinstance(p, list) and abs(p[0] - old[0]) < .001:
                st['to'] = [round(p[0] + dx, 5), round(p[1] + dy, 5), p[2]]
    for e in t['events']:
        if e.get('drop') in ('COASTER0', 'CUP0'):
            e['at'] = [round(e['at'][0] + dx, 5), round(e['at'][1] + dy, 5), e['at'][2]]
    t['sampled_coaster_positions'][0] = new


def coaster_cases():
    base = i1('BASE')
    out = dict(I1_CLEAR=finalize(base, 'I1_CLEAR', 'I1 | Đặt lần lượt ba đế ở vị trí liên tục', 'clear_motion',
                [dict(t=0, goal='serve_three_coasters')],
                [[0, 5.8, 'Người mang đế thứ nhất tới vị trí mới. Robot chuẩn bị cốc, chỉ đặt sau khi đế nằm ổn định và tay rút.'],
                 [5.8, base['duration'] + .1, 'Tiếp tục với đế thứ hai và ba ở vị trí khác. Không có vị trí đích được đánh dấu sẵn trong scene.']],
                action=1.9, reveal=3.2, review=[2.5, 4.3, 11.0, base['duration'] - .7]))
    a = i1('C2')
    b = copy.deepcopy(a)
    rng = random.Random(73)
    first = a['sampled_coaster_positions'][0]
    for _ in range(30000):
        p = [round(rng.uniform(.14, .38), 4), round(rng.uniform(-.19, .06), 4), .624]
        if .22 <= math.hypot(p[0], p[1]) <= .40 and math.dist(p[:2], first[:2]) > .14 \
                and all(math.dist(p[:2], q[:2]) > .115 for q in a['sampled_coaster_positions'][1:]):
            break
    else:
        raise RuntimeError('No valid alternate coaster position')
    retarget_first_coaster(b, p)
    for label, t in [('A', a), ('B', b)]:
        key = f'I1_AMBIG_{label}'
        out[key] = finalize(t, key, f'I1 | Mơ hồ, nhánh {label} — cùng đoạn đầu', 'ambiguous_prefix',
                           [dict(t=0, goal=f'coaster_position_{label}')],
                           [[0, 3.5, 'Hai clip A/B có cùng scene và chuyển động tới 3,5 s. Tay dừng ở giữa; robot không thể biết chắc nhánh tiếp theo.'],
                            [3.5, 7.2, f'Sau 3,5 s tay mới tiếp tục tới đích {label}; robot chuyển từ chỗ chuẩn bị tới nơi đặt.'],
                            [7.2, t['duration'] + .1, 'Người tiếp tục đặt các đế còn lại. Mơ hồ được kiểm bằng cặp continuation, không chỉ bằng động tác chậm.']],
                           action=1.9, reveal=3.5, review=[2.5, 3.75, 6.3, t['duration'] - .7], pair=('I1_PLACE', 3.5))
    return out


def t2_cases():
    out = {}
    t = c1('BASE')
    for st in t['robot']:
        if abs(st['t'][0] - 5.5) < .01:
            st['t'] = [6.1, 7.0]
    out['T2_CLEAR'] = finalize(t, 'T2_CLEAR', 'T2 | Chuyển tấm xong, đưa tay nhận lon', 'clear_motion',
        [dict(t=0, goal='finish_board_then_receive_can')],
        [[0, 3.6, 'Người làm trên tấm; robot chưa coi chậm thao tác là tín hiệu hoàn thành.'],
         [3.6, 5.3, 'Hai tay mang tấm vào khay thành phẩm. Robot chuẩn bị lấy lon khi chuyển pha đã xuất hiện trong quan sát.'],
         [5.3, 7.2, 'Tay bắt đầu đưa ra nhận. Chấm xanh là dự báo thật; robot minh họa tiếp cận điểm gặp rồi chờ tay sẵn sàng.'],
         [7.2, t['duration'] + .1, 'Người cầm và kéo lon, robot nhả. Chưa đo thời gian hay thành công của policy.']],
        action=6.1, reveal=6.7, review=[4.2, 6.35, 7.25, 8.4])
    a = c1('C2')
    b = copy.deepcopy(a)
    # Same prefix including private cup, robot staging and a partly extended relaxed hand.
    b['human'] = [s for s in b['human'] if s['t'][0] < 7.2]
    b['robot'] = [s for s in b['robot'] if s['t'][0] < 7.2]
    b['events'] = [e for e in b['events'] if e['t'] < 7.2]
    b['human'] += [point(7.2, 8.2, [.40, -.19, .69]), point(8.3, 9.0, [.41, -.31, .94]),
                   point(9.6, 10.3, [.40, -.19, .69]), point(10.7, 12.1, [.30, -.15, .78], 'offer', palm_up=1.),
                   point(12.3, 12.6, [.30, -.15, .78], 'grasp', palm_up=1.),
                   point(12.6, 13.3, [.33, -.23, .78]), point(13.3, 14.0, [.40, -.33, .90])]
    b['events'] += [dict(t=8.3, grab='OWN_CUP', by='human'), dict(t=10.3, drop='OWN_CUP', at=[.40, -.19, .62]),
                    dict(t=12.6, grab='CAN', by='human')]
    b['robot'] += [robot(10.9, 12.25, [.30, -.135, .85]), robot(12.6, 12.8, grip=.085), robot(12.9, 13.8, [.15, 0, .85])]
    b['duration'] = 14.4
    for label, t in [('A', a), ('B', b)]:
        key = f'T2_AMBIG_{label}'
        out[key] = finalize(t, key, f'T2 | Tay chưa rõ yêu cầu — nhánh {label}', 'ambiguous_prefix',
            [dict(t=0, goal='receive_next' if label == 'A' else 'own_cup_then_receive')],
            [[0, 5.3, 'Người hoàn tất phần việc trên tấm và robot chuẩn bị lon. Cốc riêng có mặt ở cả hai clip.'],
             [5.3, 7.2, 'Tay đưa ra một phần rồi dừng. A/B hoàn toàn giống nhau tới 7,2 s; robot giữ lon tại chỗ chờ.'],
             [7.2, t['duration'] + .1, 'Nhánh A: tay tiếp tục đưa ra nhận lon.' if label == 'A' else 'Nhánh B: tay tới cốc riêng để uống trước, sau đó mới xin lon. Không trao nhầm trong lúc người bận.']],
            action=5.5, reveal=7.2, review=[6.1, 7.6, 8.8, t['duration'] - .8], pair=('T2_NEXT_ACTION', 7.2))
    for natural in (True, False):
        t = c1('BASE')
        t['robot'] = [s for s in t['robot'] if s['t'][0] < 5.5 or s['t'][0] > 6.9]
        t['robot'] += [robot(5.55, 6.25, [.21, -.22, .87]), robot(6.35, 7.0, [.30, -.135, .85])]
        key = 'T2_ERROR_REAL' if natural else 'T2_ERROR_CONTROLLED'
        title = 'T2 | Lệch vị trí dự báo thật, người không đổi đích' if natural else 'T2 | Sai forecast có kiểm soát, rồi phục hồi đầu vào'
        out[key] = finalize(t, key, title, 'natural_forecast_error' if natural else 'controlled_forecast_error',
            [dict(t=0, goal='finish_board_then_receive_can')],
            [[0, 5.3, 'Cùng người, scene và đích nhận lon như clip thuận. Không có hủy yêu cầu hoặc đổi mục tiêu.'],
             [5.3, 6.35, 'Xanh: ManiCast thật dự báo điểm cổ tay chưa tới vị trí tương lai. Robot minh họa đã tiến về điểm lệch.' if natural
              else 'Đỏ: forecast bị lệch trong một cửa sổ ngắn; robot minh họa đã tiến theo điểm lệch. Đây là can thiệp kiểm soát.'],
             [6.35, t['duration'] + .1, 'Quan sát và forecast mới cho thấy điểm gặp khác; robot sửa vị trí và trao khi tay sẵn sàng. Hành vi robot chưa phải rollout ACT.']],
            action=5.55, reveal=6.35, review=[5.75, 6.4, 7.25, 8.4])
        if natural:
            out[key]['forecast_witness'] = dict(t_s=5.5333333333, joint=5, horizon_index=7, kind='position_error',
                                               screening_source='forecast_screen/C1_BASE.npz')
        else:
            out[key]['forecast_intervention'] = dict(t=[5.45, 6.35], joints=[5], offset=[-.10, -.08, 0.0])
    t = c1('C4')
    for st in t['human']:
        if st['t'][0] == 6.8:
            st['to'] = [.20, -.26, .73]
        elif st['t'][0] == 7.5:
            st['to'] = [.19, -.26, .73]
    out['T2_CANCEL'] = finalize(t, 'T2_CANCEL', 'T2 | Hủy nhận khi robot đã mang lon tới', 'local_goal_change',
        [dict(t=0, goal='receive_can'), dict(t=6.8, goal='adjust_finished_tray'), dict(t=9.7, goal='receive_can')],
        [[0, 6.8, 'Người thật sự đưa tay nhận; robot đã cầm lon và tiến tới điểm gặp.'],
         [6.8, 9.7, 'Người rút tay để chỉnh khay thành phẩm. Đây là cue bác bỏ yêu cầu cũ; robot rút lon về chỗ chờ.'],
         [9.7, t['duration'] + .1, 'Người đưa tay nhận lại; robot tiếp cận lần mới. Log đổi đích chỉ phục vụ đánh giá, không cấp cho policy.']],
        action=5.5, reveal=6.8, review=[6.2, 7.1, 10.3, 12.7])
    return out


def rack_cases():
    out = {}
    base = c2('BASE')
    out['RACK_CLEAR'] = finalize(base, 'RACK_CLEAR', 'Giá | Nâng rõ, lau rồi đặt lại', 'clear_motion',
        [dict(t=0, goal='clean_under_left_rack')],
        [[0, 3.3, 'Hai tay tới tay nắm của một giá; robot đã có dụng cụ, chuẩn bị cạnh vùng sắp trống.'],
         [3.3, 6.8, 'Giá thật sự được nhấc và giữ sang bên. Robot chỉ lau khi vùng dưới giá đã trống.'],
         [6.8, base['duration'] + .1, 'Robot rời vùng trước, người mới đặt giá lại. Mục tiêu là giảm thời gian giữ giá, không phải đoán ngón tay.']],
        action=1.9, reveal=3.3, review=[2.5, 4.2, 5.4, 8.0])
    for label, delay in [('A', 1.2), ('B', 3.2)]:
        t = c2('BASE')
        shift_after(t, 3.3, delay)
        # Grip adjustment is visible; the rack remains on the table until the actual lift.
        t['human'] += [point(3.3, 3.8, [.30, -.16, .693]), point(3.8, 4.3, [.285, -.16, .693])]
        t['human_l'] += [point(3.3, 3.8, [.10, -.16, .693]), point(3.8, 4.3, [.115, -.16, .693])]
        if label == 'B':
            t['human'] += [point(4.5, 5.1, [.295, -.16, .693]), point(5.4, 6.0, [.285, -.16, .693])]
            t['human_l'] += [point(4.5, 5.1, [.105, -.16, .693]), point(5.4, 6.0, [.115, -.16, .693])]
        key = f'RACK_GRIP_{label}'
        out[key] = finalize(t, key, f'Giá | Chỉnh tay nắm rồi mới nâng — nhánh {label}', 'ambiguous_readiness',
            [dict(t=0, goal='clean_under_left_rack')],
            [[0, 3.3, 'A/B cùng đưa hai tay tới giá. Việc chạm tay nắm chưa cho biết giá sẽ được nhấc ngay.'],
             [3.3, 3.3 + delay, 'Người điều chỉnh tay nắm; giá chưa rời bàn. Robot chờ tại vị trí chuẩn bị, không lau xuyên giá.'],
             [3.3 + delay, t['duration'] + .1, 'Chỉ khi giá được nhấc, robot mới lau; người đặt lại sau khi robot rời. Hai nhánh khác thời điểm sẵn sàng, không đổi đích.']],
            action=1.9, reveal=3.3 + delay, review=[2.5, 3.95, 5.4 + delay, t['duration'] - .8], pair=('RACK_READY', 3.3))
    return out


def mat_cases():
    out = {}
    t = c3('BASE')
    out['MAT_CLEAR'] = finalize(t, 'MAT_CLEAR', 'Khăn | Nhấc bát, trải khăn, đặt lại', 'clear_motion',
        [dict(t=0, goal='lay_mat_left')],
        [[0, 3.9, 'Người mang khăn tới khay; robot nhấc bát để giải phóng chỗ trải.'],
         [3.9, 6.6, 'Robot giữ bát ở ngoài vùng thao tác. Người trải khăn và rút hai tay.'],
         [6.6, t['duration'] + .1, 'Tay đã rời, robot đặt bát lại. Forecast hỗ trợ timing; ảnh vẫn xác nhận khăn, bát và vùng trống.']],
        action=1.2, reveal=2.5, review=[2.0, 4.0, 5.7, 7.9])
    a = c3('C2')
    b = copy.deepcopy(a)
    for arm in ('human', 'human_l', 'robot'):
        for st in b.get(arm, []):
            p = st.get('to')
            if st['t'][0] >= 2.8 and isinstance(p, list) and p != [.15, 0.0, .85]:
                if (arm == 'robot' and abs(p[0] - .15) < .001) or (arm != 'robot' and min(abs(p[0] - .21), abs(p[0] - .09), abs(p[0] - .215), abs(p[0] - .085)) < .002):
                    st['to'] = [round(p[0] + .22, 5), p[1], p[2]]
    for e in b['events']:
        if e.get('grab') == 'BOWL0':
            e['grab'] = 'BOWL1'
        if e.get('drop') == 'BOWL0':
            e['drop'] = 'BOWL1'
            e['at'][0] += .22
        if e.get('drop') == 'MAT' and e['at'][2] > 0:
            e['at'][0] += .22
    for label, t in [('A', a), ('B', b)]:
        key = f'MAT_AMBIG_{label}'
        out[key] = finalize(t, key, f'Khăn | Cùng prefix, sau đó tới khay {label}', 'ambiguous_prefix',
            [dict(t=0, goal='lay_mat_left' if label == 'A' else 'lay_mat_right')],
            [[0, 2.8, 'Hai clip có cùng chuyển động và scene tới 2,8 s; khăn còn ở giữa, robot chỉ chuẩn bị ở vùng chung.'],
             [2.8, 6.0, f'Tay mới đi tới khay {label}; robot nhấc đúng bát, người mới trải khăn.'],
             [6.0, t['duration'] + .1, 'Người rút tay rồi robot đặt bát lại. Không bắt forecaster đoán một nhánh chưa có thông tin.']],
            action=1.2, reveal=2.8, review=[2.0, 3.2, 5.5, t['duration'] - .8], pair=('MAT_TRAY', 2.8))
    t = c3('BASE')
    t['human'] = [s for s in t['human'] if s['t'][0] < 4]
    t['human_l'] = [s for s in t['human_l'] if s['t'][0] < 4]
    t['robot'] = [s for s in t['robot'] if s['t'][1] <= 4.35 + 1e-6]
    t['events'] = [e for e in t['events'] if e['t'] <= 3.75 + 1e-6]
    t['human'] += [point(4.8, 6.0, [.435, -.16, .77]), point(8.2, 9.1, [.43, -.03, .665]), point(9.4, 10.2, [.49, -.40, .70], 'relaxed')]
    t['human_l'] += [point(4.8, 6.0, [.305, -.16, .77]), point(8.2, 9.1, [.31, -.03, .665]), point(9.4, 10.2, [.13, -.40, .70], 'relaxed')]
    t['robot'] += [robot(5.1, 5.7, [.15, -.02, .80]), robot(5.7, 6.1, [.15, -.02, .663]), robot(6.1, 6.3, grip=.085),
                   robot(6.4, 6.95, [.37, -.02, .758]), robot(6.95, 7.3, [.37, -.02, .663]), robot(7.3, 7.5, grip=.06),
                   robot(7.5, 8.1, [.36, .12, .85]), robot(10.3, 10.8, [.37, -.02, .80]), robot(10.8, 11.2, [.37, -.02, .666]),
                   robot(11.2, 11.4, grip=.085), robot(11.5, 12.1, [.15, 0, .85])]
    t['events'] += [dict(t=6.3, drop='BOWL0', at=[.15, -.02, .628]), dict(t=7.5, grab='BOWL1', by='robot'),
                    dict(t=8.2, drop='TOWEL', at=[0, 0, -1]), dict(t=8.2, drop='MAT', at=[.37, -.16, .735]),
                    dict(t=8.2, grab='MAT', by='human'), dict(t=9.1, drop='MAT', at=[.37, -.03, .627]),
                    dict(t=11.4, drop='BOWL1', at=[.37, -.02, .631])]
    t['duration'] = 12.6
    out['MAT_SWITCH_HELD'] = finalize(t, 'MAT_SWITCH_HELD', 'Khăn | Đổi khay khi bát A đã được nhấc', 'local_goal_change',
        [dict(t=0, goal='lay_mat_left'), dict(t=4.8, goal='lay_mat_right')],
        [[0, 4.8, 'Người hướng tới A; robot đã nhấc bát A và đang giữ. Đây là phần chuẩn bị thực sự có chi phí.'],
         [4.8, 6.3, 'Người mang khăn đổi sang B. Robot trả bát A về chỗ cũ, khi tay người đã ở xa vùng A.'],
         [6.3, 9.4, 'Robot nhấc bát B, người mới trải khăn ở B. Không coi đổi hướng ở vùng chờ là đủ.'],
         [9.4, 12.7, 'Người rút tay, robot đặt bát B lại. Đổi đích cục bộ được log cho đánh giá.']],
        action=3.75, reveal=4.8, review=[4.3, 5.4, 7.8, 11.5])
    return out


def delay_drawer_open(t, dt):
    old_opened = 3.25 if t['robot'][0]['t'][0] == 1.25 else 3.5
    for st in t['robot']:
        st['t'] = [round(x + dt, 5) for x in st['t']]
    for st in t['human']:
        if st['t'][0] >= old_opened + .8:
            st['t'] = [round(x + dt, 5) for x in st['t']]
    for e in t['events']:
        if e.get('grab', '').startswith('DRAWER') or e.get('drop', '').startswith('DRAWER') or e.get('grab') == 'ITEM' and e.get('by', '').startswith('DRAWER') or e.get('drop') == 'ITEM':
            e['t'] = round(e['t'] + dt, 5)
    t['duration'] += dt


def drawer_cases():
    out = {}
    t = c4('BASE')
    delay_drawer_open(t, .65)
    out['DRAWER_CLEAR'] = finalize(t, 'DRAWER_CLEAR', 'Ngăn kéo | Mở đúng ngăn, đặt đồ rồi đóng', 'clear_motion',
        [dict(t=0, goal='store_left')],
        [[0, 1.9, 'Người mang đồ về ngăn trái; robot không biết đích từ cấu hình. Chấm xanh là dự báo thật từ history hiện tại.'],
         [1.9, 5.0, 'Robot chuẩn bị và mở ngăn theo hướng đã hình thành; không cần đoán từ lúc người chưa di chuyển.'],
         [5.0, t['duration'] + .1, 'Robot rời miệng ngăn, người bỏ đồ và rút tay, rồi robot đóng. So với mở sẵn nhiều ngăn là bài kiểm sau.']],
        action=1.9, reveal=2.5, review=[2.1, 3.5, 5.0, t['duration'] - .7])
    a = c4('C2')
    b = copy.deepcopy(a)
    for arm in ('human', 'robot'):
        for st in b[arm]:
            p = st.get('to')
            if st['t'][0] >= 3 and isinstance(p, list) and abs(p[0] - .17) < .001:
                st['to'][0] = .37
    for e in b['events']:
        for k in ('grab', 'drop'):
            if e.get(k) == 'DRAWER0':
                e[k] = 'DRAWER1'
        if e.get('by') == 'DRAWER0':
            e['by'] = 'DRAWER1'
        if 'at' in e and e['t'] >= 3 and abs(e['at'][0] - .17) < .001:
            e['at'][0] = .37
    for label, t in [('A', a), ('B', b)]:
        key = f'DRAWER_AMBIG_{label}'
        out[key] = finalize(t, key, f'Ngăn kéo | Chưa rõ ngăn — cùng prefix, nhánh {label}', 'ambiguous_prefix',
            [dict(t=0, goal='store_left' if label == 'A' else 'store_right')],
            [[0, 3.0, 'Scene và chuyển động A/B giống nhau tới 3 s. Người giữ đồ ở giữa; robot chưa kéo ngăn nào.'],
             [3.0, 6.0, f'Đường tay mới phân biệt ngăn {label}; robot tiếp cận tay nắm và mở.'],
             [6.0, t['duration'] + .1, 'Người đặt đồ, tay rút, robot đóng ngăn. C2 kiểm thông tin chưa đủ, không yêu cầu đoán nhánh may rủi.']],
            action=1.2, reveal=3.0, review=[2.0, 3.3, 6.0, t['duration'] - .7], pair=('DRAWER_TARGET', 3.0))
    t = c4('BASE')
    delay_drawer_open(t, 1.2)
    t['robot'] = [robot(1.35, 1.9, [.37, 0, .88]), robot(1.95, 2.35, [.17, .02, .87])] + t['robot']
    out['DRAWER_ERROR_REAL'] = finalize(t, 'DRAWER_ERROR_REAL', 'Ngăn kéo | ManiCast thật lệch phía, người giữ đích trái', 'natural_forecast_error',
        [dict(t=0, goal='store_left')],
        [[0, 1.9, 'Người luôn định cất trái. Ở khung được ghi lại, forecast 1 s còn ở phía phải trong khi tương lai thật đã sang trái.'],
         [1.9, 4.9, 'Robot minh họa đã đi phải, rồi đổi sang trái khi motion/forecast mới làm rõ. Không cần mở sai hoàn toàn để có chi phí đi sai.'],
         [4.9, t['duration'] + .1, 'Người cất ở ngăn trái và robot đóng. Đây là forecast thật trên chuyển động dựng, không phải rollout ACT.']],
        action=1.35, reveal=1.9, review=[1.6, 2.2, 5.8, t['duration'] - .7])
    out['DRAWER_ERROR_REAL']['forecast_witness'] = dict(t_s=1.4, joint=5, horizon_index=7, kind='wrong_side',
                                                       split_x=.27, true_side='left', screening_source='forecast_screen/C4_BASE.npz')
    t = c4('C3')
    t.pop('human_l', None)
    t['events'] = [e for e in t['events'] if e.get('grab') != 'OWN_PEN']
    t['human'] = [s for s in t['human'] if s['t'][0] < 1 or s['t'][0] > 4.3]
    t['human'].append(point(1.0, 3.8, [.37, -.10, .78]))
    out['DRAWER_ERROR_CONTROLLED'] = finalize(t, 'DRAWER_ERROR_CONTROLLED', 'Ngăn kéo | Forecast bị can thiệp: đã mở A, phải sửa sang B', 'controlled_forecast_error',
        [dict(t=0, goal='store_right')],
        [[0, 3.45, 'Người giữ đích B; forecast hiển thị bị đổi sang A trong cửa sổ kiểm soát. Minh họa trạng thái robot đã mở A.'],
         [3.45, 6.8, 'Dùng lại forecast thường và cue tay mới: đóng A, mở B. Bài kiểm này không ép mọi policy phải mắc lỗi mở sai.'],
         [6.8, t['duration'] + .1, 'Người đặt vào B, robot đóng B. Báo riêng với lỗi thật và kèm tỷ lệ policy đã thực sự đi sai.']],
        action=2.35, reveal=3.45, review=[3.0, 3.75, 7.8, t['duration'] - .7])
    out['DRAWER_ERROR_CONTROLLED']['forecast_intervention'] = dict(t=[1.2, 3.45], joints=[5], target_x=.19)
    t = c4('C4')
    out['DRAWER_SWITCH_OPEN'] = finalize(t, 'DRAWER_SWITCH_OPEN', 'Ngăn kéo | Đổi A sang B sau khi A đã mở', 'local_goal_change',
        [dict(t=0, goal='store_left'), dict(t=3.45, goal='store_right')],
        [[0, 3.45, 'Đích ban đầu thật sự là A; robot đã kéo A mở. Chưa có nhãn đổi ý đưa cho robot.'],
         [3.45, 6.8, 'Người chuyển đồ sang B: cue mới bác bỏ đích cũ. Robot đóng A rồi mở B.'],
         [6.8, t['duration'] + .1, 'Cất đồ vào B và đóng ngăn. C4 có đổi đích cục bộ; C3 có kiểm soát giữ B từ đầu.']],
        action=2.35, reveal=3.45, review=[3.0, 3.75, 7.8, t['duration'] - .7])
    return out


def cowork_cases():
    out = {}
    t = w1('BASE')
    # Remove the long-blocking arm and the scripted wait: safe alternatives keep work progressing.
    t.pop('human_l', None)
    t['robot'] = [s for s in t['robot'] if not 9.79 <= s['t'][0] <= 10.5]
    for st in t['robot']:
        if st['t'][0] >= 12.94:
            st['t'] = [x - 3 for x in st['t']]
    for e in t['events']:
        if e['t'] >= 12.94 and e.get('grab', e.get('drop', '')).startswith(('R0', 'R1', 'RLID')):
            e['t'] -= 3
    t['duration'] -= 3
    out['W1_ALTERNATIVE'] = finalize(t, 'W1_ALTERNATIVE', 'Cowork | Có món an toàn khác: đổi thứ tự để tiếp tục làm', 'safe_alternative',
        [dict(t=0, goal='human_own_box_and_robot_own_box')],
        [[0, 3.6, 'Người và robot làm hai hộp riêng; đồ cam và xanh xen lẫn trong vùng chung.'],
         [3.6, 10, 'Khi tay người tới một cụm, robot lấy món khác còn cần ở vùng không xung đột.'],
         [10, t['duration'] + .1, 'Hai bên tiếp tục phần việc riêng và đóng hộp. Đây là kiểm duy trì tiến độ khi có lựa chọn hợp lý.']],
        review=[2.0, 6.0, 10.5, t['duration'] - .7])
    t = w1('BASE')
    out['W1_WAIT_RESUME'] = finalize(t, 'W1_WAIT_RESUME', 'Cowork | Vùng còn lại bị chiếm: chờ ổn định rồi tiếp tục', 'wait_and_resume',
        [dict(t=0, goal='human_own_box_and_robot_own_box')],
        [[0, 9.8, 'Robot làm các món ở vùng khác trước; người vẫn xếp hộp của mình.'],
         [9.8, 12.9, 'Các món robot còn lại nằm sát vùng tay người đang chiếm. Minh họa chờ có ích thay vì giật tiến-lùi hoặc ép người né.'],
         [12.9, t['duration'] + .1, 'Tay người rời vùng, robot tiếp tục và đóng hộp. Cần kiểm đường đi/an toàn bằng physics sau; không chứng minh mọi đường 3D đều bị chặn.']],
        review=[6.0, 11.0, 13.5, t['duration'] - .7])
    t = w1('C4')
    out['W1_NEXT_ITEM_SWITCH'] = finalize(t, 'W1_NEXT_ITEM_SWITCH', 'Cowork | Người đổi món lấy tiếp theo, robot nhường', 'local_next_item_change',
        [dict(t=0, goal='take_H1_next'), dict(t=2.1, goal='take_H0_next')],
        [[0, 2.1, 'Người ban đầu với tới món khác; robot đã bắt đầu đường làm việc của mình.'],
         [2.1, 4.4, 'Người đổi món lấy tiếp theo, tay chuyển vào vùng robot sắp tới. Robot rút/nhường, người tiếp tục việc.'],
         [4.4, t['duration'] + .1, 'Hai bên hoàn tất hộp riêng. Đây là đổi mục tiêu lấy món cục bộ, không phải đổi toàn bộ mục tiêu công việc.']],
        action=1.0, reveal=2.1, review=[1.6, 2.5, 11.7, t['duration'] - .7])
    t = w1('C3')
    out['W1_TIMING_CONTROLLED'] = finalize(t, 'W1_TIMING_CONTROLLED', 'Cowork | Can thiệp forecast đường tay, kiểm nhường và phục hồi', 'controlled_forecast_error',
        [dict(t=0, goal='take_H0_next_then_pack_own_box')],
        [[0, 2.1, 'Đường tay người và đích lấy món giữ nguyên. Chấm đỏ là forecast bị làm lệch có kiểm soát, không phải output lỗi tự nhiên.'],
         [2.1, 4.4, 'Robot minh họa đã đi rồi nhường khi cue thực tế và forecast mới bác bỏ đường cũ.'],
         [4.4, t['duration'] + .1, 'Tiếp tục sau khi vùng trống; đánh giá chờ cần thiết, dừng thừa và mức cản người. Chưa chạy learned policy.']],
        action=1.0, reveal=2.1, review=[1.6, 2.5, 11.7, t['duration'] - .7])
    out['W1_TIMING_CONTROLLED']['forecast_intervention'] = dict(t=[1.0, 2.1], joints=[5], offset=[-.12, -.06, 0])
    return out


def main():
    cfg = yaml.safe_load((ROOT / 'configs/scenarios/intent_review_50.yaml').read_text())
    cfg['out_dir'] = 'outputs/natural_case_preview'
    cfg['video']['caption_h'] = 128
    cfg['tasks'] = {}
    for build in (coaster_cases, t2_cases, rack_cases, mat_cases, drawer_cases, cowork_cases):
        cfg['tasks'].update(build())
    cfg['review_groups'] = {
        'I1': dict(name='Nhiều đế lót', keys=[k for k in cfg['tasks'] if k.startswith('I1_')]),
        'T2': dict(name='Trao lon sau công việc', keys=[k for k in cfg['tasks'] if k.startswith('T2_')]),
        'RACK': dict(name='Nâng giá để robot lau', keys=[k for k in cfg['tasks'] if k.startswith('RACK_')]),
        'MAT': dict(name='Nhấc bát để trải khăn', keys=[k for k in cfg['tasks'] if k.startswith('MAT_')]),
        'DRAWER': dict(name='Mở đúng ngăn kéo', keys=[k for k in cfg['tasks'] if k.startswith('DRAWER_')]),
        'W1': dict(name='Cowork: nhường, chờ, tiếp tục', keys=[k for k in cfg['tasks'] if k.startswith('W1_')]),
    }
    for key, t in cfg['tasks'].items():
        t['forecast_overlay'] = dict(path=f'outputs/natural_case_preview/motion_audit/{key}.npz', joints=[5])
        if t.get('forecast_intervention'):
            t['annotation_legend'] = 'Xanh: checkpoint ManiCast; đỏ: quỹ đạo dự báo bị can thiệp. Robot minh họa, chưa chạy ACT/physics.'
    path = ROOT / 'configs/scenarios/natural_case_preview.yaml'
    path.write_text('# Selected task-specific cases; eval-only goals; robot kinematic, forecast overlay audited separately.\n'
                    + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, width=125))
    print(path, len(cfg['tasks']), 'clips')


if __name__ == '__main__':
    main()
