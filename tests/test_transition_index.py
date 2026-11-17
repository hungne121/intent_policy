"""intent_policy/benchmark/transition_index.py (docs/PIF.md §8.13)."""
import numpy as np

from intent_policy.benchmark.transition_index import index_episode, mixture_weights, transitions

S = dict(mix=0.5, pre_s=1.0, post_s=0.5, chunk_s=2.0, cue_labels={'demo': ['reach']})


def _episode(**kw):
    m = dict(scenario_id='demo', mode='early', n_frames=200, human_cue_onset_t=None, human_motions=[],
             intention_changes=[], forecast_fault=None, expert_commits=[])
    m.update(kw)
    return m


def test_transition_kinds_and_recovery():
    m = _episode(human_cue_onset_t=1.0, human_motions=[{'label': 'reach', 'start': 100}],
                 intention_changes=[{'frame': 60}],
                 expert_commits=[{'frame': 40, 'target': 'A'}, {'frame': 70, 'target': 'B'}, {'frame': 120, 'target': 'C'}])
    tr = transitions(m, 20.0, S)
    assert ('cue', 20) in tr and ('cue', 100) in tr and ('change', 60) in tr
    assert ('commit', 40) in tr and ('recovery', 70) in tr and ('commit', 120) in tr   # no change between 70 and 120


def test_window_and_leak_check():
    # cue at 20, commit A at 60, change at 90, recovery commit B at 95 (20 fps: pre 20, post 10, chunk 40 frames).
    # f = 60 (commit window): its chunk [60, 100] holds the recovery at 95 whose cause, the change at 90, is beyond
    # f + pre = 80: neither seen nor forecast -> dropped. f = 70: the change is within the forecast horizon -> kept.
    m = _episode(human_cue_onset_t=1.0, intention_changes=[{'frame': 90}],
                 expert_commits=[{'frame': 60, 'target': 'A'}, {'frame': 95, 'target': 'B'}])
    ix = index_episode(m, 20.0, S)
    assert ix['mask'][0] and ix['mask'][10] and ix['mask'][30] and not ix['mask'][31]   # cue window [0, 30]
    assert not ix['mask'][60] and ix['mask'][70] and ix['mask'][105] and not ix['mask'][106]
    assert ix['dropped_leak'] == 15                           # f = 55 .. 69: chunk reaches 95, change 90 > f + 20


def test_mixture_weights():
    in_t = np.array([True] * 25 + [False] * 75)
    w = mixture_weights(in_t, 0.5)
    assert np.isclose(w.sum(), 1.0) and np.isclose(w[in_t].sum(), 0.5 + 0.5 * 0.25)
    assert np.allclose(mixture_weights(in_t, 0.0), 1 / 100)   # mix 0 = uniform (D0)
