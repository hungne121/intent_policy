"""Phase 2: information-timing contract, oracle features, corruptions, intent fusion, counterfactual demonstrations."""
import numpy as np
import pytest
import torch

from intent_policy.benchmark.runner import ExpertAgent, NoisyExpertAgent, run_episode
from intent_policy.intent.corruption import Delayed, Noisy, Wrong, derangement, make_corruption
from intent_policy.intent.oracle import OracleIntentProvider
from intent_policy.intent.representation import PREFIX, feature_shapes, features
from intent_policy.policies.hri_act import HRIACTConfig
from intent_policy.policies.intent_fusion import IntentFusion
from intent_policy.scenarios.config import random_spec
from intent_policy.scenarios.scenario_registry import make_scenario
from intent_policy.utils import load_yaml

PHASE2 = {'expert': {'trigger': 'evidence'}}


@pytest.fixture(scope='module')
def handover():
    sc = make_scenario('t2_handover', PHASE2)
    yield sc
    sc.close()


def run_with_oracle(sc, seed, provider=None, agent=None, spec=None):
    provider = provider or OracleIntentProvider()
    out = []
    rec = run_episode(sc, agent or ExpertAgent(), seed, keep_trace=False, spec=spec,
                      on_frame=lambda o, d, t, s: out.append((s.time, provider.record(s), d)))
    return rec, out


def evident_time(rec):
    return next(e['timestamp'] for e in rec['events'] if e['event_type'] == 'human_intention_evident')


def test_oracle_target_is_unknown_until_the_cue_makes_it_predictable(handover):
    rec, frames = run_with_oracle(handover, 4)
    assert rec['success']
    t_ev = evident_time(rec)
    cue = next(e['timestamp'] for e in rec['events'] if e['event_type'] == 'human_cue_onset')
    assert cue < t_ev                                            # evidence comes after the cue starts
    before = [r for t, r, _ in frames if t < t_ev - 0.05]
    after = [r for t, r, _ in frames if t_ev + 0.05 < t < t_ev + 1.0]
    assert before and all(r.object_key is None for r in before)
    assert after and all(r.object_key == handover.human.selected_object for r in after)
    commits = [(t, d.extras['expert_commit']) for t, _, d in frames if d.extras.get('expert_commit')]
    assert commits and commits[0][0] >= t_ev                     # the demonstrator never commits earlier


def test_lead_makes_the_oracle_available_earlier(handover):
    rec, frames = run_with_oracle(handover, 4)
    _, lead_frames = run_with_oracle(handover, 4, OracleIntentProvider(lead_s=0.2))
    first = lambda fr: next(t for t, r, _ in fr if r.object_key is not None)
    assert first(frames) - first(lead_frames) == pytest.approx(0.2, abs=0.051)


def test_predicted_hand_follows_the_contract(handover):
    handover.reset(6)
    human = handover.human
    for _ in range(400):
        handover.step(handover.env.hold())
        seg = human.segment
        if seg is not None and seg.label == 'point_object':
            progress = (human.t_last - seg.start_t) / seg.duration
            pred = human.predicted_hand_position(0.5)
            if progress < human.evidence_fraction - 0.05:
                np.testing.assert_allclose(pred, human.pos + human.vel * 0.5)
            elif progress > human.evidence_fraction + 0.05:
                np.testing.assert_allclose(pred, seg.position(min(human.t_last + 0.5, seg.end_t)))


def test_features_are_numeric_relative_and_complete(handover):
    rec, frames = run_with_oracle(handover, 2)
    _, r, _ = frames[-40]
    tcp = np.array([0.3, 0.1, 0.8])
    f = features(r, tcp)
    assert set(f) == set(feature_shapes())
    assert all(v.dtype == np.float32 and v.shape == (feature_shapes()[k],) for k, v in f.items())
    if r.object_key is not None:
        np.testing.assert_allclose(f[PREFIX + 'object_rel'], r.object_pos - tcp, atol=1e-6)
        assert f[PREFIX + 'object_valid'][0] == 1.0 and f[PREFIX + 'object_id'].sum() == 1.0


def test_policy_bundle_has_no_semantic_input():
    cfg = load_yaml('configs/policy/act_restricted_oracle.yaml')
    keys = [k for ks in cfg['intent_branches'].values() for k in ks]
    assert set(keys) <= set(feature_shapes())
    assert not any(w in k for k in keys for w in ('state', 'stage', 'receive', 'wait', 'withdraw', 'label'))
    with pytest.raises(ValueError):
        HRIACTConfig(use_intent=True, intent_branches={'x': ['human.interaction_state']}, intent_branch_dims={'x': 1},
                     device='cpu', pretrained_backbone_weights=None)


def test_wrong_information_changes_numbers_not_just_labels(handover):
    rec, frames = run_with_oracle(handover, 4)
    _, r, _ = next(fr for fr in frames if fr[1].object_key is not None)
    w = Wrong()
    w.reset(handover, 4)
    wr = w.apply(r)
    assert wr.object_key != r.object_key and np.linalg.norm(wr.object_pos - r.object_pos) > 0.1
    np.testing.assert_allclose(wr.future_pos[0.5] - wr.hand_pos, -(r.future_pos[0.5] - r.hand_pos))
    assert r.object_key == handover.human.selected_object       # the original record is untouched


def test_delay_noise_shuffle_are_exact_and_deterministic(handover):
    rec, frames = run_with_oracle(handover, 4)
    d = Delayed(0.2)
    d.reset(handover, 4)
    out = [d.apply(r) for _, r, _ in frames]
    for i in range(10, len(frames)):
        assert out[i].t == pytest.approx(frames[i - 4][1].t)     # 0.2 s = 4 ticks at 20 Hz
    runs = []
    for _ in range(2):
        n = Noisy()
        n.reset(handover, 4)
        runs.append([n.apply(r).hand_pos for _, r, _ in frames[:30]])
    np.testing.assert_array_equal(np.array(runs[0]), np.array(runs[1]))
    assert not np.allclose(runs[0][5], frames[5][1].hand_pos)
    donors = derangement(list(range(100, 110)))
    assert sorted(donors) == sorted(donors.values()) and all(k != v for k, v in donors.items())
    corruption, lead = make_corruption({'condition': 'lead', 'lead_s': 0.1})
    assert lead == 0.1


def test_counterfactual_demonstrations_share_labels_until_the_commitment(handover):
    """Same layout with the identical pair B1 / B1' and the same seed, only the pointed-at twin differs: the
    evidence-mode expert acts identically until it commits to the target, then differently."""
    base = random_spec(handover.cfg, np.random.default_rng(3))
    layout = {'B1': 'S2', 'B1p': 'S5', 'B3': 'S1'}
    labels = []
    for target in ('B1', 'B1p'):
        spec = dict(base, target=target, layout=layout, pair=True, timing='on_time')
        rec, frames = run_with_oracle(handover, 12, spec=spec)
        assert rec['success'], rec['failure']
        labels.append(([d.action_id for _, _, d in frames],
                       next(i for i, (_, _, d) in enumerate(frames) if d.extras.get('expert_commit'))))
    (la, ca), (lb, cb) = labels
    assert ca == cb and la[:ca] == lb[:ca] and la[ca:ca + 20] != lb[ca:ca + 20]


def test_dart_noise_keeps_clean_labels(handover):
    _, frames = run_with_oracle(handover, 30, agent=NoisyExpertAgent(burst_prob=0.2, burst_ticks=(3, 5)))
    noisy = [d for _, _, d in frames if d.extras.get('noise')]
    assert noisy
    assert any(d.action_id != d.extras['expert_action'] for d in noisy)


def test_fusion_starts_as_identity_and_mask_ignores_inputs():
    torch.manual_seed(0)
    fusion = IntentFusion(32, {'spatial': 18, 'motion': 6}, hidden=16, embed=8)
    h = torch.randn(2, 5, 32)
    br = {'spatial': torch.randn(2, 18), 'motion': torch.randn(2, 6)}
    torch.testing.assert_close(fusion(h, br), h)                 # zero-initialised output projection
    torch.nn.init.normal_(fusion.out.weight)
    other = {k: v + 1.0 for k, v in br.items()}
    assert not torch.allclose(fusion(h, br), fusion(h, other))
    torch.testing.assert_close(fusion(h, br, mask=True), fusion(h, other, mask=True))


def test_commitment_events_follow_the_robot(handover):
    rec, _ = run_with_oracle(handover, 3)
    commits = [e for e in rec['events'] if e['event_type'] == 'robot_target_commit']
    assert commits and commits[0]['payload']['correct'] and commits[0]['payload']['target'] == handover.human.selected_object
    assert rec['metrics']['WCR'] == 0.0 and rec['metrics']['AM'] > 0
