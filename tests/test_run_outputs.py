"""Run folders, run logs and episode videos laid out as LeRobot does (intent_policy/utils.py,
intent_policy/benchmark/video.py)."""
import re
import sys

import av
import numpy as np

from intent_policy.benchmark.runner import ExpertAgent, run_episode
from intent_policy.benchmark.video import MAIN_H, MAIN_W, SIDE_W, EpisodeVideo, VideoWriter, compose
from intent_policy.sim.base_env import ROOT
from intent_policy.utils import checkpoint_name, log_run, run_dir


def test_run_dir_is_dated_unless_given(tmp_path):
    d = run_dir('eval', 'intent_act_late_expert')
    assert d.parent.parent == ROOT / 'outputs' / 'eval' and re.fullmatch(r'\d{4}-\d{2}-\d{2}', d.parent.name)
    assert re.fullmatch(r'\d{2}-\d{2}-\d{2}_intent_act_late_expert', d.name) and not d.exists()
    assert run_dir('eval', 'x', tmp_path) == tmp_path


def test_checkpoint_name():
    assert checkpoint_name('outputs/train/intent_act/A/checkpoints/010000/pretrained_model') == 'A-010000'
    assert checkpoint_name('/r/2026-10-04/08-00-00_intent_act_late/checkpoints/last/pretrained_model') == \
        '08-00-00_intent_act_late-last'
    assert checkpoint_name('/r/my_model') == 'my_model'


def test_log_run_keeps_console_output_and_command(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'stdout', sys.stdout)      # restored after the test
    monkeypatch.setattr(sys, 'stderr', sys.stderr)
    monkeypatch.setattr(sys, 'argv', ['scripts/evaluate.py', '--expert', '--workers', '2'])
    log_run(tmp_path, 'scripts.evaluate')
    log_run(tmp_path)                                    # again in the same process (a pool worker): no duplicates
    print('episode line')
    sys.stdout.flush()
    assert (tmp_path / 'log.txt').read_text().count('episode line') == 1
    assert './run.sh -m scripts.evaluate --expert --workers 2' in (tmp_path / 'command.txt').read_text()


def test_compose_and_writer(tmp_path):
    side = [np.full((192, 256, 3), 128, np.uint8)] * 2
    frame = compose(np.zeros((192, 256, 3), np.uint8), side, 'T1 #0 seed 1', ['0.50s object_grasp object=B1'],
                    'FAILURE: timeout', side_names=['top', 'wrist'])
    assert frame.shape == (MAIN_H, MAIN_W + SIDE_W, 3) and frame.dtype == np.uint8
    w = VideoWriter(tmp_path / 'v.mp4')
    for _ in range(5):
        w.add(frame)
    w.close()
    with av.open(str(tmp_path / 'v.mp4')) as c:
        assert sum(1 for _ in c.decode(video=0)) == 5


def test_episode_video_is_kept_or_dropped(scenarios, tmp_path):
    rec = run_episode(scenarios['t1_pick_place'], ExpertAgent(), 3, keep_trace=False,
                      on_frame=(video := EpisodeVideo(tmp_path, 'T1 seed 3')))
    path = video.finish(rec, tmp_path / 'T1__seed3__success.mp4')
    with av.open(str(path)) as c:
        n = sum(1 for _ in c.decode(video=0))
    assert rec['success'] and n >= rec['n_decisions'] and not list(tmp_path.glob('.*.part.mp4'))
    assert EpisodeVideo(tmp_path, 'dropped').finish(rec, None) is None and not list(tmp_path.glob('.*.part.mp4'))
