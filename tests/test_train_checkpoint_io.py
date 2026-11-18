import json
import subprocess
from pathlib import Path

import pytest

from tools.train_checkpoint_io import publish_checkpoint, stage_run, verify_checkpoint


def write_checkpoint(path, payload=b'optimizer-and-model'):
    def write(d):
        (d / 'training_state.pt').write_bytes(payload)
        (d / 'pretrained_model').mkdir()
        (d / 'pretrained_model/config.json').write_text('{}')
    publish_checkpoint(path, write)


def test_failed_save_preserves_previous_checkpoint(tmp_path):
    target = tmp_path / 'last'
    write_checkpoint(target)
    def fail(d):
        (d / 'training_state.pt').write_bytes(b'partial')
        raise OSError('disk full')
    with pytest.raises(OSError):
        publish_checkpoint(target, fail)
    verify_checkpoint(target)
    assert (target / 'training_state.pt').read_bytes() == b'optimizer-and-model'


def test_backup_includes_latest_optimizer_and_ignores_incomplete(tmp_path):
    run, stage = tmp_path / 'run', tmp_path / 'snapshot'
    root = run / 'M0/checkpoints'
    write_checkpoint(root / '002000', b'old')
    write_checkpoint(root / '004000', b'new optimizer')
    (root / '006000').mkdir()
    (root / '006000/training_state.pt').write_bytes(b'partial')
    assert stage_run(run, stage) == {'M0': 4000}
    restored = stage / 'M0/checkpoints/resume'
    verify_checkpoint(restored)
    assert (restored / 'training_state.pt').read_bytes() == b'new optimizer'


def test_backup_rejects_corruption(tmp_path):
    root = tmp_path / 'run/M0/checkpoints/002000'
    write_checkpoint(root)
    (root / 'training_state.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        stage_run(tmp_path / 'run', tmp_path / 'snapshot')


def test_parallel_launcher_does_not_wait_for_forever_uploader(tmp_path):
    script = (Path(__file__).parents[1] / 'workstation/rent/run_train_m0m1.sh').read_text()
    block = script[script.index('# Wait only for training children'):script.index('[ $FAIL = 0 ]')]
    prefix = '''set -euo pipefail
RUN="$1"; MODELS="M0 M1-noH M1"; PARALLEL=1; NW=1; CKPT_EVERY_S=100
run_one() { sleep 0.05; printf 'step=2\\nTRAIN_FINISHED\\n' > "$RUN/$1.log"; }
ckpt_upload() { echo backup > "$RUN/backup"; }
'''
    subprocess.run(['bash', '-c', prefix + block + '\ntest "$FAIL" = 0', '_', str(tmp_path)],
                   check=True, timeout=5, capture_output=True)
    assert (tmp_path / 'backup').exists()
