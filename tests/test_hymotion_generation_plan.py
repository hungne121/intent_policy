import sys
import types
import numpy as np
import pytest
import yaml
from workstation.rent.hymotion.gen_motions import main


def test_chunked_generation_preserves_seed_and_split_metadata_and_resume(tmp_path, monkeypatch):
    calls = []
    class Tensor:
        def __init__(self, x): self.x = x
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.x
    class Runtime:
        def __init__(self, **kwargs): calls.append('load')
        def generate_motion(self, **kw):
            seeds = list(map(int, kw['seeds_csv'].split(',')))
            calls.append(seeds)
            a = np.array(seeds, dtype=np.float32)[:, None, None, None] * np.ones((len(seeds), 3, 52, 3))
            return None, None, {k: Tensor(a) for k in ['keypoints3d', 'rot6d', 'transl', 'root_rotations_mat']}
    module = types.ModuleType('hymotion.utils.t2m_runtime')
    module.T2MRuntime = Runtime
    monkeypatch.setitem(sys.modules, 'hymotion.utils.t2m_runtime', module)
    cfg = dict(model='test', cfg_scale=5, seeds=4, generation_batch_size=2,
        seed_range=[10000, 20000], sample_splits=['train', 'train', 'val', 'test'],
        prompts={'reach': dict(d=3.5, text='Reach forward.')})
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['gen', '--config', str(path), '--model', str(tmp_path), '--out', str(out)])
    main()
    with np.load(out / 'reach.npz') as z:
        assert z['sample_splits'].tolist() == cfg['sample_splits']
        assert z['keypoints3d'][:, 0, 0, 0].tolist() == z['seeds'].tolist()
    assert len(calls) == 3  # one model load, two batches
    main()
    assert len(calls) == 3  # resume is cheap and cannot resplit clips
    cfg['prompts']['reach']['d'] = 4.5
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='generation settings mismatch'):
        main()
