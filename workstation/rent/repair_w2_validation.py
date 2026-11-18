"""QD-HL-10: add independently collected W2 validation, preserving every training row/file."""
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from tools.build_train_set import episode_grade, read_metas
from tools.train_checkpoint_io import sha256

BASE = Path('/workspace/data/train_set')
EXTRA = Path('/workspace/data/w2_val_repair')
OUT = Path('/workspace/data/train_set_w2fixed')
ROOT = Path('/workspace/hrc')


def main():
    sets = {}
    for mode in ('observed', 'forecast'):
        metas = read_metas(EXTRA / f'hrc_{mode}')
        assert len(metas) == 16, (mode, len(metas))
        assert all(m['split'] == 'val' and 300800 <= m['seed'] <= 300815 for m in metas.values())
        sets[mode] = {m['seed'] for m in metas.values() if episode_grade(EXTRA, m) == 'A'}
    pairs = sets['observed'] & sets['forecast']
    assert len(pairs) >= 10, f'Only {len(pairs)} paired grade-A W2 val seeds; collect more before proceeding'
    print('W2 grade-A validation pairs:', sorted(pairs), flush=True)
    (EXTRA / 'DONE').touch()

    # Archive the new source before changing any training launch.
    from huggingface_hub import HfApi
    token = Path('/root/.hf_token').read_text().strip()
    api = HfApi(token=token)
    archive = ROOT / 'outputs/repair_20261118/w2_val_repair.tar'
    with tarfile.open(archive, 'w') as tar:
        tar.add(EXTRA, arcname=EXTRA.name, filter=lambda ti: None if '/images/' in ti.name else ti)
    repo = 'Hungdd88/hrc-collect-20261118'
    remote = 'shards_w2val_repair/w2_val_repair.tar'
    commit = api.upload_file(path_or_fileobj=archive, path_in_repo=remote, repo_id=repo, repo_type='dataset',
                             commit_message='QD-HL-10 independent W2 validation seeds 300800-300815')
    info = api.dataset_info(repo, revision=commit.oid, files_metadata=True)
    f = next(f for f in info.siblings if f.rfilename == remote)
    assert f.size == archive.stat().st_size and f.lfs.sha256 == sha256(archive)
    receipt = dict(repo=repo, revision=commit.oid, path=remote, sha256=f.lfs.sha256, paired_A_seeds=sorted(pairs))
    (ROOT / 'outputs/repair_20261118/dataset_backup.json').write_text(json.dumps(receipt, indent=2))

    subprocess.run([sys.executable, '-m', 'tools.build_train_set', '--src', '/workspace/data/shards', str(EXTRA),
                    '--splits', 'val', '--out', str(OUT)], check=True, cwd=ROOT)
    old_rows = [json.loads(l) for l in (BASE / 'manifest.jsonl').read_text().splitlines()]
    train_rows = [r for r in old_rows if r['split'] == 'train']
    val_rows = [json.loads(l) for l in (OUT / 'manifest.jsonl').read_text().splitlines()]
    for name in ('D_observed', 'D_forecast'):
        shutil.copytree(BASE / name / 'train', OUT / name / 'train', copy_function=os.link)
        for f in (BASE / name / 'train').rglob('*'):
            if f.is_file():
                assert os.path.samefile(f, OUT / name / 'train' / f.relative_to(BASE / name / 'train'))
        train_seeds = {(r['task'], r['seed']) for r in train_rows if r['dataset'] == name}
        assert not train_seeds & {(r['task'], r['seed']) for r in val_rows if r['dataset'] == name}
    (OUT / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in train_rows + val_rows))
    report = json.loads((OUT / 'build_report.json').read_text())
    old_report = json.loads((BASE / 'build_report.json').read_text())
    report['outputs'].update({k: v for k, v in old_report['outputs'].items() if k.endswith('/train')})
    report.update(splits=['train', 'val'], unchanged_train_from=str(BASE), validation_backup=receipt)
    (OUT / 'build_report.json').write_text(json.dumps(report, indent=2))
    print('VALIDATION_REPAIR_BUILT', json.dumps({k:v['by_task'] for k,v in report['outputs'].items()}), flush=True)


if __name__ == '__main__':
    main()
