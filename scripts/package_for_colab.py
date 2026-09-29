"""Bundle the project code (and optionally one dataset) into a tar.gz for training on Google Colab.

Upload the archive to Google Drive (e.g. MyDrive/intent_policy/) and run colab/train_hri_act.ipynb.
Outputs, caches, archives and other datasets are excluded.

  ./run.sh -m scripts.package_for_colab --dataset outputs/datasets/hri_phase2_oracle --out outputs/colab/intent_policy.tar.gz
"""
import argparse
import tarfile
from env.base_env import ROOT
from scripts.common import resolve

CODE = ['assets', 'benchmark', 'configs', 'controllers', 'env', 'experts', 'human', 'intent', 'policies', 'scenarios',
        'scripts', 'colab', 'requirements.txt', 'run.sh', 'pytest.ini', 'README.md']


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', help='dataset root to include (relative to the project)')
    p.add_argument('--out', default='outputs/colab/intent_policy.tar.gz')
    args = p.parse_args()
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    skip = lambda ti: None if '__pycache__' in ti.name or ti.name.endswith('.pyc') else ti
    with tarfile.open(out, 'w:gz') as tar:
        for item in CODE:
            if (ROOT / item).exists():
                tar.add(ROOT / item, arcname=f'intent_policy/{item}', filter=skip)
        if args.dataset:
            ds = resolve(args.dataset)
            tar.add(ds, arcname=f'intent_policy/{ds.relative_to(ROOT)}', filter=skip)
    print(f'{out}  ({out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
