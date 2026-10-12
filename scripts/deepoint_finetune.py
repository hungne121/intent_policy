"""Fine-tune the released DeePoint (Nakamura et al., ICCV 2023) on simulated sensor-view data in the DP format
(scripts/deepoint_export_dp.py), with the authors' own code: `DPDataset` (use_preprocessed), `build_pointing_network`
(DP-base config), `PointingModule` (loss = cross-entropy of pointing / not + angular error of the direction; Adam,
lr 1e-4) and their class-balanced `WeightedRandomSampler` (`utils.subset_weight`).

Taken from the paper / code (docs/requirements/INTENT_ACT_v3.2.md §4.2c):
  - initialisation: the released weights (models/weight.ckpt);
  - batch size 64, Adam 1e-4 (paper §5; `configure_optimizers`);
  - pointing and non-pointing samples drawn evenly (paper; `subset_weight`);
  - the checkpoint kept = the best validation angular error (paper: "the best parameter within the epochs in terms
    of angular error measured with the validation set").
Ours:
  - that angular error aggregated as the paper defines it, the mean over all pointing windows of the epoch
    (`validation/angle_deg`). The released `validation_step` logs a per-batch mean that counts a batch without any
    pointing window as 0 deg, so with the unshuffled validation loader its `validation/direction_loss` is diluted
    (released model on the pilot validation fold: 12.4 deg logged vs the true mean); it is still logged, unchanged;
  - frame-level precision / recall of the pointing class (TP / FP / FN summed over the epoch), logged only;
  - split by seed fold (one DP venue per fold, sim-f<k>): test fold k, validation fold k+1, training the others, so
    every episode can get an out-of-fold output;
  - one epoch = one pass over the training windows (`shrink_rate` 1; the authors' 0.1 subsamples their 2M frames);
  - the paper trains "until convergence": we stop when the validation angular error has not improved for
    `--patience` epochs (early stopping, Prechelt 1998), at most `--max-epochs`.
The released model's validation / test metrics are logged first (epoch -1 baseline).

  scripts/deepoint.sh scripts/deepoint_finetune.py --data outputs/deepoint_dp/pilot_side --test-fold 0
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as TF
import torchmetrics
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'third_party' / 'deepoint'
sys.path.insert(0, str(REPO / 'src'))
import dataset as dp_dataset                       # noqa: E402
import utils as dp_utils                           # noqa: E402
from model import build_pointing_network           # noqa: E402
from pl_module import PointingModule               # noqa: E402


class GestureDataset(dp_dataset.DPDataset):
    """The authors' DPDataset with `n_classes` action classes: 0 = no gesture (id -1), class c >= 1 = action ids
    [100 (c-1), 100 c) (scripts/deepoint_export_dp.py --classes multi; n_classes = 2 is DeePoint's own table). The
    released __init__ assigns its 2-class table; the property keeps ours, so the rest of its code runs unchanged."""

    def __init__(self, keypoints_path: Path, cfg, n_classes: int = 2):
        self._classes = {0: [-1], **{c: list(range(100 * (c - 1), 100 * c)) for c in range(1, n_classes)}}
        super().__init__(keypoints_path, cfg)

    action_class_dict = property(lambda self: self._classes, lambda self, value: None)


def build_network(cfg, device: str, n_classes: int = 2, init: str | None = None):
    """DeePoint's network (`build_pointing_network`, DP-base) with the released weights; for n_classes > 2 the
    classifier is widened to n_classes rows, rows 0 / 1 (none / pointing) copied from the released one, the new rows
    PyTorch's default initialisation."""
    net = build_pointing_network(cfg, device)
    if init:
        state = torch.load(init, map_location=device)['state_dict']
        state = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in state.items()}
        n_init = state['classifier.weight'].shape[0]
        if n_init != net.classifier.out_features:
            net.classifier = torch.nn.Linear(net.classifier.in_features, n_init)
        net.load_state_dict(state)
    if net.classifier.out_features != n_classes:
        old, new = net.classifier, torch.nn.Linear(net.classifier.in_features, n_classes)
        with torch.no_grad():
            k = min(old.out_features, n_classes)
            new.weight[:k], new.bias[:k] = old.weight[:k], old.bias[:k]
        net.classifier = new
    return net.to(device)


class Module(PointingModule):
    """The authors' PointingModule (training step, loss, optimizer unchanged) whose validation / test steps also
    accumulate the angular error over every window with a direction target, the pointing-class TP / FP / FN and the
    confusion matrix of the epoch."""

    def __init__(self, model, class_names: list[str]):
        super().__init__(model)
        self.class_names = list(class_names)
        self.clasf_acc = torchmetrics.Accuracy(task='multiclass', num_classes=len(class_names))

    def _start(self):
        n = len(self.class_names)
        self._acc = dict(angle=0.0, n=0, tp=0, fp=0, fn=0, conf=np.zeros((n, n), int))

    def _step(self, batch, stage: str):
        pred = self.model(batch)
        target = batch['answer']
        classification_loss = TF.cross_entropy(pred['action'], target['action'])
        ok = target['direction'].isnan().logical_not().all(dim=1)
        eps = 1e-6
        ang = torch.arccos(torch.clamp(TF.cosine_similarity(pred['direction'][ok], target['direction'][ok]), -1 + eps, 1 - eps))
        direction_loss = ang.mean() if ok.any() else torch.tensor(0.0)
        self.log_dict({f'{stage}/classification_loss': classification_loss,          # = the released step's logs
                       f'{stage}/direction_loss': direction_loss * 180 / np.pi,
                       f'{stage}/accuracy': self.clasf_acc(pred['action'], target['action'])},
                      batch_size=batch['idx'].shape[0], sync_dist=True)
        said, truth = pred['action'].argmax(-1) == 1, target['action'] == 1
        a = self._acc
        a['angle'] += float(ang.sum()) * 180 / np.pi
        a['n'] += int(ok.sum())
        a['tp'] += int((said & truth).sum()); a['fp'] += int((said & ~truth).sum()); a['fn'] += int((~said & truth).sum())
        np.add.at(a['conf'], (target['action'].cpu().numpy(), pred['action'].argmax(-1).cpu().numpy()), 1)

    def _end(self, stage: str):
        a = self._acc
        logs = {f'{stage}/angle_deg': a['angle'] / max(a['n'], 1),
                f'{stage}/precision': a['tp'] / max(a['tp'] + a['fp'], 1),
                f'{stage}/recall': a['tp'] / max(a['tp'] + a['fn'], 1)}
        if len(self.class_names) > 2:
            c = a['conf']
            for i, name in enumerate(self.class_names):
                logs[f'{stage}/recall_{name}'] = c[i, i] / max(c[i].sum(), 1)
                logs[f'{stage}/precision_{name}'] = c[i, i] / max(c[:, i].sum(), 1)
            self.confusion = {stage: c.tolist()}
        self.log_dict(logs)

    def on_validation_epoch_start(self):
        self._start()

    def validation_step(self, batch, batch_idx):
        self._step(batch, 'validation')

    def on_validation_epoch_end(self):
        self._end('validation')

    def on_test_epoch_start(self):
        self._start()

    def test_step(self, batch, batch_idx):
        self._step(batch, 'test')

    def on_test_epoch_end(self):
        self._end('test')


def config(data: Path, venues: list[str], bs: int, workers: int):
    cfg = OmegaConf.load(REPO / 'conf' / 'base.yaml')
    cfg.model = OmegaConf.load(REPO / 'conf' / 'model' / 'DP-base.yaml')
    cfg.data = OmegaConf.load(REPO / 'conf' / 'data' / 'DP.yaml')
    cfg.hardware = OmegaConf.load(REPO / 'conf' / 'hardware' / 'full.yaml')
    cfg.data.data_root, cfg.data.label_root = f'{data}/frames/', f'{data}/labels/'
    cfg.data.keypoints_root = f'{data}/keypoints/'
    cfg.data.venue_names = venues
    cfg.use_preprocessed, cfg.shrink_rate = True, 1.0
    cfg.hardware.gpus, cfg.hardware.bs, cfg.hardware.nworkers = 1, bs, workers
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='scripts/deepoint_export_dp.py output')
    p.add_argument('--test-fold', type=int, default=0)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--init', default=str(REPO / 'models' / 'weight.ckpt'))
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--max-epochs', type=int, default=40)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--out', help='default: outputs/train/deepoint/<date>/<time>_fold<k>')
    args = p.parse_args()
    data = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    now = datetime.now()
    out = Path(args.out) if args.out else ROOT / 'outputs/train/deepoint' / now.strftime('%Y-%m-%d') / f"{now.strftime('%H-%M-%S')}_fold{args.test_fold}"
    out.mkdir(parents=True, exist_ok=True)
    (out / 'command.txt').write_text(' '.join(sys.argv) + '\n')
    export = json.loads((data / 'export.json').read_text())
    venues = sorted(export['venues'])
    class_names = export.get('class_names', ['none', 'point'])
    test_v, val_v = f'sim-f{args.test_fold}', f'sim-f{(args.test_fold + 1) % args.folds}'
    assert test_v in venues and val_v in venues, f'{test_v} / {val_v} not in {venues}'
    cfg = config(data, venues, args.batch_size, args.workers)
    (out / 'config.yaml').write_text(OmegaConf.to_yaml(cfg))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    network = build_network(cfg, device, len(class_names), args.init)
    module = Module(network, class_names)

    ds = GestureDataset(Path(cfg.data.keypoints_root), cfg, len(class_names))
    venue_of = [Path(d.venue_path).name for d in ds.datalist]
    split = {'train': [i for i, v in enumerate(venue_of) if v not in (test_v, val_v)],
             'val': [i for i, v in enumerate(venue_of) if v == val_v],
             'test': [i for i, v in enumerate(venue_of) if v == test_v]}
    cls = lambda i: ds.rev_action_class_dict[ds.datalist[i].action_id]
    info = {k: dict(windows=len(idx), **{n: int(sum(cls(i) == c for i in idx)) for c, n in enumerate(class_names)})
            for k, idx in split.items()}
    info.update(test_venue=test_v, val_venue=val_v, init=args.init)
    print(json.dumps(info, indent=1), flush=True)
    (out / 'split.json').write_text(json.dumps(info, indent=1))
    train_ds = Subset(ds, split['train'])
    sampler = WeightedRandomSampler(dp_utils.subset_weight(train_ds), len(train_ds))
    kw = dict(batch_size=cfg.hardware.bs, num_workers=cfg.hardware.nworkers,
              persistent_workers=cfg.hardware.nworkers > 0)
    train_dl = DataLoader(train_ds, sampler=sampler, **kw)
    val_dl = DataLoader(Subset(ds, split['val']), **kw)
    test_dl = DataLoader(Subset(ds, split['test']), **kw)

    ckpt = pl.callbacks.ModelCheckpoint(dirpath=out / 'checkpoints', monitor='validation/angle_deg', mode='min',
                                        save_top_k=1, save_last=True, filename='{epoch}-{step}')
    stop = pl.callbacks.EarlyStopping(monitor='validation/angle_deg', mode='min', patience=args.patience)
    logger = pl.loggers.CSVLogger(save_dir=out, name='logs')
    trainer = pl.Trainer(devices=1, max_epochs=args.max_epochs, callbacks=[ckpt, stop], logger=logger,
                         log_every_n_steps=10, num_sanity_val_steps=0)
    base = {'val': trainer.validate(module, val_dl, verbose=False)[0], 'test': trainer.test(module, test_dl, verbose=False)[0]}
    print('released weights:', json.dumps(base, indent=1), flush=True)
    trainer.fit(module, train_dl, val_dl)
    best = ckpt.best_model_path
    tuned = {'val': trainer.validate(module, val_dl, ckpt_path=best, verbose=False)[0],
             'test': trainer.test(module, test_dl, ckpt_path=best, verbose=False)[0]}
    print('fine-tuned (best):', json.dumps(tuned, indent=1), flush=True)
    (out / 'metrics.json').write_text(json.dumps(dict(released=base, finetuned=tuned, best_checkpoint=best,
                                                      epochs=trainer.current_epoch, class_names=class_names,
                                                      test_confusion=getattr(module, 'confusion', None)), indent=1))


if __name__ == '__main__':
    main()
