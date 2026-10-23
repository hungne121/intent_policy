"""Torch dataset of PIF windows (handoff §3).

A sample starts at frame s of an episode and holds L = seq_seconds * fps input frames followed by H_max = max horizon
future frames (the prediction targets of handoff §5, encoded by the target encoder, never model inputs):

  human     [L + H_max, J, 7]   objects [L + H_max, K, 6]   obj_type [K]   holding [L + H_max]
  valid     [L + H_max]  frame exists (False past the episode end -> loss masked)
  target_valid [L, n_h]  valid[tau + h] for input frame tau and horizon h
  n_input = L, horizons [n_h] (frames)

Tokens of the input frames only use frames <= tau (backward velocity uses frame s - 1 when it exists).
Training augmentation (input frames only, the future targets stay clean): Gaussian keypoint noise (sigma
`keypoint_noise_m`, before the velocity so it carries the noise), joint drop (p `joint_drop`), whole-frame human drop
(p `frame_drop`), random object order (the model must be permutation invariant).
Eval labels (pif/data/labels.py) are only returned with `cfg.supervised_warmstart` (handoff §0.4).
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

from pif.data.tokens import build_tokens, permute_objects, slice_episode


class PIFWindowDataset(Dataset):
    def __init__(self, episodes: list[dict], cfg: dict, stats: dict | None = None, train: bool = True,
                 labels: list[dict] | None = None, seed: int = 0):
        self.episodes, self.cfg, self.stats, self.train = episodes, cfg, stats, train
        self.labels = labels if cfg.get('supervised_warmstart') else None
        fps = cfg['fps']
        self.L = int(round(cfg['seq_seconds'] * fps))
        self.horizons = np.array([int(round(h * fps)) for h in cfg['horizons_s']])
        self.H = int(self.horizons.max())
        self.aug = cfg.get('augment') or {}
        self.rng = np.random.default_rng(seed)
        self.index = [(i, s) for i, e in enumerate(episodes) for s in range(max(1, len(e['holding']) - self.L + 1))]

    def __len__(self):
        return len(self.index)

    def _augment(self, ep: dict, n_in: int) -> dict:
        ep = dict(ep)
        rng, a = self.rng, self.aug
        pos, vis = ep['human_pos'].copy(), ep['human_vis'].copy()
        J = pos.shape[1]
        if a.get('keypoint_noise_m'):
            pos[:n_in] += rng.normal(0.0, a['keypoint_noise_m'], pos[:n_in].shape)
        if a.get('joint_drop'):
            vis[:n_in] &= rng.random((n_in, J)) >= a['joint_drop']
        if a.get('frame_drop'):
            vis[:n_in] &= (rng.random(n_in) >= a['frame_drop'])[:, None]
        ep['human_pos'], ep['human_vis'] = pos, vis
        if a.get('shuffle_objects'):
            ep = permute_objects(ep, rng.permutation(len(ep['obj_type'])))
        return ep

    def __getitem__(self, idx: int) -> dict:
        i, s = self.index[idx]
        ep = self.episodes[i]
        T = len(ep['holding'])
        lo = max(s - 1, 0)                      # one earlier frame for the backward velocity at s
        hi = min(s + self.L + self.H, T)
        sub = slice_episode(ep, lo, hi)
        n_in = min(s + self.L, T) - lo           # input frames of `sub` (incl. the extra one)
        if self.train:
            sub = self._augment(sub, n_in)
        tok = build_tokens(sub, self.stats)
        drop = s - lo
        n = self.L + self.H
        valid = np.zeros(n, bool)
        valid[:hi - s] = True
        pad = lambda x: np.concatenate([x[drop:], np.zeros((n - (hi - s),) + x.shape[1:], x.dtype)])
        holding = pad(tok['holding'])
        holding[~valid] = -1
        tv = np.stack([valid[np.arange(self.L) + h] for h in self.horizons], 1) & valid[:self.L, None]
        out = dict(human=torch.from_numpy(pad(tok['human'])), objects=torch.from_numpy(pad(tok['objects'])),
                   obj_type=torch.from_numpy(tok['obj_type']), holding=torch.from_numpy(holding),
                   valid=torch.from_numpy(valid), target_valid=torch.from_numpy(tv),
                   horizons=torch.from_numpy(self.horizons), n_input=self.L, episode=i, start=s)
        if self.labels is not None:
            out['labels'] = self.labels[i]
        return out


def collate(batch: list[dict]) -> dict:
    """Pad the object axis to the batch's largest K; obj_mask [B, K] is True for real objects."""
    K = max(b['obj_type'].shape[0] for b in batch)
    B, n = len(batch), batch[0]['objects'].shape[0]
    objects = torch.zeros(B, n, K, batch[0]['objects'].shape[-1])
    obj_type = torch.zeros(B, K, dtype=torch.long)
    obj_mask = torch.zeros(B, K, dtype=torch.bool)
    for j, b in enumerate(batch):
        k = b['obj_type'].shape[0]
        objects[j, :, :k] = b['objects']
        obj_type[j, :k] = b['obj_type']
        obj_mask[j, :k] = True
    out = dict(objects=objects, obj_type=obj_type, obj_mask=obj_mask)
    for key in ('human', 'holding', 'valid', 'target_valid', 'horizons'):
        out[key] = torch.stack([b[key] for b in batch])
    out['n_input'] = batch[0]['n_input']
    out['episode'] = torch.tensor([b['episode'] for b in batch])
    out['start'] = torch.tensor([b['start'] for b in batch])
    if 'labels' in batch[0]:
        out['labels'] = [b['labels'] for b in batch]
    return out
