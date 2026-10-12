"""Run the released DeePoint model (Nakamura et al., ICCV 2023; third_party/deepoint, weights models/weight.ckpt) on a
video, headless, and save its per-frame output.

DeePoint's own pipeline and modules, unchanged: OpenPifPaf (shufflenetv2k16) 2D keypoints on every frame and the
per-frame inputs of `MovieDataset`, the spatial transformer per frame, the temporal transformer over the last
`tlength` = 15 frames (1 s at the 15 fps it was trained on). Per window ending at frame t it outputs
  p_point(t)    softmax probability of 'pointing' (class 1 of `action`)
  dir(t)        3D pointing direction in the camera frame (normalised here)
Frames t < tlength - 1 have no window: NaN. Only the GUI drawing of src/demo.py is left out.

Two ways to run the same network:
  default      every frame goes through OpenPifPaf + the spatial transformer once (read sequentially), then the
               temporal transformer runs on each 15-frame window of the cached per-frame embeddings. The spatial
               embedding of a frame does not depend on the window, so the outputs equal the reference ones
               (checked with --check against --reference on the authors' demo clip).
  --reference  the released per-window loop (`MovieDataset[idx]` rebuilds all 15 frames of every window, as
               src/demo.py does): ~15x slower and ~2 GB RAM per window at 2.7K.

Runs in its own environment (python 3.10, torch 1.13.1, openpifpaf 0.13.11 as in environment/pip_freeze.txt):
  scripts/deepoint.sh scripts/deepoint_infer.py --video clip.mp4 --out clip_deepoint.npz
  scripts/deepoint.sh scripts/deepoint_infer.py --check a.npz b.npz        # compare two outputs
Several checkpoints (the fold models) on many videos in one process: OpenPifPaf and the frozen features run once per
frame, the spatial / temporal transformers of every model that needs the video (identical outputs):
  scripts/deepoint.sh scripts/deepoint_infer.py --ckpt m0.ckpt m1.ckpt --list jobs.txt
  jobs.txt: one video per line, then <model index>=<out.npz> for each model to run on it
The video should be at 15 fps (DeePoint's training rate): resample with ffmpeg -r 15 first.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from einops import repeat
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1] / 'third_party' / 'deepoint'
sys.path.insert(0, str(REPO / 'src'))
import dataset as dp_dataset                       # noqa: E402  (DeePoint's src/dataset.py)
from model import build_pointing_network           # noqa: E402  (DeePoint's src/model.py)

SPATIAL_KEYS = ('joint_id', 'rel_joint_position', 'src_key_padding_mask', 'images', 'bbox', 'keypoints')
BATCH_KEYS = ('joint_id', 'abs_joint_position', 'rel_joint_position', 'images', 'src_key_padding_mask', 'keypoints',
              'bbox')


def load_network(ckpt: Path, device: str):
    """DeePoint's network with the checkpoint's weights (Lightning prefix removed, as in src/demo.py); a gesture
    checkpoint (scripts/deepoint_finetune.py, > 2 classes) gets its wider classifier."""
    cfg = OmegaConf.load(REPO / 'conf' / 'base.yaml')
    cfg.model = OmegaConf.load(REPO / 'conf' / 'model' / 'DP-base.yaml')
    net = build_pointing_network(cfg, device)
    state = {k[len('model.'):]: v for k, v in torch.load(ckpt, map_location=device)['state_dict'].items()}
    n = state['classifier.weight'].shape[0]
    if n != net.classifier.out_features:
        net.classifier = torch.nn.Linear(net.classifier.in_features, n)
    net.load_state_dict(state)
    return net.to(device).eval(), cfg


def heads(net, final: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Class probabilities (B, C; class 1 = pointing) and the unit direction (B, 3)."""
    prob = torch.softmax(net.classifier(final), -1)
    d = torch.nn.functional.normalize(net.direction_estimator(final), dim=-1)
    return prob.cpu().numpy(), d.cpu().numpy()


def run_reference(net, ds, L: int, out: dict, batch_size: int) -> None:
    """The released loop: one MovieDataset item (15 frames) per window, PointingNetwork.forward."""
    for start in range(0, len(ds), batch_size):
        items = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
        stack = (lambda xs: xs[0].unsqueeze(0)) if len(items) == 1 else torch.stack   # no copy of the frames
        res = net({k: stack([it[k] for it in items]) for k in BATCH_KEYS})
        prob = torch.softmax(res['action'], -1)[:, 1].cpu().numpy()          # reference loop: pointing only
        d = torch.nn.functional.normalize(res['direction'], dim=-1).cpu().numpy()
        for j, it in enumerate(items):
            t = it['idx'] + L - 1
            out['p_point'][t], out['direction'][t] = prob[j], d[j]
            out['keypoints'][it['idx']:it['idx'] + L] = it['keypoints'].numpy()
        print(f'\r{min(start + batch_size, len(ds))}/{len(ds)} windows', end='', flush=True)


def run_cached(net, ds, L: int, out: dict, batch_size: int) -> None:
    """Per frame: MovieDataset's inputs of that one frame (a token_length-1 item, read sequentially) -> the spatial
    transformer's last (special) token, as PointingNetwork.forward takes it. Per window: the same temporal head."""
    run_cached_many([net], ds, L, [out], batch_size)


def run_cached_many(nets: list, ds, L: int, outs: list[dict], batch_size: int) -> None:
    """run_cached for several networks on one video: the per-frame inputs (OpenPifPaf, frozen features) once."""
    T = ds.num_frames
    embs = [[] for _ in nets]
    for i in range(T):
        ds.prev_frame = i                  # sequential read: MovieDataset seeks only when i != prev_frame
        it = ds[i]
        frame = {k: it[k].unsqueeze(0)[:, 0] for k in SPATIAL_KEYS}
        for net, out, emb in zip(nets, outs, embs):
            out['keypoints'][i] = it['keypoints'][0].numpy()
            emb.append(net.spatial_transformer(frame)[:, -1].detach())
        if i % 10 == 0 or i == T - 1:
            print(f'\r{i + 1}/{T} frames', end='', flush=True)
    print()
    for net, out, emb in zip(nets, outs, embs):
        temporal(net, emb, L, T, out, batch_size)


def temporal(net, emb: list, L: int, T: int, out: dict, batch_size: int) -> None:
    ends = list(range(L - 1, T))
    for s in range(0, len(ends), batch_size):
        tt = ends[s:s + batch_size]
        tokens = [torch.cat([emb[t - L + 1 + k] for t in tt]) for k in range(L)]           # L x (B, hidden)
        if net.use_temporal_special_token:
            tokens.append(repeat(net.temporal_special_token, f'hidden -> {len(tt)} hidden'))
        final = net.temporal_transformer(torch.stack(tokens))[:, -1]
        prob, d = heads(net, final)
        out['p_point'][tt], out['direction'][tt] = prob[:, 1], d
        if 'probs' not in out:
            out['probs'] = np.full((len(out['p_point']), prob.shape[1]), np.nan, np.float32)
        out['probs'][tt] = prob


def check(a: str, b: str) -> None:
    x, y = np.load(a), np.load(b)
    ok = np.isfinite(x['p_point']) & np.isfinite(y['p_point'])
    dp = np.abs(x['p_point'][ok] - y['p_point'][ok])
    ang = np.degrees(np.arccos(np.clip((x['direction'][ok] * y['direction'][ok]).sum(1), -1, 1)))
    print(f'{ok.sum()} common windows: |dp_point| max {dp.max():.2e} mean {dp.mean():.2e}; '
          f'direction angle max {ang.max():.3f} deg mean {ang.mean():.3f} deg')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--video')
    p.add_argument('--lr', default='r', choices=['l', 'r'], help='pointing arm (only used by the GUI demo)')
    p.add_argument('--ckpt', nargs='+', default=[str(REPO / 'models' / 'weight.ckpt')])
    p.add_argument('--out', help='.npz with p_point (T,), probs (T, C), direction (T, 3), keypoints (T, 17, 3)')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--reference', action='store_true', help='the released per-window loop (slow)')
    p.add_argument('--check', nargs=2, metavar=('A', 'B'), help='compare two output files and exit')
    p.add_argument('--list', help='jobs file: <video> <model index>=<out.npz> ... per line (several --ckpt)')
    args = p.parse_args()
    if args.check:
        return check(*args.check)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.list:
        return run_list(args, device)
    net, cfg = load_network(Path(args.ckpt[0]), device)
    L = cfg.model.tlength
    ds = dp_dataset.MovieDataset(args.video, args.lr, 1 if not args.reference else L, device)
    T = ds.num_frames
    out = dict(p_point=np.full(T, np.nan, np.float32), direction=np.full((T, 3), np.nan, np.float32),
               keypoints=np.zeros((T, 17, 3), np.float32))
    with torch.no_grad():
        (run_reference if args.reference else run_cached)(net, ds, L, out, args.batch_size)
    print()
    np.savez_compressed(args.out, **out, fps=ds.fps, width=ds.width, height=ds.height, tlength=L)
    kp = out['keypoints']
    print(f"{args.out}: {T} frames, p_point mean {np.nanmean(out['p_point']):.3f} max {np.nanmax(out['p_point']):.3f}, "
          f"frames with a person {int((kp[:, :, 2] > 0).any(1).sum())}/{T}, windows {int(np.isfinite(out['p_point']).sum())}")


def run_list(args, device: str) -> None:
    nets = [load_network(Path(c), device) for c in args.ckpt]
    L = nets[0][1].model.tlength
    jobs = [line.split() for line in Path(args.list).read_text().splitlines() if line.strip()]
    for n, (video, *targets) in enumerate(jobs):
        todo = [(int(k), o) for k, o in (t.split('=', 1) for t in targets) if not Path(o).exists()]
        if not todo:
            continue
        ds = dp_dataset.MovieDataset(video, args.lr, 1, device)
        T = ds.num_frames
        outs = [dict(p_point=np.full(T, np.nan, np.float32), direction=np.full((T, 3), np.nan, np.float32),
                     keypoints=np.zeros((T, 17, 3), np.float32)) for _ in todo]
        with torch.no_grad():
            run_cached_many([nets[k][0] for k, _ in todo], ds, L, outs, args.batch_size)
        for (k, o), out in zip(todo, outs):
            np.savez_compressed(o, **out, fps=ds.fps, width=ds.width, height=ds.height, tlength=L, ckpt=args.ckpt[k])
        print(f'[{n + 1}/{len(jobs)}] {video}: models {[k for k, _ in todo]}', flush=True)


if __name__ == '__main__':
    main()
