"""Shared helpers for experiment scripts: config loading, run folders and observation conversion."""
from __future__ import annotations
import shlex
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import yaml
from intent_policy.sim.base_env import ROOT
from intent_policy.benchmark.runner import ObservationConfig
from intent_policy.sim.restricted_action import RestrictedActionConfig


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    return yaml.safe_load((path if path.is_absolute() else ROOT / path).read_text())


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def run_dir(kind: str, name: str, output_dir: str | Path | None = None) -> Path:
    """Run folder as LeRobot lays them out: `output_dir` when given, else outputs/<kind>/<YYYY-MM-DD>/<HH-MM-SS>_<name>."""
    if output_dir:
        return resolve(output_dir)
    now = datetime.now()
    return ROOT / 'outputs' / kind / now.strftime('%Y-%m-%d') / f"{now.strftime('%H-%M-%S')}_{name}"


def checkpoint_name(checkpoint: str | Path) -> str:
    """<run>/checkpoints/<step>/pretrained_model -> '<run>-<step>' (else the folder's name): names eval / video runs."""
    parts = resolve(checkpoint).parts
    if 'checkpoints' in parts[:-1]:
        i = len(parts) - 1 - parts[::-1].index('checkpoints')
        return f'{parts[i - 1]}-{parts[i + 1]}'
    return parts[-1]


class _Tee:
    """A console stream that also writes everything into the run's log file."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, s: str) -> int:
        self.stream.write(s)
        self.log.write(s)
        return len(s)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):                # fileno, isatty, encoding, ... of the console stream
        return getattr(self.stream, name)


_LOGGED: set[Path] = set()


def log_run(out: Path, module: str | None = None) -> None:
    """Keep this process's console output in <out>/log.txt (appended: resumed runs and worker processes add to it;
    once per process and file); with `module`, also append the command line to <out>/command.txt."""
    out.mkdir(parents=True, exist_ok=True)
    path = (out / 'log.txt').resolve()
    if path not in _LOGGED:
        _LOGGED.add(path)
        log = open(path, 'a', buffering=1)
        sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
    if module:
        with open(out / 'command.txt', 'a') as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')}  ./run.sh -m {module} {shlex.join(sys.argv[1:])}\n")


def observation_config(exp: dict) -> ObservationConfig:
    o = exp['observation']
    return ObservationConfig(tuple(o['cameras']), int(o['width']), int(o['height']), bool(o.get('command_state', False)))


def controller_config(exp: dict) -> RestrictedActionConfig:
    return RestrictedActionConfig.load(resolve(exp['controller']))


def obs_to_batch(obs: dict, device: str) -> dict:
    """Scenario observation (HWC uint8 images) -> un-normalised LeRobot batch (CHW float [0,1])."""
    batch = {}
    for k, v in obs.items():
        if k.startswith('observation.images.'):
            batch[k] = torch.from_numpy(np.ascontiguousarray(v)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        else:
            batch[k] = torch.from_numpy(np.asarray(v, dtype=np.float32)).unsqueeze(0).to(device)
    return batch
