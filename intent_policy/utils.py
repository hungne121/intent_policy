"""Shared helpers for experiment scripts: config loading and observation conversion."""
from __future__ import annotations
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


def observation_config(exp: dict) -> ObservationConfig:
    o = exp['observation']
    return ObservationConfig(tuple(o['cameras']), int(o['width']), int(o['height']))


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
