"""Config-centred wiring for PIF.

Every run is described by one YAML file. A file may start with `include: [other.yaml, ...]`; included files are merged
first and the file's own keys win. `--set a.b=value` overrides on the command line.
Swappable parts (perception models, ...) register a name with `@register(kind, name)`; a config section
`{name: <registered name>, ...params}` is turned into an object with `build(kind, section)`. Adding or replacing a
part = write the class once, then change the config.
"""
from __future__ import annotations
import copy
from pathlib import Path

import yaml

from pif.utils import resolve

_REGISTRY: dict[str, dict[str, type]] = {}


def register(kind: str, name: str):
    def deco(cls):
        _REGISTRY.setdefault(kind, {})[name] = cls
        return cls
    return deco


def build(kind: str, section: dict, **extra):
    params = {k: v for k, v in section.items() if k != 'name'}
    try:
        cls = _REGISTRY[kind][section['name']]
    except KeyError:
        raise KeyError(f"no {kind} named {section.get('name')!r}; registered: {sorted(_REGISTRY.get(kind, {}))}") from None
    return cls(**params, **extra)


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = resolve(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = {}
    for inc in cfg.pop('include', []):
        base = merge(base, load_config(inc))
    cfg = merge(base, cfg)
    for a in overrides or []:
        key, value = a.split('=', 1)
        *parts, last = key.split('.')
        d = cfg
        for p in parts:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(value)
    return cfg
