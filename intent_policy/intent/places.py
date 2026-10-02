"""World positions of the intent target places (configs/layout.yaml), shared by the tracker and the sources.

Objects are aimed at their centre (a 4 cm cube: table + 2 cm), place zones at the table top, hand zones at the
receiving palm height. U1 / U2 are the T3 cube spots; the T4 regions use the robot zone centre and the T4-neg
`edge` point (configs/scenarios/t4_interrupt.yaml).
"""
from __future__ import annotations
import numpy as np
from intent_policy.utils import load_yaml

OBJECT_DZ = 0.02


def target_positions(schema: dict) -> dict[str, np.ndarray]:
    scene = load_yaml('configs/scene/common.yaml')
    layout = load_yaml(scene['layout'])
    z0 = float(scene['table_top_z'])
    zones = layout['zones']
    kinds = schema['target_kinds']
    pos = {}
    for k in schema['targets'][1:]:
        if k in zones:
            z = zones[k]
            dz = OBJECT_DZ if k in kinds['object'] else float(z.get('height', 0.0))
            pos[k] = np.array([*z['xy'], z0 + dz], float)
    u = next(z for z in zones.values() if z['kind'] == 'human_store')
    for i, xy in enumerate(u['blocks']):
        pos[f'U{i + 1}'] = np.array([*xy, z0 + OBJECT_DZ], float)
    rz = layout['robot_zone']
    pos['robot_zone'] = np.array([np.mean(rz['x']), np.mean(rz['y']), z0 + rz['z_top'] / 2], float)
    t4 = load_yaml('configs/scenarios/t4_interrupt.yaml')['human_behavior']['negative_points']
    pos['zone_edge'] = np.array([t4['edge'][0], t4['edge'][1], z0 + t4['edge'][2]], float)
    missing = [k for k in schema['targets'][1:] if k not in pos]
    if missing:
        raise KeyError(f'no position for targets {missing}')
    return pos


def target_matrix(schema: dict) -> np.ndarray:
    """(K, 3) positions in `targets` order; row 0 (`none`) is NaN."""
    pos = target_positions(schema)
    return np.array([np.full(3, np.nan)] + [pos[k] for k in schema['targets'][1:]], float)
