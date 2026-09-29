"""Ground-truth human state exposed by the scripted human (never fed to the Phase-1 policy)."""
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class HumanState:
    """All positions in the world frame [m]; velocities [m/s].

    `selected_object` / `selected_target` are the human's ground-truth choice. They exist so
    later oracle-intention phases can read them; Phase-1 observations never include them.
    """
    timestamp: float
    torso_position: np.ndarray
    torso_velocity: np.ndarray
    hand_position: np.ndarray
    hand_velocity: np.ndarray
    interaction_state: str
    selected_object: str | None = None
    selected_target: str | None = None
    holding: str | None = None
    motion_label: str | None = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ('torso_position', 'torso_velocity', 'hand_position', 'hand_velocity'):
            d[k] = np.asarray(d[k]).round(6).tolist()
        return d
