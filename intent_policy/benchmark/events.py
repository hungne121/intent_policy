"""Benchmark event vocabulary shared by every scenario and every later phase."""
from dataclasses import dataclass, field, asdict
from enum import Enum


class EventType(str, Enum):
    EPISODE_START = 'episode_start'
    EPISODE_END = 'episode_end'
    HUMAN_MOTION_START = 'human_motion_start'
    HUMAN_MOTION_END = 'human_motion_end'
    HUMAN_CUE_ONSET = 'human_cue_onset'
    HUMAN_INTENTION_CHANGE = 'human_intention_change'
    ROBOT_MOTION_START = 'robot_motion_start'
    ROBOT_MOTION_END = 'robot_motion_end'
    ROBOT_ACTION_CHANGE = 'robot_action_change'
    INTERACTION_WINDOW_START = 'interaction_window_start'
    INTERACTION_WINDOW_END = 'interaction_window_end'
    OBJECT_GRASP = 'object_grasp'
    OBJECT_RELEASE = 'object_release'
    HUMAN_ROBOT_CONTACT = 'human_robot_contact'
    SAFETY_DISTANCE_VIOLATION = 'safety_distance_violation'
    PROTOCOL_STEP_COMPLETE = 'protocol_step_complete'
    DISRUPTION_START = 'disruption_start'
    DISRUPTION_END = 'disruption_end'
    RECOVERY_START = 'recovery_start'
    RECOVERY_COMPLETE = 'recovery_complete'
    TASK_SUCCESS = 'task_success'
    TASK_FAILURE = 'task_failure'
    # Phase 2 information-timing contract: when the human's intention becomes predictable from the
    # observable cue (payload: kind + target), and when the robot's motion commits to a candidate target.
    HUMAN_INTENTION_EVIDENT = 'human_intention_evident'
    ROBOT_TARGET_COMMIT = 'robot_target_commit'


REQUIRED_EVENT_TYPES = tuple(e.value for e in EventType)


@dataclass
class Event:
    episode_id: str
    timestamp: float
    event_type: str
    scenario_id: str
    role: str
    entity_id: str
    payload: dict = field(default_factory=dict)
    seq: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Event':
        return cls(**d)
