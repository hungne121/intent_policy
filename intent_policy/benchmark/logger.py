"""Episode event logger: the single, ordered event stream every metric is computed from."""
from intent_policy.benchmark.events import Event, EventType, REQUIRED_EVENT_TYPES


class EventOrderError(ValueError):
    """Raised when an event would break the monotonic time / episode boundary contract."""


class EpisodeEventLogger:
    def __init__(self, scenario_id: str, role: str):
        self.scenario_id = scenario_id
        self.role = role
        self.episode_id = ''
        self.events: list[Event] = []
        self._open = False

    def start(self, episode_id: str, timestamp: float = 0.0, **metadata) -> None:
        self.episode_id = episode_id
        self.events = []
        self._open = True
        self.log(EventType.EPISODE_START, timestamp, entity_id='episode', **metadata)

    def log(self, event_type: str | EventType, timestamp: float, entity_id: str = 'scene', **metadata) -> Event:
        event_type = EventType(event_type).value  # rejects unknown event names
        if not self._open:
            raise EventOrderError(f'{event_type} logged outside an open episode')
        if self.events and timestamp + 1e-9 < self.events[-1].timestamp:
            raise EventOrderError(f'non-monotonic timestamp {timestamp} < {self.events[-1].timestamp} ({event_type})')
        event = Event(self.episode_id, float(timestamp), event_type, self.scenario_id, self.role, entity_id,
                      dict(metadata), seq=len(self.events))
        self.events.append(event)
        return event

    def end(self, timestamp: float, **metadata) -> None:
        """Close still-open interaction windows / disruptions, then log episode_end."""
        for start, stop in ((EventType.INTERACTION_WINDOW_START, EventType.INTERACTION_WINDOW_END),
                            (EventType.DISRUPTION_START, EventType.DISRUPTION_END)):
            if self.count(start) > self.count(stop):
                self.log(stop, timestamp, entity_id='episode', closed_by='episode_end')
        self.log(EventType.EPISODE_END, timestamp, entity_id='episode', **metadata)
        self._open = False

    def count(self, event_type: str | EventType) -> int:
        event_type = EventType(event_type).value
        return sum(e.event_type == event_type for e in self.events)

    def of_type(self, event_type: str | EventType) -> list[Event]:
        event_type = EventType(event_type).value
        return [e for e in self.events if e.event_type == event_type]

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self.events]


__all__ = ['EpisodeEventLogger', 'EventOrderError', 'REQUIRED_EVENT_TYPES']
