"""Base class of the 10 task experts (QĐ 53/QĐ-DL §3).

An expert owns the task FSM: phases, targets, grasp / place / handover conditions, observed events that
confirm phase changes, success and allowed contacts. It never runs ManiCast itself: forecast readings come from
the shared `ForecastPlanner` (`self.plan`), which returns nothing in the observed mode.
"""
import numpy as np


class BaseExpert:
    task = None
    reading = None            # 'follow' (I, C tasks) or 'avoid' (W tasks), QĐ 54/QĐ-DL

    def __init__(self, ep):
        self.ep = ep
        self.plan = ep.planner

    @property
    def forecast_mode(self):
        return self.ep.mode == 'forecast'

    def run(self):
        """Robot FSM generator: yields one action per control tick."""
        raise NotImplementedError

    def success(self):
        """(bool, failure_reason or None) from the final world state."""
        raise NotImplementedError

    def extra(self):
        return {}
