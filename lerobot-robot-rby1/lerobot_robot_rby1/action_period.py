"""Estimate the period at which actions arrive, for the EE command's minimum time.

A Cartesian impedance command asks the robot to reach its target in
``minimum_time``. That only tracks smoothly when it matches the rate the
targets arrive at: too long and the robot lags behind the stream, too short
and it reaches each target early and moves in steps. The rate is whatever the
caller runs at (``--fps``, teleoperator, policy), so it is measured.
"""

from __future__ import annotations

# The estimate is kept within these bounds (seconds): 100 Hz .. 5 Hz.
MIN_PERIOD = 0.01
MAX_PERIOD = 0.2
# Weight of a new interval in the moving average.
SMOOTHING = 0.2
# An interval longer than this many periods is a pause (episode saving, a
# stalled policy), not the action rate, and is ignored ...
PAUSE_FACTOR = 3.0
# ... unless this many arrive in a row: then the rate itself changed.
PAUSES_BEFORE_ADOPTING = 5


class ActionPeriodEstimator:
    """Exponential moving average of the interval between actions."""

    def __init__(self, initial_period: float) -> None:
        self._initial = min(max(float(initial_period), MIN_PERIOD), MAX_PERIOD)
        self.reset()

    def reset(self) -> None:
        """Forget the history: the next action starts from the initial period."""
        self._period = self._initial
        self._last_time: float | None = None
        self._pauses = 0

    @property
    def period(self) -> float:
        return self._period

    def update(self, now: float) -> float:
        """Record an action at time ``now`` (seconds, monotonic) and return the period."""
        last, self._last_time = self._last_time, now
        if last is None or now <= last:
            return self._period
        interval = now - last
        if interval > PAUSE_FACTOR * self._period:
            self._pauses += 1
            if self._pauses < PAUSES_BEFORE_ADOPTING:
                return self._period
            self._period = interval  # a lasting change of rate
        else:
            self._period += SMOOTHING * (interval - self._period)
        self._pauses = 0
        self._period = min(max(self._period, MIN_PERIOD), MAX_PERIOD)
        return self._period
