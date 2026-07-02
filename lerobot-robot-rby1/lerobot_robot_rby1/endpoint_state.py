"""Process-shared endpoint-state toggle (0/1) for the RB-Y1.

This is a tiny thread-safe holder that bridges the joystick teleoperator and
the follower robot, which are separate LeRobot objects living in the *same*
process during ``lerobot-record`` / ``lerobot-teleoperate``:

* :class:`lerobot_teleoperator_rby1.Rby1Keyboard` reads the joystick's toggle
  button and calls :func:`set` on every rising edge (and :func:`reset` to 0).
* :class:`lerobot_robot_rby1.Rby1` reads it with :func:`get` inside
  ``get_observation`` (when ``use_endpoint_state`` is enabled) so the value is
  recorded as an observation column named ``endpoint_state``.

Semantics: the value starts at 0 and flips 0<->1 each time the operator presses
the toggle button; :func:`reset` forces it back to 0 (e.g. at episode start).

Because LeRobot's teleoperator only produces *actions* and never observations,
this module is the bridge that lets a teleop-side button surface as an
observation. It lives in the robot package (the teleop already depends on it,
so there is no circular import) and defaults to ``0.0`` when no teleop writes to
it (e.g. during policy inference), leaving the observation well-defined.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_value: float = 0.0


def set(value: float) -> None:
    """Publish the current endpoint-state value (typically 0.0 or 1.0)."""
    global _value
    with _lock:
        _value = float(value)


def get() -> float:
    """Return the most recently published value (0.0 before any write)."""
    with _lock:
        return _value


def reset() -> None:
    """Reset the latch to 0.0 (e.g. at the start of a recording episode)."""
    set(0.0)
