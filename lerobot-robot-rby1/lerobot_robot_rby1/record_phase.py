"""Which phase of ``lerobot-record`` is driving the robot / teleoperator.

``lerobot-record`` runs one ``record_loop`` per episode and, between episodes,
another one for the reset phase. lerobot has no episode hook for robots or
teleoperators, so the phase is read from the calling ``record_loop`` frame: it
gets a dataset while an episode is recorded and none during the reset phase.
"""

from __future__ import annotations

import sys

RECORD = "record"
RESET = "reset"

# record_loop calls get_action / send_action directly; a few extra frames are
# allowed for wrappers.
_MAX_DEPTH = 6


def record_phase() -> str | None:
    """Phase of the ``record_loop`` on the call stack: "record", "reset" or None.

    None means the caller is not ``lerobot-record`` (lerobot-teleoperate,
    rollout, tests) or a lerobot version whose loop looks different; callers
    then keep their normal behaviour.
    """
    frame = sys._getframe(1)
    for _ in range(_MAX_DEPTH):
        frame = frame.f_back
        if frame is None:
            return None
        if frame.f_code.co_name == "record_loop" and "dataset" in frame.f_locals:
            return RESET if frame.f_locals["dataset"] is None else RECORD
    return None
