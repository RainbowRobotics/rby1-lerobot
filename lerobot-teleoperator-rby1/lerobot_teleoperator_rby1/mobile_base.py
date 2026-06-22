"""Mobile-base velocity integrator driven by VR thumbsticks.

Tiny stateful helper that turns per-step thumbstick deflections into a
``(linear[2], angular)`` velocity command for the robot's omnidirectional
base.  The integration is exponential: each step the current velocity is
nudged toward the user's input by ``acc_gain`` and decayed by ``damping``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MobileBaseGains:
    """Tunable gains for :class:`MobileBaseIntegrator`."""

    linear_acc_gain: float = 0.3
    angular_acc_gain: float = 0.7
    linear_damping: float = 0.5
    angular_damping: float = 0.5
    output_scale: float = 2.0


class MobileBaseIntegrator:
    """Integrate thumbstick inputs into a continuous base velocity command."""

    def __init__(self, gains: MobileBaseGains) -> None:
        self._gains = gains
        self._linear = np.zeros(2)
        self._angular = 0.0

    @property
    def velocity(self) -> tuple[np.ndarray, float]:
        """Latest scaled ``(linear, angular)`` velocity from :meth:`update`."""
        g = self._gains
        return g.output_scale * self._linear, g.output_scale * self._angular

    def reset(self) -> None:
        """Zero the internal velocity state (e.g. after re-init)."""
        self._linear = np.zeros(2)
        self._angular = 0.0

    def update(
        self,
        right_thumbstick: tuple[float, float] | None,
        left_thumbstick: tuple[float, float] | None,
    ) -> tuple[np.ndarray, float]:
        """Apply one integration step and return the scaled velocity command.

        Parameters
        ----------
        right_thumbstick
            ``(x, y)`` axis values from the right controller (drives linear
            velocity), or ``None`` when the controller is absent.
        left_thumbstick
            ``(x, y)`` axis values from the left controller (drives angular
            velocity), or ``None`` when the controller is absent.

        Returns
        -------
        (linear, angular)
            ``linear`` is a 2-vector (m/s, robot frame), ``angular`` is a
            scalar (rad/s), both already multiplied by ``output_scale``.
        """
        g = self._gains
        if right_thumbstick is not None:
            ax_x, ax_y = right_thumbstick
            self._linear += g.linear_acc_gain * np.array([ax_y, -ax_x])
        if left_thumbstick is not None:
            ax_x, _ = left_thumbstick
            self._angular += g.angular_acc_gain * (-ax_x)

        self._linear *= 1.0 - g.linear_damping
        self._angular *= 1.0 - g.angular_damping

        return g.output_scale * self._linear, g.output_scale * self._angular
