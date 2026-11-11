"""Grab-and-move interaction for the headset camera panels (pure geometry).

Mirrors the drag handle of the CloudXR WebXR client's control panel
(``CloudXR3DUI``: a bar above the panel, ``@react-three/handle`` with
``rotate=false`` / ``scale=false`` — position only). The camera panels are
server-side Televiz layers, so the same behaviour is re-implemented here from
the controller *aim* rays:

* **hover** — the ray of a hand hits a panel's handle bar;
* **grab**  — a *free* hand (arm not clutched) squeezes the trigger while
  hovering: the grab point stays attached to the ray at the distance it had,
  and the panel centre keeps its offset from that point;
* **release** — the trigger drops below the release threshold; the panel
  stays where it is (the caller persists the layout).

Everything is expressed in the Televiz (OpenXR stage) frame; the caller
converts panel centres back to lock-mode layouts. No session objects here so
it is unit-testable.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

HANDS = ("right", "left")


@dataclass(frozen=True)
class HandInput:
    """One controller in the Televiz frame."""

    origin: np.ndarray       # (3,) aim ray origin
    direction: np.ndarray    # (3,) unit aim direction
    trigger: float           # [0, 1]
    free: bool               # arm not clutched -> may grab
    thumb_y: float = 0.0     # push / pull while grabbing


def hand_input(
    aim_pose_base: np.ndarray,
    anchor_T_base: np.ndarray,  # noqa: N803
    trigger: float,
    free: bool,
    thumb_y: float = 0.0,
) -> HandInput:
    """Build a :class:`HandInput` from a controller aim pose given in the robot base frame.

    ``anchor_T_base`` undoes the device pipeline's ``base_T_anchor`` so the ray
    is expressed in the XR anchor frame (OpenXR aim: -Z forward).
    """
    T = np.asarray(anchor_T_base, dtype=float) @ np.asarray(aim_pose_base, dtype=float)
    return HandInput(
        origin=T[:3, 3].copy(),
        direction=T[:3, :3] @ np.array([0.0, 0.0, -1.0]),
        trigger=float(trigger),
        free=bool(free),
        thumb_y=float(thumb_y),
    )


@dataclass(frozen=True)
class QuadGeom:
    """A placed quad: centre, rotation (columns = right, up, normal) and size (w, h) in m."""

    center: np.ndarray
    R: np.ndarray
    size: tuple[float, float]


def ray_quad_hit(origin: np.ndarray, direction: np.ndarray, quad: QuadGeom) -> float | None:
    """Distance along the ray to the quad, or None when it misses / is behind."""
    n = quad.R[:, 2]
    denom = float(np.dot(direction, n))
    if abs(denom) < 1e-6:
        return None
    t = float(np.dot(quad.center - origin, n) / denom)
    if t <= 0.0:
        return None
    p = origin + direction * t
    local = quad.R.T @ (p - quad.center)
    if abs(local[0]) <= quad.size[0] / 2.0 and abs(local[1]) <= quad.size[1] / 2.0:
        return t
    return None


@dataclass
class _Grab:
    name: str
    distance: float          # ray distance of the grab point
    offset: np.ndarray       # panel centre - grab point (world)


@dataclass
class PanelGrabber:
    """Hover / grab / drag state machine over named handle quads."""

    grab_threshold: float = 0.7
    release_threshold: float = 0.3
    push_rate_mps: float = 0.5
    distance_range: tuple[float, float] = (0.3, 5.0)
    hover: dict[str, str | None] = field(default_factory=lambda: {h: None for h in HANDS})
    _grabs: dict[str, _Grab] = field(default_factory=dict)
    _prev_trigger: dict[str, float] = field(default_factory=lambda: {h: 0.0 for h in HANDS})

    @property
    def grabbing(self) -> dict[str, str | None]:
        return {h: (g.name if (g := self._grabs.get(h)) is not None else None) for h in HANDS}

    def grabbed_panels(self) -> set[str]:
        return {g.name for g in self._grabs.values()}

    def step(
        self,
        hands: dict[str, HandInput | None],
        handles: dict[str, QuadGeom],
        centers: dict[str, np.ndarray],
        dt: float,
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Advance one frame.

        ``handles`` are the handle-bar quads, ``centers`` the current panel
        centres (both Televiz frame). Returns ``(moved, released)``: new
        centres of the panels being dragged and the panels released this frame.
        """
        moved: dict[str, np.ndarray] = {}
        released: list[str] = []
        for hand in HANDS:
            inp = hands.get(hand)
            grab = self._grabs.get(hand)
            if inp is None:
                # Tracking lost: drop the grab, keep the panel where it is.
                if grab is not None:
                    released.append(grab.name)
                    del self._grabs[hand]
                self.hover[hand] = None
                self._prev_trigger[hand] = 0.0
                continue
            if grab is not None:
                if inp.trigger < self.release_threshold or not inp.free:
                    released.append(grab.name)
                    del self._grabs[hand]
                    self.hover[hand] = None
                else:
                    if abs(inp.thumb_y) > 0.15:
                        lo, hi = self.distance_range
                        grab.distance = float(np.clip(grab.distance + inp.thumb_y * self.push_rate_mps * dt, lo, hi))
                    point = inp.origin + inp.direction * grab.distance
                    moved[grab.name] = point + grab.offset
                    self.hover[hand] = grab.name
                self._prev_trigger[hand] = inp.trigger
                continue
            # Not grabbing: hover test (nearest handle along the ray).
            best: tuple[float, str] | None = None
            for name, quad in handles.items():
                if name in self.grabbed_panels():
                    continue
                t = ray_quad_hit(inp.origin, inp.direction, quad)
                if t is not None and (best is None or t < best[0]):
                    best = (t, name)
            self.hover[hand] = best[1] if best is not None else None
            rising = inp.trigger >= self.grab_threshold and self._prev_trigger[hand] < self.grab_threshold
            if best is not None and inp.free and rising:
                t, name = best
                point = inp.origin + inp.direction * t
                self._grabs[hand] = _Grab(name, t, np.asarray(centers[name], dtype=float) - point)
                logger.info("Camera panel '%s' grabbed with the %s hand.", name, hand)
            self._prev_trigger[hand] = inp.trigger
        return moved, released


# ---------------------------------------------------------------------------
# Layout persistence
# ---------------------------------------------------------------------------

LAYOUT_KEYS = ("offset_x", "offset_y", "distance", "width")


def layout_path(path: str) -> Path | None:
    if not path:
        return None
    return Path(os.path.expanduser(path))


def load_layouts(path: str, layouts: list[Any]) -> list[Any]:
    """Return ``layouts`` with the entries stored under ``path`` applied (by name)."""
    p = layout_path(path)
    if p is None or not p.is_file():
        return list(layouts)
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        logger.warning("Ignoring unreadable panel layout file %s: %s", p, e)
        return list(layouts)
    from dataclasses import replace

    out = []
    applied = []
    for lay in layouts:
        entry = data.get(lay.name) if isinstance(data, dict) else None
        if isinstance(entry, dict):
            kw = {k: float(entry[k]) for k in LAYOUT_KEYS if k in entry}
            lay = replace(lay, **kw)
            applied.append(lay.name)
        out.append(lay)
    if applied:
        logger.info("Camera panel layout restored from %s for %s.", p, applied)
    return out


def save_layouts(path: str, layouts: list[Any]) -> bool:
    p = layout_path(path)
    if p is None:
        return False
    data = {lay.name: {k: float(getattr(lay, k)) for k in LAYOUT_KEYS} for lay in layouts}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))
    except OSError as e:
        logger.warning("Could not save the panel layout to %s: %s", p, e)
        return False
    return True
