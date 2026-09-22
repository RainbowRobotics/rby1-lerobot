"""Human arm posture (IOBT shoulder / elbow / hand) → RB-Y1 shoulder + elbow angles.

Geometric retargeting used for the Cartesian solver's *nullspace hint*: the
end-effector pose is commanded separately (controller), this module only
resolves the arm's redundancy (where the elbow is) into the first four joint
angles ``arm_0..arm_3`` of the RB-Y1 arm. Pure numpy/scipy, unit-testable.

Kinematic facts (rby1m URDF v1.2 / v1.3, identical for joints 0-3), expressed
in the ``link_torso_5`` frame at q = 0:

* the shoulder is spherical (arm_0 / arm_1 / arm_2 origins coincide) at
  ``[0, ∓0.220, 0.0801]``;
* axes: ``arm_0 = (0, cos20°, ∓sin20°)`` (right / left), ``arm_1 = x``,
  ``arm_2 = z``, ``arm_3 = y`` (elbow, flexion is negative);
* upper arm (shoulder → elbow) rest vector ``[0.031, 0, -0.276]`` (≈ -Z),
  forearm (elbow → wrist centre) rest vector ``[-0.031, 0, -L]``.

Only directions are used, so the human / robot limb-length mismatch does not
matter. Solving:

1. ``u = R_tᵀ·norm(E - S)`` must equal ``R_a0(q0) R_x(q1) e``. Because
   ``R_x`` keeps the x component, ``(R_a0(q0)ᵀ u)_x = e_x`` is a
   ``A cos q0 + B sin q0 = C`` equation with two solutions; the one whose
   ``q1`` sign matches the arm_1 limit (right ≤ 0, left ≥ 0) is kept, ties go
   to the previous ``q0``. Then ``q1 = atan2(v_y, -v_z)``.
2. ``q3`` from the elbow angle ``φ = angle(u, f)``: ``cos φ = e·R_y(q3) g``
   is ``P cos q3 + Q sin q3``, solved in closed form on ``[-150°, 0]``.
3. ``q2`` from the azimuth of the forearm in the link_1 frame, corrected by
   the azimuth of ``R_y(q3) g`` (the forearm bends towards link_2 +X once the
   elbow is flexed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

DEG = math.pi / 180.0

# Shoulder tilt of arm_0 about x (URDF axis (0, cos20°, ∓sin20°)).
_ARM0_TILT = 20.0 * DEG
UPPER_ARM_VEC = np.array([0.031, 0.0, -0.276])
FOREARM_VEC = {"1.2": np.array([-0.031, 0.0, -0.256]), "1.3": np.array([-0.031, 0.0, -0.3095])}
SHOULDER_OFFSET = {
    "right": np.array([0.0, -0.220, 0.080073451539]),
    "left": np.array([0.0, 0.220, 0.080073452]),
}
ARM0_AXIS = {
    "right": np.array([0.0, math.cos(_ARM0_TILT), -math.sin(_ARM0_TILT)]),
    "left": np.array([0.0, math.cos(_ARM0_TILT), math.sin(_ARM0_TILT)]),
}
# URDF position limits of arm_0..arm_3 (rad), per side.
ARM_LIMITS = {
    "right": np.array([[-math.pi, math.pi], [-math.pi, 0.017453293], [-math.pi, math.pi], [-2.617993878, 0.017453293]]),
    "left": np.array([[-math.pi, math.pi], [-0.017453293, math.pi], [-math.pi, math.pi], [-2.617993878, 0.017453293]]),
}
ELBOW_MIN = -2.617993878  # rad (-150 deg)


def robot_reach(version: str) -> float:
    """Shoulder → wrist-centre distance of a straight RB-Y1 arm (m)."""
    return float(np.linalg.norm(UPPER_ARM_VEC + FOREARM_VEC.get(version, FOREARM_VEC["1.3"])))


def robot_shoulder_position(T_torso: np.ndarray, side: str) -> np.ndarray:  # noqa: N803
    """Shoulder centre in the robot base frame from the base→link_torso_5 pose."""
    T = np.asarray(T_torso, dtype=float)
    return T[:3, :3] @ SHOULDER_OFFSET[side] + T[:3, 3]


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(axis, dtype=float) * angle).as_matrix()


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def _nearest_branch(candidates: list[float], prev: float | None) -> float:
    if prev is None:
        return candidates[0]
    return min(candidates, key=lambda c: abs(_wrap(c - prev)))


def forearm_dir_link2(q3: float, version: str = "1.3") -> np.ndarray:
    g = FOREARM_VEC.get(version, FOREARM_VEC["1.3"])
    return _rot(np.array([0.0, 1.0, 0.0]), q3) @ (g / np.linalg.norm(g))


def _solve_q0_q1(u: np.ndarray, e: np.ndarray, side: str, prev: np.ndarray | None) -> tuple[float, float]:
    """``u = R_a0(q0) R_x(q1) e`` for a given (link_1-frame) upper-arm rest vector ``e``."""
    a0 = ARM0_AXIS[side]
    lim = ARM_LIMITS[side]
    c = np.cross(a0, u)
    # (R_a0(-q0) u)_x = u_x cos q0 - (a0 x u)_x sin q0  (a0_x = 0)  must equal e_x.
    A, B, C = float(u[0]), float(-c[0]), float(e[0])
    r = math.hypot(A, B)
    base = math.atan2(B, A)
    if r < 1e-9:
        cands = [0.0 if prev is None else float(prev[0])]
    elif abs(C) >= r:
        cands = [base if C > 0 else base + math.pi]
    else:
        d = math.acos(C / r)
        cands = [base + d, base - d]
    best: tuple[float, float] | None = None
    fallback: tuple[float, float] | None = None
    for q0 in cands:
        v = _rot(a0, -q0) @ u
        # R_x(q1) e = (e_x, e_y cos - e_z sin, e_y sin + e_z cos): solve the 2-D rotation.
        q1 = math.atan2(v[1] * e[2] - v[2] * e[1], v[1] * e[1] + v[2] * e[2]) * -1.0
        q1 = _wrap(q1)
        sign_ok = (q1 <= lim[1][1] + 1e-6) if side == "right" else (q1 >= lim[1][0] - 1e-6)
        if sign_ok:
            if best is None or (
                prev is not None and abs(_wrap(q0 - prev[0])) < abs(_wrap(best[0] - prev[0]))
            ):
                best = (q0, q1)
        elif fallback is None:
            fallback = (q0, q1)
    q0, q1 = best if best is not None else fallback  # type: ignore[misc]
    return _wrap(q0), q1


def solve_shoulder_elbow(
    u: np.ndarray,
    f: np.ndarray,
    side: str,
    q_prev: np.ndarray | None = None,
    version: str = "1.3",
    iterations: int = 6,
) -> np.ndarray | None:
    """Closed-form ``[q0, q1, q2, q3]`` from unit upper-arm / forearm directions.

    ``u`` and ``f`` are expressed in the ``link_torso_5`` frame. The upper-arm
    rest vector has a small x component that rotates with ``q2``, so ``q2`` is
    refined by a short fixed-point iteration (converges in 2-3 rounds).
    Returns None when the input is degenerate (zero-length vectors).
    """
    u = np.asarray(u, dtype=float)
    f = np.asarray(f, dtype=float)
    nu, nf = np.linalg.norm(u), np.linalg.norm(f)
    if nu < 1e-6 or nf < 1e-6:
        return None
    u, f = u / nu, f / nf
    a0 = ARM0_AXIS[side]
    e = UPPER_ARM_VEC / np.linalg.norm(UPPER_ARM_VEC)
    g = FOREARM_VEC.get(version, FOREARM_VEC["1.3"])
    g = g / np.linalg.norm(g)
    lim = ARM_LIMITS[side]
    prev = None if q_prev is None else np.asarray(q_prev, dtype=float)

    # --- q3 from the elbow angle (independent of q0..q2): cos(phi) = e . R_y(q3) g
    cos_phi = float(np.clip(np.dot(u, f), -1.0, 1.0))
    P = float(e[0] * g[0] + e[2] * g[2])
    Q = float(e[0] * g[2] - e[2] * g[0])
    rp = math.hypot(P, Q)
    ratio = cos_phi / rp if rp > 1e-9 else 1.0
    if ratio >= 1.0:
        q3_cands = [math.atan2(Q, P)]
    elif ratio <= -1.0:
        q3_cands = [math.atan2(Q, P) + math.pi]
    else:
        dq = math.acos(ratio)
        q3_cands = [math.atan2(Q, P) + dq, math.atan2(Q, P) - dq]
    q3_cands = [_wrap(x) for x in q3_cands]
    in_range = [x for x in q3_cands if ELBOW_MIN - 1e-6 <= x <= lim[3][1] + 1e-6]
    # The straight arm sits at q3* = atan2(Q, P) (about -12 deg) because of the
    # small x offsets of the rest vectors; the same elbow angle is reached on
    # both sides of q3*. A human elbow does not hyper-extend, so take the
    # flexion side (the more negative candidate).
    q3 = min(in_range) if in_range else min(q3_cands, key=lambda x: abs(x - np.clip(x, ELBOW_MIN, lim[3][1])))
    q3 = float(np.clip(q3, ELBOW_MIN, lim[3][1]))
    w2 = forearm_dir_link2(q3, version)
    w2_az = math.atan2(w2[1], w2[0]) if math.hypot(w2[0], w2[1]) >= 0.05 else None

    # --- q0, q1, q2: fixed-point on q2 (the upper-arm rest vector rotates with q2)
    q2 = float(prev[2]) if prev is not None else 0.0
    q0 = q1 = 0.0
    for _ in range(max(1, iterations)):
        e_q2 = _rot(np.array([0.0, 0.0, 1.0]), q2) @ e
        q0, q1 = _solve_q0_q1(u, e_q2, side, prev)
        f1 = _rot(np.array([1.0, 0.0, 0.0]), -q1) @ (_rot(a0, -q0) @ f)
        if w2_az is None or math.hypot(f1[0], f1[1]) < 0.05:
            q2_new = q2  # straight arm: azimuth undefined, keep the previous value
        else:
            q2_new = _wrap(math.atan2(f1[1], f1[0]) - w2_az)
        if abs(_wrap(q2_new - q2)) < 1e-10:
            q2 = q2_new
            break
        q2 = q2_new

    q = np.array([q0, q1, q2, q3], dtype=float)
    q[0] = float(np.clip(q[0], lim[0][0], lim[0][1]))
    q[1] = float(np.clip(q[1], lim[1][0], lim[1][1]))
    q[2] = float(np.clip(q[2], lim[2][0], lim[2][1]))
    return q


def forward_points(q: np.ndarray, side: str, version: str = "1.3") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shoulder / elbow / wrist-centre positions (link_torso_5 frame) for ``[q0..q3]``.

    Test-only forward model built from the same URDF constants as the solver.
    """
    q = np.asarray(q, dtype=float)
    S = SHOULDER_OFFSET[side].copy()
    R01 = _rot(ARM0_AXIS[side], q[0]) @ _rot(np.array([1.0, 0.0, 0.0]), q[1])
    R2 = R01 @ _rot(np.array([0.0, 0.0, 1.0]), q[2])
    E = S + R2 @ UPPER_ARM_VEC
    R3 = R2 @ _rot(np.array([0.0, 1.0, 0.0]), q[3])
    W = E + R3 @ FOREARM_VEC.get(version, FOREARM_VEC["1.3"])
    return S, E, W


@dataclass
class ArmPostureRetargeter:
    """Stateful wrapper: directions → smoothed, rate-limited ``[q0..q3]`` hint."""

    side: str
    version: str = "1.3"
    smoothing: float = 0.3      # EMA weight of the new sample (1 = none)
    max_vel: float = 2.0        # rad/s per joint
    hold_s: float = 1.0         # keep the last hint this long without valid input

    def __post_init__(self) -> None:
        self._q: np.ndarray | None = None
        self._last_valid_t: float | None = None

    @property
    def hint(self) -> np.ndarray | None:
        return None if self._q is None else self._q.copy()

    def reset(self) -> None:
        self._q = None
        self._last_valid_t = None

    def update(
        self,
        shoulder: np.ndarray | None,
        elbow: np.ndarray | None,
        wrist: np.ndarray | None,
        R_torso: np.ndarray,  # noqa: N803
        t: float,
        dt: float,
    ) -> np.ndarray | None:
        """Feed base-frame S/E/W (any None = invalid) and return the current hint."""
        if shoulder is None or elbow is None or wrist is None:
            if self._last_valid_t is not None and t - self._last_valid_t > self.hold_s:
                self._q = None
            return self.hint
        Rt = np.asarray(R_torso, dtype=float)[:3, :3]
        u = Rt.T @ (np.asarray(elbow, dtype=float) - np.asarray(shoulder, dtype=float))
        f = Rt.T @ (np.asarray(wrist, dtype=float) - np.asarray(elbow, dtype=float))
        q = solve_shoulder_elbow(u, f, self.side, self._q, self.version)
        if q is None:
            return self.hint
        self._last_valid_t = t
        if self._q is None:
            self._q = q
            return self.hint
        # Wrap-aware EMA + per-joint velocity limit.
        delta = np.array([_wrap(a - b) for a, b in zip(q, self._q)])
        step = self.smoothing * delta
        max_step = max(self.max_vel, 1e-6) * max(dt, 1e-3)
        step = np.clip(step, -max_step, max_step)
        lim = ARM_LIMITS[self.side]
        self._q = np.clip(self._q + step, lim[:, 0], lim[:, 1])
        return self.hint
