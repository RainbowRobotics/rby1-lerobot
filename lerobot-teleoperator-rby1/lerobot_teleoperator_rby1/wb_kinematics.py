"""RB-Y1 M v1.3 arm kinematics read from the URDF: FK and a closed-form IK.

Ported from the semi-humanoid teleoperation ``rby1_model.py`` (CES branch).
Every number comes from ``models/rby1m_model_v1.3.urdf`` (a copy of rby1-sdk
``models/rby1m/urdf/model_v1.3.urdf``, Apache-2.0); the structure the analytic
IK relies on is asserted while loading, so a different model fails loudly.

Arm structure (both sides alike, all coordinates in ``link_torso_5``):

    S   shoulder = arm_0 origin. The arm_0/1/2 axes meet here (spherical
        shoulder); the arm_0 axis is tilted ±20° about x.
    E   elbow = arm_3 origin; S→E is (0.031, 0, -0.276) in the arm_2 frame.
    W   wrist = arm_4 origin; arm_4/5/6 axes meet here (spherical wrist).
    ee  gripper point END_<side>, (0, 0, -0.125) in the arm_6 frame.

|SE| and |EW| are constant, so S, E, W plus the hand orientation determine
all 7 joints: q3 from |SW|, q0..2 from E and W, q4..6 from the hand.

Hand orientation convention: at q = 0 (arm hanging down) the arm_6 frame
equals the torso_5 frame — gripper pointing down (-z6), x6 forward.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URDF_PATH = Path(__file__).resolve().parent / "models" / "rby1m_model_v1.3.urdf"
SIDES = ("left", "right")

# When a joint limit is hit, rotate the elbow about the shoulder-wrist axis in
# steps of this size (deg) looking for an in-limit solution.
SWIVEL_STEP_DEG = 2.0

# Gimbal lock: only below this cosine of the shoulder middle axis is q0 frozen
# to the previous value (the closed form is exact above it).
GIMBAL_COS = 1e-6


# --------------------------------------------------------------------------
# Rotations
# --------------------------------------------------------------------------

def rot_axis(axis, angle: float) -> np.ndarray:
    """Rodrigues: rotation by ``angle`` (rad) about the unit ``axis``."""
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def Rx(a: float) -> np.ndarray:
    return rot_axis((1, 0, 0), a)


def Ry(a: float) -> np.ndarray:
    return rot_axis((0, 1, 0), a)


def Rz(a: float) -> np.ndarray:
    return rot_axis((0, 0, 1), a)


def rpy_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    return Rz(y) @ Ry(p) @ Rx(r)  # URDF convention (fixed axes X-Y-Z)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------
# URDF
# --------------------------------------------------------------------------

@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: np.ndarray
    R: np.ndarray  # origin rotation
    axis: np.ndarray
    lower: float
    upper: float


def load_urdf(path: Path = URDF_PATH) -> dict[str, Joint]:
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        o, a, lim = j.find("origin"), j.find("axis"), j.find("limit")
        xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
        rpy = np.array([float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()])
        axis = np.array([float(v) for v in (a.get("xyz") if a is not None else "0 0 0").split()])
        lo = float(lim.get("lower", "nan")) if lim is not None else float("nan")
        hi = float(lim.get("upper", "nan")) if lim is not None else float("nan")
        joints[j.get("name")] = Joint(
            j.get("name"), j.get("type"), j.find("parent").get("link"),
            j.find("child").get("link"), xyz, rpy_matrix(rpy), axis, lo, hi,
        )
    return joints


# --------------------------------------------------------------------------
# Arm model
# --------------------------------------------------------------------------

class Arm:
    """Constants, FK and analytic IK of one arm. All coordinates in link_torso_5."""

    def __init__(self, joints: dict[str, Joint], side: str):
        self.side = side
        names = [f"{side}_arm_{i}" for i in range(7)]
        J = [joints[n] for n in names]
        end = joints[f"END_{side}"]
        self.names = names
        self.chain = J + [end]
        self.lower = np.array([j.lower for j in J])
        self.upper = np.array([j.upper for j in J])

        # Structure the analytic IK relies on (stops here if the model changes).
        assert J[0].parent == "link_torso_5"
        for j in J + [end]:
            assert np.allclose(j.R, np.eye(3)), f"{j.name}: origin rpy is not zero"
        for i in (1, 2, 5, 6):
            assert np.allclose(J[i].xyz, 0), f"{J[i].name}: spherical joint assumption violated"
        for i, ax in ((1, (1, 0, 0)), (2, (0, 0, 1)), (3, (0, 1, 0)), (4, (0, 0, 1)),
                      (5, (0, 1, 0)), (6, (1, 0, 0))):
            assert np.allclose(J[i].axis, ax), f"{J[i].name}: axis {J[i].axis}"
        assert abs(J[0].axis[0]) < 1e-9 and np.allclose(end.xyz[:2], 0)

        self.S = J[0].xyz.copy()      # shoulder (torso_5)
        self.a = J[3].xyz.copy()      # S→E (arm_2 frame)
        self.b = J[4].xyz.copy()      # E→W (arm_3 frame)
        self.p_ee = end.xyz.copy()    # W→ee (arm_6 frame)
        self.l_se = float(np.linalg.norm(self.a))
        self.l_ew = float(np.linalg.norm(self.b))
        # arm_0 axis = Rx(tau) · ŷ
        self.tau = math.atan2(J[0].axis[2], J[0].axis[1])

        # |SW|² = |a|² + |b|² + 2 (A cos q3 + B sin q3)
        ax_, az = self.a[0], self.a[2]
        bx, bz = self.b[0], self.b[2]
        self._A = ax_ * bx + az * bz
        self._B = ax_ * bz - az * bx
        q3 = np.linspace(self.lower[3], self.upper[3], 20001)
        d = np.sqrt(self.l_se ** 2 + self.l_ew ** 2 + 2 * (self._A * np.cos(q3) + self._B * np.sin(q3)))
        self.reach_min, self.reach_max = float(d.min()), float(d.max())
        # Full extension = q3 maximising |SW| (v1.3: -12.13°).
        self.q3_straight = math.atan2(self._B, self._A)

    def dist_sw(self, q3: float) -> float:
        """Elbow angle q3 → shoulder-wrist distance |SW|."""
        return math.sqrt(self.l_se ** 2 + self.l_ew ** 2
                         + 2 * (self._A * math.cos(q3) + self._B * math.sin(q3)))

    # ---- FK -----------------------------------------------------------------

    def fk(self, q) -> dict[str, np.ndarray]:
        """7 joints → S, E, W, ee positions and R_ee (all in torso_5)."""
        R = np.eye(3)
        p = np.zeros(3)
        pts = []
        for i, j in enumerate(self.chain):
            p = p + R @ j.xyz
            R = R @ j.R
            if i < 7:
                pts.append(p.copy())  # joint i origin (before its rotation)
                R = R @ rot_axis(j.axis, q[i])
        return {"S": pts[0], "E": pts[3], "W": pts[4], "ee": p, "R_ee": R}

    # ---- IK -----------------------------------------------------------------

    def ik(self, ee_pos, R_ee, elbow_hint, prev: np.ndarray | None = None,
           swivel_max_deg: float = 180.0, no_hyperextend: bool = False) -> tuple[np.ndarray, dict]:
        """Hand target (position + orientation) + elbow direction hint → 7 joints.

        ``elbow_hint``: which side of the S-W axis the elbow should point to
        (any vector; its axial component is dropped). ``prev``: previous
        solution, used to pick the closer of the two shoulder branches and to
        carry q0 through gimbal lock. ``swivel_max_deg``: how far the elbow may
        be rotated about the axis when a limit is hit. ``no_hyperextend``:
        drop the q3 solution on the hyperextended side (> q3_straight).

        Unreachable targets / limit violations do not fail: the closest
        solution is returned and flagged in ``info``.
        """
        if prev is not None and not np.all(np.isfinite(prev)):
            prev = None
        info = {"reach_clip": False, "shoulder_clip": False, "wrist_clip": False, "gimbal": False,
                "swivel_adjust_deg": 0.0}

        # 1) wrist centre
        W = np.asarray(ee_pos, float) - R_ee @ self.p_ee
        v = W - self.S
        d = float(np.linalg.norm(v))
        w_hat = v / d if d > 1e-9 else np.array([0.0, 0.0, -1.0])
        d_c = min(max(d, self.reach_min + 1e-9), self.reach_max - 1e-9)
        if abs(d_c - d) > 1e-6:
            info["reach_clip"] = True
            W = self.S + w_hat * d_c
            d = d_c

        # 2) elbow q3: A cos + B sin = K. Out-of-limit roots are dropped (a
        #    clipped root would not match |SW|); one always survives because d
        #    was clipped to [reach_min, reach_max] above.
        K = (d * d - self.l_se ** 2 - self.l_ew ** 2) / 2
        rho = math.hypot(self._A, self._B)
        phi = math.atan2(self._B, self._A)
        delta = math.acos(max(-1.0, min(1.0, K / rho)))
        raw3 = [float(wrap(phi + s * delta)) for s in (1, -1)]
        q3s = [c for c in raw3 if self.lower[3] - 1e-6 <= c <= self.upper[3] + 1e-6]
        if not q3s:  # both slightly outside by rounding
            q3s = [float(np.clip(min(raw3, key=lambda c: max(self.lower[3] - c, c - self.upper[3])),
                                 self.lower[3], self.upper[3]))]
        if no_hyperextend:
            flexed = [c for c in q3s if c <= self.q3_straight + 1e-9]
            q3s = flexed if flexed else [min(self.q3_straight, self.upper[3])]

        # 3) elbow circle: apex S, radius set by the distance alone
        cos_al = (self.l_se ** 2 + d * d - self.l_ew ** 2) / (2 * self.l_se * d)
        cos_al = max(-1.0, min(1.0, cos_al))
        center = self.S + w_hat * self.l_se * cos_al
        radius = self.l_se * math.sqrt(1 - cos_al * cos_al)
        perp = np.asarray(elbow_hint, float) - np.dot(elbow_hint, w_hat) * w_hat
        if np.linalg.norm(perp) < 1e-6:  # hint on the S-W axis (operator arm fully straight)
            for ref in ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)):
                perp = np.array(ref) - np.dot(ref, w_hat) * w_hat
                if np.linalg.norm(perp) > 1e-6:
                    break
        perp = perp / np.linalg.norm(perp)

        # 4) solve with the operator's elbow direction; on a limit keep the hand
        #    and rotate only the elbow by ±2°, ±4°, ... (hand first).
        best = self._solve_at(center, radius, perp, W, q3s, R_ee, prev)
        if best[0][0]:
            for k in range(1, int(swivel_max_deg / SWIVEL_STEP_DEG) + 1):
                found = None
                for sgn in (1, -1):
                    ang = math.radians(sgn * k * SWIVEL_STEP_DEG)
                    cand = self._solve_at(center, radius, rot_axis(w_hat, ang) @ perp, W, q3s, R_ee, prev)
                    if not cand[0][0] and (found is None or cand[0] < found[1][0]):
                        found = (sgn * k * SWIVEL_STEP_DEG, cand)
                if found is not None:
                    info["swivel_adjust_deg"] = float(found[0])
                    best = found[1]
                    break

        _, q, viol_s, _viol_w, info["gimbal"] = best
        q = q.copy()
        if viol_s > 1e-9:
            info["shoulder_clip"] = True
            q[:3] = np.clip(q[:3], self.lower[:3], self.upper[:3])
            q[4:] = self._wrist(q[:3], q[3], R_ee)  # re-solve the wrist for the clipped shoulder
        q_w = np.clip(q[4:], self.lower[4:], self.upper[4:])
        if np.any(np.abs(q_w - q[4:]) > 1e-9):
            info["wrist_clip"] = True
            q[4:] = q_w
        return q, info

    def _solve_at(self, center, radius, perp, W, q3s, R_ee, prev):
        """Elbow on the circle along ``perp``; try every q3 root x both shoulder
        branches and return the one with the smallest (violates, violation,
        distance to prev)."""
        E = center + radius * perp
        best = None
        for q3 in q3s:
            for cand, gimbal in self._shoulder_candidates(q3, E, W, prev):
                qq = np.concatenate([cand, [q3], self._wrist(cand, q3, R_ee)])
                viol_s = np.sum(np.maximum(0, self.lower[:3] - qq[:3]) + np.maximum(0, qq[:3] - self.upper[:3]))
                viol_w = np.sum(np.maximum(0, self.lower[4:] - qq[4:]) + np.maximum(0, qq[4:] - self.upper[4:]))
                # Unwrapped difference: limits are within ±180°, so the robot
                # cannot wrap across the boundary.
                dist = 0.0 if prev is None else float(np.sum((qq - prev) ** 2))
                key = (viol_s + viol_w > 1e-9, viol_s + viol_w, dist)
                if best is None or key < best[0]:
                    best = (key, qq, viol_s, viol_w, gimbal)
        return best

    def _shoulder_candidates(self, q3, E, W, prev):
        """Rotation R2 taking the arm_2-frame (S→E, S→W) to the actual
        (E-S, W-S) → the two (q0, q1, q2) branches."""
        w_body = self.a + Ry(q3) @ self.b
        R2 = _triad(E - self.S, W - self.S) @ _triad(self.a, w_body).T
        # R2 = Rx(tau) Ry(q0) Rx(-tau) Rx(q1) Rz(q2) → Rx(-tau) R2 = Ry(q0) Rx(q1 - tau) Rz(q2)
        M = Rx(-self.tau) @ R2
        sb = max(-1.0, min(1.0, -M[1, 2]))
        cb = math.sqrt(max(0.0, 1 - sb * sb))
        if cb < GIMBAL_COS and prev is not None:
            a0 = prev[0]
            N = Ry(a0).T @ M  # = Rx(b) Rz(c)
            b = math.atan2(sb, cb)
            c = math.atan2(-N[0, 1], N[0, 0])
            raw = [(a0, b, c)]
            gimbal = True
        else:
            a0 = math.atan2(M[0, 2], M[2, 2])
            b = math.asin(sb)
            c = math.atan2(M[1, 0], M[1, 1])
            raw = [(a0, b, c), (a0 + math.pi, math.pi - b, c + math.pi)]
            gimbal = False
        return [(wrap(np.array([a_, b_ + self.tau, c_])), gimbal) for a_, b_, c_ in raw]

    def _wrist(self, q012, q3, R_ee) -> np.ndarray:
        """R3ᵀ R_ee = Rz(q4) Ry(q5) Rx(q6) (the |q5| > 130° branch is outside the limits)."""
        Mw = (_shoulder_R(self.tau, q012) @ Ry(q3)).T @ R_ee
        return np.array([math.atan2(Mw[1, 0], Mw[0, 0]),
                         math.asin(max(-1.0, min(1.0, -Mw[2, 0]))),
                         math.atan2(Mw[2, 1], Mw[2, 2])])


def _triad(u, v) -> np.ndarray:
    x = u / np.linalg.norm(u)
    y = np.cross(u, v)
    y = y / np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def _shoulder_R(tau, q012) -> np.ndarray:
    return Rx(tau) @ Ry(q012[0]) @ Rx(-tau) @ Rx(q012[1]) @ Rz(q012[2])


class Rby1Arms:
    """Both arms, upper-body frame with all torso joints at 0."""

    def __init__(self, path: Path = URDF_PATH):
        joints = load_urdf(path)
        self.arms = {s: Arm(joints, s) for s in SIDES}
        # Rotation pivot (hip) = torso_3 origin, in torso_5 coordinates at torso = 0.
        self.pivot_in_t5 = -(joints["torso_4"].xyz + joints["torso_5"].xyz)
        for n in ("torso_4", "torso_5"):
            assert np.allclose(joints[n].R, np.eye(3))
        sL, sR = self.arms["left"].S, self.arms["right"].S
        self.frame_origin = (sL + sR) / 2
        R, _ = make_frame(sL[None], sR[None], self.pivot_in_t5[None])
        assert np.allclose(R[0], np.eye(3), atol=1e-6), R[0]


def make_frame(sL: np.ndarray, sR: np.ndarray, anchor: np.ndarray):
    """WARP Algorithm 1 — both shoulders + a lower anchor → upper-body frame.

    Returns ``R`` (T,3,3) with columns x forward / y left / z up and ``p`` (T,3).
    """
    p = (sL + sR) / 2
    uy = _unit(sL - sR)
    ux = _unit(np.cross(uy, p - anchor))
    uz = np.cross(ux, uy)
    return np.stack([ux, uy, uz], axis=-1), p


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)
