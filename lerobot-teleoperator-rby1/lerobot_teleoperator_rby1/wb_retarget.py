"""Whole-body retargeting: operator body + hands → RB-Y1 torso, arms, head, base.

Ported from the semi-humanoid teleoperation ``rby1_ces_elbow.py`` (CES
branch). The control law is unchanged; the only difference is that the tick
length ``dt`` is passed in (LeRobot runs at the configured ``--fps``) instead
of the fixed 50 Hz of the original loop.

Mapping summary (what → where)

    palm position (hand tracking, U frame)  → END_<side> position (link_torso_5)
    hand orientation (landmarks)            → arm_6 orientation
    elbow direction (IOBT S/E/W)            → swivel of the analytic IK → nullspace joint targets
    operator elbow flexion (hand_map=elbow) → robot elbow angle (angle → position, no
                                              sensitivity blow-up near full extension)
    upper-body tilt (U roll/pitch/yaw)      → link_torso_5 orientation, rigid rotation
                                              about the torso_3 pivot
    pelvis planar motion / heading (W)      → base SE(2) position target → velocity
                                              (P + feed-forward, odometry closed loop)
    HMD gaze (yaw vs chest, pitch delta)    → head_0 / head_1
    thumb-index distance / pinch_open_m     → gripper (hysteresis)

Frames: the stream frame W keeps the direction faced at app start as +X.
Hand / torso / head commands are expressed in the operator's own upper-body
frame U (shoulders + pelvis), so they do not depend on where the operator
faces. The base follows the pelvis displacement since the latch, rotated by
the starting heading ψ0. A jump gate re-bases the world reference when W
itself teleports (Quest recenter), continuing the base target seamlessly.

Rejoin offset: the hand mapping is absolute, so at the latch (and whenever a
hand is re-acquired or glitches) the difference between the robot hand and
the mapped target is captured as an offset and bled to zero.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .wb_kinematics import Rby1Arms, Rx, Ry, Rz, make_frame, wrap
from .wb_stream import (
    ARM_JOINTS,
    HIPS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    PINKY_MCP,
    SIDES,
    THUMB_TIP,
    WRIST,
    BodyObs,
    HmdObs,
    Snapshot,
    lpf_alpha,
)

logger = logging.getLogger(__name__)

# Inputs older than this are treated as absent (s). Real Quest Wi-Fi streams
# drop for 0.3-0.5 s quite often; shorter timeouts make the arms hitch.
HAND_TIMEOUT = 0.5
BODY_TIMEOUT = 0.5
HMD_TIMEOUT = 0.5

# Jump gates — more than this in one tick is a glitch / Quest recenter.
BASE_JUMP_M = 0.30
BASE_JUMP_YAW = math.radians(45)
HAND_JUMP_M = 0.25

# Torso orientation clamp (increment from the latched ready pose, ZYX Euler).
TORSO_YAW_LIMIT = math.radians(120)
TORSO_PITCH_RANGE = (math.radians(-30), math.radians(40))
TORSO_ROLL_LIMIT = math.radians(20)
TORSO_SLEW_V = 0.5  # m/s

# elbow mapping: below this distance of the operator elbow from the
# shoulder-wrist axis (≈ 8° flexion) the swivel hint is kept.
ELBOW_HINT_MIN_M = 0.02
IK_SWIVEL_MAX_DEG = 30.0  # real-time: limited elbow search range
HEAD_LIMIT = math.radians(89.0)

# Tick length bounds (s) for the integrators and filters.
MIN_DT, MAX_DT = 0.005, 0.1


@dataclass
class Tuning:
    """Retargeting parameters. SI units, angles in rad."""

    # Heading low-pass (s); also the pelvis-rotation input filter time constant.
    tau_heading: float = 0.12
    # Command slew (max change of a target per second).
    hand_slew_v: float = 2.0                 # m/s
    hand_slew_w: float = 6.0                 # rad/s
    torso_slew_w: float = 2.0                # rad/s
    # Rejoin offset bleed.
    bleed_lin: float = 0.10                  # m/s, minimum speed
    bleed_ang: float = 0.6                   # rad/s
    bleed_time: float = 0.5                  # s, finish within this time
    # Base (position tracking + feed-forward).
    base_vmax: float = 0.8
    base_wmax: float = 1.5
    base_kp: float = 2.0
    base_kp_yaw: float = 3.0
    base_kff: float = 0.8
    base_tau_ff: float = 0.15
    base_err_stop: float = 0.02
    base_err_go: float = 0.04
    base_yaw_stop: float = math.radians(1.0)
    base_yaw_go: float = math.radians(2.5)
    base_max_lag: float = 0.5
    base_max_lag_yaw: float = math.radians(60)
    # Hand mapping / gripper.
    hand_map: str = "elbow"                  # elbow | palm
    tau_elbow: float = 0.10                  # s, operator flexion low-pass (elbow map)
    elbow_slew_w: float = math.radians(150)  # rad/s, flexion command rate limit (elbow map)
    arm_scale: float | None = 1.0            # None = auto (robot / operator arm length), palm map only
    arm6_offset: float = math.radians(30)    # right +, left −: gripper tips turn inwards
    pinch_open_m: float = 0.08
    pinch_threshold: float = 0.5
    pinch_hyst: float = 0.03
    # Modes.
    heading: str = "pelvis"                  # pelvis | shoulder
    torso_height: bool = False
    base_enabled: bool = True


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def yaw_of(R: np.ndarray) -> float:
    return math.atan2(R[1, 0], R[0, 0])


def rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def se3(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def euler_zyx(R: np.ndarray) -> tuple[float, float, float]:
    """R = Rz(yaw) Ry(pitch) Rx(roll)."""
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll


def from_euler_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def rotvec_of(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def R_of_rotvec(v: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(v).as_matrix()


def shrink(v: np.ndarray, step: float) -> np.ndarray:
    """Shorten ``v`` by ``step`` towards 0 (direction kept, never past 0)."""
    n = float(np.linalg.norm(v))
    if n <= step:
        return np.zeros_like(v)
    return v * (1.0 - step / n)


def slew_vec(cur: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    d = target - cur
    n = float(np.linalg.norm(d))
    if n <= max_step:
        return target.copy()
    return cur + d * (max_step / n)


def slew_rot(cur: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    d = rotvec_of(target @ cur.T)
    n = float(np.linalg.norm(d))
    if n <= max_step:
        return target.copy()
    return R_of_rotvec(d * (max_step / n)) @ cur


def clamp_norm(v: np.ndarray, limit: float) -> np.ndarray:
    """Clamp the magnitude only — per-axis clipping bends diagonal motion."""
    n = float(np.linalg.norm(v))
    if n <= limit or n < 1e-9:
        return v
    return v * (limit / n)


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def valid_se2(M) -> bool:
    """Odometry sample check (a zero matrix would silently re-base to the origin)."""
    try:
        M = np.asarray(M, dtype=float)
        if M.shape != (3, 3) or not np.all(np.isfinite(M)):
            return False
        if abs(float(np.linalg.det(M[:2, :2])) - 1.0) > 1e-3:
            return False
        return bool(np.allclose(M[2], (0.0, 0.0, 1.0), atol=1e-6))
    except Exception:  # noqa: BLE001
        return False


def se2_relative(T0: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, float]:
    """T seen from T0 → (xy, yaw)."""
    R0 = T0[:2, :2]
    rel_t = R0.T @ (T[:2, 2] - T0[:2, 2])
    rel_R = R0.T @ T[:2, :2]
    return rel_t, math.atan2(rel_R[1, 0], rel_R[0, 0])


def hand_frame(lm: np.ndarray) -> np.ndarray:
    """Hand landmarks (21, 3) → hand orientation, columns = robot arm_6 x/y/z.

    z = −(wrist → middle MCP), x = pinky MCP → index MCP (thumb side),
    y = z × x. A relaxed hanging hand (fingers down, thumb forward) equals the
    robot at q = 0 (gripper down, x forward). Same convention for both hands.
    """
    z = -unit(lm[MIDDLE_MCP] - lm[WRIST])
    lat = lm[INDEX_MCP] - lm[PINKY_MCP]
    x = unit(lat - np.dot(lat, z) * z)
    return np.stack([x, np.cross(z, x), z], axis=1)


def flexion(S: np.ndarray, E: np.ndarray, W: np.ndarray) -> float:
    """Shoulder, elbow, wrist → elbow flexion (rad, 0 = straight)."""
    a, b = E - S, W - E
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return float("nan")
    return math.acos(max(-1.0, min(1.0, float(np.dot(a, b)) / (na * nb))))


def ray_reach(u: np.ndarray, c: np.ndarray, d: float) -> float:
    """r ≥ 0 with |r·u − c| = d (larger root). u = shoulder→ee unit direction,
    c = wrist→ee vector, d = shoulder-wrist distance. d > |c| so the
    discriminant is always positive."""
    uc = float(np.dot(u, c))
    return uc + math.sqrt(max(0.0, uc * uc - float(np.dot(c, c)) + d * d))


# ---------------------------------------------------------------------------
# Operator geometry (W frame)
# ---------------------------------------------------------------------------

@dataclass
class OperatorPose:
    R_U: np.ndarray        # upper-body frame (columns x forward / y left / z up)
    o_U: np.ndarray        # shoulder midpoint
    shoulder_yaw: float    # absolute heading from the shoulder line (W)
    hips_xy: np.ndarray
    hips_z: float
    hips_R: np.ndarray


def operator_pose(body: BodyObs) -> OperatorPose:
    P = body.pos
    sL, sR = P[10], P[15]
    R, o = make_frame(sL[None], sR[None], P[HIPS][None])
    u = sL - sR  # forward = (L−R) × ẑ = (u_y, −u_x)
    return OperatorPose(R[0], o[0], math.atan2(-u[0], u[1]), P[HIPS][:2].copy(), float(P[HIPS][2]), body.hips_R)


# ---------------------------------------------------------------------------
# Retargeter
# ---------------------------------------------------------------------------

@dataclass
class Measured:
    """Measured robot state (all in the base frame)."""

    q: np.ndarray
    T_t5: np.ndarray
    T_t3: np.ndarray
    T_arm6: dict[str, np.ndarray]
    odom: np.ndarray | None
    odom_fresh: bool


@dataclass
class HandState:
    cmd_p: np.ndarray       # END target (link_torso_5)
    cmd_R: np.ndarray
    hint: np.ndarray        # elbow direction hint (torso_5)
    q_ik: np.ndarray
    off_p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    off_R: np.ndarray = field(default_factory=lambda: np.eye(3))
    last_map_p: np.ndarray | None = None
    tracking: bool = False
    info: dict = field(default_factory=dict)
    palm_err: float = 0.0
    pinch: float = float("nan")
    closed: bool = False
    rejoins: int = 0
    flex_raw: float = float("nan")
    flex_f: float = float("nan")
    flex_cmd: float = float("nan")


@dataclass
class Targets:
    T_t5: np.ndarray                 # base → link_torso_5
    T_hand: dict[str, np.ndarray]    # base → link_<side>_arm_6
    q_ns: dict[str, np.ndarray]      # 7 nullspace joint targets per arm
    base_v: np.ndarray               # body-frame linear velocity (m/s)
    base_w: float                    # yaw rate (rad/s)
    head: np.ndarray                 # head_0, head_1 (rad)
    closed: dict[str, bool]
    info: dict


class Retargeter:
    def __init__(self, tuning: Tuning, model: Rby1Arms, arm_idx: dict[str, np.ndarray], head_idx: np.ndarray):
        self.tn = tuning
        self.model = model
        self.arm_idx = arm_idx      # the 7 arm joints inside the robot joint vector
        self.head_idx = head_idx
        self.latched = False
        self.hands: dict[str, HandState] = {}
        self.k: dict[str, float] = {}
        self.base_jumps = 0

    # -- latch ---------------------------------------------------------------
    def latch(self, snap: Snapshot, meas: Measured, now: float) -> None:
        """Take the current operator pose and robot pose as Δ = 0."""
        tn = self.tn
        op = operator_pose(snap.body)
        self.psi0 = op.shoulder_yaw
        self.hips_R0 = op.hips_R.copy()
        self.hips_xy0 = op.hips_xy.copy()
        self.hips_z0 = op.hips_z
        self.R_ready = meas.T_t5[:3, :3].copy()
        self.p_ready = meas.T_t5[:3, 3].copy()
        self.pivot = meas.T_t3[:3, 3].copy()
        # Constant matching the operator's upright pose to the robot torso:
        # R_U(0)·Rnᵀ = Rz(ψ0)·R_ready
        self.Rn = self.R_ready.T @ Rz(-self.psi0) @ op.R_U
        self.torso_R = self.R_ready.copy()
        self.torso_p = self.p_ready.copy()
        self.h_vec = np.array([math.cos(self.psi0), math.sin(self.psi0)])
        # Base
        self.odom0 = meas.odom.copy() if meas.odom is not None else np.eye(3)
        self.slip = np.zeros(2)
        self.slip_yaw = 0.0
        self.p_t_prev: np.ndarray | None = None
        self.p_t_last = np.zeros(2)
        self.psi_raw_prev = self.psi0
        self.psi_prev = self.psi0
        self.v_ff = np.zeros(2)
        self.w_ff = 0.0
        self.base_moving = False
        self.body_was_lost = False
        self.lost_since: float | None = None
        self.p_b, self.theta_b = np.zeros(2), 0.0
        # Hands: start from the measured pose (Δ = 0); the rejoin offset is
        # captured on the first valid frame.
        T_t5_inv = np.linalg.inv(meas.T_t5)
        for side in SIDES:
            arm = self.model.arms[side]
            T = T_t5_inv @ meas.T_arm6[side]
            W, R = T[:3, 3], T[:3, :3]
            q = meas.q[self.arm_idx[side]].copy()
            f = arm.fk(q)
            sw = unit(f["W"] - f["S"])
            hint = (f["E"] - f["S"]) - np.dot(f["E"] - f["S"], sw) * sw
            self.hands[side] = HandState(W + R @ arm.p_ee, R.copy(), hint, q)
            if tn.arm_scale is None:
                sh, el, wr = ARM_JOINTS[side]
                P = snap.body.pos
                human = np.linalg.norm(P[el] - P[sh]) + np.linalg.norm(P[wr] - P[el])
                self.k[side] = (arm.l_se + arm.l_ew) / max(float(human), 0.2)
            else:
                self.k[side] = float(tn.arm_scale)
        # Head
        self.head_cmd = meas.q[self.head_idx].copy()
        self.head1_0 = float(self.head_cmd[1])
        self.hmd_pitch0 = self._hmd_angles(snap.hmd)[1] if snap.hmd.ok else None
        self.latched = True
        self.latch_time = now

    @staticmethod
    def _hmd_angles(hmd: HmdObs) -> tuple[float, float]:
        fwd = hmd.R[:, 0]
        return math.atan2(fwd[1], fwd[0]), math.atan2(-fwd[2], math.hypot(fwd[0], fwd[1]))

    # -- per tick --------------------------------------------------------------
    def step(self, snap: Snapshot, meas: Measured, now: float, hold: bool, dt: float) -> Targets:
        tn = self.tn
        dt = min(max(dt, MIN_DT), MAX_DT)
        info: dict = {"hold": hold}
        body_fresh = snap.body.ok and (now - snap.body.stamp) < BODY_TIMEOUT
        odom_ok = meas.odom is not None and meas.odom_fresh
        if odom_ok:
            self.p_b, self.theta_b = se2_relative(self.odom0, meas.odom)
        # Keep the last odometry when it drops out — zero would make the torso
        # yaw target jump by the base angle.
        p_b, theta_b = self.p_b, self.theta_b

        if not body_fresh or hold:
            # Body lost (or held): intent unknown — stop the base, hold arms
            # and torso. On return the base target restarts from where it is.
            if not self.body_was_lost:
                self.lost_since = now
                if not hold:
                    logger.warning(f"Body stream stale ({now - snap.body.stamp:.2f}s) -> arms/torso hold, base stop")
            self.body_was_lost = True
            self.base_moving = False
            self.v_ff[:] = 0.0
            self.w_ff = 0.0
            info["body"] = "lost" if not body_fresh else "hold"
            return self._emit(np.zeros(2), 0.0, info)
        if self.body_was_lost and self.lost_since is not None:
            logger.warning(f"Body stream back after {now - self.lost_since:.2f}s")
            self.lost_since = None

        op = operator_pose(snap.body)
        R_Up = op.R_U @ self.Rn.T

        # --- heading ψ (pre-filter) — world increment of the pelvis rotation, or the shoulder line
        if tn.heading == "pelvis":
            psi_raw = self.psi0 + yaw_of(op.hips_R @ self.hips_R0.T)
        else:
            psi_raw = op.shoulder_yaw

        # --- base SE(2) target: operator displacement rotated by ψ_ref (1:1)
        psi_ref = self.psi0 + self.slip_yaw
        p_t_raw = rot2(-psi_ref) @ (op.hips_xy - self.hips_xy0)
        v_inst, w_inst = np.zeros(2), 0.0
        jump, jump_yaw = self.body_was_lost, 0.0
        if self.p_t_prev is not None:
            # Gate on unfiltered values; a filtered jump spreads over ticks.
            jump_p = p_t_raw - self.p_t_prev
            jump_yaw = float(wrap(psi_raw - self.psi_raw_prev))
            if np.linalg.norm(jump_p) > BASE_JUMP_M or abs(jump_yaw) > BASE_JUMP_YAW:
                jump = True
                self.base_jumps += 1
                logger.warning(f"Base jump gate: |dp|={np.linalg.norm(jump_p):.2f} m "
                               f"dyaw={math.degrees(jump_yaw):.0f} deg -> world reference re-based")
            else:
                v_inst = jump_p / dt
        if jump:
            # Glitch / recenter / return from hold: re-base the world reference
            # here and keep the base target so far in the slip.
            self.slip_yaw += jump_yaw
            self.h_vec = rot2(jump_yaw) @ self.h_vec
            self.hips_xy0 = op.hips_xy.copy()
            self.slip = -self.p_t_last
            p_t_raw = np.zeros(2)
            psi_ref = self.psi0 + self.slip_yaw
        self.body_was_lost = False
        self.p_t_prev = p_t_raw.copy()
        self.psi_raw_prev = psi_raw

        # Heading filter (seam-safe: filter the unit vector, then atan2).
        a = lpf_alpha(dt, tn.tau_heading)
        self.h_vec = unit(self.h_vec + a * (np.array([math.cos(psi_raw), math.sin(psi_raw)]) - self.h_vec))
        psi = math.atan2(self.h_vec[1], self.h_vec[0])
        if not jump:
            w_inst = float(wrap(psi - self.psi_prev)) / dt
        self.psi_prev = psi
        dpsi = float(wrap(psi - self.psi0))
        p_t = p_t_raw - self.slip
        theta_t = float(wrap(psi - psi_ref))

        # Feed-forward: operator velocity (low-passed).
        a_ff = lpf_alpha(dt, tn.base_tau_ff)
        self.v_ff += a_ff * (v_inst - self.v_ff)
        self.w_ff += a_ff * (w_inst - self.w_ff)

        # Lag cap → slip (1:1 permanently given up).
        e = p_t - p_b
        n = float(np.linalg.norm(e))
        if n > tn.base_max_lag:
            excess = e * (1.0 - tn.base_max_lag / n)
            self.slip += excess
            p_t -= excess
            e = p_t - p_b
            n = float(np.linalg.norm(e))
        e_th = float(wrap(theta_t - theta_b))
        if abs(e_th) > tn.base_max_lag_yaw:
            excess = e_th - math.copysign(tn.base_max_lag_yaw, e_th)
            self.slip_yaw += excess
            e_th -= excess
        self.p_t_last = p_t.copy()

        # Schmitt dead band: standing sway must not wake the loop.
        ff_mag = float(np.linalg.norm(self.v_ff))
        if self.base_moving:
            if n < tn.base_err_stop and abs(e_th) < tn.base_yaw_stop and ff_mag < 0.05 and abs(self.w_ff) < 0.1:
                self.base_moving = False
        elif n > tn.base_err_go or abs(e_th) > tn.base_yaw_go:
            self.base_moving = True

        base_v, base_w = np.zeros(2), 0.0
        if tn.base_enabled and odom_ok and self.base_moving:
            # SE2 velocity is in the robot's current body frame: rotate by −θ_b.
            v_E = tn.base_kp * e + tn.base_kff * self.v_ff
            base_v = clamp_norm(rot2(-theta_b) @ v_E, tn.base_vmax)
            base_w = float(np.clip(tn.base_kp_yaw * e_th + tn.base_kff * self.w_ff, -tn.base_wmax, tn.base_wmax))
        info.update(base_err=n, base_yaw_err=e_th, theta_b=theta_b, dpsi=dpsi, odom=odom_ok,
                    jumps=self.base_jumps, moving=self.base_moving)

        # --- torso: chest orientation (incl. base lag) as a clamped increment from ready
        R_raw = Rz(-theta_b) @ Rz(-psi_ref) @ R_Up
        D = R_raw @ self.R_ready.T
        y, p, r = euler_zyx(D)
        y_c = float(np.clip(y, -TORSO_YAW_LIMIT, TORSO_YAW_LIMIT))
        p_c = float(np.clip(p, *TORSO_PITCH_RANGE))
        r_c = float(np.clip(r, -TORSO_ROLL_LIMIT, TORSO_ROLL_LIMIT))
        D_c = from_euler_zyx(y_c, p_c, r_c)
        R_des = D_c @ self.R_ready
        p_des = self.pivot + D_c @ (self.p_ready - self.pivot)
        if tn.torso_height:
            p_des[2] += op.hips_z - self.hips_z0
        self.torso_R = slew_rot(self.torso_R, R_des, tn.torso_slew_w * dt)
        self.torso_p = slew_vec(self.torso_p, p_des, TORSO_SLEW_V * dt)
        info["torso_ypr"] = (y_c, p_c, r_c)
        info["torso_clip"] = (abs(y - y_c) + abs(p - p_c) + abs(r - r_c)) > 1e-6

        # --- hands: U → torso_5, rejoin offset, slew, IK
        def local(x):
            return R_Up.T @ (x - op.o_U)

        for side in SIDES:
            hs = self.hands[side]
            arm = self.model.arms[side]
            # Operator flexion φ: updated every tick the body is visible.
            sh, el, wr = ARM_JOINTS[side]
            P = snap.body.pos
            phi = flexion(P[sh], P[el], P[wr])
            hs.flex_raw = phi
            if math.isfinite(phi):
                if not math.isfinite(hs.flex_f):
                    hs.flex_f = hs.flex_cmd = phi
                hs.flex_f += lpf_alpha(dt, tn.tau_elbow) * (phi - hs.flex_f)
                step_max = tn.elbow_slew_w * dt
                hs.flex_cmd += float(np.clip(hs.flex_f - hs.flex_cmd, -step_max, step_max))
            hand = snap.hands[side]
            fresh = hand.ok and (now - hand.stamp) < HAND_TIMEOUT
            if fresh:
                q_palm, q_S, q_E, q_W = local(hand.palm), local(P[sh]), local(P[el]), local(P[wr])
                map_R = R_Up.T @ hand_frame(hand.pos)
                if tn.arm6_offset != 0.0:
                    map_R = map_R @ Rx(tn.arm6_offset if side == "right" else -tn.arm6_offset)
                rel = q_palm - q_S
                op_dir = unit(rel)
                if tn.hand_map == "elbow" and math.isfinite(hs.flex_cmd):
                    theta = float(np.clip(arm.q3_straight - hs.flex_cmd, arm.lower[3], arm.q3_straight))
                    r_reach = ray_reach(op_dir, map_R @ arm.p_ee, arm.dist_sw(theta))
                    map_p = arm.S + r_reach * op_dir
                else:
                    map_p = arm.S + self.k[side] * rel
                sw = unit(q_W - q_S)
                hint = (q_E - q_S) - np.dot(q_E - q_S, sw) * sw
                # Nearly straight operator arm: the elbow sits on the S-W axis
                # and its direction is noise → keep the previous hint.
                if tn.hand_map != "elbow" or float(np.linalg.norm(hint)) >= ELBOW_HINT_MIN_M:
                    hs.hint = hint
                if not hs.tracking or (hs.last_map_p is not None
                                       and np.linalg.norm(map_p - hs.last_map_p) > HAND_JUMP_M):
                    hs.off_p = hs.cmd_p - map_p
                    hs.off_R = hs.cmd_R @ map_R.T
                    hs.rejoins += 1
                    if hs.rejoins > 1:
                        logger.info(f"{side} hand rejoin: offset {np.linalg.norm(hs.off_p) * 100:.0f} cm / "
                                    f"{math.degrees(np.linalg.norm(rotvec_of(hs.off_R))):.0f} deg")
                hs.tracking = True
                hs.last_map_p = map_p
                # Bleed the offset: at least bleed_lin/ang, finished within bleed_time.
                rv = rotvec_of(hs.off_R)
                hs.off_p = shrink(hs.off_p, max(tn.bleed_lin, float(np.linalg.norm(hs.off_p)) / tn.bleed_time) * dt)
                hs.off_R = R_of_rotvec(shrink(rv, max(tn.bleed_ang, float(np.linalg.norm(rv)) / tn.bleed_time) * dt))
                tgt_p = map_p + hs.off_p
                tgt_R = hs.off_R @ map_R
                hs.cmd_p = slew_vec(hs.cmd_p, tgt_p, tn.hand_slew_v * dt)
                hs.cmd_R = slew_rot(hs.cmd_R, tgt_R, tn.hand_slew_w * dt)
                d = float(np.linalg.norm(hand.pos[THUMB_TIP] - hand.pos[INDEX_TIP])) / tn.pinch_open_m
                hs.pinch = d
                if hs.closed and d > tn.pinch_threshold + tn.pinch_hyst:
                    hs.closed = False
                elif not hs.closed and d < tn.pinch_threshold - tn.pinch_hyst:
                    hs.closed = True
            else:
                hs.tracking = False  # the rejoin offset is re-captured on return

        # --- head
        if snap.hmd.ok and (now - snap.hmd.stamp) < HMD_TIMEOUT:
            yaw_h, pitch_h = self._hmd_angles(snap.hmd)
            if self.hmd_pitch0 is None:
                self.hmd_pitch0 = pitch_h
            self.head_cmd = np.array([
                float(np.clip(wrap(yaw_h - yaw_of(R_Up)), -HEAD_LIMIT, HEAD_LIMIT)),
                float(np.clip(self.head1_0 + (pitch_h - self.hmd_pitch0), -HEAD_LIMIT, HEAD_LIMIT))])
        return self._emit(base_v, base_w, info)

    def _emit(self, base_v: np.ndarray, base_w: float, info: dict) -> Targets:
        T_t5 = se3(self.torso_R, self.torso_p)
        T_hand, q_ns, closed = {}, {}, {}
        for side in SIDES:
            hs = self.hands[side]
            arm = self.model.arms[side]
            q, ik_info = arm.ik(hs.cmd_p, hs.cmd_R, hs.hint, hs.q_ik, swivel_max_deg=IK_SWIVEL_MAX_DEG,
                                no_hyperextend=(self.tn.hand_map == "elbow"))
            f = arm.fk(q)
            hs.q_ik, hs.info = q, ik_info
            hs.palm_err = float(np.linalg.norm(f["ee"] - hs.cmd_p))
            # The Cartesian hand target comes from the IK solution's FK too, so
            # both targets share one source even when the reach is clipped.
            T_hand[side] = T_t5 @ se3(f["R_ee"], f["W"])
            q_ns[side] = q
            closed[side] = hs.closed
        return Targets(T_t5, T_hand, q_ns, base_v, base_w, self.head_cmd.copy(), closed, info)
