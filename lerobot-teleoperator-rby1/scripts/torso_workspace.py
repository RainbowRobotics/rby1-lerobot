#!/usr/bin/env python3
"""Reachable SE3 workspace of the RB-Y1 torso (link_torso_5 in the base frame).

Pure-numpy forward kinematics of the torso chain (rby1m URDF v1.2 / v1.3,
identical for the torso) sampled over the joint ranges, to derive absolute
pose limits for the ``torso_ee.*`` teleoperation target::

    python scripts/torso_workspace.py                # table on stdout, CSV in ~/torso_workspace.csv
    python scripts/torso_workspace.py --check-sdk    # also compare the FK with rby1_sdk (if installed)
    python scripts/torso_workspace.py --joint-limits '{"torso_2": [-2.618, -0.2]}'

Chain (URDF): base -> torso_0 (roll x, z+0.2965) -> torso_1 (pitch y)
-> torso_2 (pitch y, +0.350) -> torso_3 (pitch y, +0.350) -> torso_4 (roll x)
-> torso_5 (yaw z, +0.3094) -> link_torso_5.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation

# (axis, origin offset along z of the parent link) per joint, in chain order.
CHAIN = [
    ("torso_0", np.array([1.0, 0.0, 0.0]), 0.2965),
    ("torso_1", np.array([0.0, 1.0, 0.0]), 0.0),
    ("torso_2", np.array([0.0, 1.0, 0.0]), 0.350),
    ("torso_3", np.array([0.0, 1.0, 0.0]), 0.350),
    ("torso_4", np.array([1.0, 0.0, 0.0]), 0.0),
    ("torso_5", np.array([0.0, 0.0, 1.0]), 0.309426548461),
]
URDF_LIMITS = {
    "torso_0": (-0.261799388, 0.261799388),
    "torso_1": (-0.523598776, 1.570796327),
    "torso_2": (-2.4434609528, 1.3962634016),
    "torso_3": (-0.785398163, 1.570796327),
    "torso_4": (-0.523598776, 0.523598776),
    "torso_5": (-2.35619449, 2.35619449),
}
# Solver-side limits used by lerobot_robot_rby1 (DEFAULT_TORSO_JOINT_LIMITS).
SOLVER_LIMITS = {"torso_1": (-0.523598776, 1.6), "torso_2": (-2.617993878, -0.2)}
READY_DEG = [0.0, 45.0, -90.0, 45.0, 0.0, 0.0]


def _rot(axis: np.ndarray, a: np.ndarray) -> np.ndarray:
    """(N,) angles about a unit axis -> (N, 3, 3) (Rodrigues, vectorised)."""
    c, s = np.cos(a), np.sin(a)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    I = np.eye(3)
    return I[None] + s[:, None, None] * K[None] + (1.0 - c)[:, None, None] * (K @ K)[None]


def fk(q: np.ndarray) -> np.ndarray:
    """(N, 6) joint angles -> (N, 4, 4) base_T_link_torso_5."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    n = q.shape[0]
    R = np.tile(np.eye(3), (n, 1, 1))
    p = np.zeros((n, 3))
    for i, (_name, axis, dz) in enumerate(CHAIN):
        p = p + R[:, :, 2] * dz  # offset along the parent link z
        R = R @ _rot(axis, q[:, i])
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = p
    return T


def rpy_deg(T: np.ndarray) -> np.ndarray:
    """Fixed-axis XYZ (roll about base x, then pitch y, then yaw z), degrees."""
    return Rotation.from_matrix(T[:, :3, :3]).as_euler("xyz", degrees=True)


def limits(overrides: dict | None) -> dict[str, tuple[float, float]]:
    lim = dict(URDF_LIMITS)
    for k, (lo, hi) in SOLVER_LIMITS.items():
        lim[k] = (max(lim[k][0], lo), min(lim[k][1], hi))
    for k, v in (overrides or {}).items():
        lim[k] = (float(v[0]), float(v[1]))
    return lim


def sample(lim: dict, n_roll: int, n_pitch: int, n_yaw: int) -> np.ndarray:
    grids = []
    for name, _axis, _dz in CHAIN:
        lo, hi = lim[name]
        n = {"torso_0": n_roll, "torso_4": n_roll, "torso_5": n_yaw}.get(name, n_pitch)
        g = np.linspace(lo, hi, n)
        if lo < 0.0 < hi and not np.any(g == 0.0):
            g = np.sort(np.append(g, 0.0))  # always include the zero column (sagittal slice)
        grids.append(g)
    mesh = np.meshgrid(*grids, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def table(rows: list[tuple[str, float, float, float]]) -> str:
    out = ["| 성분 | min | max | ready |", "|---|---|---|---|"]
    for name, lo, hi, ready in rows:
        out.append(f"| {name} | {lo:+.3f} | {hi:+.3f} | {ready:+.3f} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joint-limits", type=str, default=None, help='JSON overrides, e.g. {"torso_2": [-2.618, -0.2]}')
    ap.add_argument("--n-pitch", type=int, default=41)
    ap.add_argument("--n-roll", type=int, default=5)
    ap.add_argument("--n-yaw", type=int, default=9)
    ap.add_argument("--csv", type=str, default=os.path.expanduser("~/torso_workspace.csv"))
    ap.add_argument("--check-sdk", action="store_true")
    args = ap.parse_args()

    lim = limits(json.loads(args.joint_limits) if args.joint_limits else None)
    print("joint limits (deg):", {k: (round(math.degrees(v[0]), 1), round(math.degrees(v[1]), 1)) for k, v in lim.items()})

    T_ready = fk(np.deg2rad(READY_DEG))[0]
    p_ready = T_ready[:3, 3]
    rpy_ready = rpy_deg(T_ready[None])[0]
    print(f"ready link_torso_5: xyz={p_ready.round(4)} m, rpy={rpy_ready.round(1)} deg")

    if args.check_sdk:
        try:
            import rby1_sdk as rby

            urdf = os.path.join(os.path.dirname(rby.__file__), "models", "rby1m", "urdf", "model_v1.3.urdf")
            dyn = rby.dynamics.load_robot_from_urdf(urdf, "base")
            state = dyn.make_state(["base", "link_torso_5"], [f"torso_{i}" for i in range(6)])
            state.set_q(np.deg2rad(READY_DEG))
            dyn.compute_forward_kinematics(state)
            T_sdk = np.asarray(dyn.compute_transformation(state, 0, 1))
            err = float(np.linalg.norm(T_sdk[:3, 3] - p_ready))
            print(f"--check-sdk: |p_sdk - p_fk| = {err * 1000:.2f} mm, rot err = "
                  f"{math.degrees(np.linalg.norm(Rotation.from_matrix(T_sdk[:3, :3].T @ T_ready[:3, :3]).as_rotvec())):.3f} deg")
        except Exception as e:  # noqa: BLE001
            print(f"--check-sdk: skipped ({e.__class__.__name__}: {e})")

    q = sample(lim, args.n_roll, args.n_pitch, args.n_yaw)
    T = fk(q)
    p = T[:, :3, 3]
    rpy = rpy_deg(T)
    print(f"samples: {len(q)}")

    print("\n## 전체 가동범위 (base 프레임, link_torso_5)")
    rows = []
    for i, name in enumerate(("x [m]", "y [m]", "z [m]")):
        rows.append((name, p[:, i].min(), p[:, i].max(), p_ready[i]))
    for i, name in enumerate(("roll [deg]", "pitch [deg]", "yaw [deg]")):
        rows.append((name, rpy[:, i].min(), rpy[:, i].max(), rpy_ready[i]))
    print(table(rows))

    # Sagittal slice: roll / yaw joints at zero.
    sag = (q[:, 0] == 0.0) & (q[:, 4] == 0.0) & (q[:, 5] == 0.0)
    if not sag.any():  # grids without an exact zero: take the nearest column
        sag = (np.abs(q[:, 0]) == np.abs(q[:, 0]).min()) & (np.abs(q[:, 4]) == np.abs(q[:, 4]).min()) & (np.abs(q[:, 5]) == np.abs(q[:, 5]).min())
    ps, rs = p[sag], rpy[sag]
    print("\n## 사지탈 슬라이스 (torso_0 = torso_4 = torso_5 = 0)")
    print(table([("x [m]", ps[:, 0].min(), ps[:, 0].max(), p_ready[0]), ("z [m]", ps[:, 2].min(), ps[:, 2].max(), p_ready[2]),
                 ("pitch [deg]", rs[:, 1].min(), rs[:, 1].max(), rpy_ready[1])]))
    print("\n### z 구간별 x 범위 (사지탈)")
    print("| z [m] | x min | x max |", "\n|---|---|---|", sep="")
    for z0 in np.arange(math.floor(ps[:, 2].min() * 10) / 10, ps[:, 2].max() + 1e-9, 0.1):
        m = (ps[:, 2] >= z0) & (ps[:, 2] < z0 + 0.1)
        if m.any():
            print(f"| {z0:.1f}–{z0 + 0.1:.1f} | {ps[m, 0].min():+.3f} | {ps[m, 0].max():+.3f} |")
    print("\n### pitch 구간별 z 범위 (사지탈; pitch + = 앞으로 숙임)")
    print("| pitch [deg] | z min | z max | x min | x max |", "\n|---|---|---|---|---|", sep="")
    for a0 in np.arange(math.floor(rs[:, 1].min() / 15) * 15, rs[:, 1].max() + 1e-9, 15):
        m = (rs[:, 1] >= a0) & (rs[:, 1] < a0 + 15)
        if m.any():
            print(f"| {a0:+.0f}–{a0 + 15:+.0f} | {ps[m, 2].min():+.3f} | {ps[m, 2].max():+.3f} | {ps[m, 0].min():+.3f} | {ps[m, 0].max():+.3f} |")

    # Convex hull of the sagittal reach (x, z).
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(ps[:, [0, 2]])
        pts = ps[hull.vertices][:, [0, 2]]
        print("\n### 사지탈 x–z 외곽 꼭짓점 (m)")
        print(", ".join(f"({x:+.2f}, {z:+.2f})" for x, z in pts))
    except Exception as e:  # noqa: BLE001
        print(f"(convex hull skipped: {e})")

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([n for n, _, _ in CHAIN] + ["x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg"])
        for qi, pi, ri in zip(q, p, rpy):
            w.writerow([f"{v:.5f}" for v in qi] + [f"{v:.4f}" for v in pi] + [f"{v:.2f}" for v in ri])
    print(f"\nCSV: {args.csv} ({len(q)} rows)")


if __name__ == "__main__":
    main()
