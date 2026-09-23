"""URDF-based end-effector kinematics for the RB10 and RB10-E.

The public pose convention is ``[x, y, z, rx, ry, rz]`` with position in
metres and extrinsic XYZ Euler angles in radians (``Rz @ Ry @ Rx``).  The
chain is evaluated exactly from ``link0`` through the URDF's ``tcp`` link;
no additional tool transform is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ImportError as exc:
    raise ImportError(
        "lerobot_robot_rb.ee_kinematics requires SciPy in the client environment"
    ) from exc


_MODEL_URDFS = {
    "rb10": "rb10_1300e.urdf",
    "rb10e": "rb10_1300e_u.urdf",
}

# Match the existing models.py RB10 and teleoperation RB10-E envelopes.
# URDF +/-3.14 limits are generic geometry metadata, not controller travel.
# Preserve unwrapped controller angles throughout FK, seeded IK, and ServoJ.
_JOINT_ENVELOPES_DEG = {
    "rb10": ((-360, 360), (-360, 360), (-165, 165), (-360, 360), (-360, 360), (-360, 360)),
    "rb10e": ((-360, 360), (-180, 180), (-154, 154), (-360, 360), (-360, 360), (-360, 360)),
}


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray | None
    limits: tuple[float, float] | None


def _vector(element: ET.Element | None, attribute: str, default: str) -> np.ndarray:
    text = default if element is None else element.get(attribute, default)
    values = np.fromstring(text, sep=" ", dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid URDF {attribute}: {text!r}")
    return values


def _origin_transform(element: ET.Element | None) -> np.ndarray:
    xyz = _vector(element, "xyz", "0 0 0")
    rpy = _vector(element, "rpy", "0 0 0")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def _load_chain(model: str) -> tuple[tuple[_Joint, ...], np.ndarray]:
    try:
        filename = _MODEL_URDFS[model]
    except KeyError as exc:
        choices = ", ".join(sorted(_MODEL_URDFS))
        raise ValueError(
            f"unsupported model {model!r}; expected one of: {choices}"
        ) from exc

    asset = Path(__file__).with_name("urdf") / filename
    with asset.open("rb") as stream:
        root = ET.parse(stream).getroot()

    by_child: dict[str, ET.Element] = {}
    for element in root.findall("joint"):
        child = element.find("child")
        if child is None or not child.get("link"):
            raise ValueError("URDF joint is missing its child link")
        child_name = child.get("link")
        if child_name in by_child:
            raise ValueError(f"URDF has multiple parents for link {child_name!r}")
        by_child[child_name] = element

    elements: list[ET.Element] = []
    link = "tcp"
    visited: set[str] = set()
    while link != "link0":
        if link in visited or link not in by_child:
            raise ValueError("URDF does not contain a unique link0-to-tcp chain")
        visited.add(link)
        joint = by_child[link]
        elements.append(joint)
        parent = joint.find("parent")
        if parent is None or not parent.get("link"):
            raise ValueError("URDF joint is missing its parent link")
        link = parent.get("link")
    elements.reverse()

    chain: list[_Joint] = []
    active_limits: list[tuple[float, float]] = []
    for element in elements:
        name = element.get("name", "<unnamed>")
        kind = element.get("type", "")
        if kind not in {"fixed", "revolute"}:
            raise ValueError(f"unsupported joint type {kind!r} in {name!r}")

        axis = None
        limits = None
        if kind == "revolute":
            axis = _vector(element.find("axis"), "xyz", "1 0 0")
            norm = np.linalg.norm(axis)
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValueError(f"invalid axis for joint {name!r}")
            axis = axis / norm

            limit = element.find("limit")
            if limit is None or limit.get("lower") is None or limit.get("upper") is None:
                raise ValueError(f"joint {name!r} has no finite position limits")
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError(f"invalid limits for joint {name!r}")
            limits = (lower, upper)
            active_limits.append(limits)

        chain.append(
            _Joint(
                name=name,
                kind=kind,
                origin=_origin_transform(element.find("origin")),
                axis=axis,
                limits=limits,
            )
        )

    if len(active_limits) != 6:
        raise ValueError(
            "expected 6 revolute joints in link0-to-tcp chain, "
            f"found {len(active_limits)}"
        )
    return tuple(chain), np.asarray(active_limits, dtype=float)


class RBEEKinematics:
    """Bounded local FK/IK for an RB10-family ``link0`` to ``tcp`` chain.

    Args:
        model: ``"rb10"`` or ``"rb10e"``.
        max_joint_delta: Maximum absolute per-joint IK displacement from the
            seed, in radians.  The solver is bounded to this neighbourhood.
        position_tolerance: Maximum accepted translation error, in metres.
        orientation_tolerance: Maximum accepted rotation-vector error, in
            radians.
        max_nfev: Maximum residual evaluations for one IK solve.
    """

    def __init__(
        self,
        model: str = "rb10",
        *,
        max_joint_delta: float = 0.5,
        position_tolerance: float = 1e-6,
        orientation_tolerance: float = 1e-6,
        max_nfev: int = 100,
    ) -> None:
        # ``max_nfev`` bounds forward evaluations per solve (about 0.1 ms
        # each); a converged waypoint typically needs 5-15.
        if not isinstance(model, str):
            raise ValueError("model must be 'rb10' or 'rb10e'")
        model = model.lower()
        max_joint_delta = self._validate_max_joint_delta(max_joint_delta)
        if not np.isfinite(position_tolerance) or position_tolerance <= 0.0:
            raise ValueError("position_tolerance must be finite and positive")
        if not np.isfinite(orientation_tolerance) or orientation_tolerance <= 0.0:
            raise ValueError("orientation_tolerance must be finite and positive")
        if not isinstance(max_nfev, int) or max_nfev <= 0:
            raise ValueError("max_nfev must be a positive integer")

        self.model = model
        self.max_joint_delta = max_joint_delta
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.max_nfev = max_nfev
        self._chain, _ = _load_chain(model)
        self._limits = np.deg2rad(_JOINT_ENVELOPES_DEG[model])
        axes = np.asarray([joint.axis for joint in self._chain if joint.kind == "revolute"])
        self._axis_outer = axes[:, :, None] * axes[:, None, :]
        self._axis_skew = np.zeros((6, 3, 3))
        self._axis_skew[:, 0, 1] = -axes[:, 2]
        self._axis_skew[:, 0, 2] = axes[:, 1]
        self._axis_skew[:, 1, 0] = axes[:, 2]
        self._axis_skew[:, 1, 2] = -axes[:, 0]
        self._axis_skew[:, 2, 0] = -axes[:, 1]
        self._axis_skew[:, 2, 1] = axes[:, 0]

    @property
    def joint_limits(self) -> np.ndarray:
        """Return effective conservative software limits, in radians.

        These match the existing model/teleoperation deployment envelopes,
        not the URDF's generic +/-3.14 metadata. They are not an authoritative
        statement of the physical robot's full travel.
        """
        return self._limits.copy()

    @property
    def safety_joint_limits(self) -> np.ndarray:
        """Alias for the effective conservative :attr:`joint_limits`."""
        return self.joint_limits

    @staticmethod
    def _array6(value: object, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.shape != (6,):
            raise ValueError(f"{name} must be a length-6 vector")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array

    @staticmethod
    def _validate_max_joint_delta(value: object) -> float:
        try:
            delta = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_joint_delta must be finite and positive") from exc
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError("max_joint_delta must be finite and positive")
        return delta

    def _validate_joints(self, q: object, name: str) -> np.ndarray:
        joints = self._array6(q, name)
        outside = (joints < self._limits[:, 0]) | (joints > self._limits[:, 1])
        if np.any(outside):
            details = "; ".join(
                f"joint_{i}={np.rad2deg(joints[i]):.4f} deg "
                f"allowed=[{np.rad2deg(self._limits[i, 0]):.1f}, "
                f"{np.rad2deg(self._limits[i, 1]):.1f}] deg"
                for i in np.flatnonzero(outside)
            )
            raise ValueError(f"{name} is outside the effective safety joint limits: {details}")
        return joints

    def _transform(self, q: np.ndarray) -> np.ndarray:
        """Evaluate one pose or a batch of poses with the same URDF chain."""
        cosine = np.cos(q)[..., None, None]
        sine = np.sin(q)[..., None, None]
        motions = np.broadcast_to(np.eye(4), q.shape + (4, 4)).copy()
        motions[..., :3, :3] = (
            cosine * np.eye(3) + (1 - cosine) * self._axis_outer + sine * self._axis_skew
        )
        transform = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
        active_index = 0
        for joint in self._chain:
            transform = transform @ joint.origin
            if joint.kind == "revolute":
                transform = transform @ motions[..., active_index, :, :]
                active_index += 1
        return transform

    def _frames(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate one pose plus every revolute axis and origin in the base frame.

        Returns ``(transform, axes, origins)`` where ``axes[i]`` and
        ``origins[i]`` describe joint ``i`` for the geometric Jacobian.
        """
        cosine = np.cos(q)
        sine = np.sin(q)
        transform = np.eye(4)
        axes = np.empty((6, 3))
        origins = np.empty((6, 3))
        active_index = 0
        for joint in self._chain:
            transform = transform @ joint.origin
            if joint.kind == "revolute":
                axes[active_index] = transform[:3, :3] @ joint.axis
                origins[active_index] = transform[:3, 3]
                rotation = (
                    cosine[active_index] * np.eye(3)
                    + (1 - cosine[active_index]) * self._axis_outer[active_index]
                    + sine[active_index] * self._axis_skew[active_index]
                )
                transform = transform @ np.block(
                    [[rotation, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]
                )
                active_index += 1
        return transform, axes, origins

    @staticmethod
    def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
        """Rotation vector (axis * angle) of a rotation matrix, no SciPy objects."""
        cos_angle = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.arccos(cos_angle))
        axis = np.array(
            [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
        )
        if angle < 1e-6:
            # First-order: skew part already equals 2 * axis * sin(angle).
            return axis / 2.0
        if np.pi - angle < 1e-6:
            # Near pi the skew part vanishes; recover the axis from the symmetric part.
            diagonal = np.clip((np.diag(matrix) + 1.0) / 2.0, 0.0, 1.0)
            unit = np.sqrt(diagonal)
            unit *= np.sign(axis + (axis == 0.0))
            return unit / np.linalg.norm(unit) * angle
        return axis / (2.0 * np.sin(angle)) * angle

    def _solve_bounded(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        start: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Box-constrained Levenberg-Marquardt on the scaled pose residual.

        Returns ``(q, position_error, orientation_error)`` for the best
        iterate; the caller decides whether the errors are acceptable.
        Each iteration evaluates one forward pass and the analytic geometric
        Jacobian, so a typical waypoint converges in a few milliseconds.
        """
        position_scale = 1.0 / self.position_tolerance
        orientation_scale = 1.0 / self.orientation_tolerance

        def evaluate(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            transform, axes, origins = self._frames(q)
            position_error = transform[:3, 3] - target_position
            orientation_error = self._rotation_vector(target_rotation @ transform[:3, :3].T)
            residual = np.concatenate(
                (position_error * position_scale, orientation_error * orientation_scale)
            )
            if not np.all(np.isfinite(residual)):
                raise RuntimeError("inverse kinematics did not converge: non-finite residual")
            return residual, transform, axes, origins

        q = np.clip(start, lower, upper)
        residual, transform, axes, origins = evaluate(q)
        cost = float(residual @ residual)
        damping = 1e-3
        # Converge well inside the acceptance tolerance (LM is quadratic near
        # the solution, so this costs about one extra iteration) and bound the
        # total number of forward evaluations so a stuck solve cannot stall
        # the control loop.
        target_cost = 1e-6
        evaluations = 1
        stagnant = 0
        while evaluations < self.max_nfev:
            if cost <= target_cost:
                break
            if stagnant >= 3:
                # An active box bound blocks further progress; the caller
                # retries with a smaller waypoint instead of grinding here.
                break
            lever = transform[:3, 3] - origins
            jacobian = np.empty((6, 6))
            jacobian[:3] = np.cross(axes, lever).T * position_scale
            # d(rotvec(R_target R(q)^T))/dq = -J_omega to first order.
            jacobian[3:] = -axes.T * orientation_scale
            normal = jacobian.T @ jacobian
            gradient = jacobian.T @ residual
            improved = False
            for _ in range(12):
                if evaluations >= self.max_nfev:
                    break
                regularised = normal + damping * (np.diag(np.diag(normal)) + 1e-9 * np.eye(6))
                step = np.linalg.solve(regularised, -gradient)
                candidate = np.clip(q + step, lower, upper)
                candidate_residual, candidate_transform, candidate_axes, candidate_origins = evaluate(candidate)
                evaluations += 1
                candidate_cost = float(candidate_residual @ candidate_residual)
                if candidate_cost < cost:
                    stagnant = stagnant + 1 if candidate_cost > 0.9 * cost else 0
                    q, residual, cost = candidate, candidate_residual, candidate_cost
                    transform, axes, origins = candidate_transform, candidate_axes, candidate_origins
                    damping = max(damping / 4.0, 1e-9)
                    improved = True
                    break
                damping *= 6.0
            if not improved:
                break
        position_error = float(np.linalg.norm(residual[:3]) * self.position_tolerance)
        orientation_error = float(np.linalg.norm(residual[3:]) * self.orientation_tolerance)
        return q, position_error, orientation_error

    def forward(self, q: object) -> np.ndarray:
        """Return the ``link0 -> tcp`` pose for six joint angles in radians."""
        joints = self._validate_joints(q, "q")
        transform = self._transform(joints)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            euler = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")
        pose = np.concatenate((transform[:3, 3], euler))
        if not np.all(np.isfinite(pose)):
            raise RuntimeError("forward kinematics produced a non-finite pose")
        return pose

    def inverse(
        self,
        pose6: object,
        seed: object,
        *,
        max_joint_delta: float | None = None,
        joint_bounds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve bounded local IK, returning a joint vector near ``seed``.

        ``max_joint_delta`` overrides the instance default for this solve only.
        ``joint_bounds`` can further restrict (never expand) the search box.

        Raises:
            ValueError: Inputs are malformed, non-finite, or outside joint limits.
            RuntimeError: No solution satisfies the configured residual and
                seed-continuity bounds.
        """
        target = self._array6(pose6, "pose6")
        seed_array = self._validate_joints(seed, "seed")
        joint_delta = (
            self.max_joint_delta
            if max_joint_delta is None
            else self._validate_max_joint_delta(max_joint_delta)
        )
        target_rotation = Rotation.from_euler("xyz", target[3:]).as_matrix()

        lower = np.maximum(self._limits[:, 0], seed_array - joint_delta)
        upper = np.minimum(self._limits[:, 1], seed_array + joint_delta)
        if joint_bounds is not None:
            bounds = np.asarray(joint_bounds, dtype=float)
            if bounds.shape != (6, 2) or not np.all(np.isfinite(bounds)):
                raise ValueError("joint_bounds must be a finite (6, 2) array")
            lower = np.maximum(lower, bounds[:, 0])
            upper = np.minimum(upper, bounds[:, 1])
        if np.any(lower >= upper):
            raise ValueError("no safe joint search interval")

        solution, _, _ = self._solve_bounded(
            target[:3], target_rotation, seed_array, lower, upper
        )
        if not np.all(np.isfinite(solution)):
            raise RuntimeError("inverse kinematics did not converge: non-finite solution")
        if np.any(solution < self._limits[:, 0]) or np.any(solution > self._limits[:, 1]):
            raise RuntimeError("inverse kinematics returned a joint-limit violation")
        if np.max(np.abs(solution - seed_array)) > joint_delta + 1e-10:
            raise RuntimeError("inverse kinematics exceeded max_joint_delta")

        # Independent acceptance check with SciPy's rotation, not the solver's
        # own residual, so a solver bug cannot certify its own answer.
        achieved = self._transform(solution)
        position_error = np.linalg.norm(achieved[:3, 3] - target[:3])
        orientation_error = np.linalg.norm(
            Rotation.from_matrix(target_rotation @ achieved[:3, :3].T).as_rotvec()
        )
        if (
            not np.isfinite(position_error)
            or not np.isfinite(orientation_error)
            or position_error > self.position_tolerance
            or orientation_error > self.orientation_tolerance
        ):
            raise RuntimeError(
                "inverse kinematics residual exceeds tolerance "
                f"(position={position_error:.3g} m, orientation={orientation_error:.3g} rad)"
            )
        return solution

    def inverse_step(
        self, pose6: object, seed: object, *, max_joint_delta: float,
        previous_q: object | None = None,
        max_tracking_error: float | None = None,
    ) -> np.ndarray:
        """Solve a Cartesian waypoint without clipping joints independently.

        The complete goal must be reachable in the bounded local branch. Servo
        commands obey a velocity limit from the previous command and a tracking
        limit from measured joints. ``None`` retains the legacy tracking envelope
        of ``max_joint_delta``. Smaller XYZ / SO(3) waypoints are retried, never
        an approximate failed IK result.
        """
        target = self._array6(pose6, "pose6")
        measured = self._validate_joints(seed, "seed")
        step = self._validate_max_joint_delta(max_joint_delta)
        anchor = (
            measured
            if previous_q is None
            else self._validate_joints(previous_q, "previous_q")
        )
        if max_tracking_error is None:
            tracking = step
        else:
            try:
                tracking = float(max_tracking_error)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "max_tracking_error must be finite and positive"
                ) from exc
            if not np.isfinite(tracking) or tracking <= 0.0:
                raise ValueError("max_tracking_error must be finite and positive")
            if np.max(np.abs(anchor - measured)) > tracking:
                raise ValueError("previous_q exceeds max_tracking_error from measured joints")

        lower = np.maximum.reduce(
            (self._limits[:, 0], anchor - step, measured - tracking)
        )
        upper = np.minimum.reduce(
            (self._limits[:, 1], anchor + step, measured + tracking)
        )
        if np.any(lower >= upper):
            raise ValueError("measured joints and previous command have no safe step intersection")

        goal = self.inverse(target, anchor)
        if np.all(goal >= lower) and np.all(goal <= upper):
            return goal
        delta = goal - anchor
        fraction = min(1.0, 0.8 * step / float(np.max(np.abs(delta))))
        start = self._transform(anchor)
        rotation_delta = Rotation.from_matrix(
            Rotation.from_euler("xyz", target[3:]).as_matrix() @ start[:3, :3].T
        ).as_rotvec()
        last_error = None
        for _ in range(3):
            rotation = Rotation.from_rotvec(fraction * rotation_delta).as_matrix() @ start[:3, :3]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                euler = Rotation.from_matrix(rotation).as_euler("xyz")
            waypoint = np.r_[start[:3, 3] + fraction * (target[:3] - start[:3, 3]), euler]
            try:
                return self.inverse(
                    waypoint, anchor, max_joint_delta=step,
                    joint_bounds=np.column_stack((lower, upper)),
                )
            except RuntimeError as exc:
                last_error = exc
                fraction *= 0.5
        raise RuntimeError(f"no converged safe Cartesian IK step: {last_error}")


__all__ = ["RBEEKinematics"]
