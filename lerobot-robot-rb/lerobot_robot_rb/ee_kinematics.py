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
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
except ImportError as exc:
    raise ImportError(
        "lerobot_robot_rb.ee_kinematics requires SciPy in the client environment"
    ) from exc


_MODEL_URDFS = {
    "rb10": "rb10_1300e.urdf",
    "rb10e": "rb10_1300e_u.urdf",
}

# Conservative deployment envelopes from the existing RB model/teleoperation
# tables.  These are software safety bounds, not authoritative hardware range
# claims.  Every effective bound remains intersected with the packaged URDF;
# joints not listed here retain the URDF's +-3.14 rad range.
_ELBOW_ENVELOPE_DEG = {
    "rb10": 165.0,
    "rb10e": 154.0,
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
        max_joint_delta: float = 0.2,
        position_tolerance: float = 1e-6,
        orientation_tolerance: float = 1e-6,
        max_nfev: int = 400,
    ) -> None:
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
        self._chain, urdf_limits = _load_chain(model)
        self._limits = urdf_limits.copy()
        elbow_limit = np.deg2rad(_ELBOW_ENVELOPE_DEG[model])
        self._limits[2, 0] = max(self._limits[2, 0], -elbow_limit)
        self._limits[2, 1] = min(self._limits[2, 1], elbow_limit)

    @property
    def joint_limits(self) -> np.ndarray:
        """Return effective conservative software limits, in radians.

        These limits are the intersection of the packaged URDF ranges and the
        model's deployment envelope.  They are not an authoritative statement
        of the physical robot's full travel.
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
        if np.any(joints < self._limits[:, 0]) or np.any(joints > self._limits[:, 1]):
            raise ValueError(f"{name} is outside the effective safety joint limits")
        return joints

    def _transform(self, q: np.ndarray) -> np.ndarray:
        transform = np.eye(4)
        active_index = 0
        for joint in self._chain:
            transform = transform @ joint.origin
            if joint.kind == "revolute":
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(
                    joint.axis * q[active_index]
                ).as_matrix()
                transform = transform @ motion
                active_index += 1
        return transform

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
    ) -> np.ndarray:
        """Solve bounded local IK, returning a joint vector near ``seed``.

        ``max_joint_delta`` overrides the instance default for this solve only.

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

        def residual(q: np.ndarray) -> np.ndarray:
            current = self._transform(q)
            position_error = (current[:3, 3] - target[:3]) / self.position_tolerance
            rotation_error = Rotation.from_matrix(
                target_rotation @ current[:3, :3].T
            ).as_rotvec() / self.orientation_tolerance
            values = np.concatenate((position_error, rotation_error))
            if not np.all(np.isfinite(values)):
                return np.full(6, np.finfo(float).max ** 0.25)
            return values

        result = least_squares(
            residual,
            seed_array,
            bounds=(lower, upper),
            method="trf",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=self.max_nfev,
            x_scale="jac",
        )
        solution = np.asarray(result.x, dtype=float)
        if not result.success or not np.all(np.isfinite(solution)):
            raise RuntimeError(f"inverse kinematics did not converge: {result.message}")
        if np.any(solution < self._limits[:, 0]) or np.any(solution > self._limits[:, 1]):
            raise RuntimeError("inverse kinematics returned a joint-limit violation")
        if np.max(np.abs(solution - seed_array)) > joint_delta + 1e-10:
            raise RuntimeError("inverse kinematics exceeded max_joint_delta")

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


__all__ = ["RBEEKinematics"]
