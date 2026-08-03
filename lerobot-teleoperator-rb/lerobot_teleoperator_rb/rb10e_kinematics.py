"""RB10E forward kinematics, Jacobian and iterative IK.

This module preserves the numerical model and IKLM implementation from the
original hardware-tested ``RB10E_utils.py``.

Only robot communication has been removed. The LeRobot Robot adapter owns the
single control-box connection; this class performs kinematics only.

Units
-----
Joint values:
    radians

Cartesian translation:
    millimetres

Cartesian rotation:
    3x3 rotation matrix inside a 4x4 homogeneous transform
"""

from __future__ import annotations

import numpy as np


RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0


class RB10E:
    """Original RB10E six-axis kinematics and IKLM solver."""

    def __init__(self) -> None:
        self._DIM = 6

        self._q_out = np.zeros(
            (self._DIM,),
            dtype=np.float64,
        )

        # Original IK seed from RB10E_utils.py.
        #
        # This is only a numerical IK seed. It does not command the robot
        # to move to this pose.
        self._q = (
            np.array(
                [
                    140.0,
                    -50.0,
                    -100.0,
                    90.0,
                    -90.0,
                    0.0,
                ],
                dtype=np.float64,
            )
            * DEG2RAD
        )

        self._fk = np.zeros(
            (4, 4),
            dtype=np.float64,
        )

        self._jcbn = np.zeros(
            (
                self._DIM,
                self._DIM,
            ),
            dtype=np.float64,
        )

        self._error = np.empty(
            (6,),
            dtype=np.float64,
        )

        # Keep original misspelled attribute names for parity.
        self._gradiant = np.empty(
            (self._DIM,),
            dtype=np.float64,
        )
        self._hessian = np.empty(
            (
                self._DIM,
                self._DIM,
            ),
            dtype=np.float64,
        )

        # Original RB10E IK joint limits.
        self._q_min = (
            np.array(
                [
                    -360.0,
                    -180.0,
                    -154.0,
                    -360.0,
                    -360.0,
                    -360.0,
                ],
                dtype=np.float64,
            )
            * DEG2RAD
        )

        self._q_max = (
            np.array(
                [
                    360.0,
                    180.0,
                    154.0,
                    360.0,
                    360.0,
                    360.0,
                ],
                dtype=np.float64,
            )
            * DEG2RAD
        )

        self._q_mid = (
            self._q_max
            + self._q_min
        ) / 2.0

        self.T0 = np.zeros(
            (4, 4),
            dtype=np.float64,
        )
        self.T1 = np.zeros(
            (4, 4),
            dtype=np.float64,
        )
        self.T2 = np.zeros(
            (4, 4),
            dtype=np.float64,
        )
        self.T3 = np.zeros(
            (4, 4),
            dtype=np.float64,
        )
        self.T4 = np.zeros(
            (4, 4),
            dtype=np.float64,
        )
        self.T5 = np.zeros(
            (4, 4),
            dtype=np.float64,
        )

        self._We6 = np.identity(
            6,
            dtype=np.float64,
        )
        self._we = 1.0e-1

        # Initialise FK and Jacobian for the original seed.
        self.update_fk_and_jcbn(
            self._q
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_joint_vector(
        q: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(
            q,
            dtype=np.float64,
        )

        if values.shape != (6,):
            raise ValueError(
                "Joint vector must have shape (6,), "
                f"got {values.shape}."
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "Joint vector contains non-finite values."
            )

        return values

    @staticmethod
    def _validate_transform(
        transform: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        matrix = np.asarray(
            transform,
            dtype=np.float64,
        )

        if matrix.shape != (4, 4):
            raise ValueError(
                f"{name} must have shape (4, 4), "
                f"got {matrix.shape}."
            )

        if not np.all(
            np.isfinite(matrix)
        ):
            raise ValueError(
                f"{name} contains non-finite values."
            )

        return matrix

    # ------------------------------------------------------------------
    # Forward kinematics and Jacobian
    # ------------------------------------------------------------------

    def update_fk_and_jcbn(
        self,
        q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update FK and geometric Jacobian for one joint vector."""

        q = self._validate_joint_vector(
            q
        )

        self.T0 = np.array(
            [
                [
                    np.cos(q[0]),
                    -6.12323399573677e-17
                    * np.sin(q[0]),
                    -1.0 * np.sin(q[0]),
                    0.0,
                ],
                [
                    np.sin(q[0]),
                    6.12323399573677e-17
                    * np.cos(q[0]),
                    1.0 * np.cos(q[0]),
                    0.0,
                ],
                [
                    0.0,
                    -1.0,
                    6.12323399573677e-17,
                    197.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            ],
            dtype=np.float64,
        )

        self.T1 = (
            self.T0
            @ np.array(
                [
                    [
                        np.cos(q[1]),
                        -6.12323399573677e-17
                        * np.sin(q[1]),
                        1.0 * np.sin(q[1]),
                        0.0,
                    ],
                    [
                        np.sin(q[1]),
                        6.12323399573677e-17
                        * np.cos(q[1]),
                        -1.0 * np.cos(q[1]),
                        0.0,
                    ],
                    [
                        0.0,
                        1.0,
                        6.12323399573677e-17,
                        -187.5,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                ],
                dtype=np.float64,
            )
        )

        self.T2 = (
            self.T1
            @ np.array(
                [
                    [
                        np.cos(q[2]),
                        -6.12323399573677e-17
                        * np.sin(q[2]),
                        1.0 * np.sin(q[2]),
                        0.0,
                    ],
                    [
                        6.12323399573677e-17
                        * np.sin(q[2]),
                        3.74939945665464e-33
                        * np.cos(q[2])
                        + 1.0,
                        6.12323399573677e-17
                        - 6.12323399573677e-17
                        * np.cos(q[2]),
                        148.4,
                    ],
                    [
                        -1.0 * np.sin(q[2]),
                        6.12323399573677e-17
                        - 6.12323399573677e-17
                        * np.cos(q[2]),
                        1.0 * np.cos(q[2])
                        + 3.74939945665464e-33,
                        612.7,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                ],
                dtype=np.float64,
            )
        )

        self.T3 = (
            self.T2
            @ np.array(
                [
                    [
                        np.cos(q[3]),
                        -6.12323399573677e-17
                        * np.sin(q[3]),
                        1.0 * np.sin(q[3]),
                        0.0,
                    ],
                    [
                        6.12323399573677e-17
                        * np.sin(q[3]),
                        3.74939945665464e-33
                        * np.cos(q[3])
                        + 1.0,
                        6.12323399573677e-17
                        - 6.12323399573677e-17
                        * np.cos(q[3]),
                        -117.15,
                    ],
                    [
                        -1.0 * np.sin(q[3]),
                        6.12323399573677e-17
                        - 6.12323399573677e-17
                        * np.cos(q[3]),
                        1.0 * np.cos(q[3])
                        + 3.74939945665464e-33,
                        570.15,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                ],
                dtype=np.float64,
            )
        )

        self.T4 = (
            self.T3
            @ np.array(
                [
                    [
                        np.cos(q[4]),
                        -6.12323399573677e-17
                        * np.sin(q[4]),
                        -1.0 * np.sin(q[4]),
                        0.0,
                    ],
                    [
                        np.sin(q[4]),
                        6.12323399573677e-17
                        * np.cos(q[4]),
                        1.0 * np.cos(q[4]),
                        0.0,
                    ],
                    [
                        0.0,
                        -1.0,
                        6.12323399573677e-17,
                        117.15,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                ],
                dtype=np.float64,
            )
        )

        self.T5 = (
            self.T4
            @ np.array(
                [
                    [
                        np.cos(q[5]),
                        -6.12323399573677e-17
                        * np.sin(q[5]),
                        1.0 * np.sin(q[5]),
                        0.0,
                    ],
                    [
                        np.sin(q[5]),
                        6.12323399573677e-17
                        * np.cos(q[5]),
                        -1.0 * np.cos(q[5]),
                        0.0,
                    ],
                    [
                        0.0,
                        1.0,
                        6.12323399573677e-17,
                        -259.3,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                ],
                dtype=np.float64,
            )
        )

        fk = (
            self.T5
            @ np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        )

        self._fk = fk.copy()

        self._jcbn[
            0:3,
            0,
        ] = np.cross(
            -self.T0[0:3, 1],
            fk[0:3, 3]
            - self.T0[0:3, 3],
        ).astype(
            np.float64
        )

        self._jcbn[
            0:3,
            1,
        ] = np.cross(
            self.T1[0:3, 1],
            fk[0:3, 3]
            - self.T1[0:3, 3],
        ).astype(
            np.float64
        )

        self._jcbn[
            0:3,
            2,
        ] = np.cross(
            self.T2[0:3, 1],
            fk[0:3, 3]
            - self.T2[0:3, 3],
        ).astype(
            np.float64
        )

        self._jcbn[
            0:3,
            3,
        ] = np.cross(
            self.T3[0:3, 1],
            fk[0:3, 3]
            - self.T3[0:3, 3],
        ).astype(
            np.float64
        )

        self._jcbn[
            0:3,
            4,
        ] = np.cross(
            -self.T4[0:3, 1],
            fk[0:3, 3]
            - self.T4[0:3, 3],
        ).astype(
            np.float64
        )

        self._jcbn[
            0:3,
            5,
        ] = np.cross(
            self.T5[0:3, 1],
            fk[0:3, 3]
            - self.T5[0:3, 3],
        ).astype(
            np.float64
        )

        self._jcbn[
            3:6,
            0,
        ] = -self.T0[0:3, 1]

        self._jcbn[
            3:6,
            1,
        ] = self.T1[0:3, 1]

        self._jcbn[
            3:6,
            2,
        ] = self.T2[0:3, 1]

        self._jcbn[
            3:6,
            3,
        ] = self.T3[0:3, 1]

        self._jcbn[
            3:6,
            4,
        ] = -self.T4[0:3, 1]

        self._jcbn[
            3:6,
            5,
        ] = self.T5[0:3, 1]

        return (
            fk,
            self._jcbn.copy(),
        )

    # ------------------------------------------------------------------
    # Cartesian error
    # ------------------------------------------------------------------

    def update_error_vector(
        self,
        targ_6D: np.ndarray,
        curr_6D: np.ndarray,
    ) -> np.ndarray:
        """Calculate the original six-dimensional IK error vector."""

        target = self._validate_transform(
            targ_6D,
            name="targ_6D",
        )
        current = self._validate_transform(
            curr_6D,
            name="curr_6D",
        )

        rotation_target = target[
            0:3,
            0:3,
        ].copy()

        rotation_current = current[
            0:3,
            0:3,
        ].copy()

        rotation_error = (
            rotation_target
            @ rotation_current.T
        )

        self._error[0] = (
            target[0, 3]
            - current[0, 3]
        )
        self._error[1] = (
            target[1, 3]
            - current[1, 3]
        )
        self._error[2] = (
            target[2, 3]
            - current[2, 3]
        )

        self._error[3] = (
            rotation_error[2, 1]
            - rotation_error[1, 2]
        )
        self._error[4] = (
            rotation_error[0, 2]
            - rotation_error[2, 0]
        )
        self._error[5] = (
            rotation_error[1, 0]
            - rotation_error[0, 1]
        )

        return self._error

    # ------------------------------------------------------------------
    # Iterative inverse kinematics
    # ------------------------------------------------------------------

    def IKLM(
        self,
        initial_q: np.ndarray,
        targ_6D: np.ndarray,
        iter: int,
    ) -> np.ndarray:
        """Run the original iterative IKLM update.

        Parameters
        ----------
        initial_q:
            Initial six-joint seed in radians.

        targ_6D:
            Desired 4x4 TCP transform. Translation is in millimetres.

        iter:
            Number of iterations. The original VR loop uses three.

        Returns
        -------
        numpy.ndarray
            Updated six-joint solution in radians.

        Notes
        -----
        The solution is also stored in ``self._q``, matching the original
        implementation.
        """

        if not isinstance(
            iter,
            int,
        ):
            raise TypeError(
                "iter must be an integer."
            )

        if iter <= 0:
            raise ValueError(
                f"iter must be > 0, got {iter}."
            )

        q = self._validate_joint_vector(
            initial_q
        ).copy()

        target = self._validate_transform(
            targ_6D,
            name="targ_6D",
        )

        for _ in range(iter):
            fk, jcbn = (
                self.update_fk_and_jcbn(
                    q
                )
            )

            error = (
                self.update_error_vector(
                    target,
                    fk,
                )
            )

            quadratic_error = (
                error.T
                @ error
                * 0.5
            )

            gradiant = (
                jcbn.T
                @ error
            )

            hessian_approx = (
                jcbn.T
                @ jcbn
                + quadratic_error
                * self._We6
                + 2.0
                * self._We6
            )

            try:
                delta_q = np.linalg.solve(
                    hessian_approx,
                    gradiant,
                )
            except np.linalg.LinAlgError:
                # The original calculation uses solve(). This fallback is
                # only used when numerical singularity prevents a solution.
                delta_q = (
                    np.linalg.lstsq(
                        hessian_approx,
                        gradiant,
                        rcond=None,
                    )[0]
                )

            thetas_unlimited = (
                q
                + delta_q
            )

            q = np.clip(
                thetas_unlimited,
                self._q_min,
                self._q_max,
            )

            self.update_fk_and_jcbn(
                q
            )

        self._q = q.copy()

        return self._q.copy()

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def set_q(
        self,
        q: np.ndarray,
        *,
        clip_to_limits: bool = True,
    ) -> None:
        """Set the current IK seed without commanding the robot."""

        values = self._validate_joint_vector(
            q
        )

        if clip_to_limits:
            values = np.clip(
                values,
                self._q_min,
                self._q_max,
            )

        self._q = values.copy()
        self.update_fk_and_jcbn(
            self._q
        )

    def get_q(self) -> np.ndarray:
        """Return the current IK solution in radians."""

        return self._q.copy()

    def get_fk(self) -> np.ndarray:
        """Return the latest TCP forward-kinematics transform."""

        return self._fk.copy()

    def get_jacobian(self) -> np.ndarray:
        """Return the latest geometric Jacobian."""

        return self._jcbn.copy()
