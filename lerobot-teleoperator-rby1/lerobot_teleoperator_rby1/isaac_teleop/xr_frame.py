"""Raw Isaac Capture outputs -> a plain :class:`XRFrame`, and the device pipeline.

The frame data types and the parsing of the pipeline outputs live in the
framework-independent package ``rby1_xr_teleop`` (``xr_frame``, ``isaac_io``)
and are re-exported here. This module adds what needs the real
``isaaccapture`` package: building the device pipeline
(:func:`build_pipeline`) and checking that the tensor layouts mirrored in
``rby1_xr_teleop.isaac_io`` still match the installed package
(:func:`verify_index_layout`).
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Any

import numpy as np

from rby1_xr_teleop.isaac_io import (  # noqa: F401  (re-exported)
    BASE_T_ANCHOR_INPUT,
    OUT_BODY,
    OUT_CONTROLLER_LEFT,
    OUT_CONTROLLER_RIGHT,
    OUT_HEAD,
    ControllerInputIndex,
    FullBodyInputIndex,
    HeadInputIndex,
    frame_from_outputs,
)
from rby1_xr_teleop.xr_frame import (  # noqa: F401  (re-exported)
    NUM_BODY_JOINTS,
    BodyJointIndex,
    BodyState,
    ButtonEdge,
    ControllerState,
    HeadState,
    XRFrame,
    pose_to_se3,
    rotate_frame_about_z,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline construction (isaaccapture imported lazily)
# ---------------------------------------------------------------------------


# Candidate names of each layout enum across isaaccapture releases.
_REMOTE_ENUM_NAMES: dict[type[IntEnum], tuple[str, ...]] = {
    ControllerInputIndex: ("ControllerInputIndex",),
    HeadInputIndex: ("HeadInputIndex", "HeadPoseIndex"),
    FullBodyInputIndex: ("FullBodyInputIndex", "BodyInputIndex"),
    BodyJointIndex: ("BodyJointIndex",),
}


def verify_index_layout() -> dict[str, str]:
    """Check the local index enums against the installed ``isaaccapture``.

    Members present on both sides must agree (a mismatch raises, since it
    would silently corrupt every parsed pose). Enums or members missing from
    the installed package only log a warning — e.g. 1.4.x has no
    ``HeadInputIndex``/``IS_TRACKED`` — and the parsers tolerate them.
    Returns ``{local_enum_name: "verified" | "missing:<names>" | "absent"}``.
    """
    from isaaccapture.retargeting_engine.tensor_types import indices as idx

    report: dict[str, str] = {}
    for local, names in _REMOTE_ENUM_NAMES.items():
        remote = next((getattr(idx, n) for n in names if hasattr(idx, n)), None)
        if remote is None:
            logger.warning(
                "isaaccapture has none of %s; %s layout could not be verified.",
                names,
                local.__name__,
            )
            report[local.__name__] = "absent"
            continue
        missing: list[str] = []
        for member in local:
            if not hasattr(remote, member.name):
                missing.append(member.name)
                continue
            remote_value = int(getattr(remote, member.name))
            if remote_value != int(member):
                raise RuntimeError(
                    f"isaaccapture layout mismatch: {local.__name__}.{member.name} is "
                    f"{remote_value} in the installed package, {int(member)} here."
                )
        if missing:
            logger.warning(
                "isaaccapture %s lacks %s (older layout); those fields are treated as absent.",
                remote.__name__,
                missing,
            )
            report[local.__name__] = "missing:" + ",".join(missing)
        else:
            report[local.__name__] = "verified"
    return report


def build_pipeline(*, with_body: bool) -> tuple[Any, bool]:
    """Build the device pipeline.

    Returns ``(pipeline, body_rebased_in_graph)``. Controllers and head are
    rebased in-graph by the ``base_T_anchor`` external input; the body source
    is rebased in-graph when the installed ``isaaccapture`` offers
    ``FullBodySource.transformed`` and in Python otherwise.
    """
    from isaaccapture.retargeting_engine.deviceio_source_nodes import (
        ControllersSource,
        HeadSource,
    )
    from isaaccapture.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaaccapture.retargeting_engine.tensor_types import TransformMatrix

    verify_index_layout()

    xform = ValueInput(BASE_T_ANCHOR_INPUT, TransformMatrix())
    xform_out = xform.output("value")

    controllers = ControllersSource(name="controllers").transformed(xform_out)
    head = HeadSource(name="head").transformed(xform_out)
    outputs = {
        OUT_CONTROLLER_LEFT: controllers.output(OUT_CONTROLLER_LEFT),
        OUT_CONTROLLER_RIGHT: controllers.output(OUT_CONTROLLER_RIGHT),
        OUT_HEAD: head.output(OUT_HEAD),
    }

    body_in_graph = False
    if with_body:
        from isaaccapture.retargeting_engine.deviceio_source_nodes import FullBodySource

        body_source = FullBodySource(name="full_body")
        if hasattr(body_source, "transformed"):
            outputs[OUT_BODY] = body_source.transformed(xform_out).output(OUT_BODY)
            body_in_graph = True
        else:
            outputs[OUT_BODY] = body_source.output(OUT_BODY)

    return OutputCombiner(outputs), body_in_graph


def build_external_inputs(base_T_anchor: Any) -> dict[str, Any]:  # noqa: N803
    """Materialise the constant ``base_T_anchor`` external input (once, in connect)."""
    from isaaccapture.retargeting_engine.interface import TensorGroup
    from isaaccapture.retargeting_engine.tensor_types import TransformMatrix

    tg = TensorGroup(TransformMatrix())
    tg[0] = np.asarray(base_T_anchor, dtype=np.float32)
    return {BASE_T_ANCHOR_INPUT: {"value": tg}}
