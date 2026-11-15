"""Meta Quest body + hand tracking stream for the whole-body teleoperator.

The headset app (Meta_with_hands, Unity APK) runs a TCP server that sends one
NDJSON frame per line: inside-out body tracking (IOBT) joints, both hands'
landmarks and palm, and the HMD pose — no controllers. :class:`StreamReceiver`
connects to it, low-pass filters the inputs with time constants based on the
packet timestamps, and publishes an immutable :class:`Snapshot`.

Ported from the semi-humanoid teleoperation ``rby1_ces_elbow.py`` (CES branch).
"""

from __future__ import annotations

import json
import logging
import math
import socket
import threading
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

SIDES = ("left", "right")

# IOBT / hand landmark indices (Meta_with_hands SCHEMA).
HIPS = 1
ARM_JOINTS = {"left": (10, 11, 12), "right": (15, 16, 17)}  # shoulder (ARM_UPPER), elbow, wrist
BODY_NEEDED = (HIPS, 10, 11, 12, 15, 16, 17)
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17
HAND_NEEDED = (WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP)
BODY_MIN_CONFIDENCE = 0.3

# An input this long without an update resets its filter to the new value.
FILTER_GAP_RESET = 0.3
# Filter reset thresholds — more than a human moves between two packets
# (~14 ms). Larger steps pass through unfiltered so the retargeter's jump
# gates see them in one tick.
INPUT_JUMP_M = 0.20
INPUT_JUMP_ANGLE = math.radians(30)


def lpf_alpha(dt: float, tau: float) -> float:
    if tau <= 0.0 or dt <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / tau)


@dataclass
class FilterTaus:
    """Input low-pass time constants (s)."""

    hand: float = 0.04
    body: float = 0.04
    heading: float = 0.12
    hmd: float = 0.05


@dataclass(frozen=True)
class BodyObs:
    ok: bool
    pos: np.ndarray        # (18, 3) — unused joints are NaN
    hips_R: np.ndarray     # (3, 3)
    stamp: float           # monotonic time it was last ok
    fidelity: str
    confidence: float


@dataclass(frozen=True)
class HandObs:
    ok: bool
    pos: np.ndarray        # (21, 3)
    palm: np.ndarray       # (3,)
    stamp: float


@dataclass(frozen=True)
class HmdObs:
    ok: bool
    R: np.ndarray
    pos: np.ndarray
    stamp: float


@dataclass(frozen=True)
class Snapshot:
    rx: float
    stream_t: float
    body: BodyObs
    hands: dict[str, HandObs]
    hmd: HmdObs
    packets: int


class StreamReceiver:
    """Background TCP client turning headset NDJSON into filtered :class:`Snapshot` s.

    The consumer reads ``self.snapshot`` once per tick (a rebinding, so no
    lock is needed). Every line of a chunk is processed so the filters see
    the true packet times.
    """

    def __init__(self, host: str, port: int, taus: FilterTaus, allow_low_fidelity: bool = False):
        self.host, self.port = host, port
        self.taus = taus
        self.allow_low = allow_low_fidelity
        self.snapshot: Snapshot | None = None
        self.connected = False
        self._packets = 0
        self._f: dict[str, tuple] = {}
        nan3 = np.full(3, np.nan)
        self._body = BodyObs(False, np.full((18, 3), np.nan), np.eye(3), -1e9, "none", 0.0)
        self._hands = {s: HandObs(False, np.full((21, 3), np.nan), nan3, -1e9) for s in SIDES}
        self._hmd = HmdObs(False, np.eye(3), nan3, -1e9)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rby1-wb-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- filters --------------------------------------------------------------
    def _filt(self, key: str, value, t: float, tau: float) -> np.ndarray:
        value = np.asarray(value, float)
        prev = self._f.get(key)
        # Gap, time going backwards or a jump no human makes in one packet
        # (glitch / recenter) → restart the filter at the new value.
        if prev is None or t - prev[1] > FILTER_GAP_RESET or t < prev[1] \
                or float(np.linalg.norm(value - prev[0])) > INPUT_JUMP_M:
            out = value.copy()
        else:
            out = prev[0] + lpf_alpha(t - prev[1], tau) * (value - prev[0])
        self._f[key] = (out, t)
        return out

    def _filt_quat(self, key: str, q, t: float, tau: float) -> np.ndarray:
        q = np.asarray(q, float)
        q = q / max(np.linalg.norm(q), 1e-12)
        prev = self._f.get(key)
        if prev is not None and np.dot(prev[0], q) < 0:
            q = -q
        if prev is None or t - prev[1] > FILTER_GAP_RESET or t < prev[1] \
                or 2.0 * math.acos(min(1.0, abs(float(np.dot(prev[0], q))))) > INPUT_JUMP_ANGLE:
            out = q.copy()
        else:
            p = prev[0]
            out = p + lpf_alpha(t - prev[1], tau) * (q - p)
            out /= max(np.linalg.norm(out), 1e-12)
        self._f[key] = (out, t)
        return out

    # -- parsing --------------------------------------------------------------
    def apply(self, obj: dict, now: float) -> None:
        """Fold one NDJSON frame into the state (public for tests)."""
        t = float(obj.get("timestamp", now))
        tn = self.taus

        body = obj.get("body") or {}
        body_ok = False
        if body.get("tracked"):
            fid = body.get("fidelity", "unspecified")
            conf = float(body.get("confidence", 0.0))
            joints = {j.get("i"): j for j in body.get("joints", []) if j.get("valid")}
            trusted = (fid != "low" or self.allow_low) and conf >= BODY_MIN_CONFIDENCE
            if trusted and all(i in joints and "position" in joints[i] for i in BODY_NEEDED) \
                    and "rotation" in joints[HIPS]:
                pos = np.full((18, 3), np.nan)
                for i in BODY_NEEDED:
                    pos[i] = self._filt(f"b{i}", joints[i]["position"], t, tn.body)
                q = self._filt_quat("bq1", joints[HIPS]["rotation"], t, tn.heading)
                self._body = BodyObs(True, pos, Rotation.from_quat(q).as_matrix(), now, fid, conf)
                body_ok = True
        if not body_ok:
            b = self._body
            self._body = BodyObs(False, b.pos, b.hips_R, b.stamp, b.fidelity, b.confidence)

        for h in obj.get("hands") or []:
            side = h.get("side")
            if side not in SIDES:
                continue
            ok = False
            if h.get("tracked") and (h.get("palm") or {}).get("valid") and "position" in h["palm"]:
                lms = {lm.get("i"): lm for lm in h.get("landmarks", []) if lm.get("valid") and "position" in lm}
                if all(i in lms for i in HAND_NEEDED):
                    pos = np.full((21, 3), np.nan)
                    for i in HAND_NEEDED:
                        pos[i] = self._filt(f"h{side}{i}", lms[i]["position"], t, tn.hand)
                    palm = self._filt(f"h{side}palm", h["palm"]["position"], t, tn.hand)
                    self._hands[side] = HandObs(True, pos, palm, now)
                    ok = True
            if not ok:
                prev = self._hands[side]
                self._hands[side] = HandObs(False, prev.pos, prev.palm, prev.stamp)

        hmd = obj.get("hmd") or {}
        if hmd.get("tracked") and "position" in hmd and "rotation" in hmd:
            p = self._filt("hmdp", hmd["position"], t, tn.hmd)
            q = self._filt_quat("hmdq", hmd["rotation"], t, tn.hmd)
            self._hmd = HmdObs(True, Rotation.from_quat(q).as_matrix(), p, now)
        else:
            self._hmd = HmdObs(False, self._hmd.R, self._hmd.pos, self._hmd.stamp)

        self._packets += 1
        self.snapshot = Snapshot(now, t, self._body, dict(self._hands), self._hmd, self._packets)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
                    sock.settimeout(0.5)
                    self.connected = True
                    logger.info(f"Headset stream connected {self.host}:{self.port}")
                    buf = b""
                    while not self._stop.is_set():
                        try:
                            chunk = sock.recv(65536)
                        except socket.timeout:
                            continue
                        if not chunk:
                            raise ConnectionError("stream closed")
                        buf += chunk
                        *lines, buf = buf.split(b"\n")
                        now = time.monotonic()
                        for line in lines:
                            if not line.strip():
                                continue
                            try:
                                self.apply(json.loads(line), now)
                            except (ValueError, KeyError, TypeError) as exc:
                                logger.debug(f"bad frame: {exc}")
            except OSError as exc:
                if self.connected:
                    logger.warning(f"Headset stream lost ({exc}); reconnecting")
                self.connected = False
                self._stop.wait(1.0)
        self.connected = False
