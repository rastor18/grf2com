"""Treadmill speed commands and the treadmill display frame, for the live GUI.

BertecTreadmill sends belt speed commands over Bertec's remote-control RPC (JSON over ZMQ REQ/REP, port 5555), as
Bertec's BertecRemoteControl.py and the lab's Treadmill_QTM do: InitConnect once, then RunTreadmill with the same
speed, acceleration and deceleration for both belts. The Bertec software must be in remote-control mode. This is
manual speed control only: the CoM estimates never drive the treadmill.

treadmill_frame() places the display origin from the force-plate corners in the lab frame (QTM's force-plate
settings): AP at the posterior-most corner, ML midway between the belts' outer edges, vertical at the belt surface.
plate_frames() / grf_arrow() put each plate's COP and ground reaction force into the lab frame for the GRF arrows.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np


class BertecTreadmill:
    """Calls run on one worker thread (REQ/REP must alternate) and return Futures; speeds are clipped to
    [0, max_speed] and accelerations to [0.01, max_accel]."""

    def __init__(self, host: str, rpc_port: int, max_speed: float, max_accel: float = 1.0, timeout_ms: int = 3000):
        self.address = f"tcp://{host}:{rpc_port}"
        self.max_speed, self.max_accel, self.timeout_ms = float(max_speed), float(max_accel), timeout_ms
        self.sock = None
        self.id = 1
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="treadmill")

    def run(self, speed: float, accel: float, decel: float | None = None) -> Future:
        """Both belts to `speed` (m/s) at `accel` (m/s^2; `decel` when slowing, default = accel). Resolves to the
        speed actually commanded (after clipping)."""
        speed = float(np.clip(speed, 0.0, self.max_speed))
        accel = float(np.clip(accel, 0.01, self.max_accel))
        decel = float(np.clip(accel if decel is None else decel, 0.01, self.max_accel))
        params = {"leftVel": speed, "leftAccel": accel, "leftDecel": decel,
                  "rightVel": speed, "rightAccel": accel, "rightDecel": decel}
        return self.pool.submit(self._command, "RunTreadmill", params, speed)

    def stop(self, decel: float) -> Future:
        return self.run(0.0, decel, decel)

    def close(self) -> None:
        self.pool.submit(self._reset)
        self.pool.shutdown(wait=True)

    def _command(self, method: str, params: dict, result):
        if self.sock is None:
            self._connect()
        self._call(method, params)
        return result

    def _connect(self) -> None:
        import zmq

        self.sock = zmq.Context.instance().socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.address)
        try:
            self._call("InitConnect", {"ip": "127.0.0.1", "port": "5560"})
        except Exception:
            self._reset()
            raise

    def _call(self, method: str, params: dict) -> dict:
        import zmq

        msg = {"version": "v1", "id": self.id, "method": method, "params": params}
        self.id += 1
        try:
            self.sock.send_json(msg)
            res = self.sock.recv_json()
        except zmq.Again:
            self._reset()        # a REQ socket without a reply can't send again
            raise TimeoutError(f"no reply from the Bertec software on {self.address} to {method}; is it open and "
                               "in remote-control mode?") from None
        if not isinstance(res, dict) or res.get("code") != 1:
            raise RuntimeError(f"Bertec {method} was refused: {res}")
        return res

    def _reset(self) -> None:
        if self.sock is not None:
            self.sock.close(0)
            self.sock = None


def treadmill_frame(plates: list[dict]) -> dict:
    """Display frame from the force plates ([{name, corners (4, 3) in m, lab frame}]): `origin` (lab m) and `sign`
    (+1 if participants face lab +x). Facing is taken from the plate names (the left belt is on the participant's
    left, +y when facing +x); without names it assumes +x. Display coordinates: AP = sign (x - x0), forward positive;
    ML = sign (y - y0), left positive; vertical = z - z0, height above the belts."""
    corners = np.vstack([np.asarray(p["corners"], float) for p in plates])
    sign = 1.0
    named = {("left" if "left" in p["name"].lower() else "right" if "right" in p["name"].lower() else None):
             np.asarray(p["corners"], float)[:, 1].mean() for p in plates}
    if "left" in named and "right" in named and named["left"] < named["right"]:
        sign = -1.0
    x0 = corners[:, 0].min() if sign > 0 else corners[:, 0].max()
    origin = np.array([x0, (corners[:, 1].min() + corners[:, 1].max()) / 2, corners[:, 2].mean()])
    return {"origin": origin, "sign": sign}


def to_display(lab: np.ndarray, frame: dict) -> np.ndarray:
    """Lab-frame points (..., 3) in m to the treadmill display frame."""
    out = np.asarray(lab, float) - frame["origin"]
    out[..., :2] *= frame["sign"]
    return out


def plate_frames(plates: list[dict], axes: dict) -> dict[str, dict]:
    """Each plate's measurement frame in the lab, by side ("l" / "r" from the plate names): `origin` (lab m) and
    `R` (3, 3), whose columns are the lab directions of the plate's x, y, z axes (`axes`, gui.grf.plate_axes). The
    plate's `origin` setting (QTM: plate centre minus the measurement origin, in plate axes, m) places the
    measurement origin at centre - R @ origin."""
    R = np.column_stack([np.asarray(axes[a], float) for a in "xyz"])
    out = {}
    for p in plates:
        name = p["name"].lower()
        side = "l" if "left" in name else "r" if "right" in name else None
        if side is None:
            continue
        centre = np.asarray(p["corners"], float).mean(0)
        out[side] = {"origin": centre - R @ np.asarray(p.get("origin", (0.0, 0.0, 0.0)), float), "R": R}
    return out


def grf_arrow(f: np.ndarray, m: np.ndarray, frame: dict) -> tuple[np.ndarray, np.ndarray]:
    """COP (lab m) and the ground reaction force on the body (lab N) from one plate's tared force (N) and moment
    (N*m) in its own frame, the force the foot applies to the plate (as in the data: Fz > 0 when loaded).
    COP = (-My / Fz, Mx / Fz, 0), as in preprocessing."""
    cop_local = np.array([-m[1] / f[2], m[0] / f[2], 0.0])
    return frame["origin"] + frame["R"] @ cop_local, -(frame["R"] @ f)
