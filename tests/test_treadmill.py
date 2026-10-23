"""Treadmill display frame and the Bertec speed client (against a fake RPC server; the real treadmill is never used)."""
import threading

import numpy as np
import pytest

from grf2com.config import load_config
from grf2com.treadmill import BertecTreadmill, grf_arrow, plate_frames, to_display, treadmill_frame


def test_frame_from_lab_plates():
    plates = load_config()["gui"]["plates"]          # QTM's corners: lab x along the belts, left belt at +y
    f = treadmill_frame(plates)
    assert f["sign"] == 1.0
    d = np.vstack([to_display(np.asarray(p["corners"]), f) for p in plates])
    assert d[:, 0].min() == pytest.approx(0.0)                       # AP origin: posterior-most corner
    assert d[:, 1].min() == pytest.approx(-d[:, 1].max())            # ML origin: midway between the outer edges
    assert abs(d[:, 2].mean()) < 1e-9                                # vertical: belt surface
    left = to_display(np.asarray(next(p for p in plates if "Left" in p["name"])["corners"]), f)
    assert left[:, 1].mean() > 0                                     # left is +ML


def test_frame_when_facing_minus_x():
    # the same treadmill rotated 180 degrees about vertical: participants face -x, left belt at -y
    plates = [{"name": p["name"], "corners": (np.asarray(p["corners"]) * [-1, -1, 1]).tolist()}
              for p in load_config()["gui"]["plates"]]
    f = treadmill_frame(plates)
    assert f["sign"] == -1.0
    d = np.vstack([to_display(np.asarray(p["corners"]), f) for p in plates])
    assert d[:, 0].min() == pytest.approx(0.0) and d[:, 0].max() > 1.7
    left = to_display(np.asarray(plates[1]["corners"]), f)
    assert left[:, 1].mean() > 0


class FakeBertec:
    """REP server answering like the Bertec software (code 1), recording the requests."""

    def __init__(self, refuse=()):
        import zmq

        self.sock = zmq.Context.instance().socket(zmq.REP)
        self.port = self.sock.bind_to_random_port("tcp://127.0.0.1")
        self.requests, self.refuse = [], set(refuse)
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        import zmq

        self.sock.setsockopt(zmq.RCVTIMEO, 3000)
        while True:
            try:
                msg = self.sock.recv_json()
            except zmq.Again:
                return
            self.requests.append(msg)
            self.sock.send_json({"code": -1 if msg["method"] in self.refuse else 1, "id": msg["id"]})


def test_speed_commands_reach_both_belts_clipped():
    fake = FakeBertec()
    tm = BertecTreadmill("127.0.0.1", fake.port, max_speed=2.0, max_accel=1.0)
    assert tm.run(1.25, 0.1).result(timeout=5) == 1.25
    assert tm.run(9.0, 5.0).result(timeout=5) == 2.0                 # clipped
    assert tm.stop(0.3).result(timeout=5) == 0.0
    tm.close()
    methods = [r["method"] for r in fake.requests]
    assert methods == ["InitConnect", "RunTreadmill", "RunTreadmill", "RunTreadmill"]
    first, clipped, stop = (r["params"] for r in fake.requests[1:])
    assert first == {"leftVel": 1.25, "leftAccel": 0.1, "leftDecel": 0.1,
                     "rightVel": 1.25, "rightAccel": 0.1, "rightDecel": 0.1}
    assert clipped["leftVel"] == clipped["rightVel"] == 2.0 and clipped["leftAccel"] == 1.0
    assert stop["leftVel"] == stop["rightVel"] == 0.0 and stop["leftDecel"] == 0.3


def test_refused_or_silent_bertec_raises():
    fake = FakeBertec(refuse={"RunTreadmill"})
    tm = BertecTreadmill("127.0.0.1", fake.port, max_speed=2.0)
    with pytest.raises(RuntimeError, match="refused"):
        tm.run(1.0, 0.1).result(timeout=5)
    tm.close()
    import zmq
    unused = zmq.Context.instance().socket(zmq.REP)
    port = unused.bind_to_random_port("tcp://127.0.0.1")
    unused.close(0)                                                  # nobody listens there now
    tm = BertecTreadmill("127.0.0.1", port, max_speed=2.0, timeout_ms=300)
    with pytest.raises(TimeoutError):
        tm.run(1.0, 0.1).result(timeout=5)
    tm.close()


def test_grf_arrow_lands_on_the_right_belt_and_points_up():
    cfg = load_config()
    plates = cfg["gui"]["plates"]
    frames, f = plate_frames(plates, cfg["gui"]["grf"]["plate_axes"]), treadmill_frame(plates)
    # measurement origins at the belts' rear outer corners
    assert to_display(frames["l"]["origin"], f)[:2] == pytest.approx([0.0, 0.565], abs=0.01)
    assert to_display(frames["r"]["origin"], f)[:2] == pytest.approx([0.0, -0.565], abs=0.01)
    for side, copx in (("l", -0.465), ("r", 0.465)):      # training-data COPx: feet near the midline
        fz = 750.0
        force = np.array([0.05 * fz * (1 if side == "l" else -1), 0.15 * fz, fz])     # early stance, braking
        moment = np.array([0.9 * fz, -copx * fz, 0.0])                                # COPy 0.9 m, COPx copx
        cop, reaction = grf_arrow(force, moment, frames[side])
        d = to_display(cop, f)
        assert d[0] == pytest.approx(0.9, abs=0.01)                                   # AP from the rear edge
        assert (d[1] > 0) == (side == "l") and abs(d[1]) < 0.15                       # own belt, near the midline
        assert reaction[2] == fz and reaction[0] < 0                                  # up, and braking = backwards
        assert np.sign(reaction[1]) == (-1 if side == "l" else 1)                     # toward the midline
