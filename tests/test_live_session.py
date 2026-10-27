"""The live GUI's engine: tare, several bundles on one stream, toggling while streaming, logs in live_shadow format."""
import time

import numpy as np
import pandas as pd
import pytest

from grf2com.config import ROOT
from grf2com.live import SyntheticSource
from grf2com.live_session import LOG_COLS, LiveEngine
from grf2com.realtime import RAW

BUNDLE = ROOT / "deploy" / "models" / "tcn_v1"
SUBJECT = {"heights": 172.0, "mass": 76.5, "legs": 86.0, "feet": 25.0}


def wait_for(cond, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_synthetic_source_tares_to_its_offsets():
    src = SyntheticSource(pace=0)
    it = iter(src)
    first = [next(it) for _ in range(int(src.fs * 2))]
    t = np.array([s[0] for s in first])
    assert np.allclose(np.diff(t), 1 / src.fs)
    for c in RAW:
        assert all(s[1][c] == SyntheticSource.OFFSETS.get(c, 0.0) for s in first)
    walking = src.sample(src.unloaded_s + 0.2)
    assert walking["Fz_l"] + walking["Fz_r"] > 0.5 * src.bw


@pytest.mark.skipif(not BUNDLE.exists(), reason="no tcn_v1 bundle")
def test_engine_two_pipelines_and_logs(tmp_path):
    # the same bundle twice (a copy under another name) stands in for two models
    other = tmp_path / "models" / "tcn_copy"
    other.mkdir(parents=True)
    for f in BUNDLE.iterdir():
        (other / f.name).write_bytes(f.read_bytes())
    eng = LiveEngine(SyntheticSource(mass=SUBJECT["mass"], pace=0), SUBJECT, tare_s=2.0)
    eng.enable(BUNDLE)                                      # queued until the tare is done
    eng.start()
    try:
        assert wait_for(lambda: eng.state == "running" and BUNDLE.name in eng.pipes)
        assert eng.offsets["Fy_l"] == pytest.approx(SyntheticSource.OFFSETS["Fy_l"])
        assert eng.fs == pytest.approx(1000.0)
        assert eng.message == ""                            # unloaded during the tare: no warning
        eng.start_recording(tmp_path / "logs", "stamp")
        eng.enable(other)                                   # while recording: gets its own log at once
        assert wait_for(lambda: eng.snapshot()["models"]["tcn_copy"]["frames"] > 300)
        snap = eng.snapshot()
        a, b = snap["models"][BUNDLE.name], snap["models"]["tcn_copy"]
        assert a["warm"] == 1.0 and np.isfinite(a["com"]).all() and a["hist"].shape[1] == 3
        logs = eng.stop_recording()
        info = eng.disable("tcn_copy")
        assert info is None                                 # its log was already closed by stop_recording
    finally:
        eng.stop()
    assert eng.state == "stopped" and not eng.recording
    assert {i["model"] for i in logs} == {BUNDLE.name, "tcn_copy"}
    for i in logs:
        df = pd.read_csv(i["log"])
        assert list(df.columns) == LOG_COLS
        assert len(df) == i["frames"] > 100
        assert df.t.iloc[0] == pytest.approx(i["t_first"], abs=1e-3)
        assert np.isfinite(df[["com_ap", "com_ml", "com_vert"]].to_numpy()).all()
    # both copies see the same stream; the later one's frame grid starts at its own first sample, so compare the last
    # second at the nearest frame time (within half a frame): they should agree to a few mm
    la, lb = (pd.read_csv(i["log"]) for i in sorted(logs, key=lambda i: i["model"]))
    tail = lb.tail(60)
    idx = np.searchsorted(la.t.to_numpy(), tail.t.to_numpy()).clip(0, len(la) - 1)
    assert np.abs(la.t.to_numpy()[idx] - tail.t.to_numpy()).max() < 1 / 60
    assert np.abs(la.com_ap.to_numpy()[idx] - tail.com_ap.to_numpy()).max() < 0.01


@pytest.mark.skipif(not BUNDLE.exists(), reason="no tcn_v1 bundle")
def test_engine_warns_when_loaded_during_tare():
    eng = LiveEngine(SyntheticSource(mass=SUBJECT["mass"], unloaded_s=0.0, pace=0), SUBJECT, tare_s=2.0)
    eng.start()
    try:
        assert wait_for(lambda: eng.state == "running")
        assert "unloaded" in eng.message
    finally:
        eng.stop()


def test_lab_kf_pipeline_logs_steps_and_commands(tmp_path):
    from grf2com.config import load_config
    from grf2com.live_session import LabKfPipeline, LabKfSpec
    from grf2com.treadmill import plate_frames

    cfg = load_config()
    plates = plate_frames(cfg["gui"]["plates"], cfg["gui"]["grf"]["plate_axes"])
    eng = LiveEngine(SyntheticSource(mass=SUBJECT["mass"], pace=0), SUBJECT, tare_s=2.0)
    eng.enable(LabKfSpec(plates))
    eng.start()
    try:
        assert wait_for(lambda: eng.state == "running" and "lab_kf" in eng.pipes)
        eng.start_recording(tmp_path, "s")
        kf = eng.pipes["lab_kf"].kf
        n0 = kf.n_step                                      # steps may already exist: wait for 10 while recording
        assert wait_for(lambda: kf.n_step >= n0 + 10)
        logs = eng.stop_recording()
        m = eng.snapshot()["models"]["lab_kf"]
    finally:
        eng.stop()
    assert m["warm"] == 1.0 and m["note"] == "" and np.isfinite(m["com"]).all()
    assert 0.1 < m["used"].mean() < 0.5                     # opaque only around accepted heel strikes
    assert m["command"] is not None and 0.0 <= m["command"][0] <= 2.6
    df = pd.read_csv(logs[0]["log"])
    assert list(df.columns) == LOG_COLS + LabKfPipeline.extra_cols
    steps = df[df.kf_used == 1]
    assert len(steps) >= 5 and steps.kf_v_target.notna().all() and df.kf_v_target[df.kf_used == 0].isna().all()
    # the step dots sit at com_step in the mocap lab frame
    assert np.allclose(np.diff(np.array([p[0] for _, p in m["steps"]])[-4:]), 0, atol=0.05)


def test_bertec_source_spaces_queued_samples():
    """Samples that were already queued (processing fell behind) are spaced 1 / fs apart, not by their arrival, so a
    backlog can't make every sample a frame; the first fresh sample snaps back to its arrival time."""
    zmq = pytest.importorskip("zmq")
    from grf2com.config import load_config
    from grf2com.live import BertecSource

    bcfg = load_config()["realtime"]["bertec"]
    pub = zmq.Context.instance().socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")           # a fake publisher, never the Bertec software
    assert port not in (5555, 5556)
    src = BertecSource(bcfg, "127.0.0.1", port, timeout_ms=2000)
    src.fs = 1000.0
    msg = {k: 0.0 for k, _ in bcfg["channels"].values()} | {k: 0.0 for k in bcfg["speed_keys"]}
    try:
        assert wait_for(lambda: (pub.send_json(msg), src.sock.poll(10))[1], timeout=10)   # subscription is up
        it = iter(src)
        while src.sock.poll(50):                                  # drain the handshake messages
            next(it)
        for _ in range(200):                                      # a 200-sample backlog, all queued at once
            pub.send_json(msg)
        time.sleep(0.3)
        t = np.array([next(it)[0] for _ in range(200)])
    finally:
        src.close()
        pub.close(0)
    assert np.allclose(np.diff(t[:-1]), 1e-3)                   # queued: spaced by the sample period
    assert src.backlog == 0.0                                   # the last one was fresh: on arrival again
