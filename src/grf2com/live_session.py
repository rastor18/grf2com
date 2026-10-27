"""Live session engine behind the GUI (scripts/live_gui.py): one sample stream feeding several model bundles at once.

It runs in a worker thread: tare (median of each raw channel while the plates are unloaded) and the measured input
rate first, then every sample goes to each enabled bundle's own StreamingPreprocessor + StreamingEstimator (bundles
may differ in preprocessing). While recording, each bundle writes its own log in the live_shadow.py --log-raw format
(t, frame, com_ap/ml/vert, compute_ms, raw channels, speed), so check_live_mapping.py and score_live_log.py work on
it unchanged. The GUI only calls the public methods and reads snapshot(). Shadow mode: nothing is ever sent to the
treadmill.

Besides bundles, LabKfSpec adds the lab's self-paced-treadmill Kalman filter (baselines/song_kf.LabKalmanStream) as a
"model": what the lab controller would have estimated, and commanded, on this stream (LabKfPipeline).
"""
from __future__ import annotations

import csv
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .baselines.song_kf import LOOP_DT, LabKalmanStream
from .bundle import load_bundle
from .realtime import RAW, StreamingEstimator, StreamingPreprocessor
from .selfpace import LAB_CONTROLLER, TREADMILL_LENGTH, control_law
from .treadmill import grf_arrow

LOG_COLS = ["t", "frame", "com_ap", "com_ml", "com_vert", "compute_ms"] + RAW + ["speed"]
TARE_LOADED_N = 100.0      # Fz range during the tare above this: the plates were probably not unloaded
TORCH_THREADS = 1          # the models are tiny: more intra-op threads only spin, starve the GUI and, with two or
                           # more bundles, made the engine fall behind a 1000 Hz stream (lag growing to seconds)


def list_bundles(models_dir: Path) -> list[Path]:
    models_dir = Path(models_dir)
    return sorted(p for p in models_dir.iterdir() if (p / "bundle.json").exists()) if models_dir.is_dir() else []


class Pipeline:
    """One bundle on the stream: preprocessing, estimator, recent history and (while recording) its log."""

    extra_cols: list[str] = []

    def __init__(self, path: Path, subject: dict, offsets: dict, fs: float, upstream: bool, history_s: float):
        self.path = Path(path)
        self.name = self.path.name
        model, fb, cfg, self.meta = load_bundle(self.path)
        missing = [f for f in self.meta["required_subject_fields"] if f not in subject]
        if missing:
            raise ValueError(f"{self.name} needs subject fields {missing}")
        up = self.meta.get("upstream_filter") if upstream else None
        self.pre = StreamingPreprocessor(cfg["preprocess"], offsets, fs=fs, upstream_filter=up)
        self.est = StreamingEstimator(model, fb, cfg, subject)
        rate = int(cfg["preprocess"]["grf_rate"])
        self.warm_frames = int(round(float(cfg["eval"]["window_s"]) * rate))
        self._init_history(history_s, rate)

    def _init_history(self, history_s: float, rate: float) -> None:
        self.frames = 0
        self.com = np.full(3, np.nan)
        self.compute_ms = float("nan")
        self.hist_t: deque = deque(maxlen=int(history_s * rate))
        self.hist_com: deque = deque(maxlen=int(history_s * rate))
        self.log_fh = None
        self.log_writer = None
        self.log_path: Path | None = None
        self.log_t_first: float | None = None

    def warm(self) -> float:
        return min(1.0, self.frames / max(self.warm_frames, 1))

    def extra_snapshot(self) -> dict:
        return {}

    def open_log(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path, self.log_t_first, self.log_frame = path, None, 0
        self.log_fh = open(path, "w", newline="")
        self.log_writer = csv.writer(self.log_fh)
        self.log_writer.writerow(LOG_COLS + self.extra_cols)

    def close_log(self) -> dict | None:
        if self.log_fh is None:
            return None
        self.log_fh.close()
        info = {"model": self.name, "log": str(self.log_path), "frames": self.log_frame, "t_first": self.log_t_first}
        self.log_fh = self.log_writer = None
        return info

    def push(self, t: float, t_rel: float, raw: dict, speed: float) -> None:
        frame = self.pre.push(t, raw, speed)
        if frame is None:
            return
        c0 = time.perf_counter()
        com = self.est.push(frame)
        self.compute_ms = 1000 * (time.perf_counter() - c0)
        self._record(t, t_rel, com, raw, speed)

    def _record(self, t: float, t_rel: float, com: np.ndarray, raw: dict, speed: float, extra=()) -> None:
        self.com = com
        self.frames += 1
        self.hist_t.append(t_rel)
        self.hist_com.append(com)
        if self.log_writer is not None:
            if self.log_t_first is None:
                self.log_t_first = t
            self.log_writer.writerow([f"{t:.4f}", self.log_frame, *(f"{v:.5f}" for v in com),
                                      f"{self.compute_ms:.2f}", *(f"{raw[c]:.4f}" for c in RAW), f"{speed:.3f}",
                                      *extra])
            self.log_frame += 1


class LabKfSpec:
    """The lab's self-paced-treadmill Kalman filter as a model for LiveEngine.enable(). plates: plate_frames() of the
    force plates (lab frame), to place its estimate in the mocap lab frame."""

    name = "lab_kf"

    def __init__(self, plates: dict):
        self.plates = plates

    def build(self, subject: dict, offsets: dict, fs: float, upstream: bool, history_s: float) -> "LabKfPipeline":
        return LabKfPipeline(self.plates, subject, offsets, history_s)


class LabKfPipeline(Pipeline):
    """song_kf.LabKalmanStream on the tared stream as the lab runs it: a 15 ms loop on the latest sample, started when
    the belt starts, COP AP per plate = recomputed COPy - treadmill_length / 2 (the lab's frame, treadmill centre 0),
    mass from the participant fields. The lab reads the filter only on accepted step ticks: com_step / dcom_step then
    give the speed command (selfpace.control_law, lab defaults). Output per 60 Hz frame, in the mocap lab frame like
    the networks: AP = the filter's position, ML = midpoint of the last two foot placements, vertical = 0.55 x height
    (the filter estimates AP only). `used` marks the frames within USED_S after an accepted step."""

    USED_S = 0.15
    FRAME_RATE = 60
    extra_cols = ["kf_x", "kf_v", "kf_step", "kf_used", "kf_com_step", "kf_dcom_step", "kf_v_tm_mean", "kf_v_target",
                  "kf_a_target"]

    def __init__(self, plates: dict, subject: dict, offsets: dict, history_s: float):
        self.name = LabKfSpec.name
        self.meta = {"model": "lab KF (Song 2020)", "required_subject_fields": ["mass", "heights"]}
        self.plates, self.offsets = plates, offsets
        self.kf = LabKalmanStream(float(subject["mass"]))
        origin = np.mean([f["origin"] for f in plates.values()], axis=0)
        self.ap0 = TREADMILL_LENGTH / 2 + origin[0]         # lab-code AP (treadmill centre 0) -> mocap lab x
        self.ml = float(origin[1])                          # until two feet have landed: the belts' midline
        self.z = 0.55 * float(subject["heights"]) / 100 + origin[2]
        self.feet_ml: deque = deque(maxlen=2)
        self.t_start = self.next_tick = self.next_frame = None
        self.t0_rel = 0.0
        self.last = None                                    # latest sample: (t, tared dict, speed)
        self.last_used_t = -np.inf
        self.step_since_frame = None
        self.command = None                                 # (v_target, a_target) of the last accepted step
        self.warm_frames = 1
        self._init_history(history_s, self.FRAME_RATE)
        self.hist_used: deque = deque(maxlen=self.hist_t.maxlen)
        self.steps: deque = deque(maxlen=64)                # accepted steps: (t_rel, lab point of com_step)

    def warm(self) -> float:
        return 1.0 if self.t_start is not None and self.kf.n_step >= 2 else 0.0

    def extra_snapshot(self) -> dict:
        return {"used": np.fromiter(self.hist_used, bool, len(self.hist_used)),
                "steps": [(t, p.copy()) for t, p in self.steps], "command": self.command,
                "note": "waiting for the belt to start" if self.t_start is None else
                        ("waiting for steps" if self.kf.n_step < 2 else "")}

    def _lab_point(self, ap_kf: float) -> np.ndarray:
        return np.array([ap_kf + self.ap0, self.ml, self.z])

    def _tick(self, tick_t: float, tared: dict, speed: float) -> None:
        cop = {}
        for s in "lr":
            fz = tared[f"Fz_{s}"]
            cop[s] = tared[f"Mx_{s}"] / fz - TREADMILL_LENGTH / 2 if fz != 0 else 0.0
        st = self.kf.tick(tick_t - self.t_start, tared["Fy_l"] + tared["Fy_r"], tared["Fz_l"], tared["Fz_r"],
                          cop["l"], cop["r"], speed)
        if st is None:
            return
        leg = st["leg"]
        f = np.array([tared[f"F{a}_{leg}"] for a in "xyz"])
        m = np.array([tared[f"M{a}_{leg}"] for a in "xyz"])
        if leg in self.plates and f[2] > 0:
            self.feet_ml.append(grf_arrow(f, m, self.plates[leg])[0][1])
            if len(self.feet_ml) == 2:
                self.ml = float(np.mean(self.feet_ml))
        self.step_since_frame = st
        if st["used"]:
            self.last_used_t = tick_t
            v, a = control_law(st["com_step"], st["dcom_step"], st["v_tm_mean"], LAB_CONTROLLER)
            self.command = (float(v), float(a))
            self.steps.append((tick_t - self.t0_rel, self._lab_point(st["com_step"])))

    def push(self, t: float, t_rel: float, raw: dict, speed: float) -> None:
        c0 = time.perf_counter()
        tared = {c: raw[c] - self.offsets[c] for c in RAW}
        if self.t_start is None:
            if speed <= 0:
                return
            self.t_start, self.t0_rel = t, t - t_rel        # the lab initializes the filter when the belt starts
            self.next_tick, self.next_frame = t + LOOP_DT, t
        # timer ticks see the latest sample at or before the tick time
        while self.last is not None and t > self.next_tick:
            self._tick(self.next_tick, *self.last[1:])
            self.next_tick += LOOP_DT
        self.last = (t, tared, speed)
        if t + 1e-9 < self.next_frame:
            return
        self.next_frame += 1.0 / self.FRAME_RATE
        st, self.step_since_frame = self.step_since_frame, None
        used_step = bool(st and st["used"])
        kf = self.kf
        extra = [f"{kf.x0:.5f}", f"{kf.x1:.5f}", int(st is not None), int(used_step),
                 *(f"{st[k]:.5f}" if st else "" for k in ("com_step", "dcom_step", "v_tm_mean")),
                 *([f"{v:.4f}" for v in self.command] if used_step else ["", ""])]
        self.hist_used.append((self.next_tick - LOOP_DT) - self.last_used_t <= self.USED_S)
        self.compute_ms = 1000 * (time.perf_counter() - c0)
        self._record(t, t_rel, self._lab_point(kf.x0), raw, speed, extra)


class LiveEngine:
    def __init__(self, source, subject: dict, *, tare_s: float = 3.0, no_tare: bool = False, upstream: bool = True,
                 history_s: float = 10.0):
        self.source, self.subject = source, dict(subject)
        self.tare_s, self.no_tare, self.upstream, self.history_s = tare_s, no_tare, upstream, history_s
        self.state = "idle"            # idle -> taring -> running -> stopped | error
        self.message = ""
        self.offsets: dict | None = None
        self.fs: float | None = None
        self.t0: float | None = None
        self.last_t = 0.0
        self.samples = 0
        self.speed = 0.0
        self.fz = np.zeros(2)
        self.tared = np.zeros(len(RAW))       # latest raw sample minus the tare offsets (GRF arrows)
        self.tare_progress = 0.0
        self.pipes: dict[str, Pipeline] = {}
        self._wanted: list = []                 # bundles / specs enabled before the tare finished
        self._recording: dict | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ----------------------------------------------------------------- control (any thread)
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="live-engine", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 8.0) -> list[dict]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        return self.stop_recording()

    def enable(self, spec) -> None:
        """Add a bundle (its folder) or a spec with .name and .build() (LabKfSpec). Before the tare finishes it is
        queued; afterwards it is loaded here (outside the lock, so the stream doesn't stall) and starts from an empty
        window."""
        spec = spec if hasattr(spec, "build") else Path(spec)
        with self._lock:
            if spec.name in self.pipes or spec.name in {w.name for w in self._wanted}:
                return
            if self.offsets is None:
                self._wanted.append(spec)
                return
            offsets, fs = self.offsets, self.fs
        if hasattr(spec, "build"):
            pipe = spec.build(self.subject, offsets, fs, self.upstream, self.history_s)
        else:
            pipe = Pipeline(spec, self.subject, offsets, fs, self.upstream, self.history_s)
        with self._lock:
            if self._recording is not None:
                pipe.open_log(self._log_path(pipe.name))
            self.pipes[pipe.name] = pipe

    def disable(self, name: str) -> dict | None:
        with self._lock:
            self._wanted = [p for p in self._wanted if p.name != name]
            pipe = self.pipes.pop(name, None)
            if pipe is None:
                return None
            info = pipe.close_log()
            if info and self._recording is not None:
                self._recording["closed"].append(info)
            return info

    def start_recording(self, log_dir: Path, stamp: str) -> None:
        with self._lock:
            if self._recording is not None:
                raise RuntimeError("already recording")
            self._recording = {"dir": Path(log_dir), "stamp": stamp, "closed": []}
            for pipe in self.pipes.values():
                pipe.open_log(self._log_path(pipe.name))

    def stop_recording(self) -> list[dict]:
        """Close every log; returns one entry per log written during this recording."""
        with self._lock:
            if self._recording is None:
                return []
            logs = self._recording["closed"] + [i for p in self.pipes.values() if (i := p.close_log())]
            self._recording = None
            return logs

    @property
    def recording(self) -> bool:
        return self._recording is not None

    def snapshot(self) -> dict:
        with self._lock:
            models = {}
            for name, p in self.pipes.items():
                models[name] = {"com": p.com.copy(), "compute_ms": p.compute_ms, "frames": p.frames,
                                "warm": p.warm(), **p.extra_snapshot(),
                                "t": np.fromiter(p.hist_t, float, len(p.hist_t)),
                                "hist": np.array(p.hist_com).reshape(-1, 3)}
            return {"state": self.state, "message": self.message, "fs": self.fs, "samples": self.samples,
                    "t": (self.last_t - self.t0) if self.t0 is not None else 0.0, "backlog": getattr(self.source, "backlog", 0.0), "speed": self.speed,
                    "fz": self.fz.copy(), "tared": dict(zip(RAW, self.tared.tolist())), "tare_progress": self.tare_progress, "offsets": self.offsets,
                    "recording": self._recording is not None, "models": models}

    # ----------------------------------------------------------------- worker thread
    def _log_path(self, name: str) -> Path:
        return self._recording["dir"] / f"{self._recording['stamp']}_{name}.csv"

    def _run(self) -> None:
        torch.set_num_threads(TORCH_THREADS)
        try:
            it = iter(self.source)
            self.state = "taring"
            if not self._tare(it):
                return
            with self._lock:
                wanted, self._wanted = self._wanted, []
            for path in wanted:
                self.enable(path)
            self.state = "running"
            while not self._stop.is_set():
                try:
                    for t, raw, speed in it:
                        self._step(t, raw, speed)
                        if self._stop.is_set():
                            break
                    else:
                        self.message = "the source ended"
                        break
                except TimeoutError as e:    # Bertec went quiet: keep the pipelines, wait for data again
                    self.message = f"{e}; waiting"
                    it = iter(self.source)
        except Exception as e:  # noqa: BLE001  (shown in the GUI)
            self.state, self.message = "error", f"{type(e).__name__}: {e}"
        finally:
            if self.state != "error":
                self.state = "stopped"
            close = getattr(self.source, "close", None)
            if close:
                close()

    def _tare(self, it) -> bool:
        """Median offsets over tare_s (or just the input rate over 1 s with no_tare). False if stopped meanwhile."""
        seconds = 1.0 if self.no_tare else self.tare_s
        self.message = "measuring the input rate" if self.no_tare else "taring: keep both plates unloaded"
        buf, t0, last = [], None, None
        for t, raw, _ in it:
            if self._stop.is_set():
                return False
            t0 = t if t0 is None else t0
            buf.append([raw[c] for c in RAW])
            last = t
            self.tare_progress = min(1.0, (t - t0) / seconds)
            if t - t0 >= seconds:
                break
        arr = np.array(buf)
        self.fs = (len(arr) - 1) / (last - t0)
        if hasattr(self.source, "fs") and not self.source.fs:
            self.source.fs = self.fs           # BertecSource: spaces queued samples by it (sample clock)
        med = dict(zip(RAW, np.median(arr, axis=0).tolist()))
        offsets = {c: 0.0 for c in RAW} if self.no_tare else med
        fz = arr[:, [RAW.index("Fz_l"), RAW.index("Fz_r")]]
        self.message = ""
        if not self.no_tare and (fz.max(0) - fz.min(0)).max() > TARE_LOADED_N:
            self.message = "warning: Fz changed by more than 100 N during the tare; were the plates unloaded?"
        with self._lock:
            self.offsets, self.t0 = offsets, t0
        return True

    def _step(self, t: float, raw: dict, speed: float) -> None:
        with self._lock:
            self.samples += 1
            self.last_t, self.speed = t, speed
            for i, c in enumerate(RAW):
                self.tared[i] = raw[c] - self.offsets[c]
            self.fz[0], self.fz[1] = self.tared[RAW.index("Fz_l")], self.tared[RAW.index("Fz_r")]
            t_rel = t - self.t0
            for pipe in self.pipes.values():
                pipe.push(t, t_rel, raw, speed)
