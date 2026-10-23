"""Live session engine behind the GUI (scripts/live_gui.py): one sample stream feeding several model bundles at once.

It runs in a worker thread: tare (median of each raw channel while the plates are unloaded) and the measured input
rate first, then every sample goes to each enabled bundle's own StreamingPreprocessor + StreamingEstimator (bundles
may differ in preprocessing). While recording, each bundle writes its own log in the live_shadow.py --log-raw format
(t, frame, com_ap/ml/vert, compute_ms, raw channels, speed), so check_live_mapping.py and score_live_log.py work on
it unchanged. The GUI only calls the public methods and reads snapshot(). Shadow mode: nothing is ever sent to the
treadmill.
"""
from __future__ import annotations

import csv
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .bundle import load_bundle
from .realtime import RAW, StreamingEstimator, StreamingPreprocessor

LOG_COLS = ["t", "frame", "com_ap", "com_ml", "com_vert", "compute_ms"] + RAW + ["speed"]
TARE_LOADED_N = 100.0      # Fz range during the tare above this: the plates were probably not unloaded


def list_bundles(models_dir: Path) -> list[Path]:
    models_dir = Path(models_dir)
    return sorted(p for p in models_dir.iterdir() if (p / "bundle.json").exists()) if models_dir.is_dir() else []


class Pipeline:
    """One bundle on the stream: preprocessing, estimator, recent history and (while recording) its log."""

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
        self.frames = 0
        self.com = np.full(3, np.nan)
        self.compute_ms = float("nan")
        self.hist_t: deque = deque(maxlen=int(history_s * rate))
        self.hist_com: deque = deque(maxlen=int(history_s * rate))
        self.log_fh = None
        self.log_writer = None
        self.log_path: Path | None = None
        self.log_t_first: float | None = None

    def open_log(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path, self.log_t_first, self.log_frame = path, None, 0
        self.log_fh = open(path, "w", newline="")
        self.log_writer = csv.writer(self.log_fh)
        self.log_writer.writerow(LOG_COLS)

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
        self.com = com
        self.frames += 1
        self.hist_t.append(t_rel)
        self.hist_com.append(com)
        if self.log_writer is not None:
            if self.log_t_first is None:
                self.log_t_first = t
            self.log_writer.writerow([f"{t:.4f}", self.log_frame, *(f"{v:.5f}" for v in com),
                                      f"{self.compute_ms:.2f}", *(f"{raw[c]:.4f}" for c in RAW), f"{speed:.3f}"])
            self.log_frame += 1


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
        self.tare_progress = 0.0
        self.pipes: dict[str, Pipeline] = {}
        self._wanted: list[Path] = []          # bundles enabled before the tare finished
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

    def enable(self, path: Path) -> None:
        """Add a bundle. Before the tare finishes it is queued; afterwards it is loaded here (outside the lock, so the
        stream doesn't stall) and starts from an empty window."""
        path = Path(path)
        with self._lock:
            if path.name in self.pipes or path in self._wanted:
                return
            if self.offsets is None:
                self._wanted.append(path)
                return
            offsets, fs = self.offsets, self.fs
        pipe = Pipeline(path, self.subject, offsets, fs, self.upstream, self.history_s)
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
                                "warm": min(1.0, p.frames / max(p.warm_frames, 1)),
                                "t": np.fromiter(p.hist_t, float, len(p.hist_t)),
                                "hist": np.array(p.hist_com).reshape(-1, 3)}
            return {"state": self.state, "message": self.message, "fs": self.fs, "samples": self.samples,
                    "t": (self.last_t - self.t0) if self.t0 is not None else 0.0, "speed": self.speed,
                    "fz": self.fz.copy(), "tare_progress": self.tare_progress, "offsets": self.offsets,
                    "recording": self._recording is not None, "models": models}

    # ----------------------------------------------------------------- worker thread
    def _log_path(self, name: str) -> Path:
        return self._recording["dir"] / f"{self._recording['stamp']}_{name}.csv"

    def _run(self) -> None:
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
            self.fz[0] = raw["Fz_l"] - self.offsets["Fz_l"]
            self.fz[1] = raw["Fz_r"] - self.offsets["Fz_r"]
            t_rel = t - self.t0
            for pipe in self.pipes.values():
                pipe.push(t, t_rel, raw, speed)
