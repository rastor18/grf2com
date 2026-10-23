"""Live shadow-mode GUI (scripts/live_gui.py). It never commands the treadmill.

- Source: the Bertec data stream, or a synthetic one for testing without hardware. Start tares the plates (keep them
  unloaded for the first seconds), then streams into every ticked model bundle from deploy/models/.
- Models can be ticked on and off while streaming; each has its own colour in the 3D view (CoM marker, trail, drop
  line to the treadmill) and in the AP / ML / vertical time series.
- Record writes one log per ticked model (live_shadow.py --log-raw format) to deploy/logs/<stamp>_<model>.csv and,
  with "Capture in QTM", starts a QTM capture at the same time and saves it as take <stamp>. A
  deploy/logs/<stamp>_session.json records the subject, tare, models and the clock offset for score_live_log.py.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtCore, QtGui, QtWidgets

from . import live
from .config import ROOT
from .live_session import LiveEngine, list_bundles
from .qtm import QtmRecorder

COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75),
          (227, 119, 194), (23, 190, 207)]
SUBJECT_FIELDS = [("heights", "Height (cm)", 100, 220, 172), ("mass", "Mass (kg)", 30, 200, 76.5),
                  ("legs", "Leg length (cm)", 50, 130, 86), ("feet", "Foot length (cm)", 15, 35, 25)]
SOURCES = ["Bertec (live)", "Synthetic (test, no hardware)"]
LOCKED = "AE01, AE05 and AE08 are locked final-test participants; don't use them for live tests."


def _rgba(c, a=1.0):
    return (c[0] / 255, c[1] / 255, c[2] / 255, a)


class ModelView:
    """3D items and time-series curves of one model."""

    def __init__(self, view: gl.GLViewWidget, plots: list, color):
        self.marker = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=_rgba(color), size=22, pxMode=True)
        self.trail = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=_rgba(color, 0.8), width=2, antialias=True)
        self.drop = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=_rgba(color, 0.5), width=1, antialias=True)
        self.shadow = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=_rgba(color, 0.5), size=9, pxMode=True)
        self.items = [self.trail, self.drop, self.shadow, self.marker]
        for it in self.items:
            it.setVisible(False)
            view.addItem(it)
        self.plots = plots
        self.curves = [p.plot(pen=pg.mkPen(color, width=2)) for p in plots]

    def update(self, m: dict, floor_z: float, trail_n: int, scale: float = 1.0) -> None:
        """scale > 1 magnifies the 3D motion about the mean of the shown history (the plots stay true)."""
        hist, t = m["hist"], m["t"]
        if not len(hist) or not np.isfinite(m["com"]).all():
            return
        com, tr = m["com"], hist[-trail_n:]
        if scale != 1.0:
            mean = np.nanmean(hist, axis=0)
            com, tr = mean + scale * (com - mean), mean + scale * (tr - mean)
        self.marker.setData(pos=com[None])
        if len(tr) > 1:
            self.trail.setData(pos=tr)
        self.drop.setData(pos=np.array([com, [com[0], com[1], floor_z]]))
        self.shadow.setData(pos=np.array([[com[0], com[1], floor_z]]))
        for it in self.items:
            it.setVisible(True)
        for i, c in enumerate(self.curves):
            c.setData(t, hist[:, i])

    def remove(self, view: gl.GLViewWidget) -> None:
        for it in self.items:
            view.removeItem(it)
        for p, c in zip(self.plots, self.curves):
            p.removeItem(c)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: dict, models_dir: Path, log_dir: Path):
        super().__init__()
        self.cfg, self.gcfg, self.log_dir = cfg, cfg["gui"], log_dir
        self.settings = QtCore.QSettings("NeuMove", "grf2com_live_gui")
        self.engine: LiveEngine | None = None
        self.qtm: QtmRecorder | None = None
        self.rec: dict | None = None             # the current recording
        self.pending = None                      # (kind, future, payload) of a QTM call in flight
        self.views: dict[str, ModelView] = {}
        self.center = None if self.gcfg["treadmill"]["center"] == "auto" else list(self.gcfg["treadmill"]["center"])
        self.bundles = list_bundles(models_dir)
        self.setWindowTitle("grf2com live (shadow mode)")
        self._build()
        self._draw_treadmill()
        self.timer = QtCore.QTimer(self, interval=33, timeout=self.refresh)
        self.timer.start()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        s = self.settings
        side = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(side)

        src = QtWidgets.QGroupBox("Input")
        f = QtWidgets.QFormLayout(src)
        self.source = QtWidgets.QComboBox()
        self.source.addItems(SOURCES)
        self.source.setCurrentIndex(int(s.value("source", 0)))
        f.addRow("Source", self.source)
        b = self.cfg["realtime"]["bertec"]
        f.addRow("Bertec stream", QtWidgets.QLabel(f"{b['host']}:{b['data_port']}"))
        self.tare_s = QtWidgets.QDoubleSpinBox(minimum=1, maximum=10, singleStep=0.5, value=float(s.value("tare_s", 3)))
        f.addRow("Tare (s, plates unloaded)", self.tare_s)
        self.no_tare = QtWidgets.QCheckBox("Trust Bertec AUTOZERO (no tare)")
        f.addRow(self.no_tare)
        self.upstream = QtWidgets.QCheckBox("Emulate the training data's filter (recommended)")
        self.upstream.setChecked(True)
        self.upstream.setToolTip("Applies the training data's MATLAB 10 Hz filter causally (2 passes, ~45 ms lag); "
                                 "the raw live stream is unfiltered.")
        f.addRow(self.upstream)
        lay.addWidget(src)

        subj = QtWidgets.QGroupBox("Participant")
        subj.setToolTip(LOCKED)
        f = QtWidgets.QFormLayout(subj)
        self.subject = {}
        for key, label, lo, hi, default in SUBJECT_FIELDS:
            w = QtWidgets.QDoubleSpinBox(minimum=lo, maximum=hi, decimals=1, value=float(s.value(key, default)))
            self.subject[key] = w
            f.addRow(label, w)
        lay.addWidget(subj)

        mod = QtWidgets.QGroupBox("Models (deploy/models)")
        g = QtWidgets.QGridLayout(mod)
        self.model_boxes, self.model_labels = {}, {}
        checked = set(s.value("models", "", str).split(",")) - {""}
        for i, path in enumerate(self.bundles):
            meta = json.loads((path / "bundle.json").read_text())
            color = COLORS[i % len(COLORS)]
            box = QtWidgets.QCheckBox(f"{path.name}  ({meta['model']})")
            box.setStyleSheet(f"QCheckBox {{ color: rgb{color}; font-weight: bold; }}")
            loso = meta.get("development_loso") or {}
            box.setToolTip(f"trained on {', '.join(meta['train_subjects'])}\n"
                           f"dev LOSO RMSE {loso.get('rmse_mean_mm', float('nan')):.1f} mm "
                           f"(AP {loso.get('rmse_ap_mm', float('nan')):.1f}, ML {loso.get('rmse_ml_mm', float('nan')):.1f}, "
                           f"vert {loso.get('rmse_vert_mm', float('nan')):.1f})")
            box.setChecked(path.name in checked or (not checked and i == 0))
            box.toggled.connect(lambda on, p=path: self.toggle_model(p, on))
            label = QtWidgets.QLabel("")
            label.setStyleSheet("font-family: Consolas, monospace;")
            g.addWidget(box, 2 * i, 0)
            g.addWidget(label, 2 * i + 1, 0)
            self.model_boxes[path.name], self.model_labels[path.name] = box, label
        if not self.bundles:
            g.addWidget(QtWidgets.QLabel("no bundles found"), 0, 0)
        lay.addWidget(mod)

        self.start_btn = QtWidgets.QPushButton("Start streaming")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.toggle_stream)
        lay.addWidget(self.start_btn)

        rec = QtWidgets.QGroupBox("Recording")
        f = QtWidgets.QFormLayout(rec)
        self.use_qtm = QtWidgets.QCheckBox("Capture in QTM (saved as take <date_time>)")
        self.use_qtm.setChecked(s.value("use_qtm", "true") == "true")
        f.addRow(self.use_qtm)
        self.qtm_password = QtWidgets.QLineEdit(s.value("qtm_password", os.environ.get("QTM_PASSWORD", "")))
        self.qtm_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.qtm_password.setToolTip("QTM real-time control password (QTM: Project Options > Real-Time Output)")
        f.addRow("QTM password", self.qtm_password)
        self.rec_btn = QtWidgets.QPushButton("Record")
        self.rec_btn.setMinimumHeight(36)
        self.rec_btn.clicked.connect(self.toggle_record)
        f.addRow(self.rec_btn)
        self.rec_label = QtWidgets.QLabel("")
        self.rec_label.setWordWrap(True)
        f.addRow(self.rec_label)
        lay.addWidget(rec)

        self.status = QtWidgets.QLabel("idle")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-family: Consolas, monospace;")
        lay.addWidget(self.status)
        lay.addStretch(1)
        scale_row = QtWidgets.QHBoxLayout()
        scale_row.addWidget(QtWidgets.QLabel("3D motion scale"))
        self.motion_scale = QtWidgets.QComboBox()
        self.motion_scale.addItems(["x1 (true)", "x3", "x5", "x10"])
        self.motion_scale.setToolTip("Magnifies the CoM motion about its recent mean in the 3D view only; the time "
                                     "series and logs are always true scale.")
        scale_row.addWidget(self.motion_scale, 1)
        lay.addLayout(scale_row)
        view_row = QtWidgets.QHBoxLayout()
        for text, slot in (("Reset view", self.reset_view), ("Re-centre treadmill", self.recenter)):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(slot)
            view_row.addWidget(btn)
        lay.addLayout(view_row)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("w")
        self.plots_widget = pg.GraphicsLayoutWidget()
        self.plots_widget.setBackground("w")
        self.plots = []
        for i, name in enumerate(["AP", "ML", "vertical"]):
            p = self.plots_widget.addPlot(row=i, col=0)
            p.setLabel("left", name, units="m")
            p.showGrid(x=True, y=True, alpha=0.3)
            if self.plots:
                p.setXLink(self.plots[0])
            self.plots.append(p)
        self.plots[-1].setLabel("bottom", "time (s)")
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.addWidget(self.view)
        right.addWidget(self.plots_widget)
        right.setSizes([600, 350])
        scroll = QtWidgets.QScrollArea(widgetResizable=True)
        scroll.setWidget(side)
        scroll.setMinimumWidth(340)
        main = QtWidgets.QSplitter()
        main.addWidget(scroll)
        main.addWidget(right)
        main.setSizes([360, 1000])
        self.setCentralWidget(main)
        self.resize(1400, 900)

    # ------------------------------------------------------------------ 3D scene
    def _draw_treadmill(self) -> None:
        for it in getattr(self, "scene_items", []):
            self.view.removeItem(it)
        tm = self.gcfg["treadmill"]
        cx, cy = self.center or (0.0, 0.0)
        z, L, w, gap = tm["surface_z_m"], tm["length_m"], tm["belt_width_m"], tm["gap_m"]
        grid = gl.GLGridItem(color=(0, 0, 0, 40))
        grid.setSize(4, 4)
        grid.setSpacing(0.25, 0.25)
        grid.translate(cx, cy, z - 0.001)
        self.scene_items = [grid]
        for side in (-1, 1):
            y0 = cy + side * gap / 2
            y1 = y0 + side * w
            v = np.array([[cx - L / 2, y0, z], [cx + L / 2, y0, z], [cx + L / 2, y1, z], [cx - L / 2, y1, z]])
            belt = gl.GLMeshItem(meshdata=gl.MeshData(vertexes=v, faces=np.array([[0, 1, 2], [0, 2, 3]])),
                                 color=(0.35, 0.35, 0.4, 0.35), smooth=False, glOptions="translucent")
            edge = gl.GLLinePlotItem(pos=np.vstack([v, v[:1]]), color=(0.2, 0.2, 0.25, 1), width=2, antialias=True)
            self.scene_items += [belt, edge]
        axis = gl.GLAxisItem(size=QtGui.QVector3D(0.3, 0.3, 0.3))
        axis.translate(cx - L / 2 - 0.2, cy - w - 0.2, z)
        self.scene_items.append(axis)
        for text, pos in (("AP +", (cx + L / 2 + 0.1, cy, z)), ("ML +", (cx, cy + w + gap + 0.1, z))):
            self.scene_items.append(gl.GLTextItem(pos=pos, text=text, color=(60, 60, 60, 255)))
        for it in self.scene_items:
            self.view.addItem(it)
        self.reset_view()

    def reset_view(self) -> None:
        cx, cy = self.center or (0.0, 0.0)
        self.view.setCameraPosition(pos=QtGui.QVector3D(cx, cy, self.gcfg["treadmill"]["surface_z_m"] + 0.5),
                                    distance=3.8, elevation=22, azimuth=-125)

    def recenter(self) -> None:
        self.center = None
        self._auto_center(force=True)

    def _auto_center(self, snap: dict | None = None, force: bool = False) -> None:
        snap = snap or (self.engine.snapshot() if self.engine else None)
        if not snap:
            return
        for m in snap["models"].values():
            if m["warm"] >= 1 and len(m["hist"]) > 60:
                self.center = np.nanmean(m["hist"][-120:, :2], axis=0).tolist()
                self._draw_treadmill()
                return
        if force:
            self.rec_label.setText("re-centre needs a warmed-up model")

    # ------------------------------------------------------------------ actions
    def _subject(self) -> dict:
        return {k: w.value() for k, w in self.subject.items()}

    def _set_inputs_enabled(self, on: bool) -> None:
        for w in [self.source, self.tare_s, self.no_tare, self.upstream, *self.subject.values()]:
            w.setEnabled(on)

    def toggle_stream(self) -> None:
        if self.engine is not None:
            self.stop_stream()
            return
        subject = self._subject()
        if self.source.currentIndex() == 0:
            try:
                source = live.BertecSource(self.cfg["realtime"]["bertec"], timeout_ms=1000)
            except Exception as e:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(self, "Bertec", f"Could not open the Bertec stream:\n{e}")
                return
        else:
            source = live.SyntheticSource(mass=subject["mass"])
        self.engine = LiveEngine(source, subject, tare_s=self.tare_s.value(), no_tare=self.no_tare.isChecked(),
                                 upstream=self.upstream.isChecked(), history_s=float(self.gcfg["history_s"]))
        for path in self.bundles:
            if self.model_boxes[path.name].isChecked():
                self.engine.enable(path)
        self.engine.start()
        self._set_inputs_enabled(False)
        self.start_btn.setText("Stop streaming")
        self._save_settings()

    def stop_stream(self) -> None:
        if self.rec:
            self.stop_record()
        if self.engine is not None:
            self.engine.stop()
            self.engine = None
        for v in self.views.values():
            v.remove(self.view)
        self.views.clear()
        self._set_inputs_enabled(True)
        self.start_btn.setText("Start streaming")

    def toggle_model(self, path: Path, on: bool) -> None:
        self._save_settings()
        if self.engine is None:
            return
        if on:
            try:
                self.engine.enable(path)
            except Exception as e:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(self, "Model", f"Could not load {path.name}:\n{e}")
                self.model_boxes[path.name].setChecked(False)
        else:
            info = self.engine.disable(path.name)
            if info and self.rec is not None:
                self.rec["logs"].append(info)
            view = self.views.pop(path.name, None)
            if view:
                view.remove(self.view)

    def toggle_record(self) -> None:
        if self.rec is not None:
            self.stop_record()
            return
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.use_qtm.isChecked():
            self.qtm = QtmRecorder(self.cfg["realtime"]["qtm"]["host"], self.qtm_password.text())
            self.pending = ("start", self.qtm.start(), stamp)
            self.rec_btn.setEnabled(False)
            self.rec_label.setText("starting the QTM capture ...")
        else:
            self._begin_logs(stamp, None)
        self._save_settings()

    def _begin_logs(self, stamp: str, qtm_t: float | None) -> None:
        if self.engine is None or self.engine.state != "running":
            if self.qtm is not None and qtm_t is not None:
                self.pending = ("stop", self.qtm.stop(None), None)
            self.rec_label.setText("streaming stopped before the recording began")
            return
        self.engine.start_recording(self.log_dir, stamp)
        self.rec = {"stamp": stamp, "qtm_t": qtm_t, "logs": [], "started": dt.datetime.now()}
        if qtm_t is not None:
            self.qtm.event("grf2com_log_start")
        self.rec_btn.setText("Stop recording")
        self.rec_btn.setStyleSheet("background-color: #d62728; color: white; font-weight: bold;")

    def stop_record(self, wait: bool = False) -> None:
        rec, self.rec = self.rec, None
        self.rec_btn.setText("Record")
        self.rec_btn.setStyleSheet("")
        if rec is None:
            return
        rec["logs"] += self.engine.stop_recording() if self.engine else []
        rec["qtm"] = None
        if self.qtm is not None and rec["qtm_t"] is not None:
            rec["qtm"] = {"take": rec["stamp"], "capture_started_t": rec["qtm_t"], "saved": None}
            fut = self.qtm.stop(rec["stamp"])
            if wait:
                try:
                    fut.result(timeout=20)
                    rec["qtm"]["saved"] = True
                except Exception as e:  # noqa: BLE001
                    rec["qtm"]["saved"] = f"failed: {e}"
            else:
                self.pending = ("stop", fut, rec)
                self.rec_btn.setEnabled(False)
        path = self._write_session(rec)
        self.rec_label.setText(f"saved {len(rec['logs'])} log(s): {path.name}"
                               + ("; stopping and saving the QTM take ..." if rec["qtm"] and not wait else ""))

    def _write_session(self, rec: dict) -> Path:
        snap = self.engine.snapshot() if self.engine else {}
        logs = []
        for info in rec["logs"]:
            entry = dict(info)
            if rec["qtm_t"] is not None and info["t_first"] is not None:
                # CoM time (from the take's first frame) = log time (from its first row) + offset
                entry["offset_s"] = round(info["t_first"] - rec["qtm_t"], 4)
                log = Path(info["log"])
                log = log.relative_to(ROOT).as_posix() if log.is_relative_to(ROOT) else str(log)
                entry["score"] = (f"scripts/score_live_log.py --log {log} "
                                  f"--com <take {rec['stamp']} exported>_CoM.csv --offset-s {entry['offset_s']}")
            logs.append(entry)
        out = {"stamp": rec["stamp"], "started": rec["started"].isoformat(timespec="seconds"),
               "stopped": dt.datetime.now().isoformat(timespec="seconds"), "source": self.source.currentText(),
               "subject": self._subject(), "tare_offsets": snap.get("offsets"), "input_rate_hz": snap.get("fs"),
               "no_tare": self.no_tare.isChecked(), "upstream_filter_emulation": self.upstream.isChecked(),
               "qtm": rec["qtm"], "logs": logs,
               "note": "offset_s comes from QTM's capture-started event, so it is approximate; leave out --offset-s "
                       "to let score_live_log.py align the clocks itself."}
        path = self.log_dir / f"{rec['stamp']}_session.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1))
        rec["session_path"] = path
        return path

    def _check_pending(self) -> None:
        if self.pending is None or not self.pending[1].done():
            return
        kind, fut, payload = self.pending
        self.pending = None
        self.rec_btn.setEnabled(True)
        err = fut.exception()
        if kind == "start":
            if err is not None:
                self.qtm = None
                self.rec_label.setText("QTM capture did not start; nothing recorded")
                QtWidgets.QMessageBox.warning(self, "QTM", f"Could not start the QTM capture:\n{err}\n\n"
                                              "Untick 'Capture in QTM' to record the logs only.")
            else:
                self._begin_logs(payload, fut.result())
        elif kind == "stop" and isinstance(payload, dict):
            payload["qtm"]["saved"] = True if err is None else f"failed: {err}"
            self._write_session(payload)
            self.rec_label.setText(f"saved {len(payload['logs'])} log(s) and "
                                   + (f"QTM take {payload['stamp']}" if err is None else f"QTM FAILED: {err}"))
            self.qtm = None
        else:
            self.qtm = None

    # ------------------------------------------------------------------ refresh (30 Hz)
    def refresh(self) -> None:
        self._check_pending()
        running = self.engine is not None and self.engine.state == "running"
        self.rec_btn.setEnabled(self.pending is None and (self.rec is not None or
                                                          (running and bool(self.engine.pipes))))
        if self.engine is None:
            return
        snap = self.engine.snapshot()
        if snap["state"] in ("stopped", "error") and self.engine is not None:
            msg = snap["message"]
            self.stop_stream()
            self.status.setText(f"{snap['state']}: {msg}")
            if msg:
                QtWidgets.QMessageBox.warning(self, "Stream", msg)
            return
        if self.center is None:
            self._auto_center(snap)
        floor = self.gcfg["treadmill"]["surface_z_m"]
        trail_n = int(self.gcfg["trail_s"] * 60)
        scale = float(self.motion_scale.currentText().split()[0][1:])
        total_ms = 0.0
        for i, path in enumerate(self.bundles):
            name = path.name
            m = snap["models"].get(name)
            label = self.model_labels[name]
            if m is None:
                label.setText("")
                continue
            if name not in self.views:
                self.views[name] = ModelView(self.view, self.plots, COLORS[i % len(COLORS)])
            self.views[name].update(m, floor, trail_n, scale)
            if np.isfinite(m["compute_ms"]):
                total_ms += m["compute_ms"]
            c = m["com"]
            warm = "" if m["warm"] >= 1 else f"  warming {100 * m['warm']:.0f}%"
            label.setText(f"AP {c[0]:+.3f} ML {c[1]:+.3f} V {c[2]:.3f} m  {m['compute_ms']:.1f} ms{warm}")
        if len(snap["models"]):
            t = snap["t"]
            self.plots[0].setXRange(max(0.0, t - self.gcfg["history_s"]), max(t, 1.0), padding=0)
        lines = [f"state: {snap['state']}"]
        if snap["state"] == "taring":
            lines.append(f"tare {100 * snap['tare_progress']:.0f}%: keep both plates unloaded")
        if snap["fs"]:
            lines.append(f"input {snap['fs']:.0f} Hz, t {snap['t']:.1f} s, belt {snap['speed']:.2f} m/s")
            lines.append(f"Fz L {snap['fz'][0]:6.0f} N  R {snap['fz'][1]:6.0f} N")
            budget = 1000 / 60
            lines.append(f"models {total_ms:.1f} ms per frame (budget {budget:.1f})"
                         + ("  BEHIND REAL TIME" if total_ms > 0.9 * budget else ""))
        if snap["message"]:
            lines.append(snap["message"])
        self.status.setText("\n".join(lines))
        if self.rec is not None:
            secs = (dt.datetime.now() - self.rec["started"]).total_seconds()
            self.rec_label.setText(f"REC {self.rec['stamp']}  {secs:.0f} s"
                                   + ("  (+ QTM)" if self.rec["qtm_t"] is not None else ""))

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("source", self.source.currentIndex())
        s.setValue("tare_s", self.tare_s.value())
        s.setValue("use_qtm", "true" if self.use_qtm.isChecked() else "false")
        s.setValue("qtm_password", self.qtm_password.text())
        s.setValue("models", ",".join(n for n, b in self.model_boxes.items() if b.isChecked()))
        for k, w in self.subject.items():
            s.setValue(k, w.value())

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt)
        if self.rec is not None:
            self.stop_record(wait=True)
        if self.engine is not None:
            self.engine.stop()
        self._save_settings()
        super().closeEvent(event)


def main(cfg: dict, models_dir: Path, log_dir: Path) -> None:
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    win = MainWindow(cfg, models_dir, log_dir)
    win.show()
    sys.exit(app.exec_())
