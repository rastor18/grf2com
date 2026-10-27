"""Live shadow-mode GUI (scripts/live_gui.py). It never commands the treadmill.

- Source: the Bertec data stream, or a synthetic one for testing without hardware. Start tares the plates (keep them
  unloaded for the first seconds), then streams into every ticked model bundle from deploy/models/.
- Treadmill speed: manual belt speed over Bertec remote control (both belts tied; STOP / Esc ramps to 0). The CoM
  estimates never drive the treadmill.
- GRF arrows (red, like QTM's): the ground reaction force on the body at each plate's COP, from the live tared
  channels (plate axes: gui.grf in configs/base.yaml).
- Display frame: AP from the posterior-most force-plate corner (forward +), ML from midway between the belts (left +),
  height above the belts, from QTM's force-plate corners (gui.plates in configs/base.yaml when QTM isn't reachable).
  The logs stay in the mocap lab frame.
- Models can be ticked on and off while streaming; each has its own colour in the 3D view (CoM marker, trail, drop
  line to the treadmill) and in the AP / ML / vertical time series.
- "lab_kf" is the lab's self-paced-treadmill Kalman filter run on the same stream (live_session.LabKfPipeline): AP
  only. The controller reads it only at accepted heel strikes, so its marker and trail are opaque for a moment after
  each one and half transparent otherwise, and each accepted step leaves an opaque dot at the step-mean position the
  controller would have acted on (com_step); the readout shows the speed it would have commanded.
- Record writes one log per ticked model (live_shadow.py --log-raw format) to deploy/logs/<stamp>_<model>.csv and,
  with "Capture in QTM", starts a QTM capture at the same time and saves it as take <stamp>. Every speed command is a
  QTM event (speed_<m/s>) in the take; the logs carry the measured belt speed per frame. A
  deploy/logs/<stamp>_session.json records the subject, tare, models, speed commands and the clock offset for
  score_live_log.py.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtCore, QtGui, QtWidgets

from . import live
from .config import ROOT
from .live_session import LabKfSpec, LiveEngine, list_bundles
from .qtm import QtmRecorder, read_force_plates
from .treadmill import BertecTreadmill, grf_arrow, plate_frames, to_display, treadmill_frame

LAB_KF_COLOR = (255, 127, 14)                      # orange, whatever bundles are installed
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

    UNUSED_ALPHA = 0.5     # lab KF: frames the controller doesn't read

    def __init__(self, view: gl.GLViewWidget, plots: list, color):
        self.color = color
        self.marker = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=_rgba(color), size=22, pxMode=True)
        self.trail = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=_rgba(color, 0.8), width=2, antialias=True)
        self.drop = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=_rgba(color, 0.5), width=1, antialias=True)
        self.shadow = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=_rgba(color, 0.5), size=9, pxMode=True)
        self.step_dots = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=_rgba(color), size=12, pxMode=True)
        self.items = [self.trail, self.drop, self.shadow, self.marker, self.step_dots]
        for it in self.items:
            it.setVisible(False)
            view.addItem(it)
        self.plots = plots
        self.curves = [p.plot(pen=pg.mkPen(color, width=1)) for p in plots]   # 1 px: Qt's fast path (wider is slow)
        self.step_plot = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(*color), pen=None)
        plots[0].addItem(self.step_plot)

    def update(self, m: dict, frame: dict, trail_n: int, scale: float = 1.0) -> None:
        """Draws in the treadmill frame (floor = belt surface, z 0). scale > 1 magnifies the 3D motion about the mean
        of the shown history (the plots stay true)."""
        if not len(m["hist"]) or not np.isfinite(m["com"]).all():
            return
        hist, t, floor_z = to_display(m["hist"], frame), m["t"], 0.0
        com, tr = to_display(m["com"], frame), hist[-trail_n:]
        if scale != 1.0:
            mean = np.nanmean(hist, axis=0)
            com, tr = mean + scale * (com - mean), mean + scale * (tr - mean)
        used = m.get("used")
        if used is None:                                    # networks: always opaque
            self.marker.setData(pos=com[None])
            if len(tr) > 1:
                self.trail.setData(pos=tr)
        else:                                               # lab KF: opaque only where the controller reads it
            u = used[-len(tr):]
            alpha = np.where(u, 1.0, self.UNUSED_ALPHA)
            self.marker.setData(pos=com[None], color=_rgba(self.color, float(alpha[-1])))
            if len(tr) > 1:
                cols = np.tile(np.array(_rgba(self.color)), (len(tr), 1))
                cols[:, 3] = alpha
                self.trail.setData(pos=tr, color=cols)
        steps = m.get("steps") or []
        recent = [pt for ts, pt in steps if len(t) and ts >= t[-len(tr)]]
        if recent:
            dots = to_display(np.array(recent), frame)
            if scale != 1.0:
                dots = mean + scale * (dots - mean)
            self.step_dots.setData(pos=dots)
        self.drop.setData(pos=np.array([com, [com[0], com[1], floor_z]]))
        self.shadow.setData(pos=np.array([[com[0], com[1], floor_z]]))
        for it in self.items:
            it.setVisible(it is not self.step_dots or bool(recent))
        if used is None:
            for i, c in enumerate(self.curves):
                c.setData(t, hist[:, i])
        else:                                               # AP only (ML / height are placeholders)
            self.curves[0].setData(t, hist[:, 0])
            self.curves[0].setPen(pg.mkPen((*self.color, int(255 * self.UNUSED_ALPHA)), width=1))
            pts = [(ts, to_display(pt, frame)[0]) for ts, pt in steps if len(t) and ts >= t[0]]
            self.step_plot.setData(pos=pts)

    def remove(self, view: gl.GLViewWidget) -> None:
        for it in self.items:
            view.removeItem(it)
        for p, c in zip(self.plots, self.curves):
            p.removeItem(c)
        self.plots[0].removeItem(self.step_plot)


ROOM_HALF_M, ROOM_HEIGHT_M = 4.0, 3.0        # the virtual room: 8 x 8 m around the treadmill (= the floor grid)
BANNER_W_M, BANNER_H_M, BANNER_Z_M = 2.4, 0.6, 1.7


def logo_image(w: int = 1024, h: int = 256) -> np.ndarray:
    """The NeuMove wordmark (red NEU, white italic MOVE on near-black) as GLImageItem data: (x, y, RGBA), y up."""
    img = QtGui.QImage(w, h, QtGui.QImage.Format_RGBA8888)
    img.fill(QtGui.QColor(20, 20, 22))
    painter = QtGui.QPainter(img)
    painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
    neu = QtGui.QFont("Arial", 1, QtGui.QFont.Black)
    neu.setPixelSize(int(0.62 * h))
    move = QtGui.QFont(neu)
    move.setItalic(True)
    widths = [QtGui.QFontMetrics(f).horizontalAdvance(t) for f, t in ((neu, "NEU"), (move, "MOVE"))]
    x = (w - sum(widths)) // 2
    base = int(0.5 * (h + QtGui.QFontMetrics(neu).capHeight()))        # caps centred vertically
    for font, text, color, width in ((neu, "NEU", QtGui.QColor(224, 30, 43), widths[0]),
                                     (move, "MOVE", QtGui.QColor(245, 245, 245), widths[1])):
        painter.setFont(font)
        painter.setPen(color)
        painter.drawText(x, base, text)
        x += width
    painter.end()
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    arr = np.frombuffer(ptr, np.uint8).reshape(h, w, 4)
    return np.ascontiguousarray(arr[::-1].transpose(1, 0, 2))


def _placed(item, corner: np.ndarray, u: np.ndarray, v: np.ndarray, n: np.ndarray):
    """Maps the item's local x / y / z onto the lab vectors u / v / n (already scaled), from `corner`."""
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = u, v, n, corner
    item.setTransform(QtGui.QMatrix4x4(*m.flatten()))
    return item


class GrfView:
    """Red ground-reaction-force arrows, one per plate, from the COP along the force on the body."""

    RED = (0.85, 0.1, 0.1, 1.0)

    def __init__(self, view: gl.GLViewWidget):
        self.arrows = {side: gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=self.RED, width=4, antialias=True,
                                               mode="lines") for side in "lr"}
        self.cops = {side: gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=self.RED, size=10, pxMode=True)
                     for side in "lr"}
        for it in [*self.arrows.values(), *self.cops.values()]:
            it.setVisible(False)
            view.addItem(it)

    def hide(self) -> None:
        for it in [*self.arrows.values(), *self.cops.values()]:
            it.setVisible(False)

    def update(self, tared: dict, plates: dict, frame: dict, m_per_n: float, min_fz: float) -> None:
        for side in "lr":
            f = np.array([tared[f"F{a}_{side}"] for a in "xyz"])
            m = np.array([tared[f"M{a}_{side}"] for a in "xyz"])
            on = side in plates and f[2] > min_fz
            self.arrows[side].setVisible(on)
            self.cops[side].setVisible(on)
            if not on:
                continue
            cop, force = grf_arrow(f, m, plates[side])
            base = to_display(cop, frame)
            base[2] = 0.0                                  # on the drawn belt surface
            vec = force.copy()
            vec[:2] *= frame["sign"]
            tip = base + m_per_n * vec
            d = vec / np.linalg.norm(vec)
            side_dir = np.cross(d, [0.0, 1.0, 0.0])
            if np.linalg.norm(side_dir) < 1e-6:
                side_dir = np.array([1.0, 0.0, 0.0])
            side_dir /= np.linalg.norm(side_dir)
            head = 0.06
            pts = [base, tip, tip, tip - head * d + 0.5 * head * side_dir, tip, tip - head * d - 0.5 * head * side_dir]
            self.arrows[side].setData(pos=np.array(pts))
            self.cops[side].setData(pos=base[None])


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
        self.plates, self.plates_from = self.gcfg["plates"], "configs/base.yaml (QTM not read yet)"
        self.frame = treadmill_frame(self.plates)
        self.plate_frames = plate_frames(self.plates, self.gcfg["grf"]["plate_axes"])
        self.plates_pool = ThreadPoolExecutor(1, thread_name_prefix="qtm-plates")
        self.plates_fut = None
        self._request_plates()
        self.tm: BertecTreadmill | None = None
        self.speed_futs: list = []                # (future, target, accel, t_sent) of speed commands in flight
        self.commanded: float | None = None       # last acknowledged belt speed command
        self.bundles = list_bundles(models_dir) + [LabKfSpec.name]   # bundle folders, then the lab KF
        self.setWindowTitle("grf2com live (shadow mode)")
        self._build()
        self._draw_treadmill()
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self, activated=self.stop_belts,
                            context=QtCore.Qt.ApplicationShortcut)
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
        self.model_boxes, self.model_labels, self.model_colors = {}, {}, {}
        checked = set(s.value("models", "", str).split(",")) - {""}
        others = iter(c for c in itertools.cycle(COLORS) if c != LAB_KF_COLOR)   # the KF keeps orange
        for i, path in enumerate(self.bundles):
            color = LAB_KF_COLOR if path == LabKfSpec.name else next(others)
            name = self._name(path)
            self.model_colors[name] = color
            if path == LabKfSpec.name:
                box = QtWidgets.QCheckBox(f"{name}  (self-paced treadmill KF)")
                box.setToolTip("The lab's Song et al. (2020) Kalman filter, as in the self-paced treadmill code, on "
                               "this stream: AP only, starts with the belt, mass from the participant fields.\n"
                               "Opaque = the moments the controller reads it (accepted heel strikes); dots = the "
                               "step-mean position it would have acted on. ML = foot midpoint, height = 0.55 x "
                               "stature (placeholders).")
            else:
                meta = json.loads((path / "bundle.json").read_text())
                box = QtWidgets.QCheckBox(f"{name}  ({meta['model']})")
                loso = meta.get("development_loso") or {}
                box.setToolTip(f"trained on {', '.join(meta['train_subjects'])}\n"
                               f"dev LOSO RMSE {loso.get('rmse_mean_mm', float('nan')):.1f} mm "
                               f"(AP {loso.get('rmse_ap_mm', float('nan')):.1f}, "
                               f"ML {loso.get('rmse_ml_mm', float('nan')):.1f}, "
                               f"vert {loso.get('rmse_vert_mm', float('nan')):.1f})")
            box.setStyleSheet(f"QCheckBox {{ color: rgb{color}; font-weight: bold; }}")
            box.setChecked(name in checked or (not checked and i == 0))
            box.toggled.connect(lambda on, p=path: self.toggle_model(p, on))
            label = QtWidgets.QLabel("")
            label.setStyleSheet("font-family: Consolas, monospace;")
            g.addWidget(box, 2 * i, 0)
            g.addWidget(label, 2 * i + 1, 0)
            self.model_boxes[name], self.model_labels[name] = box, label
        lay.addWidget(mod)

        self.start_btn = QtWidgets.QPushButton("Start streaming")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.toggle_stream)
        lay.addWidget(self.start_btn)

        sp, scfg = QtWidgets.QGroupBox("Treadmill speed (Bertec remote control)"), self.gcfg["speed"]
        g = QtWidgets.QGridLayout(sp)
        self.target = QtWidgets.QDoubleSpinBox(minimum=0, maximum=float(scfg["max_m_s"]), decimals=2,
                                               singleStep=float(scfg["step_m_s"]), suffix=" m/s",
                                               value=float(s.value("target_speed", 1.25)))
        self.accel = QtWidgets.QDoubleSpinBox(minimum=0.01, maximum=float(scfg["max_accel_m_s2"]), decimals=2,
                                              singleStep=0.05, suffix=" m/s²",
                                              value=float(s.value("accel", scfg["accel_m_s2"])))
        g.addWidget(QtWidgets.QLabel("Target"), 0, 0)
        g.addWidget(self.target, 0, 1)
        g.addWidget(QtWidgets.QLabel("Accel"), 1, 0)
        g.addWidget(self.accel, 1, 1)
        row = QtWidgets.QHBoxLayout()
        step = float(scfg["step_m_s"])
        for text, slot in ((f"−{step:.2f}", lambda: self.nudge(-step)), ("Set speed", self.set_speed),
                           (f"+{step:.2f}", lambda: self.nudge(step))):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(slot)
            row.addWidget(btn, 2 if text == "Set speed" else 1)
        g.addLayout(row, 2, 0, 1, 2)
        self.stop_belts_btn = QtWidgets.QPushButton("STOP BELTS  (Esc)")
        self.stop_belts_btn.setMinimumHeight(40)
        self.stop_belts_btn.setStyleSheet("background-color: #d62728; color: white; font-weight: bold;")
        self.stop_belts_btn.clicked.connect(self.stop_belts)
        g.addWidget(self.stop_belts_btn, 3, 0, 1, 2)
        self.speed_label = QtWidgets.QLabel("not connected (connects on the first command)")
        self.speed_label.setWordWrap(True)
        g.addWidget(self.speed_label, 4, 0, 1, 2)
        sp.setToolTip(f"Both belts tied, 0 to {scfg['max_m_s']} m/s (gui.speed in configs/base.yaml). STOP and Esc "
                      f"ramp to 0 at {scfg['stop_decel_m_s2']} m/s². The Bertec software must be in remote control.")
        lay.addWidget(sp)

        rec = QtWidgets.QGroupBox("Recording")
        f = QtWidgets.QFormLayout(rec)
        self.use_qtm = QtWidgets.QCheckBox("Capture in QTM (saved as take <date_time>)")
        self.use_qtm.setChecked(s.value("use_qtm", "true") == "true")
        f.addRow(self.use_qtm)
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
        self.show_grf = QtWidgets.QCheckBox("Show GRF arrows (1 mm per N)")
        self.show_grf.setChecked(s.value("show_grf", "true") == "true")
        self.show_grf.setToolTip("Ground reaction force on the body at each plate's COP, from the live tared "
                                 "channels. The plate axes (gui.grf in configs/base.yaml) are inferred; compare with "
                                 "QTM's force arrows.")
        lay.addWidget(self.show_grf)
        self.frame_label = QtWidgets.QLabel("")
        self.frame_label.setWordWrap(True)
        lay.addWidget(self.frame_label)
        reset = QtWidgets.QPushButton("Reset view")
        reset.clicked.connect(self.reset_view)
        lay.addWidget(reset)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("w")
        self.grf_view = GrfView(self.view)
        self.plots_widget = pg.GraphicsLayoutWidget()
        self.plots_widget.setBackground("w")
        self.plots = []
        for i, name in enumerate(["AP", "ML (left +)", "height"]):
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
        plates = [to_display(np.asarray(p["corners"], float), self.frame) for p in self.plates]
        allc = np.vstack(plates)
        lo, hi = allc.min(0), allc.max(0)
        grid = gl.GLGridItem(color=(0, 0, 0, 40))
        grid.setSize(2 * ROOM_HALF_M, 2 * ROOM_HALF_M)
        grid.setSpacing(0.25, 0.25)
        grid.translate((lo[0] + hi[0]) / 2, 0, -0.001)
        self.scene_items = [grid]
        for v in plates:
            v = v.copy()
            v[:, 2] = 0.0                   # draw the belts flat on the display floor
            belt = gl.GLMeshItem(meshdata=gl.MeshData(vertexes=v, faces=np.array([[0, 1, 2], [0, 2, 3]])),
                                 color=(0.35, 0.35, 0.4, 0.35), smooth=False, glOptions="translucent")
            edge = gl.GLLinePlotItem(pos=np.vstack([v, v[:1]]), color=(0.2, 0.2, 0.25, 1), width=2, antialias=True)
            self.scene_items += [belt, edge]
        axis = gl.GLAxisItem(size=QtGui.QVector3D(0.3, 0.3, 0.3))     # at the origin: x AP, y ML, z up
        self.scene_items.append(axis)
        for text, pos in (("+AP", (hi[0] + 0.15, 0, 0)), ("+ML", ((lo[0] + hi[0]) / 2, hi[1] + 0.15, 0))):
            self.scene_items.append(gl.GLTextItem(pos=pos, text=text, color=(60, 60, 60, 255)))
        self._build_room((lo[0] + hi[0]) / 2)
        for it in self.scene_items:
            self.view.addItem(it)
        self.frame_label.setText(f"Origin: rear plate edge / belt midline / belt surface, from {self.plates_from}.")
        self.reset_view()

    def _build_room(self, mid_x: float) -> None:
        """Four light walls at the edges of the floor grid, each with a NeuMove banner facing in. Walls between the
        camera and the treadmill are hidden (_cull_walls), so they never block the view."""
        logo = logo_image()
        z = np.array([0.0, 0.0, 1.0])
        self.walls = []
        r = ROOM_HALF_M
        # (point on the wall, inward normal, direction that reads left -> right from inside)
        for point, normal, right in (((mid_x + r, 0, 0), (-1, 0, 0), (0, -1, 0)),
                                     ((mid_x - r, 0, 0), (1, 0, 0), (0, 1, 0)),
                                     ((mid_x, r, 0), (0, -1, 0), (1, 0, 0)),
                                     ((mid_x, -r, 0), (0, 1, 0), (-1, 0, 0))):
            point, normal, right = (np.array(a, float) for a in (point, normal, right))
            corner = point - r * right
            panel = gl.GLMeshItem(meshdata=gl.MeshData(
                vertexes=np.array([corner, corner + 2 * r * right, corner + 2 * r * right + ROOM_HEIGHT_M * z,
                                   corner + ROOM_HEIGHT_M * z]), faces=np.array([[0, 1, 2], [0, 2, 3]])),
                color=(0.87, 0.88, 0.91, 1.0), smooth=False, glOptions="opaque")
            b_corner = point + 0.01 * normal - 0.5 * BANNER_W_M * right + (BANNER_Z_M - 0.5 * BANNER_H_M) * z
            banner = _placed(gl.GLImageItem(logo, smooth=True), b_corner, right * BANNER_W_M / logo.shape[0],
                             z * BANNER_H_M / logo.shape[1], normal)
            self.walls.append(([panel, banner], point, normal))
            self.scene_items += [panel, banner]

    def _cull_walls(self) -> None:
        cam = self.view.cameraPosition()
        cam = np.array([cam.x(), cam.y(), cam.z()])
        for items, point, normal in getattr(self, "walls", []):
            inside = float(np.dot(cam - point, normal)) > 0
            for it in items:
                it.setVisible(inside)

    def reset_view(self) -> None:
        mid = np.vstack([to_display(np.asarray(p["corners"], float), self.frame) for p in self.plates]).mean(0)
        self.view.setCameraPosition(pos=QtGui.QVector3D(mid[0], 0.0, 0.7), distance=7.0, elevation=18, azimuth=-130)

    def _request_plates(self) -> None:
        """(Re)read the plate corners from QTM in the background: at start-up, on Start streaming and once a QTM
        capture runs (QTM serves them only while a measurement or preview is open)."""
        if self.plates_fut is None:
            self.plates_fut = self.plates_pool.submit(read_force_plates, self.cfg["realtime"]["qtm"]["host"])

    def _check_plates(self) -> None:
        if self.plates_fut is None or not self.plates_fut.done():
            return
        fut, self.plates_fut = self.plates_fut, None
        if fut.exception() is None:
            self.plates, self.plates_from = fut.result(), "QTM's force-plate settings"
        elif not self.plates_from.startswith("QTM"):        # keep plates read from QTM earlier
            self.plates_from = f"configs/base.yaml ({fut.exception()})"
        else:
            return
        self.frame = treadmill_frame(self.plates)
        self.plate_frames = plate_frames(self.plates, self.gcfg["grf"]["plate_axes"])
        self._draw_treadmill()

    # ------------------------------------------------------------------ treadmill speed
    def nudge(self, delta: float) -> None:
        base = self.commanded if self.commanded is not None else self.target.value()
        self.target.setValue(base + delta)
        self.set_speed()

    def set_speed(self) -> None:
        self._command_speed(self.target.value(), self.accel.value())

    def stop_belts(self) -> None:
        if self.tm is None and self.commanded is None and not self.speed_futs:
            self.speed_label.setText("belts not commanded from here")
            return
        d = float(self.gcfg["speed"]["stop_decel_m_s2"])
        self._command_speed(0.0, d, d)

    def _command_speed(self, speed: float, accel: float, decel: float | None = None) -> None:
        if self.tm is None:
            b = self.cfg["realtime"]["bertec"]
            self.tm = BertecTreadmill(b["host"], b["rpc_port"], self.gcfg["speed"]["max_m_s"],
                                      self.gcfg["speed"]["max_accel_m_s2"])
        self.speed_futs.append((self.tm.run(speed, accel, decel), speed, accel, time.perf_counter()))
        self.speed_label.setText(f"sending {speed:.2f} m/s at {accel:.2f} m/s² ...")
        self._save_settings()

    def _check_speed(self) -> None:
        while self.speed_futs and self.speed_futs[0][0].done():
            fut, speed, accel, t_sent = self.speed_futs.pop(0)
            if fut.exception() is not None:
                self.speed_label.setText(f"speed command FAILED: {fut.exception()}")
                continue
            self.commanded = fut.result()
            self.speed_label.setText(f"commanded {self.commanded:.2f} m/s at {accel:.2f} m/s²")
            if self.rec is not None:
                self.rec["speed_commands"].append({"t": round(t_sent, 4), "speed_m_s": self.commanded,
                                                   "accel_m_s2": accel})
                if self.rec["qtm_t"] is not None and self.qtm is not None and self.qtm.capturing:
                    self.qtm.event(f"speed_{self.commanded:.2f}")

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
            if self.model_boxes[self._name(path)].isChecked():
                self.engine.enable(self._spec(path))
        self.engine.start()
        self._request_plates()
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
        self.grf_view.hide()
        self._set_inputs_enabled(True)
        self.start_btn.setText("Start streaming")

    @staticmethod
    def _name(entry) -> str:
        return entry if isinstance(entry, str) else entry.name

    def _spec(self, entry):
        """What LiveEngine.enable() takes: a bundle folder, or the lab KF with the current plate frames."""
        return LabKfSpec(self.plate_frames) if entry == LabKfSpec.name else entry

    def toggle_model(self, path, on: bool) -> None:
        self._save_settings()
        if self.engine is None:
            return
        name = self._name(path)
        if on:
            try:
                self.engine.enable(self._spec(path))
            except Exception as e:  # noqa: BLE001
                QtWidgets.QMessageBox.critical(self, "Model", f"Could not load {name}:\n{e}")
                self.model_boxes[name].setChecked(False)
        else:
            info = self.engine.disable(name)
            if info and self.rec is not None:
                self.rec["logs"].append(info)
            view = self.views.pop(name, None)
            if view:
                view.remove(self.view)

    def toggle_record(self) -> None:
        if self.rec is not None:
            self.stop_record()
            return
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.use_qtm.isChecked():
            q = self.cfg["realtime"]["qtm"]
            self.qtm = QtmRecorder(q["host"], str(q.get("password") or ""))
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
        self.rec = {"stamp": stamp, "qtm_t": qtm_t, "logs": [], "started": dt.datetime.now(), "speed_commands": [],
                    "speed_at_start": self.commanded}
        if qtm_t is not None:
            self._request_plates()
            self.qtm.event("grf2com_log_start")
            if self.commanded is not None:
                self.qtm.event(f"speed_{self.commanded:.2f}")
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
               "speed_commands": [dict(c, t_from_qtm_start_s=round(c["t"] - rec["qtm_t"], 3))
                                  if rec["qtm_t"] is not None else c for c in rec.get("speed_commands", [])],
               "speed_at_start_m_s": rec.get("speed_at_start"),
               "display_frame": {"origin_lab_m": self.frame["origin"].round(5).tolist(), "sign": self.frame["sign"],
                                 "plates_from": self.plates_from},
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
        self._cull_walls()
        self._check_pending()
        self._check_plates()
        self._check_speed()
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
        trail_n = int(self.gcfg["trail_s"] * 60)
        scale = float(self.motion_scale.currentText().split()[0][1:])
        total_ms = 0.0
        for i, path in enumerate(self.bundles):
            name = self._name(path)
            m = snap["models"].get(name)
            label = self.model_labels[name]
            if m is None:
                label.setText("")
                continue
            if name not in self.views:
                self.views[name] = ModelView(self.view, self.plots, self.model_colors[name])
            self.views[name].update(m, self.frame, trail_n, scale)
            if np.isfinite(m["compute_ms"]):
                total_ms += m["compute_ms"]
            c = to_display(m["com"], self.frame)
            if "used" in m:                                 # lab KF
                if m.get("note") or not np.isfinite(c).all():
                    label.setText(m.get("note") or "waiting")
                    continue
                steps = m["steps"]
                step = f" step {to_display(steps[-1][1], self.frame)[0]:+.3f}" if steps else ""
                cmd = f"\n  would command {m['command'][0]:.2f} m/s at {m['command'][1]:.2f} m/s²" \
                    if m.get("command") else ""
                label.setText(f"AP {c[0]:+.3f}{step} m{cmd}")
                continue
            warm = "" if m["warm"] >= 1 else f"  warming {100 * m['warm']:.0f}%"
            label.setText(f"AP {c[0]:+.3f} ML {c[1]:+.3f} V {c[2]:.3f} m  {m['compute_ms']:.1f} ms{warm}")
        grf = self.gcfg["grf"]
        if self.show_grf.isChecked() and snap["state"] == "running":
            self.grf_view.update(snap["tared"], self.plate_frames, self.frame, float(grf["m_per_n"]),
                                 float(grf["min_fz_n"]))
        else:
            self.grf_view.hide()
        if len(snap["models"]):
            t = snap["t"]
            self.plots[0].setXRange(max(0.0, t - self.gcfg["history_s"]), max(t, 1.0), padding=0)
        lines = [f"state: {snap['state']}"]
        if snap["state"] == "taring":
            lines.append(f"tare {100 * snap['tare_progress']:.0f}%: keep both plates unloaded")
        if snap["fs"]:
            lines.append(f"input {snap['fs']:.0f} Hz, t {snap['t']:.1f} s, belt {snap['speed']:.2f} m/s (measured)")
            lines.append(f"Fz L {snap['fz'][0]:6.0f} N  R {snap['fz'][1]:6.0f} N")
            budget = 1000 / 60
            behind = snap["backlog"] > 0.1
            lines.append(f"models {total_ms:.1f} ms per frame (budget {budget:.1f})"
                         + (f"  BEHIND REAL TIME by {snap['backlog']:.1f} s" if behind
                            else "  BEHIND REAL TIME" if total_ms > 0.9 * budget else ""))
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
        s.remove("qtm_password")                 # older versions kept it here; it is in configs/base.yaml now
        s.setValue("target_speed", self.target.value())
        s.setValue("show_grf", "true" if self.show_grf.isChecked() else "false")
        s.setValue("accel", self.accel.value())
        s.setValue("models", ",".join(n for n, b in self.model_boxes.items() if b.isChecked()))
        for k, w in self.subject.items():
            s.setValue(k, w.value())

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt)
        if self.tm is not None:
            if self.commanded or self.speed_futs:       # the belts were commanded from here: ramp them down
                d = float(self.gcfg["speed"]["stop_decel_m_s2"])
                try:
                    self.tm.stop(d).result(timeout=5)
                except Exception as e:  # noqa: BLE001
                    QtWidgets.QMessageBox.warning(self, "Treadmill", f"Could not stop the belts:\n{e}")
            self.tm.close()
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
