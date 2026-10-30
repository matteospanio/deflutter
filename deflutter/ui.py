"""Main window: start page (recent projects) and a single workspace with a step panel.

Workflow: Analyze -> confirm References -> Detect anomalies -> Render preview -> Compare -> Export.
Long operations run in a QThreadPool worker (progress + cancel); the GUI thread only applies results.
"""
import copy
import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.signal import resample_poly

from . import __version__, audio, correction
from . import analysis as A
from .plot import PlotView
from .store import Store, cache_key

try:  # playback is optional: the app still works without a usable QtMultimedia backend
    from PySide6.QtMultimedia import QAudio, QAudioFormat, QAudioSink, QMediaDevices
except ImportError:
    QAudioSink = None

COLORS = ["#4fc3f7", "#aed581", "#f06292", "#ffd54f", "#ba68c8", "#4db6ac", "#ff8a65", "#90a4ae"]
DEFAULTS = dict(
    # analysis
    fmin=20.0, fmax=0.0, n_fft=8192, hop=1024, window="hann", prominence=6.0, search=15.0,
    # display
    freq_scale="log", dyn_range=90.0,
    # detection
    k_on=4.0, k_off=2.0, min_dur_ms=100.0, gap_ms=300.0, roll_ms=150.0, smooth_ms=0.0, coh_tol=25.0,
    # correction and export
    pv_fft=4096, pv_overlap=8, transient=True, tail_s=1.0, bit_depth="24",
)
DISPLAY_KEY = "display-v1"
AUDIO_FILTER = "Audio files (*.wav *.flac *.aif *.aiff *.ogg *.opus *.mp3);;All files (*)"


class Cancelled(Exception):
    pass


class _TaskSignals(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(int)


class _Task(QtCore.QRunnable):
    """Runs fn(progress) off the GUI thread; progress(fraction) raises Cancelled once cancel is set."""

    def __init__(self, fn, cancel):
        super().__init__()
        self.setAutoDelete(False)
        self.fn, self.cancel, self.sig = fn, cancel, _TaskSignals()

    def run(self):
        def progress(f):
            if self.cancel.is_set():
                raise Cancelled()
            self.sig.progress.emit(int(100 * f))

        try:
            result = self.fn(progress)
        except Cancelled:
            self.sig.failed.emit("Cancelled.")
        except Exception as exc:  # reported in the GUI, never lost
            traceback.print_exc()
            self.sig.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.sig.done.emit(result)


class Player(QtCore.QObject):
    """Plays part of a float array through QAudioSink as int16 PCM; optional looping."""

    def __init__(self, parent):
        super().__init__(parent)
        self.sink = self.buf = None
        self.args = None

    def play(self, x, sr, t0, t1, loop):
        self.stop()
        if QAudioSink is None:
            raise RuntimeError("QtMultimedia is not available: playback disabled")
        dev = QMediaDevices.defaultAudioOutput()
        if dev.isNull():
            raise RuntimeError("no audio output device")
        seg = x[int(t0 * sr):int(t1 * sr)]
        if seg.shape[1] > 2:
            seg = seg.mean(axis=1, keepdims=True)
        fmt = QAudioFormat()
        fmt.setSampleRate(sr)
        fmt.setChannelCount(seg.shape[1])
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        rate = sr
        if not dev.isFormatSupported(fmt):  # e.g. 96 kHz on a 48 kHz device
            rate = dev.preferredFormat().sampleRate()
            seg = resample_poly(seg, rate, sr, axis=0)
            fmt.setSampleRate(rate)
        self.bytes = QtCore.QByteArray((np.clip(seg, -1, 1) * 32767).astype("<i2").tobytes())
        self.buf = QtCore.QBuffer()
        self.buf.setData(self.bytes)
        self.buf.open(QtCore.QIODevice.OpenModeFlag.ReadOnly)
        self.sink = QAudioSink(dev, fmt)
        self.sink.stateChanged.connect(self._state)
        self.args = (x, sr, t0, t1, loop)
        self.sink.start(self.buf)

    def position(self):
        if self.sink is None:
            return None
        return self.args[2] + self.sink.processedUSecs() / 1e6

    def stop(self):
        if self.sink is not None:
            self.sink.stateChanged.disconnect(self._state)
            self.sink.stop()
        self.sink = self.buf = self.args = None

    def _state(self, state):
        if state == QAudio.State.IdleState:  # buffer drained
            if self.args and self.args[4]:
                self.play(*self.args)
            else:
                self.stop()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, data_dir=None):
        super().__init__()
        self.setWindowTitle("deflutter")
        self.resize(1500, 900)
        self.setAcceptDrops(True)
        root = data_dir or QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.StandardLocation.AppDataLocation)
        self.store = Store(root)
        self.pool = QtCore.QThreadPool.globalInstance()
        self.task = None
        self.player = Player(self)
        self._reset_project()
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_start())
        self.stack.addWidget(self._build_workspace())
        self.setCentralWidget(self.stack)
        self._build_menus()
        self._build_status()
        self.timer = QtCore.QTimer(self, interval=33, timeout=self._tick)
        self.show_start()

    # ================================================================ state
    def _reset_project(self):
        self.pid = None
        self.x = self.mono = self.sr = self.src_hash = None
        self.state = dict(params=dict(DEFAULTS), selection=None, analysis=None, tracks=[], events=[])
        self.disp = self.disp_corr = self.ana = None
        self.track_cache = {}  # (center, search, seed_t) -> (f_obs, conf) on the current analysis
        self.es, self.e_comb, self.thr, self.profiles = [], None, None, []
        self.y = self.e_after = self.corr_id = None
        self.undo_stack, self.redo_stack = [], []

    @property
    def params(self):
        return self.state["params"]

    @property
    def duration(self):
        return len(self.x) / self.sr if self.x is not None else 0.0

    def _span(self):
        sel = self.state["selection"]
        return tuple(sel) if sel else (0.0, self.duration)

    def _ana_params(self):
        p = self.params
        fmax = p["fmax"] if p["fmax"] > 0 else self.sr / 2
        return dict(n_fft=int(p["n_fft"]), hop=int(p["hop"]), window=p["window"], fmin=p["fmin"], fmax=fmax)

    # ================================================================ start page
    def _build_start(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(60, 40, 60, 40)
        title = QtWidgets.QLabel("<h1>deflutter</h1><p>Reference-tone based correction of frequency drift "
                                 "(wow, flutter, speed anomalies) in audio recordings.</p>"
                                 "<p>Drop an audio file anywhere on this window, or start a new project.</p>")
        lay.addWidget(title)
        row = QtWidgets.QHBoxLayout()
        new = QtWidgets.QPushButton("New project from audio file…")
        new.clicked.connect(lambda: self.new_project())
        row.addWidget(new)
        row.addStretch()
        lay.addLayout(row)
        lay.addWidget(QtWidgets.QLabel("<b>Recent projects</b> (double-click to open)"))
        self.recent = QtWidgets.QListWidget()
        self.recent.itemDoubleClicked.connect(lambda it: self.open_project(it.data(QtCore.Qt.ItemDataRole.UserRole)))
        lay.addWidget(self.recent, 1)
        btns = QtWidgets.QHBoxLayout()
        for text, fn in (("Open", self._open_selected), ("Remove from list", self._delete_selected)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        btns.addStretch()
        lay.addLayout(btns)
        return w

    def show_start(self):
        self.player.stop()
        self.recent.clear()
        for row in self.store.recent():
            it = QtWidgets.QListWidgetItem(f"{row['name']}    —    {row['source_path']}    ({row['updated_at']})")
            it.setData(QtCore.Qt.ItemDataRole.UserRole, row["id"])
            self.recent.addItem(it)
        self.stack.setCurrentIndex(0)

    def _open_selected(self):
        it = self.recent.currentItem()
        if it:
            self.open_project(it.data(QtCore.Qt.ItemDataRole.UserRole))

    def _delete_selected(self):
        it = self.recent.currentItem()
        if it and self._ask(f"Remove project '{it.text().split('    ')[0]}'? The audio file is not touched."):
            self.store.delete_project(it.data(QtCore.Qt.ItemDataRole.UserRole))
            self.show_start()

    # ================================================================ workspace
    def _build_workspace(self):
        self.widgets = {}  # param key -> widget
        panel = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(panel)

        g, f = self._group(col, "1 · Analysis")
        f.addRow("Band min", self._spin("fmin", 0, 200000, 10, 1, " Hz"))
        fmax = self._spin("fmax", 0, 200000, 100, 1, " Hz")
        fmax.setSpecialValueText("Nyquist")
        f.addRow("Band max", fmax)
        f.addRow("FFT size", self._combo("n_fft", [2048, 4096, 8192, 16384, 32768]))
        f.addRow("Hop", self._combo("hop", [128, 256, 512, 1024, 2048]))
        f.addRow("Window", self._combo("window", ["hann", "blackmanharris"]))
        f.addRow("Min prominence", self._spin("prominence", 1, 60, 1, 1, " dB"))
        self.span_label = QtWidgets.QLabel()
        self.span_label.setWordWrap(True)
        f.addRow(self.span_label)
        f.addRow(self._button("Analyze selection", self.analyze, "Track the tonal components inside the "
                              "selection (whole file if none) and propose references"))
        f.addRow("Frequency scale", self._combo("freq_scale", ["linear", "log", "mel"]))
        f.addRow("Display range", self._spin("dyn_range", 20, 160, 10, 0, " dB"))

        g, f = self._group(col, "2 · References")
        self.tracks_table = self._table(["On", "f0 (Hz)", "Search ±%", "Type", "Conf (dB)", "Peak (cents)"])
        self.tracks_table.itemChanged.connect(self._track_edited)
        f.addRow(self.tracks_table)
        hint = QtWidgets.QLabel("Double-click the spectrogram to add a reference. Only <i>moving</i> tones are "
                                "enabled by default; <i>fixed</i> tones (machine tones?) are excluded from the "
                                "estimate, but the global correction will shift them too during anomalies.")
        hint.setWordWrap(True)
        f.addRow(hint)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._button("Re-estimate f0", self._reestimate_f0, "Robust f0 for the selected rows"))
        row.addWidget(self._button("Remove", self._remove_tracks, "Remove the selected references"))
        f.addRow(row)

        g, f = self._group(col, "3 · Detection")
        f.addRow("Trigger", self._spin("k_on", 0.5, 50, 0.5, 1, " × σ"))
        f.addRow("Release", self._spin("k_off", 0.5, 50, 0.5, 1, " × σ"))
        f.addRow("Min duration", self._spin("min_dur_ms", 0, 10000, 10, 0, " ms"))
        f.addRow("Gap fill", self._spin("gap_ms", 0, 10000, 50, 0, " ms"))
        f.addRow("Pre/post-roll", self._spin("roll_ms", 0, 5000, 10, 0, " ms"))
        smooth = self._spin("smooth_ms", 0, 5000, 10, 0, " ms")
        smooth.setSpecialValueText("auto")
        f.addRow("Smoothing", smooth)
        f.addRow("Coherence tol.", self._spin("coh_tol", 1, 200, 5, 0, " %"))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._button("Detect anomalies", self.detect, "Hysteresis detection on the combined deviation"))
        row.addWidget(self._button("Add from selection", self._add_event, "Add an event covering the selection"))
        f.addRow(row)
        self.events_table = self._table(["On", "Start (s)", "End (s)", "Peak (cents)", "Check", "Intensity"])
        self.events_table.itemChanged.connect(self._event_edited)
        self.events_table.cellDoubleClicked.connect(self._event_zoom)
        f.addRow(self.events_table)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._button("Enable all", self._enable_events, "Force-enable every event"))
        row.addWidget(self._button("Delete", self._delete_events, "Delete the selected events"))
        f.addRow(row)

        g, f = self._group(col, "4 · Correction")
        f.addRow("FFT size", self._combo("pv_fft", [1024, 2048, 4096, 8192]))
        f.addRow("Overlap", self._combo("pv_overlap", [4, 8, 16]))
        f.addRow("Transient reset", self._check("transient"))
        f.addRow("Resync time", self._spin("tail_s", 0.1, 10, 0.1, 2, " s"))
        f.addRow(self._button("Render preview", self.render, "Phase-vocoder correction of the enabled events"))
        self.residual_label = QtWidgets.QLabel("No preview rendered.")
        self.residual_label.setWordWrap(True)
        f.addRow(self.residual_label)

        g, f = self._group(col, "5 · Compare")
        self.play_btn = self._button("Play / Stop  (Space)", self.toggle_play, "Plays the selection, else the view")
        f.addRow(self.play_btn)
        self.source = self._radios(f, "Listen", ["Original", "Corrected", "Difference"], self._source_changed)
        self.view_src = self._radios(f, "Spectrogram", ["Original", "Corrected"],
                                     lambda: (self._show_spectrogram(), self._refresh_plots()))
        self.loop = QtWidgets.QCheckBox("Loop selection  (L)")
        f.addRow(self.loop)

        g, f = self._group(col, "6 · Export")
        f.addRow("Bit depth", self._combo("bit_depth", ["16", "24", "32f"]))
        f.addRow(self._button("Export…", self.export, "Write the corrected audio (WAV or FLAC)"))
        col.addStretch()

        scroll = self.panel = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        self.plot = PlotView()
        self.plot.selectionChanged.connect(self._selection_changed)
        self.plot.referenceRequested.connect(self._add_reference)
        self.plot.eventEdited.connect(self._event_dragged)
        self.plot.cursorInfo.connect(lambda s: self.readout.setText(s))
        split = QtWidgets.QSplitter()
        split.addWidget(scroll)
        split.addWidget(self.plot)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 1080])
        return split

    def _group(self, layout, title):
        g = QtWidgets.QGroupBox(title)
        f = QtWidgets.QFormLayout(g)
        layout.addWidget(g)
        return g, f

    def _button(self, text, fn, tip=""):
        b = QtWidgets.QPushButton(text)
        b.setToolTip(tip)
        b.clicked.connect(lambda: fn())
        return b

    def _table(self, headers):
        t = QtWidgets.QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().hide()
        t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        t.setMinimumHeight(150)
        return t

    def _radios(self, form, label, names, fn):
        box = QtWidgets.QHBoxLayout()
        group = QtWidgets.QButtonGroup(self)
        for i, n in enumerate(names):
            rb = QtWidgets.QRadioButton(n)
            rb.setChecked(i == 0)
            group.addButton(rb, i)
            box.addWidget(rb)
        group.idClicked.connect(lambda _: fn())
        form.addRow(label, box)
        return group

    def _spin(self, key, lo, hi, step, decimals, suffix=""):
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(decimals)
        w.setSuffix(suffix)
        w.setKeyboardTracking(False)
        w.valueChanged.connect(lambda v: self._set_param(key, v))
        self.widgets[key] = w
        return w

    def _combo(self, key, values):
        w = QtWidgets.QComboBox()
        for v in values:
            w.addItem(str(v), v)
        w.currentIndexChanged.connect(lambda i: self._set_param(key, w.itemData(i)))
        self.widgets[key] = w
        return w

    def _check(self, key):
        w = QtWidgets.QCheckBox()
        w.toggled.connect(lambda v: self._set_param(key, v))
        self.widgets[key] = w
        return w

    def _params_to_widgets(self):
        for key, w in self.widgets.items():
            w.blockSignals(True)
            v = self.params[key]
            if isinstance(w, QtWidgets.QDoubleSpinBox):
                w.setValue(v)
            elif isinstance(w, QtWidgets.QComboBox):
                w.setCurrentIndex(max(w.findData(v), 0))
            else:
                w.setChecked(v)
            w.blockSignals(False)

    def _set_param(self, key, value):
        self.params[key] = value
        if key in ("freq_scale", "dyn_range") and self.x is not None:
            self._show_spectrogram()
            self._refresh_plots()
            if key == "freq_scale":  # the y coordinate system changed
                self.plot.set_freq_range(20.0, self.sr / 2)
        elif key in ("fmin", "fmax", "n_fft", "hop", "window") and self.ana is not None:
            self._span_info()
        self._save()

    # ================================================================ menus / status
    def _build_menus(self):
        def act(menu, text, fn, key=None):
            a = QtGui.QAction(text, self)
            if key:
                a.setShortcut(QtGui.QKeySequence(key))
            a.triggered.connect(lambda: fn())
            menu.addAction(a)
            return a

        m = self.menuBar().addMenu("&File")
        act(m, "New project from audio file…", self.new_project, "Ctrl+N")
        act(m, "Recent projects", self.show_start, "Ctrl+O")
        act(m, "Export…", self.export, "Ctrl+E")
        m.addSeparator()
        act(m, "Quit", self.close, "Ctrl+Q")
        m = self.menuBar().addMenu("&Edit")
        act(m, "Undo", self.undo, "Ctrl+Z")
        act(m, "Redo", self.redo, "Ctrl+Shift+Z")
        act(m, "Select all", lambda: self._selection_changed((0.0, self.duration)), "Ctrl+A")
        act(m, "Clear selection", lambda: self._selection_changed(None), "Esc")
        m = self.menuBar().addMenu("&View")
        act(m, "Zoom to selection", lambda: self.plot.zoom_to(*self._span()), "F")
        act(m, "Zoom to whole file", self._zoom_all, "0")
        m = self.menuBar().addMenu("&Playback")
        act(m, "Play / Stop", self.toggle_play, "Space")
        act(m, "Toggle original / corrected", self._toggle_ab, "B")
        act(m, "Toggle loop", self.loop.toggle, "L")

    def _build_status(self):
        sb = self.statusBar()
        self.readout = QtWidgets.QLabel()
        sb.addWidget(self.readout, 1)
        self.progress = QtWidgets.QProgressBar(maximumWidth=200)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(lambda: self.task and self.task.cancel.set())
        for w in (self.progress, self.cancel_btn):
            sb.addPermanentWidget(w)
            w.hide()

    def _msg(self, text, ms=8000):
        self.statusBar().showMessage(text, ms)

    def _ask(self, text):
        return QtWidgets.QMessageBox.question(self, "deflutter", text) == QtWidgets.QMessageBox.StandardButton.Yes

    def _warn(self, text):
        QtWidgets.QMessageBox.warning(self, "deflutter", text)

    # ================================================================ tasks
    def run_task(self, label, fn, on_done):
        """Run fn(progress) in the thread pool, one task at a time; on_done(result) runs in the GUI thread."""
        if self.task is not None:
            self._msg("Another operation is still running.")
            return
        self.task = _Task(fn, threading.Event())
        self.task.sig.progress.connect(self.progress.setValue)
        self.task.sig.done.connect(lambda res: self._task_finished(on_done, res))
        self.task.sig.failed.connect(lambda err: self._task_finished(None, err))
        self.progress.setValue(0)
        self.progress.setFormat(f"{label} %p%")
        self.progress.show()
        self.cancel_btn.show()
        self._msg(f"{label}…", 0)
        self.pool.start(self.task)

    def _task_finished(self, on_done, result):
        self.task = None
        self.progress.hide()
        self.cancel_btn.hide()
        self.statusBar().clearMessage()
        if on_done is None:
            self._msg(result)
            if result != "Cancelled.":
                self._warn(result)
            return
        try:
            on_done(result)
        except Exception as exc:
            traceback.print_exc()
            self._warn(f"{type(exc).__name__}: {exc}")

    def wait_idle(self, timeout_ms=600000):
        """Block until the running task has delivered its result (used by scripted/offscreen runs)."""
        timer = QtCore.QElapsedTimer()
        timer.start()
        while self.task is not None and timer.elapsed() < timeout_ms:
            QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)

    # ================================================================ projects
    def new_project(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open audio file", "", AUDIO_FILTER)
            if not path:
                return
        self._open(Path(path), None, None)

    def open_project(self, pid):
        row = self.store.project(pid)
        path = Path(row["source_path"])
        if not path.exists():
            new, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Locate {path.name}", "", AUDIO_FILTER)
            if not new:
                return
            path = Path(new)
            self.store.relocate(pid, path)
        self._open(path, row, json.loads(row["state_json"]))

    def _open(self, path, row, state):
        """Load audio, its display spectrogram and (for saved projects) the cached analysis."""
        self.player.stop()
        store = self.store

        def job(progress):
            x, sr = audio.load(path)
            h = audio.file_hash(path)
            mono = x.mean(axis=1)
            key = cache_key(h, DISPLAY_KEY)
            arr = store.load_arrays(key)
            if arr is None:
                t, f, db = A.display_spectrogram(mono, sr, progress=lambda v: progress(0.6 * v))
                store.save_arrays(key, t=t, f=f, db=db)
                arr = dict(t=t, f=f, db=db)
            ana = None
            if state and state.get("analysis"):
                p = dict(DEFAULTS, **state["params"])
                fmax = p["fmax"] if p["fmax"] > 0 else sr / 2
                ana = self._stft_cached(h, mono, sr, state["analysis"], dict(
                    n_fft=int(p["n_fft"]), hop=int(p["hop"]), window=p["window"], fmin=p["fmin"], fmax=fmax),
                    lambda v: progress(0.6 + 0.4 * v))
            return x, sr, h, (arr["t"], arr["f"], arr["db"]), ana

        def done(res):
            x, sr, h, disp, ana = res
            if row is not None and h != row["source_hash"]:
                self._warn("The audio file has changed since this project was created; cached analyses "
                           "are recomputed.")
            self._reset_project()
            self.x, self.sr, self.src_hash, self.disp = x, sr, h, disp
            self.mono = x.mean(axis=1)
            if row is None:
                self.pid = self.store.create_project(path.stem, path, h, sr, x.shape[1], len(x), self.state)
            else:
                self.pid = row["id"]
                self.state.update(state)
                self.state["params"] = dict(DEFAULTS, **state.get("params", {}))
                if ana is not None:
                    self.ana = ana
            self.setWindowTitle(f"deflutter — {path.name}  ({sr} Hz, {x.shape[1]} ch, {self.duration:.1f} s)")
            self._params_to_widgets()
            self.stack.setCurrentIndex(1)
            self.plot.duration = self.duration
            self._show_spectrogram()
            self.plot.reset_view(self.duration, 20.0, sr / 2)
            self.plot.set_selection(self.state["selection"])
            self._update(save=False)
            if row is not None and self.store.active_correction(self.pid) and self.state["events"]:
                self.render()  # restores the last preview (cache hit: no phase vocoder run)

        self.run_task(f"Loading {path.name}", job, done)

    def _stft_cached(self, h, mono, sr, span, p, progress=None):
        key = cache_key(h, round(span[0], 4), round(span[1], 4), p)
        arr = self.store.load_arrays(key)
        if arr is None:
            t, f, db = A.stft_db(mono, sr, t0=span[0], t1=span[1], progress=progress, **p)
            self.store.save_arrays(key, t=t, f=f, db=db)
            return t, f, db
        return arr["t"], arr["f"], arr["db"]

    def _save(self):
        if self.pid is not None:
            self.store.save_state(self.pid, self.state)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        urls = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.new_project(urls[0])

    # ================================================================ selection
    def _selection_changed(self, span):
        if self.x is None:
            return
        if span is not None and span[1] - span[0] < 0.5:
            span = None  # a click or a tiny drag clears the selection
            self._msg("Selection cleared (minimum length 0.5 s).")
        self.state["selection"] = list(span) if span else None
        self.plot.set_selection(span)
        self._span_info()
        self._save()

    def _span_info(self):
        t0, t1 = self._span()
        text = "Whole file" if self.state["selection"] is None else f"Selection {t0:.2f} – {t1:.2f} s"
        a = self.state["analysis"]
        if a is not None and (abs(a[0] - t0) > 1e-3 or abs(a[1] - t1) > 1e-3):
            text += f"<br><b>Analysis covers {a[0]:.2f} – {a[1]:.2f} s: press Analyze to update.</b>"
        self.span_label.setText(text)

    def _zoom_all(self):
        self.plot.spec.setXRange(0, self.duration, padding=0)

    # ================================================================ step 1: analysis
    def analyze(self):
        if self.x is None:
            return
        span, p = self._span(), self._ana_params()
        prominence, search = self.params["prominence"], self.params["search"]
        mono, sr, h = self.mono, self.sr, self.src_hash

        def job(progress):
            ana = self._stft_cached(h, mono, sr, span, p, lambda v: progress(0.7 * v))
            progress(0.75)
            return ana, A.propose_references(ana[1], ana[2], prominence, search), span

        self.run_task("Analyzing", job, self._analyzed)

    def _analyzed(self, res):
        ana, props, span = res
        self._push_undo()
        self.ana, self.track_cache = ana, {}
        self.state["analysis"] = list(span)
        tracks = [tr for tr in self.state["tracks"] if tr["user"]]  # keep what the user added or edited
        for p in props:
            if any(abs(p["f0"] / tr["f0"] - 1) < 5e-3 for tr in tracks):
                continue
            self.track_cache[(p["center"], self.params["search"], None)] = (p["f_obs"], p["conf"])
            tracks.append(dict(center=p["center"], f0=p["f0"], search=self.params["search"], kind=p["kind"],
                               enabled=p["kind"] == "moving", user=False, color=""))
        for i, tr in enumerate(tracks):
            tr["color"] = COLORS[i % len(COLORS)]
        self.state["tracks"], self.state["events"] = tracks, []
        self._clear_render()
        self._update()
        kinds = [tr["kind"] for tr in tracks]
        self._msg(f"{len(tracks)} references: {kinds.count('moving')} moving, {kinds.count('fixed')} fixed.")
        if "moving" not in kinds:
            self._warn("No moving reference found in the analysed range. If the selection lies inside an "
                       "anomaly, include some stable audio around it, or double-click the drifting tone on "
                       "the spectrogram and type its nominal f0 in the References table.")

    # ================================================================ derived data and refresh
    def _update(self, save=True):
        """Recompute everything derived from state + cached analysis (cheap: no STFT), then refresh."""
        self.es, self.e_comb, self.thr, self.profiles = [], None, None, []
        if self.ana is not None:
            for tr in self.state["tracks"]:
                f_obs = self._track(tr)[0]
                tr["kind"] = A.classify(f_obs, tr["f0"])
                self.es.append(f_obs / tr["f0"] - 1)
            on = [e for e, tr in zip(self.es, self.state["tracks"]) if tr["enabled"]]
            if on:
                p = self.params
                self.e_comb = A.combine(on)
                self.thr = A.thresholds(self.e_comb, p["k_on"], p["k_off"])
                self._derive_events()
        self._refresh_tables()
        self._refresh_plots()
        self._span_info()
        if save:
            self._save()

    def _derive_events(self):
        t = self.ana[0]
        p = self.params
        tracks = self.state["tracks"]
        labels = [f"{tr['f0']:.0f} Hz" for tr in tracks if tr["enabled"]]
        es = [e for e, tr in zip(self.es, tracks) if tr["enabled"]]
        fixed = [f"{tr['f0']:.0f}" for tr in tracks if tr["kind"] == "fixed"]
        a0, a1 = self.state["analysis"]
        for ev in self.state["events"]:
            status, msg = A.coherence(t, es, labels, self.e_comb, ev["start"], ev["end"], self.thr[1],
                                      tol=p["coh_tol"] / 100)
            if fixed:
                msg += f". Fixed tones not following the deviation: {', '.join(fixed)} Hz"
            ev["status"], ev["msg"] = status, msg
            open_ends = (ev["start"] <= t[0] + 1e-6 and a0 <= 0, ev["end"] >= t[-1] - 1e-6 and a1 >= self.duration)
            tt, ee = A.fit_profile(t, self.e_comb, ev["start"], ev["end"], p["roll_ms"] / 1000,
                                   p["smooth_ms"] / 1000, open_ends, (0.0, self.duration))
            self.profiles.append((tt, ee * ev["gain"]))

    def _refresh_plots(self):
        if self.x is None:
            return
        tracks = [dict(f_obs=self._track(tr)[0], f0=tr["f0"], color=tr["color"],
                       enabled=tr["enabled"], e=e) for tr, e in zip(self.state["tracks"], self.es)]  # es: [] w/o analysis
        on = [p for p, ev in zip(self.profiles, self.state["events"]) if ev["enabled"]]
        t = self.ana[0] if self.ana is not None else None
        corrected_view = self.view_src.checkedId() == 1 and self.disp_corr is not None
        self.plot.set_curves(tracks, t, self.e_comb, self.thr, on, self.e_after, show_tracks=not corrected_view)
        self.plot.set_events([(ev["start"], ev["end"], ev["enabled"]) for ev in self.state["events"]])

    def _show_spectrogram(self):
        if self.disp is None:
            return
        corrected = self.view_src.checkedId() == 1 and self.disp_corr is not None
        t, f, db = self.disp_corr if corrected else self.disp
        self.plot.set_spectrogram(t, f, db, self.params["freq_scale"], self.params["dyn_range"])

    def _refresh_tables(self):
        tb = self.tracks_table
        tb.blockSignals(True)
        tb.setRowCount(len(self.state["tracks"]))
        for i, (tr, e) in enumerate(zip(self.state["tracks"], self.es or [None] * len(self.state["tracks"]))):
            conf = self._track(tr)[1] if self.ana is not None else None
            peak = np.nanmax(np.abs(A.cents(e))) if e is not None and np.isfinite(e).any() else float("nan")
            cells = ["", f"{tr['f0']:.2f}", f"{tr['search']:.1f}", tr["kind"],
                     f"{np.median(conf):.1f}" if conf is not None else "", f"{peak:.1f}"]
            for c, text in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(text)
                flags = QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
                if c in (1, 2):
                    flags |= QtCore.Qt.ItemFlag.ItemIsEditable
                if c == 0:
                    flags |= QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    it.setCheckState(QtCore.Qt.CheckState.Checked if tr["enabled"] else QtCore.Qt.CheckState.Unchecked)
                    it.setBackground(QtGui.QColor(tr["color"]))
                it.setFlags(flags)
                tb.setItem(i, c, it)
        tb.blockSignals(False)

        tb = self.events_table
        tb.blockSignals(True)
        tb.setRowCount(len(self.state["events"]))
        colors = dict(coherent="#2e7d32", unverified="#9e7a00", incoherent="#b71c1c")
        for i, (ev, (tt, ee)) in enumerate(zip(self.state["events"], self.profiles or [((), ())] * len(self.state["events"]))):
            peak = A.cents(ee[np.argmax(np.abs(ee))]) if len(ee) else 0.0
            cells = ["", f"{ev['start']:.3f}", f"{ev['end']:.3f}", f"{peak:+.1f}", ev.get("status", ""),
                     f"{ev['gain']:.2f}"]
            for c, text in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(text)
                flags = QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
                if c in (1, 2, 5):
                    flags |= QtCore.Qt.ItemFlag.ItemIsEditable
                if c == 0:
                    flags |= QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    it.setCheckState(QtCore.Qt.CheckState.Checked if ev["enabled"] else QtCore.Qt.CheckState.Unchecked)
                if c == 4:
                    it.setToolTip(ev.get("msg", ""))
                    it.setForeground(QtGui.QColor("white"))
                    it.setBackground(QtGui.QColor(colors.get(ev.get("status"), "#555")))
                it.setFlags(flags)
                tb.setItem(i, c, it)
        tb.blockSignals(False)

    # ================================================================ step 2: references
    def _track(self, tr):
        """(f_obs, conf) of a reference on the current analysis, cached."""
        key = (tr["center"], tr["search"], tr.get("seed_t"))
        if key not in self.track_cache:
            t, f, db = self.ana
            seed = None if key[2] is None else int(np.clip(np.searchsorted(t, key[2]), 0, len(t) - 1))
            self.track_cache[key] = A.track(f, db, tr["center"], tr["search"], seed=seed)
        return self.track_cache[key]

    def _add_reference(self, t_click, f):
        """Double-click: follow the tone under the cursor, starting from the clicked frame."""
        if self.ana is None:
            self._msg("Run Analyze first, then double-click a tone to add it as a reference.")
            return
        freqs = self.ana[1]
        if not freqs[0] < f < freqs[-1]:
            self._msg(f"{f:.0f} Hz is outside the analysis band: widen it and Analyze again.")
            return
        n = len(self.state["tracks"])
        tr = dict(center=f, f0=0.0, search=self.params["search"], seed_t=t_click, kind="", enabled=True,
                  user=True, color=COLORS[n % len(COLORS)])
        tr["f0"] = A.estimate_f0(self._track(tr)[0])
        if not np.isfinite(tr["f0"]):
            self._msg(f"No tone found near {f:.0f} Hz.")
            return
        self._push_undo()
        self.state["tracks"].append(tr)
        self._update()

    def _track_edited(self, item):
        tr = self.state["tracks"][item.row()]
        c = item.column()
        try:
            if c == 0:
                new = dict(enabled=item.checkState() == QtCore.Qt.CheckState.Checked)
            elif c == 1:
                new = dict(f0=float(item.text()), user=True)
            elif c == 2:
                new = dict(search=min(max(float(item.text()), 0.1), 50.0), user=True)
            else:
                return
        except ValueError:
            self._update(save=False)
            return
        self._push_undo()
        tr.update(new)
        self._update()

    def _selected_rows(self, table):
        return sorted({i.row() for i in table.selectedIndexes()})

    def _reestimate_f0(self):
        rows = self._selected_rows(self.tracks_table) or range(len(self.state["tracks"]))
        self._push_undo()
        for i in rows:
            tr = self.state["tracks"][i]
            tr["f0"] = A.estimate_f0(self._track(tr)[0])
        self._update()

    def _remove_tracks(self):
        rows = self._selected_rows(self.tracks_table)
        if rows:
            self._push_undo()
            self.state["tracks"] = [tr for i, tr in enumerate(self.state["tracks"]) if i not in rows]
            self._update()

    # ================================================================ step 3: detection
    def detect(self):
        if self.e_comb is None:
            self._msg("Analyze first and enable at least one reference.")
            return
        p = self.params
        t = self.ana[0]
        spans = A.detect_events(t, self.e_comb, *self.thr, p["min_dur_ms"] / 1000, p["gap_ms"] / 1000,
                                p["roll_ms"] / 1000)
        self._push_undo()
        self.state["events"] = [dict(start=s, end=e, enabled=True, gain=1.0, user=False) for s, e in spans]
        self._update(save=False)
        # global correction only where the references agree; a single reference is trusted only if no
        # fixed tone contradicts it (doc: never apply a global correction to an inconsistent deviation)
        fixed = any(tr["kind"] == "fixed" for tr in self.state["tracks"])
        for ev in self.state["events"]:
            ev["enabled"] = ev["status"] == "coherent" or (ev["status"] == "unverified" and not fixed)
        self._clear_render()
        self._update()
        off = sum(not ev["enabled"] for ev in self.state["events"])
        self._msg(f"{len(spans)} anomalies detected" + (f", {off} left disabled (see Check column)." if off else "."))

    def _add_event(self):
        if self.e_comb is None or self.state["selection"] is None:
            self._msg("Analyze and select a time range first.")
            return
        t0, t1 = self._span()
        self._push_undo()
        self.state["events"].append(dict(start=t0, end=t1, enabled=True, gain=1.0, user=True))
        self.state["events"].sort(key=lambda ev: ev["start"])
        self._update()

    def _event_edited(self, item):
        ev = self.state["events"][item.row()]
        c = item.column()
        try:
            if c == 0:
                new = dict(enabled=item.checkState() == QtCore.Qt.CheckState.Checked)
            elif c in (1, 2):
                new = {("start", "end")[c - 1]: float(item.text()), "user": True}
            elif c == 5:
                new = dict(gain=min(max(float(item.text()), 0.0), 2.0), user=True)
            else:
                return
        except ValueError:
            self._update(save=False)
            return
        self._push_undo()
        ev.update(new)
        if ev["end"] <= ev["start"]:
            ev["start"], ev["end"] = ev["end"], ev["start"]
        self._update()

    def _event_dragged(self, i, start, end):
        self._push_undo()
        self.state["events"][i].update(start=start, end=end, user=True)
        self._update()

    def _event_zoom(self, row, col):
        if col in (3, 4):
            ev = self.state["events"][row]
            self.plot.zoom_to(ev["start"], ev["end"])

    def _enable_events(self):
        self._push_undo()
        for ev in self.state["events"]:
            ev["enabled"] = True
        self._update()

    def _delete_events(self):
        rows = self._selected_rows(self.events_table)
        if rows:
            self._push_undo()
            self.state["events"] = [ev for i, ev in enumerate(self.state["events"]) if i not in rows]
            self._update()

    # ================================================================ undo
    def _push_undo(self):
        self.undo_stack = self.undo_stack[-99:] + [copy.deepcopy((self.state["tracks"], self.state["events"]))]
        self.redo_stack.clear()

    def _restore(self, src, dst):
        if src:
            dst.append(copy.deepcopy((self.state["tracks"], self.state["events"])))
            self.state["tracks"], self.state["events"] = src.pop()
            self._update()

    def undo(self):
        self._restore(self.undo_stack, self.redo_stack)

    def redo(self):
        self._restore(self.redo_stack, self.undo_stack)

    # ================================================================ step 4: correction
    def _clear_render(self):
        self.y = self.disp_corr = self.e_after = self.corr_id = None
        self.residual_label.setText("No preview rendered.")
        if self.view_src.checkedId() == 1:
            self._show_spectrogram()

    def render(self):
        on = [(p, ev) for p, ev in zip(self.profiles, self.state["events"]) if ev["enabled"]]
        if not on:
            self._msg("No enabled anomaly to correct.")
            return
        p = self.params
        t = self.ana[0]
        profiles = [pr for pr, _ in on]
        spans = [(float(tt[0]), float(tt[-1])) for tt, _ in profiles]
        r = A.ratio_curve(t, profiles)
        pv = dict(n_fft=int(p["pv_fft"]), overlap=int(p["pv_overlap"]), tail_s=p["tail_s"],
                  transient_db=6.0 if p["transient"] else 0.0)
        key = cache_key(self.src_hash, t, r, spans, pv)
        x, sr, store = self.x, self.sr, self.store
        span = self.state["analysis"]
        ap = self._ana_params()
        ap["hop"] = int(round((t[1] - t[0]) * sr))  # the effective analysis hop (may have been raised)
        refs = [(tr["center"], tr["search"], tr["f0"]) for tr in self.state["tracks"] if tr["enabled"]]

        def job(progress):
            arr = store.load_arrays(key)
            if arr is None:
                y, regions = correction.render(x, sr, t, r, spans, progress=lambda v: progress(0.8 * v), **pv)
                store.save_arrays(key, regions=np.array(regions, dtype=np.int64).reshape(-1, 2),
                                  data=np.concatenate([y[a:b] for a, b in regions]))
            else:  # cached: only the rewritten sample ranges are stored
                y, off = x.copy(), 0
                for a, b in arr["regions"]:
                    y[a:b] = arr["data"][off:off + b - a]
                    off += b - a
            progress(0.85)
            ymono = y.mean(axis=1)
            with ThreadPoolExecutor(2) as ex:  # corrected display + residual analysis in parallel
                fd = ex.submit(A.display_spectrogram, ymono, sr)
                fa = ex.submit(A.stft_db, ymono, sr, t0=span[0], t1=span[1], **ap)
                disp, (_, f2, db2) = fd.result(), fa.result()
            e_after = A.combine([A.track(f2, db2, c, s)[0] / f0 - 1 for c, s, f0 in refs])
            return y, disp, e_after

        def done(res):
            self.y, self.disp_corr, self.e_after = res
            before, after, red = A.residual(t, self.e_comb, self.e_after, spans)
            self.residual_label.setText(
                f"Residual deviation in the corrected events: {A.cents(before):.2f} → {A.cents(after):.2f} "
                f"cents RMS (−{red:.1f} dB). Length unchanged: {len(self.y)} samples.")
            self.corr_id = self.store.add_correction(self.pid, key, dict(params=pv, spans=spans),
                                                     self.store.cache_path(key))
            self._show_spectrogram()
            self._refresh_plots()

        self.run_task("Rendering", job, done)

    # ================================================================ step 5: compare / playback
    def _listen_data(self):
        src = self.source.checkedId()
        if src == 0 or self.y is None:
            if src:
                self._msg("No preview rendered yet: playing the original.")
            return self.x
        return self.y if src == 1 else self.x - self.y

    def toggle_play(self):
        if self.x is None:
            return
        if self.player.sink is not None:
            self.player.stop()
            self.timer.stop()
            self.plot.set_playhead(None)
            return
        if self.state["selection"]:
            t0, t1 = self._span()
        else:
            t0, t1 = max(self.plot.spec.vb.viewRange()[0][0], 0.0), self.duration
        self._play(t0, t1)

    def _play(self, t0, t1):
        try:
            self.player.play(self._listen_data(), self.sr, t0, t1, self.loop.isChecked() and bool(self.state["selection"]))
        except RuntimeError as exc:
            self._warn(str(exc))
            return
        self.timer.start()

    def _toggle_ab(self):
        self.source.button(0 if self.source.checkedId() else 1).setChecked(True)
        self._source_changed()

    def _source_changed(self):
        pos = self.player.position()
        if pos is not None and pos < self.player.args[3]:  # switch source at the current position
            self._play(pos, self.player.args[3])

    def _tick(self):
        pos = self.player.position()
        self.plot.set_playhead(pos)
        if pos is None:
            self.timer.stop()

    # ================================================================ step 6: export
    def export(self):
        if self.y is None:
            self._msg("Render a preview first: the export writes the corrected signal.")
            return
        name = Path(self.store.project(self.pid)["source_path"]).stem + "_deflutter.wav"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export corrected audio", name,
                                                        "WAV (*.wav);;FLAC (*.flac)")
        if path:
            self.export_to(path)

    def export_to(self, path):
        path, bit = Path(path), self.params["bit_depth"]
        if path.suffix.lower() == ".flac" and bit == "32f":
            self._warn("FLAC does not support 32-bit float: choose 16 or 24 bit.")
            return
        y, sr = self.y, self.sr

        def done(clipped):
            self.store.add_export(self.pid, self.corr_id, path, path.suffix.lstrip(".").lower(), sr, bit)
            self._msg(f"Exported {path.name}" + (f" ({clipped} samples clipped)" if clipped else ""))

        self.run_task("Exporting", lambda progress: audio.export(path, y, sr, bit), done)

    def closeEvent(self, ev):
        self.player.stop()
        if self.task is not None:
            self.task.cancel.set()
        self.pool.waitForDone(5000)
        super().closeEvent(ev)
