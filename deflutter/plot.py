"""Spectrogram and deviation plots (pyqtgraph) with drag-to-select time ranges.

Mouse: left-drag selects a time range (drag the edges to adjust it), middle-drag pans, right-drag or
the wheel zooms (over an axis: that axis only), double-click on the spectrogram adds a reference there.
The frequency axis can be linear, log or mel.
"""
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from .analysis import cents

ORANGE = (255, 150, 30)
SEL_BRUSH = (90, 170, 255, 45)

# frequency scales: Hz -> plot y coordinate, and back
SCALES = {
    "linear": (lambda f: np.asarray(f, float), lambda y: np.asarray(y, float)),
    "log": (lambda f: np.log10(np.maximum(f, 1.0)), lambda y: 10.0 ** np.asarray(y, float)),
    "mel": (lambda f: 2595 * np.log10(1 + np.asarray(f, float) / 700),
            lambda y: 700 * (10 ** (np.asarray(y, float) / 2595) - 1)),
}


def _hz(f):
    return f"{f / 1000:g}k" if f >= 1000 else f"{f:g}"


class FreqAxis(pg.AxisItem):
    """Left axis labelled in Hz whatever the scale; log/mel ticks sit on round 1-2-5 frequencies."""

    def __init__(self):
        super().__init__("left")
        self.fscale = "log"  # (AxisItem already uses self.scale)

    def tickValues(self, minVal, maxVal, size):
        if self.fscale == "linear":
            return super().tickValues(minVal, maxVal, size)
        fwd, inv = SCALES[self.fscale]
        nice = np.array([m * 10.0 ** e for e in range(1, 6) for m in (1, 2, 5)])
        ys = fwd(nice[(nice >= inv(minVal)) & (nice <= inv(maxVal))])
        ticks, gap = [], (maxVal - minVal) / 12  # skip ticks that would crowd the previous one
        for y in ys:
            if not ticks or y - ticks[-1] >= gap:
                ticks.append(float(y))
        return [(1.0, ticks)]

    def tickStrings(self, values, scale, spacing):
        return [_hz(float(SCALES[self.fscale][1](v))) for v in values]


class SelectViewBox(pg.ViewBox):
    sigSelect = QtCore.Signal(float, float, bool)  # t0, t1, drag finished
    sigDoubleClick = QtCore.Signal(float, float)  # view coordinates

    def mouseDragEvent(self, ev, axis=None):
        if axis is not None or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mouseDragEvent(ev, axis)
        ev.accept()
        x0 = self.mapSceneToView(ev.buttonDownScenePos()).x()
        x1 = self.mapSceneToView(ev.scenePos()).x()
        self.sigSelect.emit(min(x0, x1), max(x0, x1), ev.isFinish())

    def mouseClickEvent(self, ev):
        if ev.double() and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            p = self.mapSceneToView(ev.scenePos())
            self.sigDoubleClick.emit(p.x(), p.y())
        else:
            super().mouseClickEvent(ev)


class EdgeRegion(pg.LinearRegionItem):
    """Region with draggable edges whose body lets drags through, so a new selection can start inside."""

    def mouseDragEvent(self, ev):
        pass  # not accepted: the ViewBox below receives the drag

    def hoverEvent(self, ev):
        pass


def _curve(color, width=1, style=QtCore.Qt.PenStyle.SolidLine):
    return pg.PlotDataItem(pen=pg.mkPen(color, width=width, style=style), connect="finite")


class PlotView(pg.GraphicsLayoutWidget):
    selectionChanged = QtCore.Signal(object)  # (t0, t1) when a drag/edge move finishes
    referenceRequested = QtCore.Signal(float, float)  # time s, frequency Hz
    eventEdited = QtCore.Signal(int, float, float)  # index, start, end
    cursorInfo = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.faxis = FreqAxis()
        self.spec = self.addPlot(row=0, col=0, viewBox=SelectViewBox(), axisItems={"left": self.faxis})
        self.dev = self.addPlot(row=1, col=0, viewBox=SelectViewBox())
        self.ci.layout.setRowStretchFactor(0, 3)
        self.ci.layout.setRowStretchFactor(1, 1)
        self.dev.setXLink(self.spec)
        self.spec.setLabel("left", "Frequency (Hz)")
        self.dev.setLabel("left", "Deviation (cents)")
        self.dev.setLabel("bottom", "Time", units="s")
        for p in (self.spec, self.dev):
            p.hideButtons()
            p.setMenuEnabled(False)
            p.vb.sigSelect.connect(self._dragged)
        self.dev.showGrid(y=True, alpha=0.2)
        self.spec.getAxis("bottom").setStyle(showValues=False)  # the deviation plot below carries the time axis
        self.spec.vb.sigDoubleClick.connect(self._double_clicked)
        self.img = pg.ImageItem(axisOrder="col-major")
        self.img.setColorMap(pg.colormap.get("inferno"))
        self.spec.addItem(self.img)
        self.sel = [EdgeRegion(brush=SEL_BRUSH, hoverBrush=SEL_BRUSH) for _ in range(2)]
        for p, r in zip((self.spec, self.dev), self.sel):
            p.addItem(r)
            r.setZValue(10)
            r.hide()
            r.sigRegionChanged.connect(self._sync_selection)
            r.sigRegionChangeFinished.connect(self._selection_edited)
        self.heads = [pg.InfiniteLine(angle=90, pen=pg.mkPen("w", width=1)) for _ in range(2)]
        for p, h in zip((self.spec, self.dev), self.heads):
            p.addItem(h)
            h.setZValue(20)
            h.hide()
        self._dyn = []  # overlay items rebuilt on every refresh: (plot, item)
        self._event_items = []
        self._data = None
        self.fscale = "log"  # (QGraphicsView already has a scale() method)
        self.duration = 1.0
        self.scene().sigMouseMoved.connect(self._mouse_moved)

    # ---- spectrogram
    def set_spectrogram(self, times, freqs, db, scale, dyn_range):
        """Show a spectrogram (frames x bins, dB) on a 'linear', 'log' or 'mel' frequency axis.

        Every item on this plot lives in the scale's y coordinate (self.fy converts Hz). For log/mel
        the bins are regrouped into rows uniformly spaced on that scale, keeping the max of the bins
        in each row so that narrow tonal lines survive where many bins share one row.
        """
        self._data, self.fscale = (times, freqs, db), scale
        self.faxis.fscale = scale
        fwd, inv = SCALES[scale]
        hop = times[1] - times[0] if len(times) > 1 else 0.0
        x0, w = times[0] - hop / 2, times[-1] - times[0] + hop
        if scale == "linear":
            img, y0, h = db.astype(np.float32), freqs[0], freqs[-1] - freqs[0]
        else:
            lo = max(freqs[1], 10.0) if scale == "log" else freqs[0]
            edges = inv(np.linspace(fwd(lo), fwd(freqs[-1]), 2049))
            starts = np.minimum(np.searchsorted(freqs, edges[:-1]), len(freqs) - 1)
            img = np.maximum.reduceat(db.astype(np.float32), starts, axis=1)
            y0, h = float(fwd(lo)), float(fwd(freqs[-1]) - fwd(lo))
        self.faxis.picture = None  # ticks depend on the scale
        self.faxis.update()
        top = float(np.percentile(img, 99.9))
        self.img.setImage(img, autoLevels=False, levels=(top - dyn_range, top))
        self.img.setRect(QtCore.QRectF(x0, y0, w, h))
        self.spec.vb.setLimits(xMin=0, xMax=self.duration, yMin=y0, yMax=y0 + h)
        self.dev.vb.setLimits(xMin=0, xMax=self.duration)

    def reset_view(self, duration, fmin, fmax):
        self.duration = duration
        self.spec.vb.setLimits(xMin=0, xMax=duration)
        self.spec.setXRange(0, duration, padding=0)
        self.set_freq_range(fmin, fmax)

    def fy(self, f):
        """Hz -> y coordinate of the spectrogram plot."""
        return SCALES[self.fscale][0](f)

    def set_freq_range(self, fmin, fmax):
        self.spec.setYRange(float(self.fy(max(fmin, 10.0))), float(self.fy(fmax)), padding=0)

    def zoom_to(self, t0, t1):
        self.spec.setXRange(t0, t1, padding=0.02)

    # ---- overlays
    def set_curves(self, tracks, t=None, combined=None, thresholds=None, profiles=(), residual=None,
                   show_tracks=True):
        """tracks: [dict(f_obs, f0, color, enabled, e)] on time grid t (e = relative deviation).
        combined / residual: relative deviation curves; thresholds: (on, off); profiles: [(t, e)].
        show_tracks=False draws only the nominal f0 lines on the spectrogram (corrected view)."""
        for plot, item in self._dyn:
            plot.removeItem(item)
        self._dyn = []

        def add(plot, item, x, y):
            plot.addItem(item)  # clip/downsample only once the item has its ViewBox
            item.setDownsampling(auto=True, method="peak")
            item.setClipToView(True)
            item.setData(x, y)
            self._dyn.append((plot, item))

        for tr in tracks:
            style = QtCore.Qt.PenStyle.SolidLine if tr["enabled"] else QtCore.Qt.PenStyle.DotLine
            if show_tracks:
                add(self.spec, _curve(tr["color"], 2 if tr["enabled"] else 1, style), t, self.fy(tr["f_obs"]))
            add(self.spec, _curve(tr["color"], 1, QtCore.Qt.PenStyle.DashLine), t[[0, -1]], self.fy([tr["f0"]] * 2))
            if tr["enabled"]:
                add(self.dev, _curve(tr["color"]), t, cents(tr["e"]))
        if combined is not None:
            add(self.dev, _curve("w", 2), t, cents(combined))
        if thresholds is not None:
            for v in (thresholds[0], -thresholds[0], thresholds[1], -thresholds[1]):
                line = pg.InfiniteLine(cents(v), angle=0, pen=pg.mkPen((160, 160, 160), style=QtCore.Qt.PenStyle.DashLine))
                self.dev.addItem(line)
                self._dyn.append((self.dev, line))
        for tt, ee in profiles:
            add(self.dev, _curve(ORANGE, 3), tt, cents(ee))
        if residual is not None:
            add(self.dev, _curve((90, 230, 120), 2), t, cents(residual))

    def set_events(self, events):
        """events: [(start, end, enabled)]. Editable (edge-drag) in the deviation plot, shaded on the
        spectrogram."""
        for plot, item in self._event_items:
            plot.removeItem(item)
        self._event_items = []
        for i, (s, e, on) in enumerate(events):
            alpha = 60 if on else 20
            shade = pg.LinearRegionItem((s, e), movable=False, brush=(*ORANGE, alpha // 3), pen=pg.mkPen(None))
            edit = EdgeRegion((s, e), brush=(*ORANGE, alpha), hoverBrush=(*ORANGE, alpha), pen=pg.mkPen(ORANGE))
            edit.sigRegionChangeFinished.connect(lambda r, i=i: self.eventEdited.emit(i, *r.getRegion()))
            for plot, item in ((self.spec, shade), (self.dev, edit)):
                item.setZValue(5)
                plot.addItem(item)
                self._event_items.append((plot, item))

    # ---- selection and playhead
    def set_selection(self, span):
        for r in self.sel:
            r.blockSignals(True)
            if span is None:
                r.hide()
            else:
                r.setRegion(span)
                r.show()
            r.blockSignals(False)

    def set_playhead(self, t):
        for h in self.heads:
            h.setVisible(t is not None)
            if t is not None:
                h.setValue(t)

    def _dragged(self, t0, t1, finished):
        span = (max(t0, 0.0), min(t1, self.duration))
        self.set_selection(span)
        if finished:
            self.selectionChanged.emit(span)

    def _sync_selection(self, region):
        other = self.sel[1] if region is self.sel[0] else self.sel[0]
        other.blockSignals(True)
        other.setRegion(region.getRegion())
        other.blockSignals(False)

    def _selection_edited(self, region):
        t0, t1 = region.getRegion()
        self.selectionChanged.emit((max(t0, 0.0), min(t1, self.duration)))

    def _double_clicked(self, x, y):
        self.referenceRequested.emit(float(x), float(SCALES[self.fscale][1](y)))

    def _mouse_moved(self, pos):
        for plot in (self.spec, self.dev):
            if plot.vb.sceneBoundingRect().contains(pos):
                p = plot.vb.mapSceneToView(pos)
                if plot is self.dev:
                    self.cursorInfo.emit(f"t = {p.x():.3f} s   deviation = {p.y():+.1f} cents")
                    return
                f = float(SCALES[self.fscale][1](p.y()))
                text = f"t = {p.x():.3f} s   f = {f:.1f} Hz"
                if self._data is not None:
                    times, freqs, db = self._data
                    i = min(np.searchsorted(times, p.x()), len(times) - 1)
                    k = min(np.searchsorted(freqs, f), len(freqs) - 1)
                    text += f"   level = {float(db[i, k]):.1f} dB"
                self.cursorInfo.emit(text)
