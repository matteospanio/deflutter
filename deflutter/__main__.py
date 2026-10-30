"""Entry point: `deflutter [audio file]`, `python -m deflutter`, and the PyInstaller script.

Absolute imports, so the file also works as a top-level script in the frozen (PyInstaller) build.
`--smoke-test` builds the main window and quits: CI uses it to check that a packaged binary starts.
"""
import multiprocessing
import sys


def main():
    multiprocessing.freeze_support()  # frozen builds start worker processes through this entry point
    import pyqtgraph as pg
    from PySide6 import QtCore, QtWidgets

    from deflutter.ui import MainWindow

    pg.setConfigOptions(antialias=True)
    QtCore.QLocale.setDefault(QtCore.QLocale.c())  # English UI: "." decimals everywhere
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("deflutter")
    app.setStyle("Fusion")
    args = app.arguments()[1:]
    if "--smoke-test" in args:
        import numpy as np

        from deflutter import audio, correction

        win = MainWindow(data_dir=QtCore.QDir.tempPath() + "/deflutter-smoke-test")
        QtCore.QTimer.singleShot(0, app.quit)
        app.exec()
        # two separate events -> rendered by worker processes: checks multiprocessing in frozen builds
        x = audio.synth(audio.PROFILES["multi"], dur=20.0)
        t = np.linspace(0, 20, 2000)
        y, regions = correction.render(x, 48000, t, audio.PROFILES["multi"](t), [(2.5, 6.5), (8, 12)])
        assert y.shape == x.shape and len(regions) == 2 and np.isfinite(y).all()
        print("deflutter smoke test OK")
        return 0
    win = MainWindow()
    win.show()
    if args:
        win.new_project(args[0])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
