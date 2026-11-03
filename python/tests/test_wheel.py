"""Smoke test of the neutrino wheel with PySide6 (cibuildwheel test-command): python test_wheel.py

Exit status 0 if all the checks pass. Offscreen (no display needed)."""
import faulthandler
import os
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
faulthandler.dump_traceback_later(180, exit=True)  # e.g. a modal dialog would hang the test

import numpy as np
from PySide6.QtCore import QTimer, qVersion
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QPushButton

import neutrino

app = QApplication([])
status = 1


def checks():
    print("neutrino", neutrino.version(), "Qt", neutrino.qtVersion(), "| PySide6 Qt", qVersion(), flush=True)
    # one Qt in the process: the one of PySide6
    assert neutrino.qtVersion() == qVersion()
    plugins = neutrino.plugins()
    print(len(plugins), "plugins", flush=True)
    assert len(plugins) > 20 and "Wavelet" in plugins

    w = neutrino.Window()
    means = []
    w.bufferChanged.connect(lambda: means.append(w.currentBuffer().mean()))
    a = np.random.rand(64, 128)
    img = w.addShowPhys(a, "random")
    assert tuple(img.shape) == (64, 128) and np.array_equal(img.array(), a)
    w.addShowPhys(np.ones((10, 10)) * 3, "threes")
    assert means and means[-1] == 3.0, means

    # the PySide objects of the window and of a pan
    assert w.widget().metaObject().className() == "neutrino"
    pan = w.openPan("Wavelet")
    spin = pan.widget().findChild(QDoubleSpinBox, "minThick")
    assert spin is not None and spin.value() == pan.get("minThick")

    # a PySide button driving Neutrino
    button = QPushButton("noise")
    button.clicked.connect(lambda: w.addShowPhys(np.random.rand(8, 8), "from button"))
    button.click()
    assert w.currentImage().name == "from button"

    img.close()
    assert not img.valid


def run():
    global status
    try:
        checks()
        status = 0
        print("all checks passed", flush=True)
    except Exception:
        traceback.print_exc()
    app.quit()


QTimer.singleShot(0, run)
app.exec()
sys.exit(status)
