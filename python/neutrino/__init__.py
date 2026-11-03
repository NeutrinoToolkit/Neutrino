"""Neutrino image analysis in python: windows, images (numpy arrays) and pans (plugins).

    from PySide6.QtWidgets import QApplication
    import neutrino, numpy as np
    app = QApplication([])
    w = neutrino.Window()
    w.addShowPhys(np.random.rand(512, 512))
    app.exec()
"""

# PySide6 first: its Qt is the one used by the module (a wheel has no Qt, it finds the one of the PySide6 package).
# Without PySide6 (a module built against another Qt) neutrino.app() creates the QApplication
try:
    import PySide6.QtWidgets  # noqa: F401
except ImportError:
    pass

from ._neutrino import *  # noqa: F401,F403
