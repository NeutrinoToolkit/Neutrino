"""Neutrino image analysis in python: windows, images (numpy arrays) and pans (plugins).

    from PySide6.QtWidgets import QApplication
    import neutrino, numpy as np
    app = QApplication([])
    w = neutrino.Window()
    w.addShowPhys(np.random.rand(512, 512))
    neutrino.run()  # app.exec(), or the console stays usable with python -i / ipython -i
"""

# PySide6 first: its Qt is the one used by the module (a wheel has no Qt, it finds the one of the PySide6 package).
# Without PySide6 (a module built against another Qt) neutrino.app() creates the QApplication
try:
    import PySide6.QtWidgets  # noqa: F401
    import sys as _sys
    if _sys.platform == "win32":
        # no run paths on Windows: the Qt DLLs not loaded yet by PySide6 (e.g. QtMultimedia) are found in
        # its folder (the handle is kept: closing it removes the folder from the search)
        import os as _os
        _pyside_dlls = _os.add_dll_directory(_os.path.dirname(PySide6.__file__))
except ImportError:
    pass

from ._neutrino import *  # noqa: F401,F403


def run():
    """Runs the Qt event loop until the last window is closed (app.exec()), except in an interactive console:
    with python -i (PySide6 runs the Qt events while the console waits) and in IPython (ipython -i script.py: its Qt
    integration, %gui qt, is switched on) it returns at once and the windows work while the console is used"""
    import sys
    try:
        ipython = get_ipython()  # noqa: F821 (a builtin in IPython)
    except NameError:
        ipython = None
    if ipython is not None:
        ipython.enable_gui("qt")
        return 0
    if sys.flags.interactive:
        return 0
    from PySide6.QtWidgets import QApplication
    return QApplication.instance().exec()
