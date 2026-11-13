"""Neutrino image analysis in python: windows, images (numpy arrays) and pans (plugins).

    from PySide6.QtWidgets import QApplication
    import neutrino, numpy as np
    app = QApplication([])
    w = neutrino.Window()
    w.addShowPhys(np.random.rand(512, 512))
    neutrino.run()  # app.exec(), or the console stays usable with python -i / ipython -i

or, at the prompt of python or IPython:

    >>> import neutrino
    >>> n = neutrino.run()   # a window (and the QApplication), the prompt stays usable
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
from ._neutrino import __version__  # noqa: F401 (not in the * import)
from . import _neutrino

# help(neutrino) lists what is in __all__ (without it, only what is defined here: not the classes of _neutrino)
__all__ = [name for name in dir(_neutrino) if not name.startswith("_") and not name.endswith("_")] + ["run"]


_app = None  # the QApplication made by run() (kept: PySide deletes it with its last reference)


def _interactive():
    """True in a console waiting for commands: python (its prompt), python -i, IPython"""
    import sys
    return hasattr(sys, "ps1") or bool(sys.flags.interactive)


def run(*files):
    """Opens Neutrino: creates the QApplication if there is none, and a window if there is none or files are given
    (they are opened in it).

    In an interactive console (python, python -i, IPython) it returns the window at once and the console stays
    usable while the window works (PySide6 runs the Qt events while the console waits for a command; in IPython
    it switches on %gui qt):

        >>> import neutrino
        >>> n = neutrino.run()          # or neutrino.run("shot.neus")
        >>> n.currentImage().shape

    In a script it runs the Qt event loop until the last window is closed (app.exec()) and returns its exit code"""
    global _app
    import sys
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        QApplication = None
    if QApplication is not None:
        if QApplication.instance() is None:
            _app = QApplication(sys.argv[:1] or ["neutrino"])
    else:
        _neutrino.app()

    win = None
    if files or not _neutrino.windows():
        win = _neutrino.Window()
        for f in files:
            win.fileOpen(str(f))
    else:
        win = _neutrino.windows()[-1]

    try:
        ipython = get_ipython()  # noqa: F821 (a builtin in IPython)
    except NameError:
        ipython = None
    if ipython is not None:
        ipython.enable_gui("qt")
        return win
    if _interactive():
        return win
    if QApplication is not None:
        return QApplication.instance().exec()
    return _neutrino.exec()

# old: only the event loop (the QApplication and the windows made by the caller), 0 in a console
# def run():
#     import sys
#     try:
#         ipython = get_ipython()  # noqa: F821 (a builtin in IPython)
#     except NameError:
#         ipython = None
#     if ipython is not None:
#         ipython.enable_gui("qt")
#         return 0
#     if sys.flags.interactive:
#         return 0
#     from PySide6.QtWidgets import QApplication
#     return QApplication.instance().exec()
