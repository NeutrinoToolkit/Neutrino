Neutrino in python
==================

The `neutrino` module opens Neutrino windows from python: images are numpy arrays, the tools (pans) are driven by
the names of their widgets, and the windows are Qt widgets of PySide6. It replaces the PyShell plugin (PythonQt).

```python
from PySide6.QtWidgets import QApplication
import neutrino, numpy as np

app = QApplication([])
w = neutrino.Window()
w.addShowPhys(np.random.rand(512, 512))
w.bufferChanged.connect(lambda: print(w.currentBuffer().mean()))
neutrino.run()
```

At the prompt of python or IPython, `neutrino.run()` does everything and returns the window:

```python
>>> import neutrino
>>> n = neutrino.run()                 # or neutrino.run("shot.neus", ...): the files open in a new window
>>> n.currentImage().shape
```

`neutrino.run(*files)` creates the QApplication if there is none, and a window if there is none or files are given
(otherwise it takes the last window). In an interactive console (`python`, `python -i script.py`, IPython) it returns
the window at once and the console stays usable while the windows work (PySide6 runs the Qt events while the console
waits; in IPython it switches on `%gui qt`). In a script it is `app.exec()`: it returns the exit code when the last
window is closed.

Installation
------------

```
pip install neutrino-toolkit
```

or a wheel of the CI (`pip install neutrino_toolkit-*.whl`). It needs python >= 3.10 and installs PySide6 and numpy:
the module uses the Qt of PySide6 (the wheels have no Qt). Wheels: Linux x86_64 and aarch64, macOS >= 15
(arm64 and x86_64), Windows x86_64.

From the sources (`pip install .` in the Neutrino folder, or the cmake build with `-DNEUTRINO_PYTHON_MODULE=ON`:
`PYTHONPATH=<build>/python`), see
`CMakeLists.txt` and `Wheel.cmake` here.

Without PySide6 (a module built against another Qt), `neutrino.app()` creates the QApplication and
`neutrino.exec()` runs it.

Windows: `neutrino.Window`
--------------------------

* `w = neutrino.Window()` a new window; `neutrino.windows()` all of them
* `w.addPhys(array, name)`, `w.addShowPhys(array, name)` add a 2D array (copied, as float64) and return the `Image`
* `w.currentBuffer()` the pixels of the shown image (a numpy copy), `w.currentImage()` the shown `Image`,
  `w.images()` all of them
* `w.fileOpen(filename)` returns the list of the `Image`s of the file, `w.fileSave(...)`
* `w.openPan(name)` opens a tool (`neutrino.plugins()` the names; `ValueError` for an unknown name), `w.getPan(name)` (`None` if it is not open), `w.pans()`,
  `w.newPan("name")` an empty tool with that name (its title, `w.getPan("name")`), for PySide widgets: see
  `Pan` below
* `w.close(ask=False)` closes it; `w.askCloseUnsaved` (False for the windows made in python) asks to save the images
  not saved
* `w.widget()` the PySide6 `QMainWindow`
* the slots of the window by name (e.g. `w.getZoom()`), Qt properties with `w.property(name)`,
  `w.setProperty(name, value)`

Signals (`connect(callable)`, `disconnect(callable=None)`; the callable gets the arguments it accepts):
`bufferChanged` (the `Image`), `physAdd`, `physDel`, `physChanged`, `physReordered`, `panAdd`, `panDel`,
`mouseAtMatrix`, `mouseAtWorld` and `mouseClickAtWorld` (`(x, y)`), `nZoom`, `colorValue`, `closeAll`, `destroyed`.

Images: `neutrino.Image`
------------------------

* `img.array()` (or `np.asarray(img)`) the pixels as a numpy copy, shape `(height, width)`; `img.shape`
* `img.update(array)` replaces the pixels (the shape can change)
* `img.name`, `img.shortName`, `img.origin`, `img.scale` (read and write)
* `img.scalePoly = ([c0, c1, c2, ...], [c0, c1, ...])` non-linear axes: along x (and y) the world coordinate is
  P(p) = c0 + c1·p + c2·p² + ... of the absolute pixel p with the origin at 0, P(p) − P(origin) + c0 otherwise (c0 is
  the value at the origin). `[]`: linear axis; degree 1 sets the scale and the origin. `img.scale` is then the mean
  slope over the image. `img.toWorld(x, y)` and `img.toPixel(wx, wy)` convert with them
* `img.properties()`, `img.property(name)`, `img.setProperty(name, value)` the properties of the image (e.g. read
  from the file)
* `img.show()`, `img.close(ask=False)`, `img.window()`, `img.valid` (False once closed)

Tools: `neutrino.Pan`
---------------------

* `p.get("name")`, `p.set("name", value)`, `p.button("name")` (waits for the calculation it starts): the name of a
  widget is in square brackets in its tooltip. `p.get()` without a name: all of them, `{name: value}`;
  `p.entries()` their kinds, `{name: class}`; an unknown name is a `ValueError` (listing the names).
  The objects of a pan by their name (their tooltip, e.g. `"Box lineoutrect"`), in pixels of the shown image:
  rectangles and ellipses `(x, y, width, height)`, lines `[(x, y), ...]`, points `(x, y)`; `p.get(name)` gives them
  back the same way. `p.widgets()` the object names of all the widgets (also those that get and set don't use)
* `p.getData(...)`, `p.show()`, `p.close()`, `p.raise_()`, `p.window()`
* `p.results(what=None)`: the data calculated by the pan, a dict of numpy arrays. By default the curves of its plots
  (`{plot: {graph: {"x": ..., "y": ...}}}`, `what` the name of a plot); Visar gives
  `{"visar": [{"time", "velocity", "velocityError", "reflectivity", ...}, ...], "sop": {...}}`, with `what` "sop",
  a VISAR number (0 for the first) or its name ("Visar1"). A plugin gives its own by reimplementing
  `nGenericPan::results(QVariant what)`
* `p.widget()` the PySide6 `QMainWindow` of the tool (e.g. `p.widget().findChild(QSpinBox, "name")`, or
  `p.widget().setCentralWidget(my_widget)` in a pan of `w.newPan`); `p.show()` and not `p.widget().show()`: it also
  sets the title and the saved values, as for the other pans. A `QToolBar` added before `p.show()` with
  `setObjectName("toolBar")` gets the buttons of the pans (preferences, help and the gear menu: raise viewer, copy as python, widget names, screenshot, reset)
* its slots by name, Qt properties with `p.property(name)`, `p.setProperty(name, value)`

From PyShell
------------

| PyShell | `neutrino` module |
|---|---|
| `neu` (the window of the shell) | `w = neutrino.Window()` or `neutrino.windows()[0]` |
| `nApp.neus()` | `neutrino.windows()` |
| `neutrino()` | `neutrino.Window()` |
| `n.getBufferList()` | `w.images()` |
| `n.getCurrentBuffer()` | `w.currentImage()`, pixels `w.currentBuffer()` |
| `nPhysD(...)`, `array2phys(arr)` + `n.addShowPhys(a)` | `w.addShowPhys(arr, name)` |
| `phys2array(a)`, `a.getData()`, `a.getBytes()` | `img.array()` |
| `a.getShape()` | `img.shape` |
| `a.get(x, y)` | `img.array()[y, x]` |
| `nPhysD.open(file)` | `w.fileOpen(file)` |
| `a.properties()`, `a.getProperty(n)`, `a.setProperty(n, v)` | `img.properties()`, `img.property(n)`, `img.setProperty(n, v)` |
| `n.getPanList()`, `n.getPan(name)` | `w.pans()`, `w.getPan(name)` |
| `n.newPan()`, `n.newPan('my.ui')` | `w.newPan('name')` + PySide widgets (a .ui file: `PySide6.QtUiTools.QUiLoader`) |
| `p.getIt`/`p.get`, `p.setIt`/`p.set`, `p.button` | `p.get`, `p.set`, `p.button` |
| `n.connect('mouseAtMatrix(QPointF)', f)` | `w.mouseAtMatrix.connect(f)` |
| `PythonQt.QtGui` widgets | PySide6 (`p.widget()`, `w.widget()`) |
| the modules folder of PyShell | the python environment (pip, venv, conda...) |

Examples
--------

`examples/`: `mouse_follow.py` (a PySide window following the mouse), `blur.py` (scipy filters in new windows),
`custom_pan.py` (a tool with a PySide widget). `tests/test_wheel.py` is the test of the wheels.
