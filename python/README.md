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
app.exec()
```

Installation
------------

```
pip install neutrino-imaging
```

or a wheel of the CI (`pip install neutrino_imaging-*.whl`). It needs python >= 3.12 and installs PySide6 and numpy:
the module uses the Qt of PySide6 (the wheels have no Qt). Wheels: Linux x86_64 and aarch64, macOS >= 15
(arm64 and x86_64).

From the sources (`pip install .` in the Neutrino folder, or the cmake build: `PYTHONPATH=<build>/python`), see
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
* `w.openPan(name)` opens a tool (`neutrino.plugins()` the names), `w.getPan(name)`, `w.pans()`,
  `w.newPan()` an empty tool, `w.newPan("my.ui")` one from a Qt Designer file
* `w.close(ask=False)` closes it; `w.askCloseUnsaved` (False for the windows made in python) asks to save the images
  not saved
* `w.widget()` the PySide6 `QMainWindow`
* the slots of the window by name (e.g. `w.getZoom()`), Qt properties with `w.property(name)`,
  `w.setProperty(name, value)`

Signals (`connect(callable)`, `disconnect(callable=None)`; the callable gets the arguments it accepts):
`bufferChanged` (the `Image`), `physAdd`, `physDel`, `physChanged`, `physReordered`, `panAdd`, `panDel`,
`mouseAtMatrix` and `mouseAtWorld` (`(x, y)`), `nZoom`, `colorValue`, `closeAll`, `destroyed`.

Images: `neutrino.Image`
------------------------

* `img.array()` (or `np.asarray(img)`) the pixels as a numpy copy, shape `(height, width)`; `img.shape`
* `img.update(array)` replaces the pixels (the shape can change)
* `img.name`, `img.shortName`, `img.origin`, `img.scale` (read and write)
* `img.properties()`, `img.property(name)`, `img.setProperty(name, value)` the properties of the image (e.g. read
  from the file)
* `img.show()`, `img.close(ask=False)`, `img.window()`, `img.valid` (False once closed)

Tools: `neutrino.Pan`
---------------------

* `p.get("name")`, `p.set("name", value)`, `p.button("name")` (waits for the calculation it starts): the name of a
  widget is in square brackets in its tooltip; `p.widgets()` all the names
* `p.getData(...)`, `p.close()`, `p.raise_()`, `p.window()`
* `p.widget()` the PySide6 `QMainWindow` of the tool (e.g. `p.widget().findChild(QSpinBox, "name")`)
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
| `n.newPan()`, `n.newPan('my.ui')` | `w.newPan()`, `w.newPan('my.ui')` |
| `p.getIt`/`p.get`, `p.setIt`/`p.set`, `p.button` | `p.get`, `p.set`, `p.button` |
| `n.connect('mouseAtMatrix(QPointF)', f)` | `w.mouseAtMatrix.connect(f)` |
| `PythonQt.QtGui` widgets | PySide6 (`p.widget()`, `w.widget()`) |
| the modules folder of PyShell | the python environment (pip, venv, conda...) |

Examples
--------

`examples/`: `mouse_follow.py` (a PySide window following the mouse), `blur.py` (scipy filters in new windows),
`custom_pan.py` (a tool with a PySide widget). `tests/test_wheel.py` is the test of the wheels.
