Python scripting
================

Neutrino includes a scripting console to interact with.

It creates several python clases:

1. `neutrino` is a generic main window
2. `nPhysD` is a generic image
3. `nPan` is a generic tool window

other classes (less used) are:

1. `nLine` for lines
2. `nRect` for rectangles
3. `nEllisse` for ovals
4. `nPoint` for single points
5. `nPlot` for plots

nApp
----

This is a static variable the main use is to collect all the main window present:
* `nApp.neus()` will return a list of all the main window present

main window: `neutrino` class and neu variable
----------------------------------------------

You can create a new top-level window with `n=neutrino()` or you can get the one that opened the python PyShell via the variable `neu` 

available methods (here `n` is a generic `neutrino` and a a generic image `nPhysD`, see below):

* `n.getBufferList()` will return the list of the images.
* `n.getCurrentBuffer()` will return the current image
* `n.addShowPhys(a)` will display an image


Generic image: `nPhysD`
-----------------------

There are several constructors:

* empty constructor `a=nPhysD()` (empty image)
* from shape: `a=nPhysD(10,10,3,"ABC")` will create a 10x10 image filled with the value 3 and named ABC
* from list and shape: `a=nPhysD(range(100),(10,10))` will create a 10x10 image
* from raw doubles and shape: `a=nPhysD(data, (height, width))` where `data` are `bytes` (row major order)
* from a numpy array: `a=array2phys(np.array([[1, 2, 3], [4, 5, 6]]))` (see *numpy and other modules* below)

There is one static method to open a file (which returns a list of `nPhysD`): `b=nPhysD.open(`filename`)`

There are several method for each image here (we note with a a generic `nPhysD`):

* to get the data
  * `a.getData()` will return a linearised vector of floats and  `a.getShape()` will return a couple of int with the shape (height, width)
  * `a.getBytes()` will return the data as raw doubles (row major order)
  * `phys2array(a)` will return a numpy array (a copy of the data)

* other methods (not exhaustive list)
  * `a.properties()` will return a list of strings of properties
  * `a.getPropery('name')` will return the image property name
  * `a.setPropery('name', 'my image name')` will set the image property name to 'my image name'
 
 
numpy and other modules
-----------------------

The packaged Neutrino ships its own python with `numpy` and `scipy` (macOS, Linux AppImage; on Windows only `numpy`).
The Linux deb/rpm packages use the python of the system: install `numpy` and `scipy` with the package manager
(`python3-numpy`, `python3-scipy`); the system and user (`~/.local/lib/python3.X`) site-packages are in `sys.path`.
`init.py` (run when PyShell starts) defines two functions to exchange images with numpy:

* `phys2array(a)` returns a 2D numpy array (height, width) from the image `a`
* `array2phys(arr)` returns a new image from a 2D numpy array

```python
import scipy.ndimage
a=neu.getCurrentBuffer()
b=array2phys(scipy.ndimage.gaussian_filter(phys2array(a), 5))
neu.addShowPhys(b)
```

Other modules can be installed in the user modules folder, which is added to `sys.path`:
`~/Library/Application Support/polytechnique/Neutrino/python3.X` on macOS,
`~/.local/share/polytechnique/Neutrino/python3.X` on Linux, `%APPDATA%\polytechnique\Neutrino\python3.X` on Windows
(`python3.X` is the PyShell python version: `import sys; print(sys.version)`; `sys.path[...]` shows the folder).
Use a python with the same version to install them:

```
python3.X -m pip install --target "<user modules folder>" <module>
```

Folders added in the PyShell settings (site folder, `:` separated, `;` on Windows) come first in `sys.path`, then the user modules folder, then the shipped modules.


Generic tool window: `nPan`
---------------------------

retrieve a tool window

* `n.getPanList()` get the list of opened tool windows:
* `n.getPan('PyShell')` will return the python Shell window
 
create new window:
* `n.newPan()` create a generic empty tool window
* `n.newPan('myfile.ui')` create a generic tool from Qt .ui file

operate `p` is a tool window:
* `p.getIt('name')` retrieve the 'name' widget value
* `p.setIt('name',3)`
* `p.button('name')` will press the button 'name'

The name of the wigdets inside a `nPan` is written in square bracket in tooltip window that appears when hoovering over the widget


Example
-------

```python
p=neu.newPan()
widget=PythonQt.QtGui.QSpinBox()
widget.setObjectName('my_widget')
p.setCentralWidget(widget)
p.get('my_widget')
p.set('my_widget', 5)
p.show()
```
