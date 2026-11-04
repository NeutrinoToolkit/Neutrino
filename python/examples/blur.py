"""An image blurred with increasing widths, each in a new Neutrino window (needs scipy).

(the PyShell example testblur.py, with the neutrino module and PySide6)

    python blur.py [image file]"""
import sys

import numpy as np
import scipy.ndimage
from PySide6.QtWidgets import QApplication

import neutrino

app = QApplication([])
source = neutrino.Window()
if len(sys.argv) > 1:
    image = source.fileOpen(sys.argv[1])[0]
else:
    image = source.addShowPhys(np.random.rand(256, 256), "random")

windows = []
for sigma in range(1, 10):
    win = neutrino.Window()
    win.addShowPhys(scipy.ndimage.gaussian_filter(image.array(), sigma), f"{image.name} blur {sigma}")
    windows.append(win)

app.exec()
