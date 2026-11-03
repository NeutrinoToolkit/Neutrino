"""A tool window (pan) of Neutrino with a PySide widget, driven by name like the other pans.

(the "Example" of the PyShell README, with the neutrino module and PySide6)"""
import numpy as np
from PySide6.QtWidgets import QApplication, QSpinBox

import neutrino

app = QApplication([])
w = neutrino.Window()
w.addShowPhys(np.random.rand(128, 128), "random")

pan = w.newPan()                    # an empty pan (w.newPan("my.ui") loads a Qt Designer file)
spin = QSpinBox()
spin.setObjectName("my_widget")     # the name used by get/set
pan.widget().setCentralWidget(spin)
pan.set("my_widget", 5)
print("my_widget:", pan.get("my_widget"))
pan.widget().show()

app.exec()
