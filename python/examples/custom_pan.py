"""A tool window (pan) of Neutrino with a PySide widget, driven by name like the other pans.

(the "Example" of the PyShell README, with the neutrino module and PySide6)"""
import numpy as np
from PySide6.QtWidgets import QApplication, QSpinBox

import neutrino

# app = QApplication([])
# the QApplication already there if the script is run again (e.g. %run in IPython)
app = QApplication.instance() or QApplication([])
w = neutrino.Window()
w.addShowPhys(np.random.rand(128, 128), "random")

# pan = w.newPan()                    # an empty pan (w.newPan("my.ui") loads a Qt Designer file)
pan = w.newPan("Custom pan")        # an empty pan with that name (title, w.getPan("Custom pan"))
spin = QSpinBox()
spin.setObjectName("my_widget")     # the name used by get/set
pan.widget().setCentralWidget(spin)
pan.set("my_widget", 5)
print("my_widget:", pan.get("my_widget"))
# pan.widget().show()
pan.show()                          # as Neutrino shows its pans (title, saved values)

# app.exec()
# app.exec(), or with python -i / ipython -i the console stays usable while the windows work
neutrino.run()
