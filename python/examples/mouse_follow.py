"""A PySide window that follows the mouse click on the Neutrino image: position and pixel value.

(the PyShell example mouse_follow.py, with the neutrino module and PySide6)"""
import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QToolBar, QVBoxLayout, QWidget

import neutrino

app = QApplication.instance() or QApplication([])
w = neutrino.Window()
w.addShowPhys(np.random.rand(256, 256), "random")

pan = w.newPan("Mouse follow")

toolbar = QToolBar("My main toolbar")
toolbar.setObjectName("toolBar")


box = QWidget()
layout = QVBoxLayout(box)
pos_lbl = QLabel("(..)")
val_lbl = QLabel("(..)")
layout.addWidget(QLabel("Mouse Position"))
layout.addWidget(pos_lbl)
layout.addWidget(QLabel("Point Value"))
layout.addWidget(val_lbl)

# the pixels of the current image: a copy, taken again only when the image changes
pixels = {"array": w.currentBuffer()}
w.bufferChanged.connect(lambda: pixels.update(array=w.currentBuffer()))

val=0

def mousemove(point):
    x, y = point
    pos_lbl.setText(f"x: {x:.1f} y: {y:.1f}")
    val_lbl.setText(f"value: {val:.4g}")

def mousecolor(valin):
    global val
    val = valin


w.mouseClickAtWorld.connect(mousemove)
w.colorValue.connect(mousecolor)

pan.widget().setCentralWidget(box)
pan.widget().addToolBar(toolbar)
pan.widget().setUnifiedTitleAndToolBarOnMac(True)
pan.show()
neutrino.run()
