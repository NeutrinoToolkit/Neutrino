"""A PySide window that follows the mouse on the Neutrino image: position and pixel value.

(the PyShell example mouse_follow.py, with the neutrino module and PySide6)"""
import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

import neutrino

app = QApplication([])
w = neutrino.Window()
w.addShowPhys(np.random.rand(256, 256), "random")

box = QWidget()
box.setWindowTitle("Mouse Follow")
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


def mousemove(point):
    x, y = point
    pos_lbl.setText(f"x: {x:.1f} y: {y:.1f}")
    a = pixels["array"]
    i, j = int(y), int(x)
    if 0 <= i < a.shape[0] and 0 <= j < a.shape[1]:
        val_lbl.setText(f"value: {a[i, j]:.4g}")


w.mouseAtMatrix.connect(mousemove)
box.show()
app.exec()
