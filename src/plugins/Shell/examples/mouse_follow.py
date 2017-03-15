import sys


sys.path.append("/usr/lib/python2.7/dist-packages")
sys.path.append("/usr/lib/pyshared/python2.7")



import PythonQt.QtGui as qt


def mousemove(mm):
    x = mm.x()
    y = mm.y()
    pos_lbl.setText("x: "+repr(x)+" y: "+repr(y))
    val = neu.getBuffer().get(x, y)
    val_lbl.setText("value: "+repr(val))

def buffer_changed():
    print "------------------------ buffer changed"


w = qt.QWidget()
w.setWindowTitle("Mouse Follow")

box = qt.QVBoxLayout(w)


lbl = qt.QLabel(w)
lbl.setText("Mouse Position")
box.addWidget(lbl)

pos_lbl = qt.QLabel(w)
pos_lbl.setText("(..)")
box.addWidget(pos_lbl)

lbl2 = qt.QLabel(w)
lbl2.setText("Point Value")
box.addWidget(lbl2)

val_lbl = qt.QLabel(w)
val_lbl.setText("(..)")
box.addWidget(val_lbl)

neu.connect('mouseAtMatrix(QPointF)', mousemove)
neu.connect('bufferChanged(nPhysD *)', buffer_changed)

w.show()
