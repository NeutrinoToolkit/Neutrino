import sys

#sys.path.append("/Library/Python/2.6/site-packages")
sys.path.append("/usr/lib/python2.7/dist-packages")
sys.path.append("/usr/lib/pyshared/python2.7")
from PyQt4 import QtGui as qt
from PyQt4 import QtCore
#import numpy as np

class DoSomething(QtCore.QThread):
    def __init__(self):
        self.thcount = 0
        QtCore.QThread.__init__(self)

    def run(self):
        while True:
            self.thcount += 1
            self.thread().sleep(1)
            print "pling"



def mousemove(mm):
    x = mm.x()
    y = mm.y()
    pos_lbl.setText("x: "+repr(x)+" y: "+repr(y))
    val = n1.getBuffer().get(x, y)
    #val_lbl.setText("value: "+repr(val))
    val_lbl.setText(repr(myth.thcount))

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

try:
    n1
except NameError:
    n1=None

if (n1 == None) :
    n1 = neutrino(1)


n1.connect('mouseAtMatrix(QPointF)', mousemove)
n1.connect('bufferChanged(nPhysD *)', buffer_changed)
#img = n1.getPhys()


if __name__ == "__main__":
    myth = DoSomething()
    myth.start()

w.show()
