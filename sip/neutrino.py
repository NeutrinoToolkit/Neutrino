from PyQt4 import QtCore, QtGui, uic

from PyNeutrino import *
import sys

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

class neuConsole(nGenericPan):
    def __init__(self, neu=None, name="Console"):
        nGenericPan.__init__(self,neu,name)
        self.ui=uic.loadUi('nPython.ui',self)
        self.ui.show()
        self.ui.runScript.released.connect(self.evaluate)
        self.ui.loadScript.released.connect(self.load)
        self.decorate()
        self.show()
        
    def evaluate(self):
        exec(str(self.ui.plainTextEdit.toPlainText()))

    def load(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self, 'Open File',self.ui.scriptFile.text(),'Scripts (*.py)')
        if QtCore.QFile(fileName).exists() :
            self.ui.scriptFile.setText(fileName)
            self.ui.plainTextEdit.setPlainText(open(fileName).read())


a = QtGui.QApplication(sys.argv)
n = neutrino()
n.show()
c = neuConsole(n)
a.exec_()
