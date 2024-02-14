import sys

sys.path.append("/usr/lib/python2.7/dist-packages")
sys.path.append("/usr/lib/pymodules/python2.7")
#sys.path.append("/Library/Python/2.6/site-packages")
from PyQt4 import QtGui as qt
import numpy as np

import os

nll = list()

my_n1 = neutrino(1)

for ii in np.arange(1,10):
    nll.append(neutrino())
    nll[-1].showPhys(my_n1.getBuffer(1).blur(ii))
