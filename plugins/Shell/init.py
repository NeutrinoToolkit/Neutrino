from PythonQt import *
from PythonQt.neutrino import *
import PythonQt.private

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

try:
	import numpy as np
except ImportError:
	pass
else:
	def phys2array(my_phys) :
		"return numpy array from nPhysD"
		return np.array(my_phys.getData()).reshape(my_phys.getShape())

	def array2phys(my_array) :
		"return nPhysD from numpy array"
		shape=my_array.shape
		return nPhysD(tuple(my_array.reshape(1,my_array.size).tolist()[0]),shape)
