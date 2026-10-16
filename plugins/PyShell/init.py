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
	# def phys2array(my_phys) :
	# 	"return numpy array from nPhysD"
	# 	return np.array(my_phys.getData()).reshape(my_phys.getShape())

	# def array2phys(my_array) :
	# 	"return nPhysD from numpy array"
	# 	shape=my_array.shape
	# 	return nPhysD(tuple(my_array.reshape(1,my_array.size).tolist()[0]),shape)

	def phys2array(my_phys) :
		"return numpy array (a copy) from nPhysD"
		return np.frombuffer(my_phys.getBytes(), dtype=np.float64).reshape(my_phys.getShape()).copy()

	def array2phys(my_array) :
		"return nPhysD from 2D numpy array"
		my_array=np.ascontiguousarray(my_array, dtype=np.float64)
		return nPhysD(my_array.tobytes(), my_array.shape)
