import os
from PyQt4 import pyqtconfig
if os.path.dirname(__file__)!='' :
	os.chdir(os.path.dirname(__file__))

import sys

build_file = "neutrino.sbf"
config = pyqtconfig.Configuration()
pyqt_sip_flags = config.pyqt_sip_flags

command=" ".join([ \
    config.sip_bin, \
    "-c", ".", \
    "-b", build_file, \
    "-I", config.pyqt_sip_dir, \
    pyqt_sip_flags, \
    "neutrino.sip" \
])
print command
os.system(command)

installs = []
installs.append(["neutrino.sip", os.path.join(config.default_sip_dir, "neutrino")])
installs.append(["neutrinoConfig.py", config.default_mod_dir])

makefile = pyqtconfig.QtGuiModuleMakefile(
    configuration=config,
    build_file=build_file,
    installs=installs
)

if sys.platform == 'win32' :
	makefile.extra_libs = ["Neutrino1","qwt"]
else :
	makefile.extra_libs = ["Neutrino","qwt"]

makefile.extra_lib_dirs = [".."]
if sys.platform == 'win32' :
    makefile.extra_lib_dirs += ["..\\lib","C:\\compile\\qwt-6.0.1\\qwt-6.0.1\\lib"]
makefile.extra_include_dirs = ["../src",\
"../nPhysImage",\
"../build",\
"../src/pans/colorbar",\
"../src/graphics"]
if sys.platform == 'win32' :
	makefile.extra_include_dirs += ["C:\\compile\\qwt-6.0.1\\qwt-6.0.1\\src","C:\\compile\\GnuWin32\\include","C:\\Qt\\4.7.4\\include\\QtSvg"]

if sys.platform == 'darwin' :
    makefile.extra_include_dirs += ["/opt/local/Library/Frameworks/qwt.framework/Versions/Current/Headers","/opt/local/Library/Frameworks/QtSvg.framework/Versions/Current/Headers"]

if sys.platform.startswith('linux'):
    makefile.extra_libs += ["hdf5"]
    makefile.extra_include_dirs += ["/usr/include/hdf","/usr/include/qwt","/usr/include/qt4/QtSvg"]

makefile.generate()

# import sipconfig
# content = {
#     "neutrino_sip_dir":    config.default_sip_dir,
#     "neutrino_sip_flags":  pyqt_sip_flags
# }
# sipconfig.create_config_module("neutrinoConfig.py", "neutrinoConfig.py.in", content)

