include (../lib/lib.pro)

#CONFIG += debug_and_release

TARGET = Neutrino

TEMPLATE = app

SOURCES += main.cc

OBJECTS_DIR = build
MOC_DIR = build
RCC_DIR = build
UI_DIR = build

DEFINES += __phys_debug=10 __old_neu_format
DEFINES -= QT_NO_DEBUG_OUTPUT

QMAKE_CC = /opt/local/bin/gcc
QMAKE_CXX = /opt/local/bin/g++ 
QMAKE_LINK       = $$QMAKE_CXX
QMAKE_LINK_SHLIB = $$QMAKE_CXX
QMAKE_CXXFLAGS_X86_64 = -mmacosx-version-min=10.6
QMAKE_LFLAGS_X86_64 = $$QMAKE_CXXFLAGS_X86_64
