include (../lib/lib.pro)

#CONFIG += debug_and_release

TARGET = Neutrino

TEMPLATE = app

SOURCES += main.cc

OBJECTS_DIR = build
MOC_DIR = build
RCC_DIR = build
UI_DIR = build

DEFINES += __phys_debug=10
DEFINES -= QT_NO_DEBUG_OUTPUT

