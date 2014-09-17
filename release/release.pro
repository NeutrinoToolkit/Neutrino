include (../lib/lib.pro)

TARGET = Neutrino

TEMPLATE = app

SOURCES += main.cc

CONFIG += release
CONFIG -= debug

DEFINES -= __phys_debug=10

OBJECTS_DIR = build
MOC_DIR = build
RCC_DIR = build
UI_DIR = build


