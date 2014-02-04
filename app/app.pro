include (../neutrino.pri)

TEMPLATE = app

SOURCES += main.cc
LIBS += -L.. -lNeutrino

OBJECTS_DIR = ../build
MOC_DIR = ../build
RCC_DIR = ../build
UI_DIR = ../build

INCLUDEPATH += ../lib