
QT += svg


# ui_* should be included in the packaging
#include (../neutrino-release/neutrino/plugins/neutrino-shared.pri)
include (../neutrino-release/neutrino/plugins/neutrino-shared.pri)
#include ( ../neutrino-release/neutrino/neutrino.pri )

TEMPLATE = lib
CONFIG += plugin qwt
TARGET = $$qtLibraryTarget(npMag)

#INCLUDEPATH += ../neutrino/build ../nPhysImage
INCLUDEPATH += ../neutrino-release/neutrino/nPhysImage ../neutrino-release/neutrino/src
#INCLUDEPATH +=  ../neutrino/src/pans \
#		../neutrino/src/pans/winlist \
#		../neutrino/src/pans/colorbar \
#		../neutrino/src/pans/VISAR \
#		../neutrino/src/graphics \ 
#		../neutrino/src/python \ 
#		../neutrino/src/registration \ 
#		../neutrino/resources/colormaps \ 
#		../neutrino/src

DEFINES += __phys_debug=10

HEADERS += npMag.h
SOURCES += npMag.cc
FORMS += npMag.ui

LIBS += -lnPhysImageF -lNeutrino -L../neutrino-release/neutrino -L../neutrino-release/neutrino/nPhysImage


#LIBS += ../neutrino/build/*o -lPythonQt

