
QT += svg


include (../neutrino-shared.pri)
include (../neutrino-plugins.pri)

TEMPLATE = lib
CONFIG += plugin qwt

#TARGET = $$qtLibraryTarget(__your_library_name__)
TARGET = $$qtLibraryTarget(skel)

HEADERS += skel.h
SOURCES += skel.cc
FORMS += skel.ui


