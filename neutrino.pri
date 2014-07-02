TARGET = ../Neutrino

CONFIG += qt qwt debug_and_release windows

CONFIG += neutrino-HDF

QT += svg xml network core gui


VERSION = 1.0.0

# VERSION STUFF
NVERSION=$$system(git describe)

macx {
	DEFINES += __VER=\'\"$${NVERSION}\"\'
} else {
	DEFINES += __VER=\\\"$${NVERSION}\\\"
}

message($${NVERSION})
# nPhysImage compilation
nPhys.target = nPhys
CONFIG(debug, debug|release) {
    nPhys.commands = make -C ../nPhysImage debug
} else {
    nPhys.commands = make -C ../nPhysImage release
}
QMAKE_EXTRA_TARGETS += nPhys
PRE_TARGETDEPS = nPhys


CONFIG(debug, debug|release) {
    DEFINES += __phys_debug=10
    message("DEBUG!")
} else {
    DEFINES += QT_NO_DEBUG_OUTPUT
    message("RELEASE!")
}

#this accounts for a bug in qmake for xcode
macx-xcode {
    DEFINES += __phys_debug=10
}

# base
INCLUDEPATH += ../src
DEPENDPATH += ../src ../resources ../UIs

RESOURCES=../resources/neutrino.qrc

win32:RC_FILE=../resources/neutrino.rc

QMAKE_INFO_PLIST=../resources/neutrino.plist

macx {
	neutrino.path = .
	neutrino.files = neutrino.app
	INSTALLS += neutrino

	ICON = ../resources/icons/icon.icns

	LIBS += -L/opt/local/lib
	INCLUDEPATH += /opt/local/include /opt/local/include/netpbm
				
# workaround for a bug in macdeployqt: http://stackoverflow.com/questions/3454001/macdeployqt-not-copying-plugins
# sudo ln -sf /Developer/Applications/Qt/plugins /usr/plugins

    # this is required from src/osxApp.h
    LIBS += -framework IOKit -framework CoreFoundation
}


unix:!macx {
	INCLUDEPATH += /usr/include/qwt
	LIBS += -lqwt

    neutrino.path = /usr/local/bin
	neutrino.files = neutrino
    INSTALLS += neutrino

# gsl
	LIBS += -L/usr/lib -lgsl -lgslcblas -lm

}        

win32 {
	ICON = ../resources/icons/icon.ico
	QWT_ROOT = "C:\\compile\\qwt-6.0.1\\qwt-6.0.1"
	
	LIBS += -L../lib	
		
	DEFINES += QT_NO_DEBUG
	
	# GNU32 subsys
	INCLUDEPATH += "C:\\compile\\GnuWin32\\include"
	LIBS +=  -L"C:\\compile\\GnuWin32\\lib"
	
	# qwt
	INCLUDEPATH += "C:\\compile\\qwt-6.0.1\\qwt-6.0.1\\src"
	LIBS +=  -lqwt -L"C:\\compile\\qwt-6.0.1\\qwt-6.0.1\\lib" -lfftw3-3
	
	# fits
    LIBS += -L"C:\\compile\\cfit3310"

	DEFINES    += QT_DLL QWT_DLL
	
	qtAddLibrary(qwt)
	
} else {
	LIBS += -lfftw3
}

# physImage
INCLUDEPATH += ../nPhysImage 
#LIBS += -L../nPhysImage -lnPhysImageF -lnetpbm  -L/usr/lib -lgsl -lgslcblas -lm -ltiff -ljpeg -lm -ldf -lcfitsio 
LIBS += -L../nPhysImage -lnPhysImageF -lnetpbm  -L/usr/lib -lgsl -lgslcblas -lm -ltiff -ljpeg -lm 

neutrino-HDF {
    LIBS += -lmfhdf -ldf -lhdf5
	message ( HDF enabled )
	DEFINES += __phys_HDF

	macx {
		LIBS+=-lmfhdf -ldf -ljpeg -lz -lm -lhdf5
	}

	win32 {
		INCLUDEPATH += "C:\\compile\\HDF_Group\\HDFshared\\4.2.8\\include"
	        LIBS += -L"C:\\compile\\HDF_Group\\HDFshared\\4.2.8\\lib"
	        LIBS += -lmfhdfdll -lhdfdll -lhdf5 -l hdf5_hl
	        LIBS -= -ljpeg -lm -lmfhdf -ldf
	        INCLUDEPATH += "C:\\compile\\HDF_Group\\HDF5\\1.8.10\\include"
	        LIBS += -L"C:\\compile\\HDF_Group\\HDF5\\1.8.10\\lib"
	}

	unix:!macx {
		INCLUDEPATH += /usr/include/hdf
		LIBS+=-lmfhdf -ldf -ljpeg -lz -lm -lhdf5 -lhdf5_hl
	}
} 

# paths and mouse and tics
INCLUDEPATH += ../src/graphics
DEPENDPATH += ../src/graphics

# pans
INCLUDEPATH += ../src/pans
DEPENDPATH += ../src/pans

# colorbar
INCLUDEPATH += ../src/pans/colorbar
DEPENDPATH += ../src/pans/colorbar

# VISAR
INCLUDEPATH += ../src/pans/VISAR
DEPENDPATH += ../src/pans/VISAR

#winlist pan
INCLUDEPATH += ../src/pans/winlist
DEPENDPATH += ../src/pans/winlist


FORMS += neutrino.ui nSbarra.ui
FORMS += nLine.ui nObject.ui nPoint.ui

# external colormaps
SOURCES += neutrinoPalettes.cc

DEPENDPATH += ../nPhysImage
HEADERS += config.h

HEADERS += nGenericPan.h  neutrino.h 
SOURCES += nGenericPan.cc neutrino.cc

HEADERS += nView.h 
SOURCES += nView.cc

HEADERS += nPlug.h 
SOURCES += nPlug.cc

FORMS += nColorBarWin.ui
HEADERS += nColorBarWin.h  nHistogram.h
SOURCES += nColorBarWin.cc nHistogram.cc

FORMS += nPhysProperties.ui
HEADERS += nPhysProperties.h 
SOURCES += nPhysProperties.cc


macx {
	HEADERS += osxApp.h
}
