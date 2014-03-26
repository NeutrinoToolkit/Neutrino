include (../neutrino.pri)

TEMPLATE = lib

OBJECTS_DIR = ../build
MOC_DIR = ../build
RCC_DIR = ../build
UI_DIR = ../build

## padellume
FORMS += nOpenRAW.ui
HEADERS += nOpenRAW.h
SOURCES += nOpenRAW.cc


SOURCES += nLine.cc nRect.cc nEllipse.cc 
HEADERS += nLine.h  nRect.h  nEllipse.h  
HEADERS += nMouse.h  nTics.h
SOURCES += nMouse.cc nTics.cc


FORMS += nMouseInfo.ui
HEADERS += nMouseInfo.h 
SOURCES += nMouseInfo.cc

HEADERS += nWinList.h  nTreeWidget.h
SOURCES += nWinList.cc nTreeWidget.cc
FORMS += nWinList.ui


## focal spot
HEADERS += focalspot_pan.h
SOURCES += focalspot_pan.cc
FORMS += focalspot.ui

## lineout
HEADERS += nLineout.h  nLineoutBoth.h
SOURCES += nLineout.cc nLineoutBoth.cc
FORMS += nLineout.ui nLineoutBoth.ui 

## operators +-*/
HEADERS += nOperator.h
SOURCES += nOperator.cc
FORMS += nOperator.ui

## cutoff mask
HEADERS += nCutoffMask.h
SOURCES += nCutoffMask.cc
FORMS += nCutoffMask.ui

## rotation
HEADERS += nRotate.h
SOURCES += nRotate.cc
FORMS += nRotate.ui

## Blur
HEADERS += nBlur.h
SOURCES += nBlur.cc
FORMS += nBlur.ui

#Auto align two images tool
FORMS += nAutoAlign.ui
SOURCES += nAutoAlign.cc
HEADERS += nAutoAlign.h 

#HDF stuff 
neutrino-HDF {
	FORMS += nHDF5.ui
	SOURCES += nHDF5.cc
	HEADERS += nHDF5.h 
} 

#Box Lineout
FORMS += nBoxLineout.ui
SOURCES += nBoxLineout.cc
HEADERS += nBoxLineout.h 

#Find peaks
FORMS += nFindPeaks.ui
SOURCES += nFindPeaks.cc
HEADERS += nFindPeaks.h 

#VISAR
FORMS += nVISAR1.ui nVISAR2.ui  nVISAR3.ui 
SOURCES += nVisar.cc nVisarZoomer.cc
HEADERS += nVisar.h  nVisarZoomer.h 

# Wavelet
FORMS += nWavelet.ui
SOURCES += nWavelet.cc
HEADERS += nWavelet.h 

# Spectral Analysis
FORMS += nSpectralAnalysis.ui
SOURCES += nSpectralAnalysis.cc
HEADERS += nSpectralAnalysis.h

# Inversions
FORMS += nIntegralInversion.ui
SOURCES += nIntegralInversion.cc
HEADERS += nIntegralInversion.h 

# Monitor directory
FORMS += nMonitor.ui
SOURCES += nMonitor.cc
HEADERS += nMonitor.h 

# RegionPath
FORMS += nRegionPath.ui
SOURCES += nRegionPath.cc
HEADERS += nRegionPath.h 

# Preferences Panel
FORMS += nPreferences.ui
SOURCES += nPreferences.cc
HEADERS += nPreferences.h 

# Preferences Panel
FORMS += nCompareLines.ui
SOURCES += nCompareLines.cc
HEADERS += nCompareLines.h 
 
# Shortcuts
FORMS += nShortcuts.ui
SOURCES += nShortcuts.cc
HEADERS += nShortcuts.h 


