# src CMakeLists.txt

# ref: http://wiki.qt.io/Using_CMake_build_system 

set (CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -Wall -D__phys_debug=10")

cmake_minimum_required (VERSION 3.0)
project (neutrino)
add_definitions(-D__VER="--test--")

set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

list (APPEND CMAKE_CXX_FLAGS "-fopenmp")


## sources
set (SOURCES nScene.cc neutrino.cc neutrinoPalettes.cc graphics/nRect.cc graphics/nMouse.cc graphics/nLine.cc graphics/nTics.cc graphics/nPoint.cc graphics/nEllipse.cc panThread.cc nPlug.cc pans/nFocalSpot.cc pans/nPhysProperties.cc  pans/VISAR/nVisarZoomer.cc pans/VISAR/nVisar.cc pans/nFindPeaks.cc pans/nInterpolatePath.cc pans/nOpenRAW.cc pans/nBlur.cc pans/nSpectralAnalysis.cc pans/colorbar/nHistogram.cc pans/colorbar/nColorBarWin.cc pans/nInterferometry.cc pans/nGhost.cc pans/nMouseInfo.cc pans/nBoxLineout.cc pans/nPreferences.cc pans/nAffine.cc pans/nMonitor.cc pans/nRotate.cc pans/nWavelet.cc pans/nRegionPath.cc pans/nIntegralInversion.cc pans/nLineoutBoth.cc pans/winlist/nTreeWidget.cc pans/winlist/nWinList.cc pans/nCutoffMask.cc pans/nCompareLines.cc pans/nShortcuts.cc pans/nOperator.cc pans/nLineout.cc nView.cc nGenericPan.cc nApp.cc main.cc 
)

## UIs
set (UIS ../UIs/nCompareLines.ui ../UIs/nFocalSpot.ui ../UIs/nPhysProperties.ui ../UIs/nWinList.ui ../UIs/nInterferometry1.ui ../UIs/nVISAR2.ui ../UIs/nSpectralAnalysis.ui ../UIs/nVISAR3.ui ../UIs/nLine.ui ../UIs/nRegionPath.ui ../UIs/nGhost.ui ../UIs/nInterpolatePath.ui ../UIs/nIntegralInversion.ui ../UIs/nPreferences.ui ../UIs/nOpenRAW.ui ../UIs/nColorBarWin.ui ../UIs/nOperator.ui ../UIs/nCutoffMask.ui ../UIs/nBoxLineout.ui ../UIs/nLineout.ui ../UIs/nPoint.ui ../UIs/nVISAR1.ui ../UIs/nSbarra.ui ../UIs/nMonitor.ui ../UIs/nFindPeaks.ui ../UIs/nShortcuts.ui ../UIs/nAffine.ui ../UIs/nBlur.ui ../UIs/nLineoutBoth.ui ../UIs/nWavelet.ui ../UIs/nInterferometry.ui ../UIs/nRotate.ui ../UIs/nMouseInfo.ui ../UIs/nObject.ui ../UIs/neutrino.ui ../UIs/nFileInfo.ui	
)

## resources
set (RESOURCES ../resources/neutrino.qrc)

# sub-structure (wrong: we should probably replicate CMakeLists.txt in every subfolder)
include_directories(graphics doc pans pans/VISAR pans/colorbar pans/winlist)

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})
include_directories(${CMAKE_BINARY_DIR})

# using Qt meta-system (precompiler)
QT4_ADD_RESOURCES( RES_SOURCES ${RESOURCES} )
QT4_WRAP_UI( UI_HEADERS ${UIS} )
QT4_WRAP_CPP( MOC_SRCS ${MOC_HEADERS} ) # probabilmente auto_moc interferisce con questo

add_executable(Neutrino ${SOURCES} ${MOC_SRCS} ${RES_SOURCES} ${UI_HEADERS} )
target_link_libraries(Neutrino ${QT_LIBRARIES} Qt4::QtSvg)

# nPhysImage
include_directories(../nPhysImage)
target_link_libraries(Neutrino "-L../nPhysImage -lnPhysImageF -lfftw3 -lfftw3_threads -lgsl -lgslcblas -lm")

list(APPEND CMAKE_MODULE_PATH ".")
find_package(Qwt REQUIRED)
if (QWT_FOUND)
	include_directories(${QWT_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${QWT_LIBRARIES})
endif(QWT_FOUND)

find_package(TIFF REQUIRED)
if (TIFF_FOUND)
	include_directories(${TIFF_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${TIFF_LIBRARIES})
	add_definitions(-DHAVE_TIFF)
endif()

find_package(HDF5)
if (HDF5_FOUND)
	include_directories(${HDF5_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${HDF5_LIBRARIES})
	add_definitions(-DHAVE_HDF5)

	# add nHDF5 sources
	set (SOURCES ${SOURCES} pans/nHDF5.cc)
	set (UIS ${UIS} ../UIs/nHDF5.ui)
endif (HDF5_FOUND)


target_link_libraries(Neutrino ${LIBS})


## ------- TODO ----------
# carefully check for openmp
