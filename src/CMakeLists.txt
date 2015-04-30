# src CMakeLists.txt

# ref: http://wiki.qt.io/Using_CMake_build_system 

cmake_minimum_required (VERSION 3.0)

if (APPLE) 
    message(STATUS "Apple : use macports !")
    set (CMAKE_C_COMPILER "/opt/local/bin/gcc")
    set (CMAKE_CXX_COMPILER "/opt/local/bin/g++")
    set (CMAKE_PREFIX_PATH "/opt/local")
    SET (CMAKE_EXE_LINKER_FLAGS  "-headerpad_max_install_names" )
    add_compile_options(-fPIC)
    add_compile_options(-mmacosx-version-min=10.6)
endif()

if (UNIX AND (NOT APPLE))
    message(STATUS "Linux")
    set (CMAKE_PREFIX_PATH "/usr")
    include_directories(BEFORE "/usr/include")
endif()


add_compile_options(-std=c++0x)

project (neutrino)
add_definitions(-D__VER="--test--")

set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set (LIBPATHS /usr/lib /opt/local/lib)

## sources
set (SOURCES nScene.cc neutrino.cc neutrinoPalettes.cc graphics/nRect.cc graphics/nMouse.cc graphics/nLine.cc graphics/nTics.cc graphics/nPoint.cc graphics/nEllipse.cc panThread.cc nPlug.cc pans/nFocalSpot.cc pans/nPhysProperties.cc  pans/VISAR/nVisarZoomer.cc pans/VISAR/nVisar.cc pans/nFindPeaks.cc pans/nInterpolatePath.cc pans/nOpenRAW.cc pans/nBlur.cc pans/nSpectralAnalysis.cc pans/colorbar/nHistogram.cc pans/colorbar/nColorBarWin.cc pans/nInterferometry.cc pans/nGhost.cc pans/nMouseInfo.cc pans/nBoxLineout.cc pans/nPreferences.cc pans/nAffine.cc pans/nMonitor.cc pans/nRotate.cc pans/nWavelet.cc pans/nRegionPath.cc pans/nIntegralInversion.cc pans/nLineoutBoth.cc pans/winlist/nTreeWidget.cc pans/winlist/nWinList.cc pans/nCutoffMask.cc pans/nCompareLines.cc pans/nShortcuts.cc pans/nOperator.cc pans/nLineout.cc nView.cc nGenericPan.cc osxApp.cc nApp.cc main.cc 
)

## UIs
set (UIS ../UIs/nCompareLines.ui ../UIs/nFocalSpot.ui ../UIs/nPhysProperties.ui ../UIs/nWinList.ui ../UIs/nInterferometry1.ui ../UIs/nVISAR2.ui ../UIs/nSpectralAnalysis.ui ../UIs/nVISAR3.ui ../UIs/nLine.ui ../UIs/nRegionPath.ui ../UIs/nGhost.ui ../UIs/nInterpolatePath.ui ../UIs/nIntegralInversion.ui ../UIs/nPreferences.ui ../UIs/nOpenRAW.ui ../UIs/nColorBarWin.ui ../UIs/nOperator.ui ../UIs/nCutoffMask.ui ../UIs/nBoxLineout.ui ../UIs/nLineout.ui ../UIs/nPoint.ui ../UIs/nVISAR1.ui ../UIs/nSbarra.ui ../UIs/nMonitor.ui ../UIs/nFindPeaks.ui ../UIs/nShortcuts.ui ../UIs/nAffine.ui ../UIs/nBlur.ui ../UIs/nLineoutBoth.ui ../UIs/nWavelet.ui ../UIs/nInterferometry.ui ../UIs/nRotate.ui ../UIs/nMouseInfo.ui ../UIs/nObject.ui ../UIs/neutrino.ui ../UIs/nFileInfo.ui	
)

## resources
set (RESOURCES ../resources/neutrino.qrc)

# sub-structure (wrong: we should probably replicate CMakeLists.txt in every subfolder)
include_directories(graphics doc pans pans/VISAR pans/colorbar pans/winlist)



SET(QT_LIBRARY_DIR "/opt/local/Library/Frameworks/")
find_package(Qt4 REQUIRED)

message(STATUS "qt use frmaeworks: ${QT_USE_FRAMEWORKS}")

include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})
include_directories(${CMAKE_BINARY_DIR})

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

# fftw_threads
find_library(FFTW_THREADS NAMES fftw3_threads)
if (NOT ${FFTW_THREADS} STREQUAL "FFTW_THREADS-NOTFOUND")
	message (STATUS "using FFTW_THREADS: ${FFTW_THREADS}")
	set(LIBS ${LIBS} ${FFTW_THREADS}) 
	add_definitions(-DHAVE_LIBFFTW_THREADS)
endif()

#fftw
find_library(FFTW NAMES fftw3)
if (NOT ${FFTW} STREQUAL "FFTW-NOTFOUND")
	message (STATUS "using FFTW: ${FFTW}")
	set(LIBS ${LIBS} ${FFTW}) 
	add_definitions(-DHAVE_LIBFFTW)
endif()

# gsl
find_library(GSL NAMES gsl)
if (NOT ${GSL} STREQUAL "GSL-NOTFOUND")
	message (STATUS "using gsl: ${GSL}")
	set(LIBS ${LIBS} ${GSL}) 
	add_definitions(-DHAVE_LIBGSL)
endif()

# gslcblas
find_library(GSLCBLAS NAMES gslcblas)
if (NOT ${GSLCBLAS} STREQUAL "GSLCBLAS-NOTFOUND")
	message (STATUS "using gslcblas: ${GSL}")
	set(LIBS ${LIBS} ${GSLCBLAS}) 
	add_definitions(-DHAVE_LIBGSLCBLAS)
endif()

# hdf4
find_library (HDF4 NAMES mfhdf)
if (NOT ${HDF4} STREQUAL "HDF4-NOTFOUND")
	message (STATUS "using hdf4: ${HDF4}")
	set(LIBS ${LIBS} ${HDF4})
	include_directories(BEFORE "/usr/include/hdf")
	add_definitions(-DHAVE_LIBMFHDF)
endif()
find_library (DF NAMES df)
if (NOT ${DF} STREQUAL "DF-NOTFOUND")
	message (STATUS "using df: ${DF}")
	set(LIBS ${LIBS} ${DF})
	add_definitions(-DHAVE_LIBDF)
endif()

#libhdf5_hl
find_library(HDF5HL NAMES hdf5_hl)
if (NOT ${HDF5HL} STREQUAL "HDF5HL-NOTFOUND")
	message (STATUS "using libhdf5_hl: ${HDF5HL}")
	set(LIBS ${LIBS} ${HDF5HL}) 
	add_definitions(-DHAVE_LIBHDF5HL)
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

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# using Qt meta-system (precompiler)
QT4_ADD_RESOURCES( RES_SOURCES ${RESOURCES} )
QT4_WRAP_UI( UI_HEADERS ${UIS} )
QT4_WRAP_CPP( MOC_SRCS ${MOC_HEADERS} ) # probabilmente auto_moc interferisce con questo

add_executable(Neutrino MACOSX_BUNDLE
 ${SOURCES} ${MOC_SRCS} ${RES_SOURCES} ${UI_HEADERS} )

target_link_libraries(Neutrino ${LIBS})
target_link_libraries(Neutrino ${QT_LIBRARIES} Qt4::QtSvg)

# nPhysImage
include_directories(../nPhysImage)
target_link_libraries(Neutrino "-L../nPhysImage -lnPhysImageF -lm")
if (APPLE) 
    target_link_libraries(Neutrino "-framework IOKit -framework CoreFoundation")
endif()



## ------- TODO ----------
# carefully check for openmp
