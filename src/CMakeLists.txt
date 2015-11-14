# src CMakeLists.txt

# ref: http://wiki.qt.io/Using_CMake_build_system 

cmake_minimum_required (VERSION 3.0)

project (Neutrino CXX)

set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../resources/cmake)

include(FindGit)
set (VERSION_UPDATE_FROM_GIT ON)
include(GetVersionFromGitTag)

message (STATUS "${PROJECT_NAME} version : ${${PROJECT_NAME}_VERSION_STRING_FULL}")
set (CMAKE_CXX_FLAGS "-std=c++11")
set (CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -Wall -D__phys_debug=10")

add_definitions(-D__VER="${${PROJECT_NAME}_VERSION_STRING_FULL}")

if (APPLE) 
    message(STATUS "Apple : remember to use macports !")
    set (CMAKE_C_COMPILER "/opt/local/bin/gcc")
    set (CMAKE_CXX_COMPILER "/opt/local/bin/g++")
    set (CMAKE_PREFIX_PATH "/opt/local")
    SET (CMAKE_EXE_LINKER_FLAGS  "-headerpad_max_install_names" )
    add_compile_options(-fPIC)
    add_compile_options(-mmacosx-version-min=10.6)
    
    SET (MACOSX_BUNDLE_BUNDLE_VERSION "${${PROJECT_NAME}_VERSION_AHEAD}")
    SET (MACOSX_BUNDLE_SHORT_VERSION_STRING "${${PROJECT_NAME}_VERSION_AHEAD}")
    SET (MACOSX_BUNDLE_COPYRIGHT "GPL")
    SET (MACOSX_BUNDLE_GUI_IDENTIFIER "edu.polytechnique.neutrino")

    SET (MACOSX_BUNDLE_BUNDLE_NAME "Neutrino")
    SET (MACOSX_BUNDLE_LONG_VERSION_STRING "${${PROJECT_NAME}_VERSION_STRING_FULL}")
    SET (MACOSX_BUNDLE_INFO_STRING  "Neutrino program" )
    SET (MACOSX_BUNDLE_ICON_FILE  "icon.icns" )
    set(myApp_ICON ${CMAKE_CURRENT_SOURCE_DIR}/../resources/icons/icon.icns)
    set_source_files_properties(${myApp_ICON} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
endif()

if (UNIX AND (NOT APPLE))
    message(STATUS "Linux")
    set (CMAKE_PREFIX_PATH "/usr")
    include_directories(BEFORE "/usr/include")
endif()



add_compile_options(-std=c++0x)


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set (LIBPATHS /usr/lib /opt/local/lib ../lib)

## sources
set (SOURCES 
    nScene.cc neutrino.cc neutrinoPalettes.cc graphics/nRect.cc graphics/nMouse.cc 
    graphics/nLine.cc graphics/nTics.cc graphics/nPoint.cc graphics/nEllipse.cc 
    panThread.cc nPlug.cc pans/nFocalSpot.cc pans/nPhysProperties.cc  
    pans/VISAR/nVisarZoomer.cc pans/VISAR/nVisar.cc pans/nFindPeaks.cc 
    pans/nInterpolatePath.cc pans/nOpenRAW.cc pans/nBlur.cc pans/nSpectralAnalysis.cc 
    pans/colorbar/nHistogram.cc pans/colorbar/nColorBarWin.cc pans/nInterferometry.cc 
    pans/nGhost.cc pans/nMouseInfo.cc pans/nBoxLineout.cc pans/nPreferences.cc 
    pans/nAffine.cc pans/nMonitor.cc pans/nRotate.cc pans/nWavelet.cc pans/nRegionPath.cc
    pans/nIntegralInversion.cc pans/nLineoutBoth.cc pans/winlist/nTreeWidget.cc 
    pans/winlist/nWinList.cc pans/nCutoffMask.cc pans/nCompareLines.cc 
    pans/nShortcuts.cc pans/nOperator.cc pans/nLineout.cc nView.cc nGenericPan.cc 
    osxApp.cc nApp.cc main.cc 
)

## UIs
set (UIS 
    ../UIs/nCompareLines.ui ../UIs/nFocalSpot.ui ../UIs/nPhysProperties.ui 
    ../UIs/nWinList.ui ../UIs/nInterferometry1.ui ../UIs/nVISAR2.ui 
    ../UIs/nSpectralAnalysis.ui ../UIs/nVISAR3.ui ../UIs/nLine.ui ../UIs/nRegionPath.ui 
    ../UIs/nGhost.ui ../UIs/nInterpolatePath.ui ../UIs/nIntegralInversion.ui 
    ../UIs/nPreferences.ui ../UIs/nOpenRAW.ui ../UIs/nColorBarWin.ui 
    ../UIs/nOperator.ui ../UIs/nCutoffMask.ui ../UIs/nBoxLineout.ui ../UIs/nLineout.ui 
    ../UIs/nPoint.ui ../UIs/nVISAR1.ui ../UIs/nSbarra.ui ../UIs/nMonitor.ui 
    ../UIs/nFindPeaks.ui ../UIs/nShortcuts.ui ../UIs/nAffine.ui ../UIs/nBlur.ui 
    ../UIs/nLineoutBoth.ui ../UIs/nWavelet.ui ../UIs/nInterferometry.ui ../UIs/nRotate.ui
    ../UIs/nMouseInfo.ui ../UIs/nObject.ui ../UIs/neutrino.ui ../UIs/nFileInfo.ui	
)

## when NEUTRINO_CHECKREG, __neutrino_check_registration is defined
## add relevant sources
if (GENERATE_NHASH)
	set (SOURCES ${SOURCES}
		neutrino-key/nHash.cc
		)

	include_directories(neutrino-key)
	
	if (UNIX AND (NOT APPLE))
		add_definitions(-D__neutrino_reg_linux)
	endif()

endif()

## resources
set (RESOURCES "${${PROJECT_NAME}_SOURCE_DIR}/../resources/neutrino.qrc")

# sub-structure (wrong: we should probably replicate CMakeLists.txt in every subfolder)
include_directories(graphics doc pans pans/VISAR pans/colorbar pans/winlist)

## find qt -- search for 5.x first, fallback to 4.x
find_package(Qt5 COMPONENTS Core Gui Sql Widgets Svg PrintSupport QUIET)
if (Qt5_FOUND)
	# qt5
	SET (USE_QT5 True)
	include_directories(${Qt5Core_INCLUDE_DIRS} ${Qt5Gui_INCLUDE_DIRS} ${Qt5Sql_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${Qt5Svg_INCLUDE_DIRS} ${Qt5PrintSupport_INCLUDE_DIRS})
	
	QT5_ADD_RESOURCES( RES_SOURCES ${RESOURCES} )
	QT5_WRAP_UI( UI_HEADERS ${UIS} )
	add_definitions(-DUSE_QT5)
else()
	# some incompatibilities between 4.x and 5.x
	# qt4
	SET (USE_QT4 True)
	message(STATUS "Qt5 not found, searching for Qt4 instead")
	find_package(Qt4 4.7.0 COMPONENTS QtMain QtCore QtGui QtSQL REQUIRED)
	include(UseQt4)
	include(${QT_USE_FILE})
	
	QT4_ADD_RESOURCES( RES_SOURCES ${RESOURCES} )
	QT4_WRAP_UI( UI_HEADERS ${UIS} )
	add_definitions(-DUSE_QT4)
endif()


add_definitions(${QT_DEFINITIONS})
include_directories(${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})


list(APPEND CMAKE_MODULE_PATH "../lib")

# cmake cannot look for nPhysImageF at this stage, for it will ALWAYS fail..
include_directories(../nPhysImage)

set(LIBS ${LIBS} "nPhysImageF") 
add_definitions(-DHAVE_NPHYSIMAGE)

#endif()

find_package(Qwt REQUIRED)
if (QWT_FOUND)
	include_directories(${QWT_INCLUDE_DIRS})
	message(STATUS "qwt link: " ${QWT_LIBRARIES})
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
else()
	#message ("----------------- non ho trovato hdf4 della fungia")
endif()


find_library (DF NAMES df)
if (NOT ${DF} STREQUAL "DF-NOTFOUND")
	message (STATUS "using df: ${DF}")
	set(LIBS ${LIBS} ${DF})
	add_definitions(-DHAVE_LIBDF)
endif()


# pgm
find_library(PBM NAMES netpbm)
if (NOT ${PBM} STREQUAL "PBM-NOTFOUND")
	message (STATUS "using netpbm: ${PBM}")
	set(LIBS ${LIBS} ${PBM}) 
	add_definitions(-DHAVE_LIBNETPBM)
	if (APPLE) 
		include_directories(BEFORE "/opt/local/include/netpbm")
	endif()
endif()

find_package(JPEG REQUIRED)
if (JPEG_FOUND)
	include_directories(${JPEG_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${JPEG_LIBRARIES})
	add_definitions(-DHAVE_JPEG)
endif()

find_package(TIFF REQUIRED)
if (TIFF_FOUND)
	include_directories(${TIFF_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${TIFF_LIBRARIES})
	add_definitions(-DHAVE_TIFF)
endif()


#libhdf5_hl
message ("---- looking for the entire HDF5 mess...")
find_package(HDF5)
if (HDF5_FOUND)

	message(STATUS "HDF5 Found, now looking for HL")

	# IF HDF5 is there, THEN look for hl...
	find_library(HDF5HL NAMES hdf5_hl PATHS ${HDF5_LIBRARY_DIRS})
	if (${HDF5HL} STREQUAL "HDF5HL-NOTFOUND")
		message (STATUS "Cannot find HDF5_HL: disabling HDF5 support")
		message (STATUS "Search dir: " ${HDF5_LIBRARY_DIRS})

	else()

		#hdf5 libs
		include_directories(${HDF5_INCLUDE_DIRS})
		set(LIBS ${LIBS} ${HDF5_LIBRARIES})
		add_definitions(-DHAVE_HDF5)

		# hdf5_hl
		message (STATUS "using libhdf5_hl: ${HDF5HL}")
		set(LIBS ${LIBS} ${HDF5HL}) 
		add_definitions(-DHAVE_LIBHDF5HL)

	
		# add nHDF5 sources
		set (SOURCES ${SOURCES} pans/nHDF5.cc)
		set (UIS ${UIS} ../UIs/nHDF5.ui)

	endif()

endif (HDF5_FOUND)


if(APPLE) 
#     add_executable(Neutrino MACOSX_BUNDLE ${myApp_ICON} ${SOURCES} ${MOC_SRCS} ${RES_SOURCES} ${UI_HEADERS} )
    add_executable(${PROJECT_NAME} WIN32 MACOSX_BUNDLE ${myApp_ICON} ${SOURCES} ${MOC_SRCS} ${UI_HEADERS} ${RES_SOURCES})
else()
    add_executable(Neutrino ${SOURCES} ${MOC_SRCS} ${RES_SOURCES} ${UI_HEADERS} )
endif()

if (APPLE) 
    target_link_libraries(Neutrino "-framework IOKit -framework CoreFoundation")
endif()

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


target_link_libraries(Neutrino ${LIBS})
if (USE_QT5)
	qt5_use_modules(Neutrino Core Gui Sql Widgets Svg PrintSupport)
elseif(USE_QT4)
	target_link_libraries(Neutrino ${QT_LIBRARIES} Qt4::QtCore Qt4::QtGui Qt4::QtSvg)
endif()


# output paths
if (APPLE)
    set (EXECUTABLE_OUTPUT_PATH "..")
else()
    set (EXECUTABLE_OUTPUT_PATH "../bin")
    install (TARGETS Neutrino COMPONENT binary DESTINATION bin)
endif()


if(APPLE AND CMAKE_INSTALL_PREFIX MATCHES "/usr/local")
	set(CMAKE_INSTALL_PREFIX "/Applications")
endif()

if(NOT CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]")
	add_definitions(-DQT_NO_DEBUG_OUTPUT)
	set(CMAKE_BUILD_RELEASE TRUE)
else()
	set(CMAKE_BUILD_RELEASE FALSE)
endif()

message(STATUS "Building ${PROJECT_NAME} in ${CMAKE_BUILD_TYPE} mode")

set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
set(PROJECT_VENDOR "neutrino inc.")
set(PROJECT_COPYRIGHT_YEAR "2011")
set(PROJECT_DOMAIN_FIRST "neutrino")
set(PROJECT_DOMAIN_SECOND "com")
set(PROJECT_DOMAIN "${PROJECT_DOMAIN_FIRST}.${PROJECT_DOMAIN_SECOND}")

if(UNIX AND NOT APPLE)
	string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWERCASE)
	set(BIN_INSTALL_DIR "bin")
	set(DOC_INSTALL_DIR "share/doc/${PROJECT_NAME_LOWERCASE}/")
else()
	set(BIN_INSTALL_DIR ".")
	set(DOC_INSTALL_DIR ".")
endif()
set(ICONS_DIR "${${PROJECT_NAME}_SOURCE_DIR}/../resources/icons")

if(CMAKE_COMPILER_IS_GNUCXX OR APPLE)
	add_definitions(-Wall -Wextra)
	if(CMAKE_BUILD_RELEASE)
#		add_definitions(-Werror)
	endif()
endif()


if(APPLE)
	set(MACOSX_BUNDLE_INFO_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_NAME} ${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_LONG_VERSION_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
	set(MACOSX_BUNDLE_COPYRIGHT "${PROJECT_COPYRIGHT_YEAR} ${PROJECT_VENDOR}")
	set(MACOSX_BUNDLE_ICON_FILE "icon.icns")
	set(MACOSX_BUNDLE_GUI_IDENTIFIER "${PROJECT_DOMAIN_SECOND}.${PROJECT_DOMAIN_FIRST}")
	set(MACOSX_BUNDLE_BUNDLE_NAME "${PROJECT_NAME}")

	set(MACOSX_BUNDLE_RESOURCES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.app/Contents/Resources")
	set(MACOSX_BUNDLE_ICON "${ICONS_DIR}/${MACOSX_BUNDLE_ICON_FILE}")
	execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${MACOSX_BUNDLE_RESOURCES})
	execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MACOSX_BUNDLE_ICON} ${MACOSX_BUNDLE_RESOURCES})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})

if(APPLE)
	set(CMAKE_INSTALL_PREFIX "/Applications")
endif()

message(STATUS "${PROJECT_NAME} will be installed to ${CMAKE_INSTALL_PREFIX}")
install(TARGETS ${PROJECT_NAME} DESTINATION ${BIN_INSTALL_DIR})

set(LICENSE_FILE "LICENSE.txt")

if(NOT APPLE)
	install(FILES "${LICENSE_FILE}" "${README_FILE}" DESTINATION ${DOC_INSTALL_DIR})
endif()

set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VENDOR "${PROJECT_VENDOR}")

message(STATUS ${${PROJECT_NAME}_SOURCE_DIR})

set(CPACK_RESOURCE_FILE_README "${README_FILE}")

if(WIN32)
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_PACKAGE_EXECUTABLES "${PROJECT_NAME}" "${PROJECT_NAME}")
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "${PROJECT_NAME}")
    set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME} ${PROJECT_VERSION}")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/${LICENSE_FILE}")
    set(CPACK_NSIS_EXECUTABLES_DIRECTORY "${BIN_INSTALL_DIR}")
    #set(CPACK_NSIS_MUI_ICON "${PROJECT_ICONS_DIRECTORY}/NSIS.ico")
    #set(CPACK_PACKAGE_ICON "${PROJECT_ICONS_DIRECTORY}\\\\NSISHeader.bmp")
    set(CPACK_NSIS_URL_INFO_ABOUT "http://${PROJECT_DOMAIN}")
    set(CPACK_NSIS_INSTALLED_ICON_NAME "${PROJECT_NAME}${CMAKE_EXECUTABLE_SUFFIX}")
    set(CPACK_NSIS_MENU_LINKS "${LICENSE_FILE}" "License" "${README_FILE}" "Readme")
    set(CPACK_NSIS_MUI_FINISHPAGE_RUN "${CPACK_NSIS_INSTALLED_ICON_NAME}")
elseif(APPLE)
     set(CMAKE_INSTALL_PREFIX "/Applications")
     set(CPACK_GENERATOR "DragNDrop")
     set(CPACK_DMG_FORMAT "UDBZ")
     set(CPACK_DMG_VOLUME_NAME "${PROJECT_NAME}")
     set(CPACK_SYSTEM_NAME "OSX")
     set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${${PROJECT_NAME}_VERSION_STRING_FULL}")
     set(CPACK_PACKAGE_ICON "${ICONS_DIR}/icon.icns")
     set(CPACK_DMG_DS_STORE "${ICONS_DIR}/../macPackage/DS_Store")
     set(CPACK_DMG_BACKGROUND_IMAGE "${ICONS_DIR}/../macPackage/sfondo.png")
elseif(UNIX)
    set(CPACK_GENERATOR "TBZ2")
    set(CPACK_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
endif()


set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION "${BIN_INSTALL_DIR}")
include(InstallRequiredSystemLibraries)

if(APPLE)
    set(EXECUTABLE "${PROJECT_NAME}.app")
elseif(WIN32)
    set(EXECUTABLE "${PROJECT_NAME}${CMAKE_EXECUTABLE_SUFFIX}")
else()
    set(EXECUTABLE "${BIN_INSTALL_DIR}/${PROJECT_NAME}${CMAKE_EXECUTABLE_SUFFIX}")
endif()

if(APPLE OR WIN32)
    include(DeployQt4)
    install_qt4_executable("${EXECUTABLE}")
endif()

include (CPack)
