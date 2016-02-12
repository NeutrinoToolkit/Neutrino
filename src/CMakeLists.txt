# src CMakeLists.txt

# ref: http://wiki.qt.io/Using_CMake_build_system 

cmake_minimum_required (VERSION 3.0)

include(FindGit)

message (STATUS "${PROJECT_NAME} version : ${${PROJECT_NAME}_VERSION_STRING_FULL}")
set (CMAKE_CXX_FLAGS "-O3 -std=c++11 -Wall")
set (CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -Wextra -D__phys_debug=10")

add_definitions(-D__VER="${${PROJECT_NAME}_VERSION_STRING_FULL}")

if (APPLE) 
    message(STATUS "Apple : remember to use brew !")
    SET (CMAKE_EXE_LINKER_FLAGS  "-headerpad_max_install_names" )
    add_compile_options(-fPIC)
    add_compile_options(-mmacosx-version-min=10.6)
    
    set(MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/../resources/neutrino.plist")

    set(MACOSX_BUNDLE_INFO_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
    set(MACOSX_BUNDLE_ICON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../resources/icons/icon.icns")
    set(MACOSX_BUNDLE_GUI_IDENTIFIER "${PROJECT_DOMAIN_SECOND}.${PROJECT_DOMAIN_FIRST}")
    set(MACOSX_BUNDLE_LONG_VERSION_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
    set(MACOSX_BUNDLE_BUNDLE_NAME "${PROJECT_NAME}")
    set(MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
    set(MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_NAME} ${PROJECT_VERSION}")
    set(MACOSX_BUNDLE_COPYRIGHT "${PROJECT_COPYRIGHT_YEAR} ${PROJECT_VENDOR}")

    set(MACOSX_BUNDLE_RESOURCES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.app/Contents/Resources")
    set(MACOSX_BUNDLE_ICON "${ICONS_DIR}/${MACOSX_BUNDLE_ICON_FILE}")
    set(MACOSX_BUNDLE_ICON_FILE "icon.icns")

    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${MACOSX_BUNDLE_RESOURCES})
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MACOSX_BUNDLE_ICON} ${MACOSX_BUNDLE_RESOURCES})

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
    message(STATUS "OpenMP CXX: ${OpenMP_CXX_FLAGS} C: ${OpenMP_C_FLAGS} ")
endif()

if (WIN32)
	if (CMAKE_COMPILER_IS_GNUCXX)
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mwindows")
	endif()
endif()


set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set (LIBPATHS /usr/lib /usr/local/lib ../lib)

include(FindNeutrinoGuiComponents)

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
    nApp.cc main.cc 
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
        set (SOURCES ${SOURCES} neutrino-key/nHash.cc)

	include_directories(neutrino-key)
	
	if (UNIX AND (NOT APPLE))
		add_definitions(-D__neutrino_reg_linux)
	endif()

endif()

# HDF5 sources and uis
if (HDF5_FOUND_COMPLETE)
        # add nHDF5 sources
        list (APPEND SOURCES pans/nHDF5.cc)
        list (APPEND UIS ../UIs/nHDF5.ui)
endif()

if (PYTHONQT_FOUND_COMPLETE)
    MESSAGE(STATUS "adding python wrappers")
        include_directories(python)
        list (APPEND SOURCES python/nPhysPyWrapper.cc python/nPython.cc)
        list (APPEND UIS ../UIs/nPython.ui)
endif()


# all dependencies search now moved to FindNeutrinoGuiComponents.cmake
# (see neutrino cmake resources folder)
if (USE_QT5)
	# add here QT5 only sources
	list (APPEND SOURCES pans/nCamera.cc)
	list (APPEND UIS ../UIs/nCamera.ui)

	QT5_ADD_RESOURCES( RES_SOURCES ${RESOURCES} )
	QT5_WRAP_UI( UI_HEADERS ${UIS} )
elseif (USE_QT4)
	QT4_ADD_RESOURCES( RES_SOURCES ${RESOURCES} )
	QT4_WRAP_UI( UI_HEADERS ${UIS} )
endif()

# sub-structure (wrong: we should probably replicate CMakeLists.txt in every subfolder)
include_directories(graphics doc pans pans/VISAR pans/colorbar pans/winlist)




list(APPEND CMAKE_MODULE_PATH "../lib")

# cmake cannot look for nPhysImageF at this stage, for it will ALWAYS fail..
include_directories(../nPhysImage)

set(LIBS ${LIBS} "nPhysImageF") 
add_definitions(-DHAVE_NPHYSIMAGE)

if(APPLE) 
#     add_executable(Neutrino MACOSX_BUNDLE ${myApp_ICON} ${SOURCES} ${MOC_SRCS} ${RES_SOURCES} ${UI_HEADERS} )
    add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${myApp_ICON} ${SOURCES} ${MOC_SRCS} ${UI_HEADERS} ${RES_SOURCES})
elseif(WIN32)
	set (RC_WIN "${${PROJECT_NAME}_SOURCE_DIR}/resources/neutrino.rc")
	add_executable(${PROJECT_NAME} WIN32 ${SOURCES} ${MOC_SRCS} ${UI_HEADERS} ${RES_SOURCES} ${RC_WIN})
else()
    add_executable(Neutrino ${SOURCES} ${MOC_SRCS} ${RES_SOURCES} ${UI_HEADERS} )
endif()

if (APPLE) 
    target_link_libraries(Neutrino "-framework IOKit -framework CoreFoundation")
endif()

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


target_link_libraries(Neutrino ${LIBS})
if (USE_QT5)
        qt5_use_modules(Neutrino Core Gui Sql Widgets Svg PrintSupport UiTools Multimedia MultimediaWidgets)
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
set(ICONS_DIR "${${PROJECT_NAME}_SOURCE_DIR}/resources/icons")

if(CMAKE_COMPILER_IS_GNUCXX OR APPLE)
	#add_definitions(-Wall -Wextra)
	# -Wxx are compiler flags and NOT definitions (otherwise: messes up resource compiler)
	if(CMAKE_BUILD_RELEASE)
#		add_definitions(-Werror)
	endif()
endif()


include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})

if(APPLE)
	set(CMAKE_INSTALL_PREFIX "/Applications")
endif()

message(STATUS "${PROJECT_NAME} will be installed to ${CMAKE_INSTALL_PREFIX}")
#install(TARGETS ${PROJECT_NAME} DESTINATION ${BIN_INSTALL_DIR})

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
#    set(CPACK_GENERATOR "NSIS")
#    set(CPACK_PACKAGE_EXECUTABLES "${PROJECT_NAME}" "${PROJECT_NAME}")
#    set(CPACK_PACKAGE_INSTALL_DIRECTORY "${PROJECT_NAME}")
#    set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME} ${PROJECT_VERSION}")
#    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/${LICENSE_FILE}")
#    set(CPACK_NSIS_EXECUTABLES_DIRECTORY "${BIN_INSTALL_DIR}")
#    #set(CPACK_NSIS_MUI_ICON "${PROJECT_ICONS_DIRECTORY}/NSIS.ico")
#    #set(CPACK_PACKAGE_ICON "${PROJECT_ICONS_DIRECTORY}\\\\NSISHeader.bmp")
#    set(CPACK_NSIS_URL_INFO_ABOUT "http://${PROJECT_DOMAIN}")
#    set(CPACK_NSIS_INSTALLED_ICON_NAME "${PROJECT_NAME}${CMAKE_EXECUTABLE_SUFFIX}")
#    set(CPACK_NSIS_MENU_LINKS "${LICENSE_FILE}" "License" "${README_FILE}" "Readme")
#    set(CPACK_NSIS_MUI_FINISHPAGE_RUN "${CPACK_NSIS_INSTALLED_ICON_NAME}")
elseif(APPLE)
     set(CMAKE_INSTALL_PREFIX "/Applications")
     set(CPACK_GENERATOR "Bundle")
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

#if(APPLE OR WIN32)
if(APPLE)
    include(DeployQt4)
    install_qt4_executable("${EXECUTABLE}")
endif()


#include (CPack)
