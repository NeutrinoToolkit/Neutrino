# src CMakeLists.txt

# ref: http://wiki.qt.io/Using_CMake_build_system 

cmake_minimum_required (VERSION 3.0)

include(FindGit)

message (STATUS "${PROJECT_NAME} version : ${${PROJECT_NAME}_VERSION_STRING_FULL}")

add_definitions(-D__VER="${${PROJECT_NAME}_VERSION_STRING_FULL}")

set(ICONS_DIR "${${PROJECT_NAME}_SOURCE_DIR}/resources/icons")


if (APPLE)
    message(STATUS "Apple : remember to use brew !")
    SET (CMAKE_EXE_LINKER_FLAGS  "-headerpad_max_install_names" )
    add_compile_options(-fvisibility=hidden)

    set(MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/../resources/neutrino.plist")

    set(MACOSX_BUNDLE_INFO_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
    set(MACOSX_BUNDLE_ICON_FILE "icon.icns")
    set(MACOSX_BUNDLE_GUI_IDENTIFIER "${PROJECT_DOMAIN_SECOND}.${PROJECT_DOMAIN_FIRST}")
    set(MACOSX_BUNDLE_LONG_VERSION_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
    set(MACOSX_BUNDLE_BUNDLE_NAME "${PROJECT_NAME}")
    set(MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
    set(MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_NAME} ${PROJECT_VERSION}")
    set(MACOSX_BUNDLE_COPYRIGHT "${PROJECT_COPYRIGHT_YEAR} ${PROJECT_VENDOR}")

    set(MACOSX_BUNDLE_RESOURCES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.app/Contents/Resources")
    set(MACOSX_BUNDLE_ICON "${ICONS_DIR}/${MACOSX_BUNDLE_ICON_FILE}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${MACOSX_BUNDLE_RESOURCES})
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MACOSX_BUNDLE_ICON} ${MACOSX_BUNDLE_RESOURCES})

    set(myApp_ICON ${CMAKE_CURRENT_SOURCE_DIR}/../resources/icons/icon.icns)
    set_source_files_properties(${myApp_ICON} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
endif()

if (LINUX)
    set (CMAKE_PREFIX_PATH "/usr")
    include_directories(BEFORE "/usr/include")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
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


## sources
file(GLOB UIS ${CMAKE_CURRENT_SOURCE_DIR}/../UIs/*.ui)

list (APPEND SOURCES
    neutrino.cc
    nGenericPan.cc panThread.cc nPluginLoader.cc nPlug.cc nApp.cc main.cc

    graphics/neutrinoPalettes.cc graphics/nView.cc graphics/nRect.cc graphics/nMouse.cc
    graphics/nLine.cc graphics/nTics.cc graphics/nPoint.cc graphics/nEllipse.cc 
    graphics/spline.cc graphics/qcustomplot.cpp graphics/nCustomPlots.cpp

    pans/nFocalSpot.cc pans/nPhysProperties.cc pans/nVisar.cc pans/nFindPeaks.cc pans/nContours.cc
    pans/nInterpolatePath.cc pans/nOpenRAW.cc pans/nSpectralAnalysis.cc
    pans/colorbar/nHistogram.cc pans/colorbar/nColorBarWin.cc pans/nInterferometry.cc 
    pans/nGhost.cc pans/nMouseInfo.cc pans/nBoxLineout.cc pans/nPreferences.cc 
    pans/nAffine.cc pans/nMonitor.cc pans/nRotate.cc pans/nWavelet.cc pans/nRegionPath.cc
    pans/nIntegralInversion.cc pans/nLineoutBoth.cc pans/winlist/nTreeWidget.cc 
    pans/winlist/nWinList.cc pans/nCutoffMask.cc pans/nCompareLines.cc 
    pans/nShortcuts.cc pans/nOperator.cc pans/nLineout.cc pans/nZoomWin.cc
)

file(GLOB UIS ${CMAKE_CURRENT_SOURCE_DIR}/../UIs/*.ui)

#moving it to main file
#include(FindNeutrinoGuiComponents)

if (PYTHONQT_FOUND_COMPLETE)
    MESSAGE(STATUS "adding python wrappers")
    include_directories(python)
    list (APPEND SOURCES python/nPhysPyWrapper.cc python/nPython.cc python/codeeditor.cpp)
endif()


## when NEUTRINO_CHECKREG, __neutrino_check_registration is defined
## add relevant sources
if (GENERATE_NHASH)
    set (SOURCES ${SOURCES} neutrino-key/nHash.cc)

    include_directories(neutrino-key)

    if (UNIX AND (NOT APPLE))
        add_definitions(-D__neutrino_reg_linux)
    endif()

endif()


set(QRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/../qrc)
file(MAKE_DIRECTORY ${QRCDIR})

# write automatically the qrc file for icons
set(ICONSDIR ${CMAKE_CURRENT_SOURCE_DIR}/../resources/icons)
file(GLOB_RECURSE ICONS ${ICONSDIR}/*.png)
set(ICONS_QRC ${QRCDIR}/icons.qrc)
file(WRITE ${ICONS_QRC} "<RCC>\n    <qresource prefix=\"/icons/\">\n")
foreach(my_file ${ICONS})
    file(RELATIVE_PATH my_file_relative_path ${CMAKE_CURRENT_SOURCE_DIR}/../src ${my_file})
    GET_FILENAME_COMPONENT(my_file_basename ${my_file} NAME)
    file(APPEND ${ICONS_QRC} "        <file alias=\"${my_file_basename}\">${my_file_relative_path}</file>\n")
endforeach()
file(APPEND ${ICONS_QRC} "    </qresource>\n</RCC>")
list(LENGTH ICONS LIST_LENGTH)
message(STATUS "${LIST_LENGTH} icon files: ${ICONS_QRC}")

# write automatically the qrc file for html
set(HTMLDIR ${CMAKE_CURRENT_SOURCE_DIR}/../resources/html)
file(GLOB HTMLS ${HTMLDIR}/*.html)
set(HTML_QRC ${QRCDIR}/html.qrc)
file(WRITE ${HTML_QRC} "<RCC>\n    <qresource prefix=\"/html/\">\n")
foreach(my_file ${HTMLS})
    file(RELATIVE_PATH my_file_relative_path ${CMAKE_CURRENT_SOURCE_DIR}/../src ${my_file})
    GET_FILENAME_COMPONENT(my_file_basename ${my_file} NAME)
    file(APPEND ${HTML_QRC} "        <file alias=\"${my_file_basename}\">${my_file_relative_path}</file>\n")
endforeach()
file(APPEND ${HTML_QRC} "    </qresource>\n</RCC>")
list(LENGTH HTMLS LIST_LENGTH)
message(STATUS "${LIST_LENGTH} html files: ${HTML_QRC}")

file(GLOB_RECURSE HTMLLICENSES ${HTMLDIR}/licenses/*)
set(LICENSES_QRC ${QRCDIR}/licenses.qrc)
file(WRITE ${LICENSES_QRC} "<RCC>\n    <qresource prefix=\"/licenses/\">\n")
foreach(my_file ${HTMLLICENSES})
    file(RELATIVE_PATH my_file_relative_path ${CMAKE_CURRENT_SOURCE_DIR}/../src ${my_file})
    GET_FILENAME_COMPONENT(my_file_basename ${my_file} NAME)
    file(APPEND ${LICENSES_QRC} "        <file alias=\"${my_file_basename}\">${my_file_relative_path}</file>\n")
endforeach()
file(APPEND ${LICENSES_QRC} "    </qresource>\n</RCC>")
list(LENGTH HTMLLICENSES LIST_LENGTH)
message(STATUS "${LIST_LENGTH} license files: ${LICENSES_QRC}")

file(GLOB_RECURSE HTMLHELP ${HTMLDIR}/help/*)
set(HELP_QRC ${QRCDIR}/help.qrc)
file(WRITE ${HELP_QRC} "<RCC>\n    <qresource prefix=\"/help/\">\n")
foreach(my_file ${HTMLHELP})
    file(RELATIVE_PATH my_file_relative_path ${CMAKE_CURRENT_SOURCE_DIR}/../src ${my_file})
    GET_FILENAME_COMPONENT(my_file_basename ${my_file} NAME)
    GET_FILENAME_COMPONENT(my_file_dir ${my_file} DIRECTORY)
    GET_FILENAME_COMPONENT(my_file_dir_clean ${my_file_dir} NAME)
    file(APPEND ${HELP_QRC} "        <file alias=\"${my_file_dir_clean}/${my_file_basename}\">${my_file_relative_path}</file>\n")
endforeach()
file(APPEND ${HELP_QRC} "    </qresource>\n</RCC>")
list(LENGTH HTMLHELP LIST_LENGTH)
message(STATUS "${LIST_LENGTH} help files: ${HELP_QRC}")

# all dependencies search now moved to FindNeutrinoGuiComponents.cmake
# (see neutrino cmake resources folder)
if (USE_QT5)
    # add here QT5 only sources
    list (APPEND SOURCES pans/nCamera.cc)
    list (APPEND UIS ../UIs/nCamera.ui)

    qt5_add_resources(RES_SOURCES ${ICONS_QRC})
    qt5_add_resources(RES_SOURCES ${HTML_QRC})
    qt5_add_resources(RES_SOURCES ${LICENSES_QRC})
    qt5_add_resources(RES_SOURCES ${HELP_QRC})


    if (PYTHONQT_FOUND_COMPLETE)
        qt5_add_resources(RES_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../resources/python/python.qrc)
    endif()

    QT5_WRAP_UI( UI_HEADERS ${UIS} )
elseif (USE_QT4)
    qt4_add_resources(RES_SOURCES ${ICONS_QRC})
    qt4_add_resources(RES_SOURCES ${HTML_QRC})
    qt4_add_resources(RES_SOURCES ${LICENSES_QRC})
    qt4_add_resources(RES_SOURCES ${HELP_QRC})

    if (PYTHONQT_FOUND_COMPLETE)
        qt4_add_resources(RES_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../resources/python/python.qrc)
    endif()

    QT4_WRAP_UI( UI_HEADERS ${UIS} )
endif()

# sub-structure (wrong: we should probably replicate CMakeLists.txt in every subfolder)
include_directories(graphics doc pans pans/colorbar pans/winlist)


#translate ui
if (APPLE)
  SET(Qt5LinguistTools_DIR "${Qt5_DIR}/../Qt5LinguistTools")
endif(APPLE)

find_package(Qt5LinguistTools)

if (Qt5LinguistTools_FOUND)
  option (UPDATE_TRANSLATIONS "Update source translation translations/*.ts" off)
  SET(LANGUAGES fr_FR it_IT ko_KP)

  SET(LANGUAGE_TS_FILES)
  FOREACH(LANGUAGE ${LANGUAGES})
    message(STATUS "Language : add ${LANGUAGE}")
    SET(TS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../resources/translations/neutrino_${LANGUAGE}.ts")
    message(STATUS "Language : ${TS_FILE}")
    SET(LANGUAGE_TS_FILES ${LANGUAGE_TS_FILES} ${TS_FILE})
  ENDFOREACH(LANGUAGE ${LANGUAGES})

  IF(LANGUAGE_TS_FILES)

    if (UPDATE_TRANSLATIONS)
      qt5_create_translation(qm_files ${SOURCES} ${UIS} ${LANGUAGE_TS_FILES})
      message(STATUS "qm_files created: ${qm_files}")
    else (UPDATE_TRANSLATIONS)
      qt5_add_translation(qm_files ${LANGUAGE_TS_FILES})
       message(STATUS "qm_files added : ${qm_files}")
    endif (UPDATE_TRANSLATIONS) 


    set(TRANSL_QRC ${QRCDIR}/translations.qrc)
    file(WRITE ${TRANSL_QRC} "<RCC>\n    <qresource prefix=\"/translations/\">\n")
    foreach(my_file ${qm_files})
        file(RELATIVE_PATH my_file_relative_path ${CMAKE_CURRENT_SOURCE_DIR}/../src ${my_file})
        GET_FILENAME_COMPONENT(my_file_basename ${my_file} NAME)
        file(APPEND ${TRANSL_QRC} "        <file alias=\"${my_file_basename}\">${my_file_relative_path}</file>\n")
    endforeach()
    file(APPEND ${TRANSL_QRC} "    </qresource>\n</RCC>")
    list(LENGTH LANGUAGE_TS_FILES LIST_LENGTH)
    message(STATUS "${LIST_LENGTH} translation files: ${TRANSL_QRC}")



    qt5_add_resources(RES_SOURCES ${TRANSL_QRC})
  ENDIF(LANGUAGE_TS_FILES)


endif(Qt5LinguistTools_FOUND)



list(APPEND CMAKE_MODULE_PATH "../lib")

# cmake cannot look for nPhysImageF at this stage, for it will ALWAYS fail..
include_directories(../nPhysImage)

set(LIBS ${LIBS} "nPhysImageF") 
add_definitions(-DHAVE_NPHYSIMAGE)

if(APPLE) 
    add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${myApp_ICON} ${SOURCES} ${MOC_SRCS} ${UI_HEADERS} ${RES_SOURCES} ${ICONS_QRC} ${HTML_QRC} ${qm_files})
    target_link_libraries(Neutrino "-framework IOKit -framework CoreFoundation")
endif()

if(WIN32)
    set (RC_WIN "${${PROJECT_NAME}_SOURCE_DIR}/resources/neutrino.rc")
    add_executable(${PROJECT_NAME} WIN32 ${SOURCES} ${MOC_SRCS} ${UI_HEADERS} ${RES_SOURCES}  ${ICONS_QRC} ${HTML_QRC} ${RC_WIN} ${qm_files})
endif()

if(LINUX)
    add_executable(${PROJECT_NAME} ${SOURCES} ${MOC_SRCS} ${UI_HEADERS} ${RES_SOURCES} ${ICONS_QRC} ${HTML_QRC} ${qm_files})
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

if(LINUX)
    string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWERCASE)
    set(BIN_INSTALL_DIR "bin")
    set(DOC_INSTALL_DIR "share/doc/${PROJECT_NAME_LOWERCASE}/")
else()
    set(BIN_INSTALL_DIR ".")
    set(DOC_INSTALL_DIR ".")
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR APPLE)
    #add_definitions(-Wall -Wextra)
    # -Wxx are compiler flags and NOT definitions (otherwise: messes up resource compiler)
    if(CMAKE_BUILD_RELEASE)
#        add_definitions(-Werror)
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

set(CPACK_RESOURCE_FILE_README "${README_FILE}")


if(LINUX)
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

if (APPLE)
set (CMAKE_INSTALL_RPATH "/usr/local/lib")
set (CMAKE_MACOSX_RPATH off)
endif()


