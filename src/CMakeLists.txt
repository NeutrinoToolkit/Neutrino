
# if (WIN32 AND CMAKE_COMPILER_IS_GNUCXX)
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mwindows")
# endif()
# -mwindows is a link option: add_executable(... WIN32 ...) below already gives it (Debug keeps the console)


find_package(Git REQUIRED)
execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	OUTPUT_VARIABLE VERSION_GIT_SHA
	OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

## sources
# CONFIGURE_DEPENDS: new files are picked up by the build, without re-running cmake by hand
# file(GLOB_RECURSE UIS *.ui)
file(GLOB_RECURSE UIS CONFIGURE_DEPENDS *.ui)

# file(GLOB_RECURSE SOURCES *.cc *.cpp *.cxx *.h)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cc *.cpp *.cxx *.h)

# add_compile_definitions(QT_DISABLE_DEPRECATED_BEFORE=0x050F00)
# now in the top CMakeLists.txt, for the plugins too

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE("SOURCES:\n${SOURCES}")
endif()

list(LENGTH SOURCES LIST_LENGTH)
message(STATUS "${LIST_LENGTH} source files")

# include(FindNeutrinoDeps)
# the dependencies come from the nPhysImageF usage requirements (nPhysImage/CMakeLists.txt)

set(LIBS ${LIBS} "nPhysImageF")
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nPhysImage)

# add_definitions(-DSHAVERSION__="${VERSION_GIT_SHA}")
# the sha is written at build time in nVersion.cc, only when it changes (see nVersion.h):
# a commit recompiles one file instead of all of src, and the sha is always the current one
# (VERSION_GIT_SHA above, from configure time, is still used for the macOS Info.plist)
set(VERSION_CC ${CMAKE_CURRENT_BINARY_DIR}/nVersion.cc)
add_custom_target(NeutrinoVersion
    COMMAND ${CMAKE_COMMAND} -DGIT_EXECUTABLE=${GIT_EXECUTABLE} -DSOURCE_DIR=${CMAKE_SOURCE_DIR} -DOUT=${VERSION_CC} -P ${CMAKE_SOURCE_DIR}/resources/cmake/GitVersion.cmake
    BYPRODUCTS ${VERSION_CC}
    VERBATIM)
set_source_files_properties(${VERSION_CC} PROPERTIES GENERATED TRUE SKIP_AUTOGEN ON)

set(MODULES Core Gui Sql Widgets Svg PrintSupport Network UiTools Multimedia MultimediaWidgets Qml DBus Charts)

find_package(Qt6 COMPONENTS ${MODULES} REQUIRED)

set(QRCDIR ${CMAKE_CURRENT_BINARY_DIR}/qrc)

# write automatically the qrc file for icons
set(ICONSDIR ${CMAKE_CURRENT_SOURCE_DIR}/../resources/icons)
# the qrc files are written only when their content changes (file(WRITE) at every configure made rcc run again
# and relinked Neutrino): the text is collected in QRC_CONTENT and written by file(CONFIGURE)
# file(GLOB_RECURSE ICONS ${ICONSDIR}/*.png)
file(GLOB_RECURSE ICONS CONFIGURE_DEPENDS ${ICONSDIR}/*.png)
set(ICONS_QRC ${QRCDIR}/icons.qrc)
# file(WRITE ${ICONS_QRC} "<RCC>\n    <qresource prefix=\"/icons/\">\n")
set(QRC_CONTENT "<RCC>\n    <qresource prefix=\"/icons/\">\n")
foreach(my_file ${ICONS})
    file(RELATIVE_PATH my_file_relative_path ${QRCDIR} ${my_file})
    GET_FILENAME_COMPONENT(my_file_basename ${my_file} NAME)
    # file(APPEND ${ICONS_QRC} "        <file alias=\"${my_file_basename}\">${my_file_relative_path}</file>\n")
    string(APPEND QRC_CONTENT "        <file alias=\"${my_file_basename}\">${my_file_relative_path}</file>\n")
endforeach()
# file(APPEND ${ICONS_QRC} "    </qresource>\n</RCC>")
string(APPEND QRC_CONTENT "    </qresource>\n</RCC>")
file(CONFIGURE OUTPUT ${ICONS_QRC} CONTENT "${QRC_CONTENT}" @ONLY)


# write automatically the qrc file for html
set(LICENSESDIR ${CMAKE_CURRENT_SOURCE_DIR}/../resources/licenses)
#parse neutrino license file
# file(GLOB_RECURSE HTMLLICENSES ${LICENSESDIR}/*.txt)
file(GLOB_RECURSE HTMLLICENSES CONFIGURE_DEPENDS ${LICENSESDIR}/*.txt)
# list(APPEND HTMLLICENSES "${CPACK_RESOURCE_FILE_LICENSE}")
# CPACK_RESOURCE_FILE_LICENSE is set later (Package.cmake) and would clash with the alias of resources/licenses/Neutrino.txt
set(LICENSES_QRC ${QRCDIR}/licenses.qrc)
# file(WRITE ${LICENSES_QRC} "<RCC>\n    <qresource prefix=\"/licenses/\">\n")
set(QRC_CONTENT "<RCC>\n    <qresource prefix=\"/licenses/\">\n")
foreach(my_file ${HTMLLICENSES})
    file(RELATIVE_PATH my_file_relative_path ${QRCDIR} ${my_file})
    GET_FILENAME_COMPONENT(my_file_basename ${my_file} NAME)
    # file(APPEND ${LICENSES_QRC} "        <file alias=\"${my_file_basename}\">${my_file_relative_path}</file>\n")
    string(APPEND QRC_CONTENT "        <file alias=\"${my_file_basename}\">${my_file_relative_path}</file>\n")
endforeach()
# file(APPEND ${LICENSES_QRC} "    </qresource>\n</RCC>")
string(APPEND QRC_CONTENT "    </qresource>\n</RCC>")
file(CONFIGURE OUTPUT ${LICENSES_QRC} CONTENT "${QRC_CONTENT}" @ONLY)

#parse neutrino colormap files
set(CMAPDIR ${CMAKE_CURRENT_SOURCE_DIR}/../resources/colormaps/cmaps)
# file(GLOB_RECURSE CMAPS ${CMAPDIR}/*)
file(GLOB_RECURSE CMAPS CONFIGURE_DEPENDS ${CMAPDIR}/*)
set(CMAP_QRC ${QRCDIR}/cmap.qrc)
# file(WRITE ${CMAP_QRC} "<RCC>\n    <qresource prefix=\"/cmaps/\">\n")
set(QRC_CONTENT "<RCC>\n    <qresource prefix=\"/cmaps/\">\n")
foreach(my_file ${CMAPS})
    file(RELATIVE_PATH my_file_relative_path ${QRCDIR} ${my_file})
    GET_FILENAME_COMPONENT(my_file_basename ${my_file} NAME)
    # file(APPEND ${CMAP_QRC} "        <file alias=\"${my_file_basename}\">${my_file_relative_path}</file>\n")
    string(APPEND QRC_CONTENT "        <file alias=\"${my_file_basename}\">${my_file_relative_path}</file>\n")
endforeach()
# file(APPEND ${CMAP_QRC} "    </qresource>\n</RCC>")
string(APPEND QRC_CONTENT "    </qresource>\n</RCC>")
file(CONFIGURE OUTPUT ${CMAP_QRC} CONTENT "${QRC_CONTENT}" @ONLY)


# sub-structure (wrong: we should probably replicate CMakeLists.txt in every subfolder)
include_directories(graphics pans)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${QRCDIR} ${ICONSDIR})

# set(filedeps ${SOURCES} ${UIS} ${ICONS_QRC} ${LICENSES_QRC} ${CMAP_QRC})
set(filedeps ${SOURCES} ${UIS} ${ICONS_QRC} ${LICENSES_QRC} ${CMAP_QRC} ${VERSION_CC})

if(APPLE)


    set(MACOSX_BUNDLE_EXECUTABLE_NAME "${PROJECT_NAME}")
    set(MACOSX_BUNDLE_INFO_STRING "${PROJECT_NAME} ${VERSION_GIT_SHA}")
    set(MACOSX_BUNDLE_GUI_IDENTIFIER "${PROJECT_DOMAIN_SECOND}.${PROJECT_DOMAIN_FIRST}")

    set(MACOSX_BUNDLE_BUNDLE_NAME "${PROJECT_NAME}")
    set(MACOSX_BUNDLE_BUNDLE_VERSION "${VERSION_GIT_SHA}")
    set(MACOSX_BUNDLE_SHORT_VERSION_STRING "${VERSION_GIT_SHA}")
    set(MACOSX_BUNDLE_LONG_VERSION_STRING "${VERSION_GIT_SHA}")
    set(MACOSX_BUNDLE_COPYRIGHT "Copyright ${PROJECT_COPYRIGHT_YEAR} Tommaso Vinci, Alessandro Flacco. GNU General Public License v2.0")

    # LSMinimumSystemVersion: deployment target if given, otherwise the build host version (brew libraries are built for it)
    if (CMAKE_OSX_DEPLOYMENT_TARGET)
        set(MACOSX_MINIMUM_SYSTEM_VERSION "${CMAKE_OSX_DEPLOYMENT_TARGET}")
    else()
        execute_process(COMMAND sw_vers -productVersion OUTPUT_VARIABLE MACOSX_MINIMUM_SYSTEM_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
        # string(REGEX REPLACE "^([0-9]+)\\.([0-9]+).*" "\\1.\\2" MACOSX_MINIMUM_SYSTEM_VERSION "${MACOSX_MINIMUM_SYSTEM_VERSION}")
        # brew bottles are built for <major>.0
        string(REGEX REPLACE "^([0-9]+).*" "\\1.0" MACOSX_MINIMUM_SYSTEM_VERSION "${MACOSX_MINIMUM_SYSTEM_VERSION}")
    endif()
    message(STATUS "LSMinimumSystemVersion ${MACOSX_MINIMUM_SYSTEM_VERSION}")

    set(MACOSX_BUNDLE_ICON_FILE icon.icns)
    set(${PROJECT_NAME}_ICON ${CMAKE_CURRENT_SOURCE_DIR}/../resources/icons/icon.icns)
    set(${PROJECT_NAME}_ICONTYPE ${CMAKE_CURRENT_SOURCE_DIR}/../resources/macPackage/filetype.icns)

    set_source_files_properties(${${PROJECT_NAME}_ICON} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
    set_source_files_properties(${${PROJECT_NAME}_ICONTYPE} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

    add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${${PROJECT_NAME}_ICON} ${${PROJECT_NAME}_ICONTYPE} ${filedeps})
    target_link_libraries(${PROJECT_NAME} "-framework IOKit -framework CoreFoundation")

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../resources/macPackage/neutrino.plist.cmake" ${CMAKE_CURRENT_BINARY_DIR}/Info.plist)
    set_target_properties(${PROJECT_NAME} PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_BINARY_DIR}/Info.plist)

    install (TARGETS ${PROJECT_NAME} BUNDLE DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/.." COMPONENT binary)
    add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD COMMAND
        ${CMAKE_INSTALL_NAME_TOOL} -add_rpath "@executable_path/../Frameworks/"
        $<TARGET_FILE:${PROJECT_NAME}>)

elseif(WIN32)
    set (RC_WIN "${${PROJECT_NAME}_SOURCE_DIR}/resources/winPackage/neutrino.rc")
    # set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--export-all-symbols -Wl,-allow-multiple-definition")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_executable(${PROJECT_NAME} ${RC_WIN} ${filedeps})
    else()
        add_executable(${PROJECT_NAME} WIN32 ${RC_WIN} ${filedeps})
    endif()
    target_link_options(${PROJECT_NAME} PRIVATE -Wl,--export-all-symbols -Wl,-allow-multiple-definition)

    set_property(TARGET ${PROJECT_NAME} PROPERTY ENABLE_EXPORTS ON)
    set_property(TARGET ${PROJECT_NAME} PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin COMPONENT binary)

elseif(LINUX)
    add_executable(${PROJECT_NAME} ${filedeps})
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin COMPONENT binary)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../resources/linuxPackage/Neutrino.desktop DESTINATION share/applications)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../resources/icons/icon.svg DESTINATION share/applications RENAME Neutrino.svg)

endif()

set_target_properties(${PROJECT_NAME} PROPERTIES ENABLE_EXPORTS ON POSITION_INDEPENDENT_CODE ON)
add_dependencies(${PROJECT_NAME} NeutrinoVersion)

foreach(MODULE ${MODULES})
    set(MODULES_TWEAK "${MODULES_TWEAK};Qt6::${MODULE}")
endforeach()

if(APPLE)
# add_definitions(-DQT_NO_OPENGL)
target_compile_definitions(${PROJECT_NAME} PRIVATE QT_NO_OPENGL)
endif()

#target_compile_definitions(${PROJECT_NAME} PRIVATE NEU_PLUGIN_LIST=${NEU_PLUGIN_LIST})

target_link_libraries(${PROJECT_NAME} ${LIBS} ${MODULES_TWEAK})
