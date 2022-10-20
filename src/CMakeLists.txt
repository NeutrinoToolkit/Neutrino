cmake_minimum_required (VERSION 3.0)

if (WIN32 AND CMAKE_COMPILER_IS_GNUCXX)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mwindows")
endif()


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
file(GLOB_RECURSE UIS *.ui)

file(GLOB_RECURSE SOURCES *.cc *.cpp *.cxx *.h)

add_compile_definitions(QT_DISABLE_DEPRECATED_BEFORE=0x050F00)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE("SOURCES:\n${SOURCES}")
endif()

list(LENGTH SOURCES LIST_LENGTH)
message(STATUS "${LIST_LENGTH} source files")

include(FindNeutrinoDeps)

set(LIBS ${LIBS} "nPhysImageF")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nPhysImage)

add_definitions(-DSHAVERSION__="${VERSION_GIT_SHA}")

set(MODULES Core Gui Sql Widgets Svg PrintSupport Network UiTools Multimedia MultimediaWidgets Qml DBus)

find_package(Qt6 COMPONENTS ${MODULES} REQUIRED)

set(QRCDIR ${CMAKE_CURRENT_BINARY_DIR}/qrc)

# write automatically the qrc file for icons
set(ICONSDIR ${CMAKE_CURRENT_SOURCE_DIR}/../resources/icons)
file(GLOB_RECURSE ICONS ${ICONSDIR}/*.png)
set(ICONS_QRC ${QRCDIR}/icons.qrc)
file(WRITE ${ICONS_QRC} "<RCC>\n    <qresource prefix=\"/icons/\">\n")
foreach(my_file ${ICONS})
    file(RELATIVE_PATH my_file_relative_path ${QRCDIR} ${my_file})
    GET_FILENAME_COMPONENT(my_file_basename ${my_file} NAME)
    file(APPEND ${ICONS_QRC} "        <file alias=\"${my_file_basename}\">${my_file_relative_path}</file>\n")
endforeach()
file(APPEND ${ICONS_QRC} "    </qresource>\n</RCC>")


# write automatically the qrc file for html
set(LICENSESDIR ${CMAKE_CURRENT_SOURCE_DIR}/../resources/licenses)
#parse neutrino license file
file(GLOB_RECURSE HTMLLICENSES ${LICENSESDIR}/*.txt)
list(APPEND HTMLLICENSES "${CPACK_RESOURCE_FILE_LICENSE}")
set(LICENSES_QRC ${QRCDIR}/licenses.qrc)
file(WRITE ${LICENSES_QRC} "<RCC>\n    <qresource prefix=\"/licenses/\">\n")
foreach(my_file ${HTMLLICENSES})
    file(RELATIVE_PATH my_file_relative_path ${QRCDIR} ${my_file})
    GET_FILENAME_COMPONENT(my_file_basename ${my_file} NAME)
    file(APPEND ${LICENSES_QRC} "        <file alias=\"${my_file_basename}\">${my_file_relative_path}</file>\n")
endforeach()
file(APPEND ${LICENSES_QRC} "    </qresource>\n</RCC>")

#parse neutrino colormap files
set(CMAPDIR ${CMAKE_CURRENT_SOURCE_DIR}/../resources/colormaps/cmaps)
file(GLOB_RECURSE CMAPS ${CMAPDIR}/*)
set(CMAP_QRC ${QRCDIR}/cmap.qrc)
file(WRITE ${CMAP_QRC} "<RCC>\n    <qresource prefix=\"/cmaps/\">\n")
foreach(my_file ${CMAPS})
    file(RELATIVE_PATH my_file_relative_path ${QRCDIR} ${my_file})
    GET_FILENAME_COMPONENT(my_file_basename ${my_file} NAME)
    file(APPEND ${CMAP_QRC} "        <file alias=\"${my_file_basename}\">${my_file_relative_path}</file>\n")
endforeach()
file(APPEND ${CMAP_QRC} "    </qresource>\n</RCC>")


# sub-structure (wrong: we should probably replicate CMakeLists.txt in every subfolder)
include_directories(graphics pans pans/colorbar pans/winlist)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${QRCDIR} ${ICONSDIR})

set(filedeps ${SOURCES} ${UIS} ${ICONS_QRC} ${LICENSES_QRC} ${CMAP_QRC})

if(APPLE)


    set(MACOSX_BUNDLE_EXECUTABLE_NAME "${PROJECT_NAME}")
    set(MACOSX_BUNDLE_INFO_STRING "${PROJECT_NAME} ${VERSION_GIT_SHA}")
    set(MACOSX_BUNDLE_GUI_IDENTIFIER "${PROJECT_DOMAIN_SECOND}.${PROJECT_DOMAIN_FIRST}")

    set(MACOSX_BUNDLE_BUNDLE_NAME "${PROJECT_NAME}")
    set(MACOSX_BUNDLE_BUNDLE_VERSION "${VERSION_GIT_SHA}")
    set(MACOSX_BUNDLE_SHORT_VERSION_STRING "${VERSION_GIT_SHA}")
    set(MACOSX_BUNDLE_LONG_VERSION_STRING "${VERSION_GIT_SHA}")
    set(MACOSX_BUNDLE_COPYRIGHT "Copyright ${PROJECT_COPYRIGHT_YEAR} Tommaso Vinci, Alessandro Flacco. GNU General Public License v2.0")

    set(MACOSX_BUNDLE_ICON_FILE icon.icns)
    set(${PROJECT_NAME}_ICON ${CMAKE_CURRENT_SOURCE_DIR}/../resources/icons/icon.icns)
    set(${PROJECT_NAME}_ICONTYPE ${CMAKE_CURRENT_SOURCE_DIR}/../resources/macPackage/filetype.icns)

    set_source_files_properties(${${PROJECT_NAME}_ICON} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
    set_source_files_properties(${${PROJECT_NAME}_ICONTYPE} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

    add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${${PROJECT_NAME}_ICON} ${${PROJECT_NAME}_ICONTYPE} ${filedeps})
    target_link_libraries(${PROJECT_NAME} "-framework IOKit -framework CoreFoundation")

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../resources/macpackage/neutrino.plist.cmake" ${CMAKE_CURRENT_BINARY_DIR}/Info.plist)
    set_target_properties(${PROJECT_NAME} PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_BINARY_DIR}/Info.plist)


    SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set (CMAKE_INSTALL_RPATH "@executable_path/../Frameworks")
    set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_RPATH} INSTALL_RPATH_USE_LINK_PATH TRUE)

    install (TARGETS ${PROJECT_NAME} BUNDLE DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/.." COMPONENT binary)

elseif(WIN32)
    set (RC_WIN "${${PROJECT_NAME}_SOURCE_DIR}/resources/winPackage/neutrino.rc")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--export-all-symbols -Wl,-allow-multiple-definition")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_executable(${PROJECT_NAME} ${RC_WIN} ${filedeps})
    else()
        add_executable(${PROJECT_NAME} WIN32 ${RC_WIN} ${filedeps})
    endif()

    set_property(TARGET ${PROJECT_NAME} PROPERTY ENABLE_EXPORTS ON)
    set_property(TARGET ${PROJECT_NAME} PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin COMPONENT binary)

elseif(LINUX)
    add_executable(${PROJECT_NAME} ${filedeps})
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin COMPONENT binary)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../resources/linuxPackage/Neutrino.desktop DESTINATION share/applications)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../resources/icons/icon.svg DESTINATION share/applications RENAME Neutrino.svg)

endif()

foreach(MODULE ${MODULES})
    set(MODULES_TWEAK "${MODULES_TWEAK};Qt6::${MODULE}")
endforeach()

if(APPLE)
add_definitions(-DQT_NO_OPENGL)
endif()

#target_compile_definitions(${PROJECT_NAME} PRIVATE NEU_PLUGIN_LIST=${NEU_PLUGIN_LIST})

target_link_libraries(${PROJECT_NAME} ${LIBS} ${MODULES_TWEAK})
