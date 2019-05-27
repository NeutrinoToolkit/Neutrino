cmake_minimum_required (VERSION 3.0)

if (WIN32 AND CMAKE_COMPILER_IS_GNUCXX)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mwindows")
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

## sources
file(GLOB UIS ${CMAKE_CURRENT_SOURCE_DIR}/../UIs/*.ui)
set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../UIs)

file(GLOB_RECURSE SOURCES *.cc *.cpp *.h)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(file_to_remove "${CMAKE_CURRENT_SOURCE_DIR}/grabStream.h")
    list(FIND SOURCES "${file_to_remove}" file_to_remove_found)
    if (${file_to_remove_found})
        list(REMOVE_ITEM SOURCES "${file_to_remove}")
    endif()
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE("SOURCES:\n${SOURCES}")
endif()

list(LENGTH SOURCES LIST_LENGTH)
message(STATUS "${LIST_LENGTH} source files")

include(FindNeutrinoDeps)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/nPhysImage")
set(LIBS ${LIBS} "nPhysImageF")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nPhysImage)


add_definitions(-D__VER="${${PROJECT_NAME}_VERSION_AHEAD}")
add_definitions(-D__VER_SHA="${${PROJECT_NAME}_VERSION_GIT_SHA}")

if (APPLE AND NOT DEFINED Qt5_DIR)
    set(Qt5_DIR "/usr/local/opt/qt5/lib/cmake/Qt5")
    MESSAGE(STATUS "Qt5_DIR set to ${Qt5_DIR}")
endif()

set(MODULES Core Gui Sql Widgets Svg PrintSupport UiTools OpenGL Network)

find_package(Qt5 COMPONENTS ${MODULES} REQUIRED)

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
file(GLOB_RECURSE HTMLLICENSES ${LICENSESDIR}/*)
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

#translate ui

SET(Qt5LinguistTools_DIR "${Qt5_DIR}/../Qt5LinguistTools")
find_package(Qt5LinguistTools)
if (Qt5LinguistTools_FOUND)
    SET(LANGUAGES fr_FR it_IT ko_KP)
    SET(LANGUAGE_TS_FILES)
    FOREACH(LANGUAGE ${LANGUAGES})
        SET(TS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../resources/translations/neutrino_${LANGUAGE}.ts")
        if(EXISTS ${TS_FILE})
            SET(LANGUAGE_TS_FILES ${LANGUAGE_TS_FILES} ${TS_FILE})
            qt5_add_translation(qm_files ${TS_FILE})
        else ()
            qt5_create_translation(qm_files ${SOURCES} ${UIS} ${TS_FILE})
            if (CMAKE_BUILD_TYPE STREQUAL "Debug")
                message (STATUS "[Debug] translation file ${TS_FILE} will be created, commit it if you create the translations.")
            endif()
        endif()
    ENDFOREACH()

    IF(LANGUAGE_TS_FILES)
        set(TRANSL_QRC ${QRCDIR}/translations.qrc)
        file(WRITE ${TRANSL_QRC} "<RCC>\n    <qresource prefix=\"/translations/\">\n")
        foreach(my_file ${TRASL_FILES})
            file(RELATIVE_PATH my_file_relative_path ${QRCDIR} ${my_file})
            GET_FILENAME_COMPONENT(my_file_basename ${my_file} NAME)
            file(APPEND ${TRANSL_QRC} "        <file alias=\"${my_file_basename}\">${my_file_relative_path}</file>\n")
        endforeach()
        file(APPEND ${TRANSL_QRC} "    </qresource>\n</RCC>")
    ENDIF(LANGUAGE_TS_FILES)

endif(Qt5LinguistTools_FOUND)


include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${QRCDIR} ${ICONSDIR})

set(filedeps ${SOURCES} ${UIS} ${ICONS_QRC} ${LICENSES_QRC} ${TRANSL_QRC} ${CMAP_QRC} ${TRASL_FILES})

if(APPLE)
    set(MACOSX_BUNDLE_EXECUTABLE_NAME "${PROJECT_NAME}")
    set(MACOSX_BUNDLE_INFO_STRING "${PROJECT_NAME} ${${PROJECT_NAME}_VERSION}")
    set(MACOSX_BUNDLE_GUI_IDENTIFIER "${PROJECT_DOMAIN_SECOND}.${PROJECT_DOMAIN_FIRST}")

    set(MACOSX_BUNDLE_BUNDLE_NAME "${PROJECT_NAME}")
    set(MACOSX_BUNDLE_BUNDLE_VERSION "${${PROJECT_NAME}_VERSION_AHEAD}")
    set(MACOSX_BUNDLE_SHORT_VERSION_STRING "${${PROJECT_NAME}_VERSION}")
    set(MACOSX_BUNDLE_LONG_VERSION_STRING "${${PROJECT_NAME}_VERSION_AHEAD} sha: ${${PROJECT_NAME}_VERSION_GIT_SHA}")
    set(MACOSX_BUNDLE_COPYRIGHT "Copyright ${PROJECT_COPYRIGHT_YEAR} Tommaso Vinci, Alessandro Flacco. GNU General Public License v2.0")

    set(MACOSX_BUNDLE_ICON_FILE icon.icns)
    set(${PROJECT_NAME}_ICON ${CMAKE_CURRENT_SOURCE_DIR}/../resources/icons/icon.icns)
    set(${PROJECT_NAME}_ICONTYPE ${CMAKE_CURRENT_SOURCE_DIR}/../resources/macPackage/filetype.icns)

    set_source_files_properties(${${PROJECT_NAME}_ICON} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
    set_source_files_properties(${${PROJECT_NAME}_ICONTYPE} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")

    add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${${PROJECT_NAME}_ICON} ${${PROJECT_NAME}_ICONTYPE} ${filedeps})
    target_link_libraries(Neutrino "-framework IOKit -framework CoreFoundation")

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/../resources/macpackage/neutrino.plist.cmake" ${CMAKE_CURRENT_BINARY_DIR}/Info.plist)
    set_target_properties(${PROJECT_NAME} PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_BINARY_DIR}/Info.plist)

    set (CMAKE_INSTALL_RPATH "@executable_path/../Frameworks")
    set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_RPATH} INSTALL_RPATH_USE_LINK_PATH TRUE)

    install (TARGETS ${PROJECT_NAME} BUNDLE DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/.." COMPONENT binary)

elseif(WIN32)
    set (RC_WIN "${${PROJECT_NAME}_SOURCE_DIR}/resources/winPackage/neutrino.rc")
    list (APPEND CMAKE_EXE_LINKER_FLAGS "-Wl,--export-all-symbols")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_executable(${PROJECT_NAME} ${RC_WIN} ${filedeps})
    else()
        add_executable(${PROJECT_NAME} WIN32 ${RC_WIN} ${filedeps})
    endif()

    set_property(TARGET ${PROJECT_NAME} PROPERTY ENABLE_EXPORTS ON)
    set_property(TARGET ${PROJECT_NAME} PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "." COMPONENT binary)

elseif(LINUX)
    add_executable(${PROJECT_NAME} ${filedeps})
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin COMPONENT binary)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../resources/linuxPackage/Neutrino.desktop DESTINATION share/applications)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../resources/icons/icon.svg DESTINATION share/applications RENAME Neutrino.svg)

endif()


SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

foreach(MODULE ${MODULES})
    set(MODULES_TWEAK "${MODULES_TWEAK};Qt5::${MODULE}")
endforeach()

target_link_libraries(${PROJECT_NAME} ${LIBS} ${MODULES_TWEAK})

