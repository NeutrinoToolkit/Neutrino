# cmake_minimum_required (VERSION 3.9)
cmake_minimum_required (VERSION 3.18...3.31)

option(NEUTRINOPLUGINS "Compile Neutrino plugins" ON)
if (${NEUTRINOPLUGINS})

    include(NeutrinoPlugin)

    # the ui headers included by the neutrino headers (neutrino.h, nLine.h, nObject.h), generated once for all the plugins
    find_package(Qt6 COMPONENTS Widgets REQUIRED)
    qt6_wrap_ui(NEUTRINO_PLUGINS_UIS ${CMAKE_SOURCE_DIR}/src/neutrino.ui ${CMAKE_SOURCE_DIR}/src/graphics/nLine.ui
                ${CMAKE_SOURCE_DIR}/src/graphics/nObject.ui ${CMAKE_SOURCE_DIR}/src/graphics/nPoint.ui)
    add_custom_target(neutrino_ui_headers DEPENDS ${NEUTRINO_PLUGINS_UIS})
    set(NEUTRINO_PLUGINS_UI_DIR ${CMAKE_CURRENT_BINARY_DIR})

    # PyShell (PythonQt) is replaced by the neutrino python module: not built any more, to be removed
    list(APPEND NEUTRINO_SKIP_PLUGINS PyShell)

    message(STATUS "Scanning plugins directory")

    # FILE(GLOB allfiles "*" LIST_DIRECTORIES true)
    # CONFIGURE_DEPENDS: a new plugin directory is picked up by the build
    FILE(GLOB allfiles CONFIGURE_DEPENDS "*" LIST_DIRECTORIES true)

    FOREACH(oneplugin ${NEUTRINO_SKIP_PLUGINS})
        list(FILTER allfiles EXCLUDE REGEX ".*/plugins/${oneplugin}$")
    ENDFOREACH()

    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        MESSAGE("PLUGINS:\n${allfiles}")
    endif()

    FOREACH(oneplugin ${allfiles})
        IF(IS_DIRECTORY ${oneplugin})
            add_subdirectory(${oneplugin})
        ENDIF()
    ENDFOREACH()

#    set(NEU_PLUGIN_LIST "${NEU_PLUGIN_LIST}" PARENT_SCOPE)
endif()
