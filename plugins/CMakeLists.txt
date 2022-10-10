cmake_minimum_required (VERSION 3.9)

option(NEUTRINOPLUGINS "Compile Neutrino plugins" ON)

if (${NEUTRINOPLUGINS})

    include(NeutrinoPlugin)

    message(STATUS "Scanning plugins directory")

    FILE(GLOB allfiles "*" LIST_DIRECTORIES true)

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
