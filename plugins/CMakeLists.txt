cmake_minimum_required (VERSION 3.9)

option(NEUTRINOPLUGINS "Compile Neutrino plugins" ON)

set (NEUTRINO_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")

if (${NEUTRINOPLUGINS})
    message(STATUS "Scanning plugins directory")

    FILE(GLOB allfiles "*")
    FOREACH(oneplugin ${allfiles})
        IF(IS_DIRECTORY ${oneplugin})
            add_subdirectory(${oneplugin})
        ENDIF()
    ENDFOREACH()
endif()
