cmake_minimum_required (VERSION 3.9)

option(NEUTRINOPLUGINS "Compile Neutrino plugins" ON)

set (NEUTRINO_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")

if (${NEUTRINOPLUGINS})
message(STATUS "Scanning plugins directory")


SET(plugins_list "")
FILE(GLOB allfiles "*")
FOREACH(oneplugin ${allfiles})
    IF(IS_DIRECTORY ${oneplugin})
        get_filename_component(namedir ${oneplugin} NAME)
        if(NOT ( ${namedir} STREQUAL "testPlugin") OR (CMAKE_BUILD_TYPE STREQUAL "Debug"))
            if (EXISTS "${oneplugin}/CMakeLists.txt")
                add_subdirectory(${oneplugin})
            endif()
        endif()
    ENDIF()
ENDFOREACH()
endif()
