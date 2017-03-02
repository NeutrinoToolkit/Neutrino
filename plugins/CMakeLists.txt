cmake_minimum_required (VERSION 3.0)

message(STATUS "Scanning plugins directory")

SET(plugins_list "")
FILE(GLOB allfiles "*")
FOREACH(oneplugin ${allfiles})
    IF(IS_DIRECTORY ${oneplugin})
        get_filename_component(namedir ${oneplugin} NAME)
        if(NOT ( ${namedir} STREQUAL "skel"))
            LIST(APPEND plugins_list ${oneplugin})
        endif()
    ENDIF()
ENDFOREACH()

list(LENGTH plugins_list plugins_list_len)

message(STATUS "Found ${plugins_list_len} plugins")

if(APPLE)
    set (LIBRARY_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/../Neutrino.app/Contents/Resources/plugins")
elseif(LINUX)
    set (LIBRARY_OUTPUT_PATH share/neutrino/plugins)
endif()

foreach (plugin_name ${plugins_list})
    if (EXISTS "${plugin_name}/CMakeLists.txt")
        add_subdirectory(${plugin_name})
    else()
        message (STATUS "Discarding plugin ${plugin_name}: no CmakeLists.txt found")
    endif()
endforeach()

if (WIN32)
    install (FILES ${CMAKE_CURRENT_BINARY_DIR}/plugins DESTINATION bin/plugins)
endif()
