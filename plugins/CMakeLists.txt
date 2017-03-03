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

foreach (plugin_name ${plugins_list})
    if (EXISTS "${plugin_name}/CMakeLists.txt")
        add_subdirectory(${plugin_name})
    else()
        message (STATUS "Discarding plugin ${plugin_name}: no CmakeLists.txt found")
    endif()
endforeach()

