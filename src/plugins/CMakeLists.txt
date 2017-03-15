message(STATUS "Scanning plugins directory")

list(APPEND CMAKE_MODULE_PATH ${NEUTRINO_ROOT}/../resources/cmake)

if (NOT DEFINED NEUTRINO_ROOT)
    set (NEUTRINO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
endif()

if (NOT IS_ABSOLUTE ${NEUTRINO_ROOT})
    set (NEUTRINO_ROOT "${CMAKE_BINARY_DIR}/${NEUTRINO_ROOT}")
endif()

include(NeutrinoPlugin)

SET(plugins_list "")
FILE(GLOB allfiles "*")
FOREACH(oneplugin ${allfiles})
    IF(IS_DIRECTORY ${oneplugin})
        get_filename_component(namedir ${oneplugin} NAME)
        if(NOT ( ${namedir} STREQUAL "testPlugin") OR (CMAKE_BUILD_TYPE STREQUAL "Debug"))
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

