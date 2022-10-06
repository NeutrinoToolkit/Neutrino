cmake_minimum_required (VERSION 3.9)

option(NEUTRINOPLUGINS "Compile Neutrino plugins" ON)

if (${NEUTRINOPLUGINS})

    set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/resources/cmake)

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
            get_filename_component(MY_PROJECT_NAME ${oneplugin} NAME)
#             set(NEU_PLUGIN_LIST "${NEU_PLUGIN_LIST} ${MY_PROJECT_NAME}")

        ENDIF()
    ENDFOREACH()

#     message(STATUS "${NEU_PLUGIN_LIST}")
#     target_compile_definitions(Neutrino PRIVATE NEU_PLUGIN_LIST=${NEU_PLUGIN_LIST})

endif()
