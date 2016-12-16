
cmake_minimum_required (VERSION 3.0)

message(STATUS "Scanning plugins directory")

SET(plugins_list "")
FILE(GLOB allfiles "*")
FOREACH(onefile ${allfiles})
  IF(IS_DIRECTORY ${onefile})
    get_filename_component(namedir ${onefile} NAME)
    if(NOT ( ${namedir} STREQUAL "skel"))
      LIST(APPEND plugins_list ${onefile})
    endif()
  ENDIF()
ENDFOREACH()

list(LENGTH plugins_list plugins_list_len)

message(STATUS "Found ${plugins_list_len} plugins")

if (NOT DEFINED LIBRARY_OUTPUT_PATH)
    if (APPLE)
        set (LIBRARY_OUTPUT_PATH "../../Neutrino.app/Contents/Resources/plugins")
    else()
        set (LIBRARY_OUTPUT_PATH "../../bin/plugins")
    endif(APPLE)
endif()

foreach (plugin_name ${plugins_list})
	if (EXISTS "${plugin_name}/CMakeLists.txt")
		message (STATUS "Adding plugin: ${plugin_name}")
		add_subdirectory(${plugin_name})
	else()
		message (STATUS "Discarding plugin ${plugin_name}: no CmakeLists.txt found")
	endif()
endforeach()

