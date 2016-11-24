
cmake_minimum_required (VERSION 3.0)

message(STATUS "Scanning plugins directory")

file(GLOB plugins_list LIST_DIRECTORIES true "plugin_*")

message(STATUS "Found plugins: ${plugins_list}")
set (LIBRARY_OUTPUT_PATH "../../bin/plugins")

foreach (plugin_name ${plugins_list})
	if (EXISTS "${plugin_name}/CMakeLists.txt")
		message (STATUS "Adding plugin: ${plugin_name}")
		add_subdirectory(${plugin_name})
	else()
		message (STATUS "Discarding plugin ${plugin_name}: no CmakeLists.txt found")
	endif()
endforeach()

