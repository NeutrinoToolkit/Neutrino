cmake_minimum_required (VERSION 3.9)

message(STATUS "Scanning plugins directory")

FILE(GLOB allfiles "*")
FOREACH(oneplugin ${allfiles})
	IF(IS_DIRECTORY ${oneplugin})
		add_subdirectory(${oneplugin})
	ENDIF()
ENDFOREACH()
