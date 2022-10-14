cmake_minimum_required(VERSION 3.9)

project (Neutrino CXX)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

## default build type is release
if (CMAKE_BUILD_TYPE STREQUAL "")
	set (CMAKE_BUILD_TYPE "Release")
endif()

set(PROJECT_DOMAIN_FIRST "neutrino")
set(PROJECT_DOMAIN_SECOND "com")

string(TIMESTAMP CURRENT_YEAR "%Y")
set(PROJECT_COPYRIGHT_YEAR "2013-${CURRENT_YEAR}")

set(DEBUG_LEVEL "10" CACHE STRING "DEBUG verbosity")

## global compile options
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -ggdb -D__phys_debug=${DEBUG_LEVEL}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -DQT_NO_DEBUG_OUTPUT")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/resources/cmake)

add_subdirectory (nPhysImage)
add_subdirectory (plugins)
add_subdirectory (src)

include(Package)
