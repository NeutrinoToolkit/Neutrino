# this is the frigging toplevel cmakelists

cmake_minimum_required(VERSION 3.9)

## before project https://stackoverflow.com/a/34208904
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.10" CACHE STRING "Minimum OS X deployment version")

### top level compile options
project (Neutrino CXX)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

## default build type is release
if (CMAKE_BUILD_TYPE STREQUAL "")
	set (CMAKE_BUILD_TYPE "Release")
endif()

#versioning
find_package(Git REQUIRED)
if (GIT_FOUND)
    #How many commits since last version tag
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --match "v[0-9]*\\.[0-9]*" --abbrev=0
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE ${PROJECT_NAME}_VERSION_GIT_VER
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    string(REGEX REPLACE "^v([0-9]*)\\.([0-9]*)" "\\1.\\2" ${PROJECT_NAME}_VERSION "${${PROJECT_NAME}_VERSION_GIT_VER}")

    execute_process(COMMAND ${GIT_EXECUTABLE} rev-list --count ${${PROJECT_NAME}_VERSION_GIT_VER}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE ${PROJECT_NAME}_VERSION_AHEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(${PROJECT_NAME}_VERSION_AHEAD "${${PROJECT_NAME}_VERSION}+${${PROJECT_NAME}_VERSION_AHEAD}")

    # Get current commit SHA from git
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE ${PROJECT_NAME}_VERSION_GIT_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE)

endif(GIT_FOUND)


set(PROJECT_DOMAIN_FIRST "neutrino")
set(PROJECT_DOMAIN_SECOND "com")
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
string(TIMESTAMP CURRENT_YEAR "%Y")
set(PROJECT_COPYRIGHT_YEAR "2011-${CURRENT_YEAR}")



## global compile options
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -D__phys_debug=10 -DQT_DEBUG_PLUGINS=1")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-Wall)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	add_compile_options(-stdlib=libc++)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-static-libgcc -static-libstdc++)
 endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/resources/cmake)

include(Package)

add_subdirectory (nPhysImage)
add_subdirectory (src)
add_subdirectory (plugins)

