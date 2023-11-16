cmake_minimum_required(VERSION 3.9)

project (Neutrino CXX)

message(STATUS "path: $ENV{PATH}")
if (APPLE)
    find_program(BREW NAMES brew)
    if(BREW)
        message(STATUS "brew: ${BREW}")
        execute_process(COMMAND ${BREW} --prefix libomp OUTPUT_VARIABLE BREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
        message(STATUS "BREW_LIBOMP_PREFIX: ${BREW_LIBOMP_PREFIX}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${BREW_LIBOMP_PREFIX}/lib/libomp.dylib")
        include_directories("${BREW_LIBOMP_PREFIX}/include")
    else()
        message(STATUS "This program supports OpenMP on Mac through Homebrew, installing Homebrew recommmended https://brew.sh")
    endif()
endif()


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
