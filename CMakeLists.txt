# this is the frigging toplevel cmakelists

cmake_minimum_required (VERSION 3.0)

project (neutrino)

add_subdirectory (nPhysImage)
add_subdirectory (src)

# for debug use cmake . -DCMAKE_BUILD_TYPE=Debug
set (CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -Wall -D__phys_debug=10")


