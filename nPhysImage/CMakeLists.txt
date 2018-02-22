# this is the frigging nPhysImage cmakelists
cmake_minimum_required (VERSION 3.9)

add_library (nPhysImageF SHARED mcomplex.h bidimvec.h tools.h nPhysTools.cc anymap.cc NaNStream.h  nPhysFormats.cc  nPhysImageF.h  nPhysMaths.cc  nPhysWave.cc anymap.cc anymap.h unwrapping/unwrap_goldstein.cc  unwrapping/unwrap_quality.cc  unwrapping/unwrap_simple.cc  unwrapping/unwrap_util.cc  )

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../resources/cmake)
include(FindNeutrinoDeps)

target_link_libraries(nPhysImageF m ${LIBS})

# install
if (WIN32)
    install (TARGETS nPhysImageF COMPONENT binary DESTINATION bin)
elseif(LINUX)
    install (TARGETS nPhysImageF COMPONENT binary DESTINATION lib)
elseif (APPLE)
    install (TARGETS nPhysImageF COMPONENT binary DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/../Neutrino.app/Contents/Frameworks")
endif()


