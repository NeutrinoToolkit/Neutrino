# this is the frigging nPhysImage cmakelists
cmake_minimum_required (VERSION 3.9)

file(GLOB_RECURSE SOURCES *.cc *.cpp *.cxx *.h)

add_library (nPhysImageF SHARED ${SOURCES})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include(FindNeutrinoDeps)

target_link_libraries(nPhysImageF m ${LIBS})

# install
if (WIN32)
    install (TARGETS nPhysImageF COMPONENT binary DESTINATION "bin")
elseif(LINUX)
    install (TARGETS nPhysImageF COMPONENT binary DESTINATION lib)
elseif (APPLE)
    install (TARGETS nPhysImageF COMPONENT binary DESTINATION "${PROJECT_BINARY_DIR}/Neutrino.app/Contents/Frameworks")
endif()


