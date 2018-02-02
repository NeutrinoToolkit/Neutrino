# this is the frigging nPhysImage cmakelists
cmake_minimum_required (VERSION 3.9)

if(LINUX)
	set (CMAKE_PREFIX_PATH "/usr")
endif()

add_library (nPhysImageF SHARED mcomplex.h bidimvec.h tools.h nPhysTools.cc anymap.cc NaNStream.h  nPhysFormats.cc  nPhysImageF.h  nPhysMaths.cc  nPhysWave.cc anymap.cc unwrapping/unwrap_goldstein.cc  unwrapping/unwrap_quality.cc  unwrapping/unwrap_simple.cc  unwrapping/unwrap_util.cc  )

target_include_directories (nPhysImageF PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../resources/cmake)
include(FindNeutrinoDeps)

target_link_libraries(nPhysImageF m ${LIBS})

# install
if (WIN32)
	install (TARGETS nPhysImageF COMPONENT binary DESTINATION bin)
endif()

if(LINUX)
    install (TARGETS nPhysImageF COMPONENT binary DESTINATION lib)
endif()


if (APPLE)
    set (CMAKE_SKIP_RPATH ON)
    set (CMAKE_INSTALL_RPATH "@executable_path/../Frameworks")

#	set (LIB_INSTALL_DIR "/usr/local/lib")
    set_target_properties(nPhysImageF PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_RPATH})
    install (TARGETS nPhysImageF COMPONENT binary DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/../Neutrino.app/Contents/Frameworks")
endif()


