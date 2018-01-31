# this is the frigging nPhysImage cmakelists
cmake_minimum_required (VERSION 3.0)

if (LINUX)
	message(STATUS "Linux")
	set (CMAKE_PREFIX_PATH "/usr")
endif()

add_library (nPhysImageF SHARED mcomplex.h bidimvec.h tools.h nPhysTools.cc anymap.cc NaNStream.h  nPhysFormats.cc  nPhysImageF.h  nPhysMaths.cc  nPhysWave.cc anymap.cc unwrapping/unwrap_goldstein.cc  unwrapping/unwrap_quality.cc  unwrapping/unwrap_simple.cc  unwrapping/unwrap_util.cc  )

target_include_directories (nPhysImageF PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../resources/cmake)
include(FindNeutrinoDeps)

SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")

target_link_libraries(nPhysImageF m ${LIBS})

# install
if (WIN32)
	install (TARGETS nPhysImageF COMPONENT binary DESTINATION bin)
else()
	install (TARGETS nPhysImageF COMPONENT binary DESTINATION lib)
endif()

if (APPLE)
	set (LIB_INSTALL_DIR "/usr/local/lib")
	set (CMAKE_INSTALL_RPATH "/usr/local/lib")
	set (CMAKE_MACOSX_RPATH off)
	set_target_properties(nPhysImageF PROPERTIES INSTALL_RPATH ${LIB_INSTALL_DIR})
endif()


