# this is the frigging nPhysImage cmakelists
cmake_minimum_required (VERSION 3.0)

if (LINUX)
    message(STATUS "Linux")
    set (CMAKE_PREFIX_PATH "/usr")
endif()

add_library (nPhysImageF SHARED bidimvec.cc tools.cc nPhysTools.cc anymap.cc NaNStream.cc  nPhysFormats.cc  nPhysImageF.cc  nPhysMaths.cc  nPhysWave.cc anymap.cc unwrapping/unwrap_goldstein.cc  unwrapping/unwrap_quality.cc  unwrapping/unwrap_simple.cc  unwrapping/unwrap_util.cc  )

target_include_directories (nPhysImageF PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nPhysImageF "-lm")


message(STATUS "cmake_prefix_path: ${CMAKE_PREFIX_PATH}")


#SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")



# force necessary defines on config.h
add_definitions(-DHAVE_LIBPTHREAD)

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../resources/cmake)
include(FindNeutrinoDeps)


SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")

target_link_libraries(nPhysImageF ${LIBS})

# output
set (LIBRARY_OUTPUT_PATH "../lib")

#set (CMAKE_SKIP_RPATH true)


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

if (NOT APPLE)
    install (DIRECTORY "." COMPONENT dev DESTINATION include/nPhysImage FILES_MATCHING PATTERN "*.h")
endif()



