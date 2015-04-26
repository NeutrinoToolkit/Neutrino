# this is the frigging nPhysImage cmakelists
cmake_minimum_required (VERSION 3.0)


add_library (nPhysImageF SHARED NaNStream.cc  nPhysFormats.cc  nPhysImageF.cc  nPhysMaths.cc  nPhysWave.cc anymap.cc unwrapping/unwrap_goldstein.cc  unwrapping/unwrap_quality.cc  unwrapping/unwrap_simple.cc  unwrapping/unwrap_util.cc  )


target_include_directories (nPhysImageF PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nPhysImageF "-lfftw3 -lgsl -lgslcblas -lm")


# force necessary defines on config.h
add_definitions(-DHAVE_LIBFFTW3 -DHAVE_LIBGSLCBLAS -DHAVE_LIBJPEG -DHAVE_LIBNETPBM -DHAVE_LIBPTHREAD -DHAVE_LIBTIFF -DHAVE_LIBZ)

# libraries to check for
# HAVE_LIBFFTW3_THREADS  

# hdf4
find_library (HDF4 NAMES mfhdf PATHS /usr/lib)
if (NOT ${HDF4} STREQUAL "HDF4-NOTFOUND")
	message (STATUS "using hdf4: ${HDF4}")
	target_link_libraries (nPhysImageF ${HDF4})
	include_directories(BEFORE "/usr/include/hdf")
	add_definitions(-DHAVE_LIBMFHDF)
endif()

# list include_directories
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()
