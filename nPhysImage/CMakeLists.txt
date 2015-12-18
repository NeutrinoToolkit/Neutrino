# this is the frigging nPhysImage cmakelists
cmake_minimum_required (VERSION 3.0)



if (APPLE) 
    message(STATUS "Apple : use macports !")
    set (CMAKE_C_COMPILER "/opt/local/bin/gcc")
    set (CMAKE_CXX_COMPILER "/opt/local/bin/g++")
    set (CMAKE_PREFIX_PATH "/opt/local")
endif()

if (UNIX AND (NOT APPLE))
    message(STATUS "Linux")
    set (CMAKE_PREFIX_PATH "/usr")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_compile_options(-std=c++0x)
add_compile_options(-fPIC)


set (CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -Wall -D__phys_debug=10")
set (CMAKE_CXX_FLAGS "-O3")



add_library (nPhysImageF SHARED NaNStream.cc  nPhysFormats.cc  nPhysImageF.cc  nPhysMaths.cc  nPhysWave.cc anymap.cc unwrapping/unwrap_goldstein.cc  unwrapping/unwrap_quality.cc  unwrapping/unwrap_simple.cc  unwrapping/unwrap_util.cc  )


target_include_directories (nPhysImageF PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nPhysImageF "-lm")


message(STATUS "cmake_prefix_path: ${CMAKE_PREFIX_PATH}")

#SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")



# force necessary defines on config.h
add_definitions(-DHAVE_LIBPTHREAD)

include_directories(BEFORE "/opt/local/include")
include_directories(BEFORE "/usr/local/include")

# gsl
find_library(GSL NAMES gsl)
if (NOT ${GSL} STREQUAL "GSL-NOTFOUND")
	message (STATUS "using gsl: ${GSL}")
	set(LIBS ${LIBS} ${GSL}) 
	add_definitions(-DHAVE_LIBGSL)
endif()

# gslcblas
find_library(GSLCBLAS NAMES gslcblas)
if (NOT ${GSLCBLAS} STREQUAL "GSLCBLAS-NOTFOUND")
	message (STATUS "using gslcblas: ${GSL}")
	set(LIBS ${LIBS} ${GSLCBLAS}) 
	add_definitions(-DHAVE_LIBGSLCBLAS)
endif()

# fftw_threads
find_library(FFTW_THREADS NAMES fftw3_threads)
if (NOT ${FFTW_THREADS} STREQUAL "FFTW_THREADS-NOTFOUND")
	message (STATUS "using FFTW_THREADS: ${FFTW_THREADS}")
	set(LIBS ${LIBS} ${FFTW_THREADS}) 
	add_definitions(-DHAVE_LIBFFTW_THREADS)
endif()

#fftw
find_library(FFTW NAMES fftw3 fftw3-3)
if (NOT ${FFTW} STREQUAL "FFTW-NOTFOUND")
	message (STATUS "using FFTW: ${FFTW}")
	set(LIBS ${LIBS} ${FFTW}) 
	add_definitions(-DHAVE_LIBFFTW)
endif()


# pgm
find_library(PBM NAMES netpbm)
if (NOT ${PBM} STREQUAL "PBM-NOTFOUND")
	message (STATUS "using netpbm: ${PBM}")
	set(LIBS ${LIBS} ${PBM}) 
	add_definitions(-DHAVE_LIBNETPBM)
	if (APPLE) 
		include_directories(BEFORE "/opt/local/include/netpbm")
	endif()
endif()

# hdf4
find_library (HDF4 NAMES mfhdf)
if (NOT ${HDF4} STREQUAL "HDF4-NOTFOUND")
	message (STATUS "using hdf4: ${HDF4}")
	set(LIBS ${LIBS} ${HDF4})
	include_directories(BEFORE "/usr/include/hdf")
	add_definitions(-DHAVE_LIBMFHDF)
endif()

find_package(JPEG REQUIRED)
if (JPEG_FOUND)
	include_directories(${JPEG_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${JPEG_LIBRARIES})
	add_definitions(-DHAVE_JPEG)
endif()

find_package(ZLIB REQUIRED)
if (ZLIB_FOUND)
	include_directories(${ZLIB_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${ZLIB_LIBRARIES})
	add_definitions(-DHAVE_ZLIB)
endif()


find_package(TIFF REQUIRED)
if (TIFF_FOUND)
	include_directories(${TIFF_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${TIFF_LIBRARIES})
	add_definitions(-DHAVE_LIBTIFF)
endif()

# list include_directories


SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS} -static-libgcc -static-libstdc++")
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
set (LIB_INSTALL_DIR "/usr/lib")
set (CMAKE_INSTALL_RPATH "/usr/lib")
set (MACOSX_RPATH off)
set_target_properties(nPhysImageF PROPERTIES INSTALL_RPATH ${LIB_INSTALL_DIR} MACOSX_RPATH off)
endif()

if (NOT APPLE)
    install (DIRECTORY "." COMPONENT dev DESTINATION include/nPhysImage FILES_MATCHING PATTERN "*.h")
endif()


