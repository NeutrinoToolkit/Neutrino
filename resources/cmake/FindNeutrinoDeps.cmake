
find_package(OPENMP REQUIRED)
if(OPENMP_FOUND)
	set(LIBS ${LIBS} OpenMP::OpenMP_CXX)
endif() 

find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
	include_directories( ${ZLIB_INCLUDE_DIRS} )
	set(LIBS ${LIBS} ${ZLIB_LIBRARIES} )
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		message (STATUS "using ZLIB: ${ZLIB_LIBRARIES}")
	endif()
endif()

find_package(TIFF REQUIRED)
if(TIFF_FOUND)
	include_directories(${TIFF_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${TIFF_LIBRARIES})
	add_definitions(-DHAVE_LIBTIFF)
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		message (STATUS "using TIFF: ${TIFF_LIBRARIES}")
	endif()
endif()

#fftw
find_library(FFTW NAMES fftw3 fftw3-3 REQUIRED)
if (NOT ${FFTW} STREQUAL "FFTW-NOTFOUND")
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		message (STATUS "using FFTW: ${FFTW}")
	endif()
	set(LIBS ${LIBS} ${FFTW})
	add_definitions(-DHAVE_LIBFFTW)
endif()

#in precompiled win dlls the threads are included
if(WIN32)
	if( ${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
		add_compile_options(-Wa,-mbig-obj)
	endif()
	add_definitions(-DNT_THREADS) 
else()
	# fftw_threads
	find_library(FFTW_THREADS NAMES fftw3_threads REQUIRED)
	if (NOT ${FFTW_THREADS} STREQUAL "FFTW_THREADS-NOTFOUND")
		if (CMAKE_BUILD_TYPE STREQUAL "Debug")
			message (STATUS "using FFTW_THREADS: ${FFTW_THREADS}")
		endif()
		set(LIBS ${LIBS} ${FFTW_THREADS})
	endif()
endif()

#gsl
find_library(GSL NAMES gsl)
if (NOT ${GSL} STREQUAL "GSL-NOTFOUND")
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		message (STATUS "using gsl: ${GSL}")
	endif()
	set(LIBS ${LIBS} ${GSL})
	add_definitions(-DHAVE_LIBGSL)

	FIND_PATH(GSL_INCLUDE_DIR gsl/gsl_math.h
		/usr/local/include/
		/usr/include
		)
	IF (GSL_INCLUDE_DIR)
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		message (STATUS "gsl header dir: ${GSL_INCLUDE_DIR}")
	endif()
	include_directories(BEFORE "${GSL_INCLUDE_DIR}")
	ENDIF (GSL_INCLUDE_DIR)
endif()

# gslcblas
find_library(GSLCBLAS NAMES gslcblas)
if (NOT ${GSLCBLAS} STREQUAL "GSLCBLAS-NOTFOUND")
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		message (STATUS "using gslcblas: ${GSL}")
	endif()
	set(LIBS ${LIBS} ${GSLCBLAS})
	add_definitions(-DHAVE_LIBGSLCBLAS)
else()
	message(FATAL_ERROR "Missing gslcblas.")
endif()

# hdf4
find_library (HDF4 NAMES mfhdf)
if (NOT ${HDF4} STREQUAL "HDF4-NOTFOUND")
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		message (STATUS "using hdf4: ${HDF4}")
	endif()
	set(LIBS ${LIBS} ${HDF4})
	add_definitions(-DHAVE_LIBMFHDF)

	FIND_PATH(HDF4_INCLUDE_DIR hdf.h
		/usr/local/include/
		/usr/include
		/usr/local/include/hdf
		/usr/include/hdf
		${CMAKE_FIND_ROOT_PATH}/include
		)

	IF (HDF4_INCLUDE_DIR)
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		message (STATUS "hdf4 header dir: ${HDF4_INCLUDE_DIR}")
	endif()
	include_directories(BEFORE "${HDF4_INCLUDE_DIR}")
	ENDIF (HDF4_INCLUDE_DIR)
endif()

find_library (DF NAMES df)
if (NOT ${DF} STREQUAL "DF-NOTFOUND")
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		message (STATUS "using df: ${DF}")
	endif()
	set(LIBS ${LIBS} ${DF})
	add_definitions(-DHAVE_LIBDF)
endif()

find_package(JPEG REQUIRED)
if (JPEG_FOUND)
	include_directories(${JPEG_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${JPEG_LIBRARIES})
	add_definitions(-DHAVE_JPEG)
endif()

find_library(CFITS NAMES cfitsio)
if (NOT ${CFITS} STREQUAL "CFITS-NOTFOUND")
	FIND_PATH(CFITS_INCLUDE_DIR fitsio.h
		/usr/include
		/usr/include/cfitsio
		/usr/local/include
		/usr/local/include/cfitsio
		${CMAKE_FIND_ROOT_PATH}/include
		${CMAKE_FIND_ROOT_PATH}/include/cfitsio
		)
	IF (CFITS_INCLUDE_DIR)
	IF (CMAKE_BUILD_TYPE STREQUAL "Debug")
	    message(STATUS "using cfits: ${CFITS}")
	ENDIF()
	list(APPEND LIBS ${CFITS})
	add_definitions(-DHAVE_LIBCFITSIO)
	include_directories(BEFORE ${CFITS_INCLUDE_DIR})
	ENDIF()
endif()

# opencl
find_package(OpenCL QUIET)
if (OpenCL_FOUND)
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		message (STATUS "using OpenCL: ${OpenCL_LIBRARIES}")
	endif()

	include_directories(${OpenCL_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${OpenCL_LIBRARIES})
	add_definitions(-DHAVE_OPENCL)

	find_library (CLFFT NAMES clFFT)
    if (NOT ${CLFFT} STREQUAL "CLFFT-NOTFOUND")
		if (CMAKE_BUILD_TYPE STREQUAL "Debug")
			message (STATUS "using clFFT: ${CLFFT}")
		endif()
		set(LIBS ${LIBS} ${CLFFT})
		add_definitions(-DHAVE_LIBCLFFT)
	endif()
endif()
