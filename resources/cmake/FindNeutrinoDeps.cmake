# this macro to find libraries that trigger components in BOTH neutrino and nPhysImage
# CAVEAT: source inclusion must not be done here (but in FindNeutrinoGuiComponents.cmake or
# in src/CMakeLists.txt)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	add_definitions(-DHAVE_OPENMP)
endif()

find_package(TIFF REQUIRED)
if (TIFF_FOUND)
	include_directories(${TIFF_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${TIFF_LIBRARIES})
	add_definitions(-DHAVE_LIBTIFF)
endif()

# fftw_threads
find_package(FFTW REQUIRED)
if (FFTW_FOUND AND FFTW_THREADS_LIB)
	include_directories(${FFTW_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${FFTW_LIBRARIES} ${FFTW_THREADS_LIB})
	add_definitions(-DHAVE_LIBFFTW_THREADS)
else()
    message( FATAL_ERROR "You need fftw_threads library" )
endif()

find_library(GSL NAMES gsl)
if (NOT ${GSL} STREQUAL "GSL-NOTFOUND")
	message (STATUS "using gsl: ${GSL}")
	set(LIBS ${LIBS} ${GSL}) 
	add_definitions(-DHAVE_LIBGSL)

	FIND_PATH(GSL_INCLUDE_DIR gsl/gsl_math.h
  		/usr/local/include/
  		/usr/include
  	)
    IF (GSL_INCLUDE_DIR)
      message (STATUS "gsl header dir: ${GSL_INCLUDE_DIR}")
      include_directories(BEFORE "${GSL_INCLUDE_DIR}")
    ENDIF (GSL_INCLUDE_DIR)
	
endif()


# gslcblas
find_library(GSLCBLAS NAMES gslcblas)
if (NOT ${GSLCBLAS} STREQUAL "GSLCBLAS-NOTFOUND")
	message (STATUS "using gslcblas: ${GSL}")
	set(LIBS ${LIBS} ${GSLCBLAS}) 
	add_definitions(-DHAVE_LIBGSLCBLAS)
else()
	message(FATAL_ERROR "Missing libgsl. Stop.")
endif()

# hdf4
find_library (HDF4 NAMES mfhdf)
if (NOT ${HDF4} STREQUAL "HDF4-NOTFOUND")
	message (STATUS "using hdf4: ${HDF4}")
	set(LIBS ${LIBS} ${HDF4})
	add_definitions(-DHAVE_LIBMFHDF)
	
	FIND_PATH(HDF4_INCLUDE_DIR hdf.h
  		/usr/local/include/
  		/usr/include
  		/usr/local/include/hdf
  		/usr/include/hdf
  	)
    IF (HDF4_INCLUDE_DIR)
      message (STATUS "hdf4 header dir: ${HDF4_INCLUDE_DIR}")
      include_directories(BEFORE "${HDF4_INCLUDE_DIR}")
    ENDIF (HDF4_INCLUDE_DIR)
endif()
find_library (DF NAMES df)
if (NOT ${DF} STREQUAL "DF-NOTFOUND")
	message (STATUS "using df: ${DF}")
	set(LIBS ${LIBS} ${DF})
	add_definitions(-DHAVE_LIBDF)
endif()


# pgm
find_library(NETPBM NAMES netpbm)
if (NOT ${NETPBM} STREQUAL "NETPBM-NOTFOUND")
	message (STATUS "using netpbm: ${NETPBM}")
	set(LIBS ${LIBS} ${NETPBM}) 
	add_definitions(-DHAVE_LIBNETPBM)

  FIND_PATH(NETPBM_INCLUDE_DIR pgm.h
  /usr/include
  /usr/include/netpbm
  /usr/local/include
  /usr/local/include/netpbm
  )
  IF (NETPBM_INCLUDE_DIR)
	message (STATUS "netpbm header dir: ${NETPBM_INCLUDE_DIR}")
	include_directories(BEFORE ${NETPBM_INCLUDE_DIR})
  ENDIF (NETPBM_INCLUDE_DIR)

endif()

find_package(JPEG REQUIRED)
if (JPEG_FOUND)
	include_directories(${JPEG_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${JPEG_LIBRARIES})
	add_definitions(-DHAVE_JPEG)
endif()

find_library(CFITS NAMES cfitsio)
if (NOT ${CFITS} STREQUAL "CFITS-NOTFOUND")
	message(STATUS "using cfits: ${CFITS}")
	list(APPEND LIBS ${CFITS})
	add_definitions(-DHAVE_LIBCFITSIO)
endif()

# opencl
if (NOT CMAKE_MINOR_VERSION LESS "5")
	find_package(OpenCL QUIET)
else()
	# opencl-config.cmake not available until cmake 3.5.x
	find_library(OpenCL_LIBRARIES NAMES OpenCL)
	find_path(OpenCL_INCLUDE_DIRS opencl.h PATH_SUFFIXES CL)
	if (NOT ${OpenCL_LIBRARIES} STREQUAL "OpenCL_LIBRARIES-NOTFOUND")
		message(STATUS "Found OpenCL")
		set (OpenCL_FOUND true)
	endif()
endif()

if (OpenCL_FOUND)
    message (STATUS "using OpenCL: ${OpenCL_LIBRARIES}")
    include_directories(${OpenCL_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${OpenCL_LIBRARIES})
    add_definitions(-DHAVE_OPENCL)

    find_library (CLFFT NAMES clFFT)
    if (NOT ${CLFFT} STREQUAL "CLFFT-NOTFOUND")
        message (STATUS "using clFFT: ${CLFFT}")
        set(LIBS ${LIBS} ${CLFFT})
        add_definitions(-DHAVE_LIBCLFFT)
    endif()
endif()
