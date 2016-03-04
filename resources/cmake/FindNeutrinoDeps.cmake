# this macro to find libraries that trigger components in BOTH neutrino and nPhysImage
# CAVEAT: source inclusion must not be done here (but in FindNeutrinoGuiComponents.cmake or
# in src/CMakeLists.txt)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(TIFF REQUIRED)
if (TIFF_FOUND)
	include_directories(${TIFF_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${TIFF_LIBRARIES})
	add_definitions(-DHAVE_LIBTIFF)
endif()

# fftw_threads
find_library(FFTW_THREADS NAMES fftw3_threads)
if (NOT ${FFTW_THREADS} STREQUAL "FFTW_THREADS-NOTFOUND")
	message (STATUS "using FFTW_THREADS: ${FFTW_THREADS}")
	set(LIBS ${LIBS} ${FFTW_THREADS}) 
	add_definitions(-DHAVE_LIBFFTW_THREADS)
endif()

#fftw
find_library(FFTW NAMES fftw3)
if (NOT ${FFTW} STREQUAL "FFTW-NOTFOUND")
	message (STATUS "using FFTW: ${FFTW}")
	set(LIBS ${LIBS} ${FFTW}) 
	add_definitions(-DHAVE_LIBFFTW)
endif()

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
# hdf4
find_library (HDF4 NAMES mfhdf)
if (NOT ${HDF4} STREQUAL "HDF4-NOTFOUND")
	message (STATUS "using hdf4: ${HDF4}")
	set(LIBS ${LIBS} ${HDF4})
	include_directories(BEFORE "/usr/include/hdf")
	add_definitions(-DHAVE_LIBMFHDF)
else()
	#message ("----------------- non ho trovato hdf4 della fungia")
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
	include_directories(${NETPBM_INCLUDE_DIR})
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

#libhdf5_hl
message ("---- looking for the entire HDF5 mess...")
find_package(HDF5)
if (HDF5_FOUND)

	message(STATUS "HDF5 Found, now looking for HL")

	# IF HDF5 is there, THEN look for hl...
	find_library(HDF5HL NAMES hdf5_hl PATHS ${HDF5_LIBRARY_DIRS})
	if (${HDF5HL} STREQUAL "HDF5HL-NOTFOUND")
		message (STATUS "Cannot find HDF5_HL: disabling HDF5 support")
		message (STATUS "Search dir: " ${HDF5_LIBRARY_DIRS})

	else()

		#hdf5 libs
		include_directories(${HDF5_INCLUDE_DIRS})
		set(LIBS ${LIBS} ${HDF5_LIBRARIES})
		add_definitions(-DHAVE_HDF5)

		# hdf5_hl
		message (STATUS "using libhdf5_hl: ${HDF5HL}")
		set(LIBS ${LIBS} ${HDF5HL}) 
		add_definitions(-DHAVE_LIBHDF5HL)

		set (HDF5_FOUND_COMPLETE "TRUE")

	endif()

endif (HDF5_FOUND)

if (APPLE)

    include(FindPythonLibs)
    
    if(PYTHONLIBS_FOUND)
        list(APPEND LIBS ${PYTHON_LIBRARIES})
        include_directories(${PYTHON_INCLUDE_DIRS})    
    
        INCLUDE_DIRECTORIES(../../pythonqt-code/src ../../pythonqt-code/src/gui)
        LINK_DIRECTORIES(/Users/tommaso/pythonqt-code/lib)

        find_library(PYTHONQT NAMES PythonQt PATHS ../pythonqt-code/lib)
        find_library(PYTHONQTALL NAMES PythonQt_QtAll PATHS ../pythonqt-code/lib)
        
        if (NOT (${PYTHONQT} STREQUAL "PYTHONQT-NOTFOUND" OR ${PYTHONQTALL} STREQUAL "PYTHONQTALL-NOTFOUND"))

            message(STATUS "[PYTHONQT] using pythonqt : ${PYTHONQT} ${PYTHONQTALL}")
            list(APPEND LIBS ${PYTHONQT} ${PYTHONQTALL})
            add_definitions(-DHAVE_PYTHONQT)
        
            FIND_PATH(PYTHONQT_INCLUDE_DIR PythonQt.h ../pythonqt-code/src)
            IF (PYTHONQT_INCLUDE_DIR)
                  message (STATUS "[PYTHONQT] header dir: ${PYTHONQT_INCLUDE_DIR}")
                  include_directories(${PYTHONQT_INCLUDE_DIR})
            ENDIF ()

            FIND_PATH(PYTHONQTGUI_INCLUDE_DIR PythonQtScriptingConsole.h ../pythonqt-code/src/gui)
            IF (PYTHONQTGUI_INCLUDE_DIR)
                  message (STATUS "[PYTHONQT] gui header dir: ${PYTHONQTGUI_INCLUDE_DIR}")
                  include_directories(${PYTHONQTGUI_INCLUDE_DIR})
            ENDIF ()

            FIND_PATH(PYTHONQTALL_INCLUDE_DIR PythonQt_QtAll.h ../pythonqt-code/extensions/PythonQt_QtAll)
            IF (PYTHONQTALL_INCLUDE_DIR)
                  message (STATUS "[PYTHONQT] all header dir: ${PYTHONQTALL_INCLUDE_DIR}")
                  include_directories(${PYTHONQTALL_INCLUDE_DIR})
            ENDIF ()

            set (PYTHONQT_FOUND_COMPLETE "TRUE")
        
        endif()


    endif()
endif()
