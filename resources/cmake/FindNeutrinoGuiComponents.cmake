

find_package(Qwt REQUIRED)
if (QWT_FOUND)
	include_directories(${QWT_INCLUDE_DIRS})
	message(STATUS "qwt link: " ${QWT_LIBRARIES})
	set(LIBS ${LIBS} ${QWT_LIBRARIES})
endif(QWT_FOUND)

find_package(TIFF REQUIRED)
if (TIFF_FOUND)
	include_directories(${TIFF_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${TIFF_LIBRARIES})
	add_definitions(-DHAVE_TIFF)
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
find_library(PBM NAMES netpbm)
if (NOT ${PBM} STREQUAL "PBM-NOTFOUND")
	message (STATUS "using netpbm: ${PBM}")
	set(LIBS ${LIBS} ${PBM}) 
	add_definitions(-DHAVE_LIBNETPBM)
	if (APPLE) 
		include_directories(BEFORE "/opt/local/include/netpbm")
	endif()
endif()

find_package(JPEG REQUIRED)
if (JPEG_FOUND)
	include_directories(${JPEG_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${JPEG_LIBRARIES})
	add_definitions(-DHAVE_JPEG)
endif()

find_package(TIFF REQUIRED)
if (TIFF_FOUND)
	include_directories(${TIFF_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${TIFF_LIBRARIES})
	add_definitions(-DHAVE_TIFF)
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

	
		# add nHDF5 sources
		set (SOURCES ${SOURCES} pans/nHDF5.cc)
		set (UIS ${UIS} ../UIs/nHDF5.ui)

	endif()

endif (HDF5_FOUND)


## find qt MUST be LAST to all modifications to SOURCES list
## (otherwise automoc and autoui won't take new sources in account)

## find qt -- search for 5.x first, fallback to 4.x
find_package(Qt5 COMPONENTS Core Gui Sql Widgets Svg PrintSupport QUIET)
if (Qt5_FOUND)
	# qt5
	SET (USE_QT5 True)
	message(STATUS "Using Qt5: ${Qt5Core_INCLUDE_DIRS}")
	include_directories(${Qt5Core_INCLUDE_DIRS} ${Qt5Gui_INCLUDE_DIRS} ${Qt5Sql_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${Qt5Svg_INCLUDE_DIRS} ${Qt5PrintSupport_INCLUDE_DIRS})
	
	QT5_ADD_RESOURCES( RES_SOURCES ${RESOURCES} )
	QT5_WRAP_UI( UI_HEADERS ${UIS} )
	add_definitions(-DUSE_QT5)
else()
	# some incompatibilities between 4.x and 5.x
	# qt4
	SET (USE_QT4 True)
	message(STATUS "Qt5 not found, searching for Qt4 instead")
	find_package(Qt4 4.7.0 COMPONENTS QtMain QtCore QtGui QtSQL REQUIRED)
	include(UseQt4)
	include(${QT_USE_FILE})
	
	QT4_ADD_RESOURCES( RES_SOURCES ${RESOURCES} )
	QT4_WRAP_UI( UI_HEADERS ${UIS} )
	add_definitions(-DUSE_QT4)
endif()


add_definitions(${QT_DEFINITIONS})
include_directories(${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})


