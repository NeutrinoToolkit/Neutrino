# find components specific to Neutrino GUI: qwt/qt/additional files here!

if(DEFINED ENV{QTDIR})
  set(CMAKE_PREFIX_PATH $ENV{QTDIR} ${CMAKE_PREFIX_PATH})
endif()

if(DEFINED QTDIR)
  set(CMAKE_PREFIX_PATH ${QTDIR} ${CMAKE_PREFIX_PATH})
endif()


find_package(Qwt REQUIRED)
if (QWT_FOUND)
	include_directories(${QWT_INCLUDE_DIRS})
	message(STATUS "qwt link: " ${QWT_LIBRARIES})
	set(LIBS ${LIBS} ${QWT_LIBRARIES})
endif(QWT_FOUND)

#libhdf5_hl
find_package(HDF5)
if (HDF5_FOUND)

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

if (HDF5_FOUND_COMPLETE)
	# add nHDF5 sources
	list (APPEND SOURCES pans/nHDF5.cc)
	list (APPEND UIS ${UIS} ../UIs/nHDF5.ui)
endif()


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

            set (PYTHONQT_FOUND_COMPLETE "TRUE")

            message(STATUS "[PYTHONQT] using pythonqt : ${PYTHONQT} ${PYTHONQTALL}")
            list(APPEND LIBS ${PYTHONQT} ${PYTHONQTALL})
            add_definitions(-DHAVE_PYTHONQT)

            FIND_PATH(PYTHONQT_INCLUDE_DIR PythonQt.h ../../pythonqt-code/src)
            IF (PYTHONQT_INCLUDE_DIR)
                  message (STATUS "[PYTHONQT] header dir: ${PYTHONQT_INCLUDE_DIR}")
                  include_directories(${PYTHONQT_INCLUDE_DIR})
            ELSE()
                set (PYTHONQT_FOUND_COMPLETE "FALSE")
            ENDIF ()

            FIND_PATH(PYTHONQTALL_INCLUDE_DIR PythonQt_QtAll.h ../../pythonqt-code/extensions/PythonQt_QtAll)
            IF (PYTHONQTALL_INCLUDE_DIR)
                  message (STATUS "[PYTHONQT] all header dir: ${PYTHONQTALL_INCLUDE_DIR}")
                  include_directories(${PYTHONQTALL_INCLUDE_DIR})
            ELSE()
                set (PYTHONQT_FOUND_COMPLETE "FALSE")
            ENDIF ()

        endif()


    endif()
endif()


if (PYTHONQT_FOUND_COMPLETE)
    MESSAGE(STATUS "adding python wrappers")
    include_directories(python)
    list (APPEND SOURCES python/nPhysPyWrapper.cc python/nPython.cc python/PythonQtScriptingConsole.cpp)
    list (APPEND UIS ../UIs/nPython.ui)
endif()

if (OpenCL_FOUND)
    MESSAGE(STATUS "adding opencl win")
    include_directories(python)
    list (APPEND SOURCES python/nPhysPyWrapper.cc python/nPython.cc python/PythonQtScriptingConsole.cpp)
    list (APPEND UIS ../UIs/nOpenCL.ui)
endif()

## find qt -- search for 5.x first, fallback to 4.x
find_package(Qt5 COMPONENTS Core Gui Sql Widgets Svg PrintSupport UiTools Multimedia MultimediaWidgets QUIET)
if (Qt5_FOUND)
	# qt5
	SET (USE_QT5 True)
	message(STATUS "Using Qt5: ${Qt5Core_INCLUDE_DIRS}")
        include_directories(${Qt5Core_INCLUDE_DIRS} ${Qt5Gui_INCLUDE_DIRS} ${Qt5Sql_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${Qt5Svg_INCLUDE_DIRS} ${Qt5PrintSupport_INCLUDE_DIRS} ${Qt5UiTools_INCLUDE_DIRS} ${Qt5Multimedia_INCLUDE_DIRS} ${Qt5MultimediaWidgets_INCLUDE_DIRS})
	
	add_definitions(-DUSE_QT5)
else()
	# some incompatibilities between 4.x and 5.x
	# qt4
	SET (USE_QT4 True)
	message(STATUS "Qt5 not found, searching for Qt4 instead")
	find_package(Qt4 4.7.0 COMPONENTS QtMain QtCore QtGui QtSQL REQUIRED)
	include(UseQt4)
	include(${QT_USE_FILE})
	
	add_definitions(-DUSE_QT4)
endif()


add_definitions(${QT_DEFINITIONS})
include_directories(${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})


