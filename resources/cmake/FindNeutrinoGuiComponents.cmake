# find components specific to Neutrino GUI: qwt/qt/additional files here!

find_package(Qwt REQUIRED)
if (QWT_FOUND)
	include_directories(${QWT_INCLUDE_DIRS})
	message(STATUS "qwt link: " ${QWT_LIBRARIES})
	set(LIBS ${LIBS} ${QWT_LIBRARIES})
endif(QWT_FOUND)

## find qt MUST be LAST to all modifications to SOURCES list
## (otherwise automoc and autoui won't take new sources in account)
set (RESOURCES "${${PROJECT_NAME}_SOURCE_DIR}/resources/neutrino.qrc")

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


# PYTHNOQT

if (APPLE)

    include(FindPythonLibs)

    if(PYTHONLIBS_FOUND)
        list(APPEND LIBS ${PYTHON_LIBRARIES})
        include_directories(${PYTHON_INCLUDE_DIRS})

        set(pythonqt_src "${CMAKE_CURRENT_SOURCE_DIR}/../../pythonqt-code")

        INCLUDE_DIRECTORIES(${pythonqt_src}/src ${pythonqt_src}/src/gui)
        LINK_DIRECTORIES(${pythonqt_src}/lib)

        find_library(PYTHONQT NAMES PythonQt PATHS ${pythonqt_src}/lib)
        find_library(PYTHONQTALL NAMES PythonQt_QtAll PATHS ${pythonqt_src}/lib)

        if (NOT (${PYTHONQT} STREQUAL "PYTHONQT-NOTFOUND" OR ${PYTHONQTALL} STREQUAL "PYTHONQTALL-NOTFOUND"))

            message(STATUS "[PYTHONQT] using pythonqt : ${PYTHONQT} ${PYTHONQTALL}")
            list(APPEND LIBS ${PYTHONQT} ${PYTHONQTALL})
            add_definitions(-DHAVE_PYTHONQT)

            FIND_PATH(PYTHONQT_INCLUDE_DIR PythonQt.h ${pythonqt_src}/src)
            IF (PYTHONQT_INCLUDE_DIR)
                  message (STATUS "[PYTHONQT] header dir: ${PYTHONQT_INCLUDE_DIR}")
                  include_directories(${PYTHONQT_INCLUDE_DIR})
            ENDIF ()

            FIND_PATH(PYTHONQTGUI_INCLUDE_DIR PythonQtScriptingConsole.h ${pythonqt_src}/src/gui)
            IF (PYTHONQTGUI_INCLUDE_DIR)
                  message (STATUS "[PYTHONQT] gui header dir: ${PYTHONQTGUI_INCLUDE_DIR}")
                  include_directories(${PYTHONQTGUI_INCLUDE_DIR})
            ENDIF ()

            FIND_PATH(PYTHONQTALL_INCLUDE_DIR PythonQt_QtAll.h ${pythonqt_src}/extensions/PythonQt_QtAll)
            IF (PYTHONQTALL_INCLUDE_DIR)
                  message (STATUS "[PYTHONQT] all header dir: ${PYTHONQTALL_INCLUDE_DIR}")
                  include_directories(${PYTHONQTALL_INCLUDE_DIR})
            ENDIF ()

            set (PYTHONQT_FOUND_COMPLETE "TRUE")

        endif()


    endif()
endif()





add_definitions(${QT_DEFINITIONS})
include_directories(${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})



