# find components specific to Neutrino GUI files here!

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(DEFINED ENV{QTDIR})
  set(CMAKE_PREFIX_PATH $ENV{QTDIR} ${CMAKE_PREFIX_PATH})
endif()

if(DEFINED QTDIR)
  set(CMAKE_PREFIX_PATH ${QTDIR} ${CMAKE_PREFIX_PATH})
endif()

if (OPTION_USE_PYTHON)
	include(FindPythonLibs)
else()
	message(STATUS "Python disabled")
endif()

if(PYTHONLIBS_FOUND)
	list(APPEND LIBS ${PYTHON_LIBRARIES})
	include_directories(${PYTHON_INCLUDE_DIRS})

	find_library(PYTHONQT NAMES PythonQt PATH_SUFFIXES lib)

	if (NOT ${PYTHONQT} STREQUAL "PYTHONQT-NOTFOUND" )

		message(STATUS "Using pythonqt : ${PYTHONQT}")
	    	list(APPEND LIBS ${PYTHONQT})    	

		FIND_PATH(PYTHONQT_INCLUDE_DIR PythonQt.h PATH_SUFFIXES PythonQt)
		IF (NOT ${PYTHONQT_INCLUDE_DIR} STREQUAL "PYTHONQT_INCLUDE_DIR-NOTFOUND")
                        set (PYTHONQT_FOUND_COMPLETE "TRUE")
    			add_definitions(-DHAVE_PYTHONQT)
			message (STATUS "PythonQt header dir: ${PYTHONQT_INCLUDE_DIR}")
			include_directories(${PYTHONQT_INCLUDE_DIR})

		ELSE()
			set (PYTHONQT_FOUND_COMPLETE "FALSE")
			message(STATUS "PythonQt.h NOT FOUND (perhaps you forgot -DCMAKE_INCLUDE_PATH)")
		ENDIF ()
	else()
		message(STATUS "PythonQt NOT FOUND (perhaps you forgot -DCMAKE_LIBRARY_PATH)")
	endif()
else()
	message(STATUS "No python libraries found: python subsystem is DISABLED!")

endif()

## find qt -- search for 5.x first, fallback to 4.x
find_package(Qt5 COMPONENTS Core Gui Sql Widgets Svg PrintSupport UiTools Multimedia MultimediaWidgets OpenGL)
if (Qt5_FOUND)
	# qt5
	SET (USE_QT5 True)
	message(STATUS "Using Qt5: ${Qt5Core_INCLUDE_DIRS}")
	include_directories(${Qt5Core_INCLUDE_DIRS} ${Qt5Gui_INCLUDE_DIRS} ${Qt5Sql_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${Qt5Svg_INCLUDE_DIRS} ${Qt5PrintSupport_INCLUDE_DIRS} ${Qt5UiTools_INCLUDE_DIRS} ${Qt5Multimedia_INCLUDE_DIRS} ${Qt5MultimediaWidgets_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS})
	
	add_definitions(-DUSE_QT5)
else()
	# some incompatibilities between 4.x and 5.x
	# qt4
	SET (USE_QT4 True)
	message(STATUS "Qt5 not found, searching for Qt4 instead")
	find_package(Qt4 4.7.0 COMPONENTS QtMain QtCore QtGui QtSQL QtSvg QtUiTools REQUIRED)

        include(${QT_USE_FILE})
	
	add_definitions(-DUSE_QT4)
endif()


add_definitions(${QT_DEFINITIONS})
include_directories(${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})

if(NOT DEFINED PANDOC)
  message(STATUS "Looking for pandoc")
  find_program(PANDOC pandoc)
  if(PANDOC)
    message(STATUS "Looking for pandoc - found")
  else(PANDOC)
    message(STATUS "Looking for pandoc - not found")
  endif(PANDOC)
  mark_as_advanced(PANDOC)
endif(NOT DEFINED PANDOC)

MACRO(ADD_PLUGIN_HELP)
    if(PANDOC AND (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md"))

        set(PANDOC_QRC ${CMAKE_CURRENT_BINARY_DIR}/pandoc.qrc)
        file(WRITE ${PANDOC_QRC} "<RCC>\n    <qresource prefix=\"/${MY_PROJECT_NAME}/\">\n")
        file(APPEND ${PANDOC_QRC} "        <file>README.html</file>\n")
        file(APPEND ${PANDOC_QRC} "    </qresource>\n</RCC>")

        qt5_add_resources(RES_SOURCES ${PANDOC_QRC})

        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/README.html
            COMMAND ${PANDOC} -f markdown -t html -s -S README.md --self-contained -o ${CMAKE_CURRENT_BINARY_DIR}/README.html
            MAIN_DEPENDENCY "README.md"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )

        add_custom_target(pandoc${MY_PROJECT_NAME} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/README.html SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
    endif()
ENDMACRO()

SET(Qt5LinguistTools_DIR "${Qt5_DIR}/../Qt5LinguistTools")
find_package(Qt5LinguistTools)

MACRO(ADD_PLUGIN_TRANSLATIONS)
    if (Qt5LinguistTools_FOUND)
        SET(LANGUAGES fr_FR it_IT ko_KP)

        SET(LANGUAGE_TS_FILES)
        FOREACH(LANGUAGE ${LANGUAGES})
        SET(TS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${MY_PROJECT_NAME}_${LANGUAGE}.ts")
        message(STATUS "Language : ${TS_FILE}")
        SET(LANGUAGE_TS_FILES ${LANGUAGE_TS_FILES} ${TS_FILE})

        if(EXISTS ${TS_FILE})
            qt5_add_translation(qm_files ${TS_FILE})
        else ()
            qt5_create_translation(qm_files ${SOURCES} ${UIS} ${TS_FILE})
        endif()

        ENDFOREACH()

        IF(LANGUAGE_TS_FILES)
        set(TRANSL_QRC ${CMAKE_CURRENT_BINARY_DIR}/translations.qrc)
        file(WRITE ${TRANSL_QRC} "<RCC>\n    <qresource prefix=\"/translations/\">\n")
        foreach(my_file ${qm_files})
            file(RELATIVE_PATH my_file_relative_path ${CMAKE_CURRENT_BINARY_DIR} ${my_file})
            file(APPEND ${TRANSL_QRC} "        <file>${my_file_relative_path}</file>\n")
        endforeach()
        file(APPEND ${TRANSL_QRC} "    </qresource>\n</RCC>")
        list(LENGTH LANGUAGE_TS_FILES LIST_LENGTH)
        message(STATUS "${LIST_LENGTH} translation files: ${TRANSL_QRC}")

        qt5_add_resources(RES_SOURCES ${TRANSL_QRC})
        ENDIF(LANGUAGE_TS_FILES)

    endif(Qt5LinguistTools_FOUND)
ENDMACRO()


