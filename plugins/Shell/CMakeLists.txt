# check for neutrino root
if (OPTION_USE_PYTHON)
    include(FindPythonLibs)
else()
    message(STATUS "Python disabled")
endif()

if(PYTHONLIBS_FOUND)
    list(APPEND LOCAL_LIBS ${PYTHON_LIBRARIES})
    include_directories(${PYTHON_INCLUDE_DIRS})

    find_library(PYTHONQT NAMES PythonQt PATH_SUFFIXES lib)

    if (NOT ${PYTHONQT} STREQUAL "PYTHONQT-NOTFOUND" )

        list(APPEND LOCAL_LIBS ${PYTHONQT})

        FIND_PATH(PYTHONQT_INCLUDE_DIR PythonQt.h PATH_SUFFIXES PythonQt)
        IF (NOT ${PYTHONQT_INCLUDE_DIR} STREQUAL "PYTHONQT_INCLUDE_DIR-NOTFOUND")
            set (PYTHONQT_FOUND_COMPLETE "TRUE")
            add_definitions(-DHAVE_PYTHONQT)
            include_directories(${PYTHONQT_INCLUDE_DIR})

            if (NOT DEFINED NEUTRINO_ROOT)
                set (NEUTRINO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
            endif()

            if (NOT IS_ABSOLUTE ${NEUTRINO_ROOT})
                set (NEUTRINO_ROOT "${CMAKE_BINARY_DIR}/${NEUTRINO_ROOT}")
            endif()

            set (CMAKE_MODULE_PATH ${NEUTRINO_ROOT}/resources/cmake)

            include(NeutrinoPlugin)
            ADD_NEUTRINO_PLUGIN()

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



