# check for neutrino root
IF(APPLE)
    set(PYVER 2.7)
ENDIF()

find_package(PythonLibs ${PYVER})

if(PYTHONLIBS_FOUND)
    list(APPEND LOCAL_LIBS ${PYTHON_LIBRARIES})
    include_directories(${PYTHON_INCLUDE_DIRS})

    message(STATUS "Found Python includes: ${PYTHON_INCLUDE_DIRS}")

    include(FindPythonInterp)
    if(PYTHONINTERP_FOUND)
        message(STATUS "Found Python interpreter: ${PYTHON_EXECUTABLE}")
        if (NOT NUMPY_PATH AND PYTHON_EXECUTABLE)
            file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/determineNumpyPath.py "try: import numpy; print(numpy.get_include())\nexcept: pass\n")
            exec_program("${PYTHON_EXECUTABLE}" ARGS "\"${CMAKE_CURRENT_BINARY_DIR}/determineNumpyPath.py\"" OUTPUT_VARIABLE NUMPY_PATH)
        endif (NOT NUMPY_PATH AND PYTHON_EXECUTABLE)
        if (WIN32)
#            install (DIRECTORY ${PYTHON_INCLUDE_DIRS}/../../lib/python2.7 DESTINATION bin PATTERN "python2.7/test" EXCLUDE)
            if (NOT DEFINED NUMPY_PATH)
                set(NUMPY_PATH "${PYTHON_INCLUDE_PATH}/../../lib/python2.7/site-packages/numpy/core/include")
            endif()
        endif()
        find_path(
            NUMPY_INCLUDE_DIR
            NAMES "numpy/arrayobject.h"
            PATHS "${NUMPY_PATH}" "${PYTHON_INCLUDE_PATH}" "${MY_NUMPY_PATH}"
            DOC "Directory where the arrayobject.h header file can be found."
        )
        if (NUMPY_INCLUDE_DIR)
            message(STATUS "Found NumPy includes: ${NUMPY_INCLUDE_DIR}")
            set(NUMPY_FOUND 1 CACHE INTERNAL "NumPy found")
            include_directories(${NUMPY_INCLUDE_DIR})
            add_definitions(-DHAVE_NUMPY -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
        endif ()
    endif(PYTHONINTERP_FOUND)

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


            file (GLOB PYTHONQT_EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/examples/*py)


            if (WIN32)
                install(FILES ${PYTHONQT_EXAMPLES} DESTINATION python/examples)
            endif()

            if (LINUX)
                message (STATUS "PythonQt lib ${PYTHONQT} is copied along")
                set(CPACK_RPM_COMPONENT_INSTALL ON)
                install (FILES ${PYTHONQT} DESTINATION lib COMPONENT binary)
                list(APPEND CPACK_DEBIAN_PACKAGE_CONFLICTS "libpythonqt3.0")
                message (STATUS "Warning: will conflict with ${CPACK_DEBIAN_PACKAGE_CONFLICTS} package")

                install(FILES ${PYTHONQT_EXAMPLES} DESTINATION share/neutrino/python/examples)
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



