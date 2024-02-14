find_package(Python COMPONENTS Interpreter Development)
if(Python_FOUND)
    include_directories(${Python_INCLUDE_DIRS})

    find_library(PYTHONQT    NAMES PythonQt-Qt6       PATHS "${CMAKE_CURRENT_SOURCE_DIR}/local/lib")
    find_library(PYTHONQTALL NAMES PythonQt_QtAll-Qt6 PATHS "${CMAKE_CURRENT_SOURCE_DIR}/local/lib")

    if (NOT (${PYTHONQT} STREQUAL "PYTHONQT-NOTFOUND" OR ${PYTHONQTALL} STREQUAL "PYTHONQTALL-NOTFOUND"))

        list(APPEND LOCAL_LIBS ${PYTHONQT})
        list(APPEND LOCAL_LIBS ${PYTHONQTALL})

        FIND_PATH(PYTHONQT_INCDIR    PythonQt.h       PATHS "${CMAKE_CURRENT_SOURCE_DIR}/local/include")
        FIND_PATH(PYTHONQTALL_INCDIR PythonQt_QtAll.h PATHS "${CMAKE_CURRENT_SOURCE_DIR}/local/include")
                
        IF (NOT (${PYTHONQT_INCDIR} STREQUAL "PYTHONQT_INCDIR-NOTFOUND" OR ${PYTHONQTALL_INCDIR} STREQUAL "PYTHONQTALL_INCDIR-NOTFOUND"))
            set (PYTHONQT_FOUND_COMPLETE "TRUE")
            add_definitions(-DHAVE_PYTHONQT)
            include_directories(${PYTHONQT_INCDIR})

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



