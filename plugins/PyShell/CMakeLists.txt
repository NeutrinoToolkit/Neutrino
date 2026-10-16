if (CMAKE_CROSSCOMPILING AND NOT Python_EXECUTABLE)
    # cross build: the python of the build machine is not the target one. Fedora's mingw python packages
    # have a python running on the build machine that describes the target installation
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        find_program(Python_EXECUTABLE NAMES mingw64-python3)
    else()
        find_program(Python_EXECUTABLE NAMES mingw32-python3)
    endif()
endif()

if (CMAKE_CROSSCOMPILING AND NOT Python_EXECUTABLE)
    message(STATUS "No python for the target found (e.g. mingw64-python3): python subsystem is DISABLED!")
else()
find_package(Python COMPONENTS Interpreter Development)
endif()
if(Python_FOUND)
    # include_directories(${Python_INCLUDE_DIRS})
    # list(APPEND LOCAL_LIBS ${Python_LIBRARIES})
    # Python comes with the PythonQt targets (Python3::Python): linking ${Python_LIBRARIES} here
    # could add a second libpython when the installed PythonQt was built with another Python

    # PythonQt: use an installed one (CMake package) or build it from source.
    # To work on a local PythonQt checkout use -DFETCHCONTENT_SOURCE_DIR_PYTHONQT=<path>
    find_package(PythonQt 3 CONFIG QUIET)
    if (PythonQt_FOUND)
        message(STATUS "PythonQt: using installed package ${PythonQt_DIR}")
    else()
        message(STATUS "PythonQt: building from source")
        include(FetchContent)
        set(Python3_EXECUTABLE ${Python_EXECUTABLE}) # same python as PyShell
        set(PYTHONQT_QT_VERSION_MAJOR 6)
        set(PYTHONQT_BUILD_SHARED OFF)    # linked into the plugin: nothing to ship
        set(PYTHONQT_BUILD_GENERATOR OFF) # wrappers are still generated while configuring
        set(PYTHONQT_BUILD_TESTS OFF)
        set(PYTHONQT_BUILD_EXAMPLES OFF)
        set(PYTHONQT_INSTALL OFF)
        # all the QtAll modules but WebEngineWidgets (it would bring QtWebEngine into the packages)
        set(PYTHONQTALL_CONFIG "PythonQtCore;PythonQtGui;PythonQtSvg;PythonQtSql;PythonQtNetwork;PythonQtXml;PythonQtMultimedia;PythonQtQml;PythonQtQuick;PythonQtUiTools")
        FetchContent_Declare(pythonqt
            GIT_REPOSITORY https://github.com/iltommi/pythonqt.git
            GIT_TAG 3793f8999edd4e0b9c6841096bd7cf94bceacb8a
        )
        FetchContent_MakeAvailable(pythonqt)
    endif()

    if (TARGET PythonQt::PythonQt AND TARGET PythonQt::QtAll)

        set (PYTHONQT_FOUND_COMPLETE "TRUE")
        add_definitions(-DHAVE_PYTHONQT)
        list(APPEND LOCAL_LIBS PythonQt::PythonQt PythonQt::QtAll)

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
            install(FILES ${PYTHONQT_EXAMPLES} DESTINATION share/neutrino/python/examples)
        endif()

        list(APPEND CMAKE_MODULE_PATH ${NEUTRINO_ROOT}/resources/cmake)
        include(NeutrinoPlugin)
        ADD_NEUTRINO_PLUGIN()

        # an installed PythonQt is usually shared: ship its libraries (with their soname) next to nPhysImageF
        foreach(pythonqt_target PythonQt::PythonQt PythonQt::QtAll)
            get_target_property(pythonqt_type ${pythonqt_target} TYPE)
            get_target_property(pythonqt_imported ${pythonqt_target} IMPORTED)
            if (pythonqt_imported AND pythonqt_type STREQUAL "SHARED_LIBRARY")
                get_target_property(pythonqt_location ${pythonqt_target} LOCATION)
                get_target_property(pythonqt_soname ${pythonqt_target} IMPORTED_SONAME_RELEASE)
                if (pythonqt_soname)
                    get_filename_component(pythonqt_soname ${pythonqt_soname} NAME)
                else()
                    get_filename_component(pythonqt_soname ${pythonqt_location} NAME)
                endif()
                if (APPLE)
                    install(FILES ${pythonqt_location} DESTINATION "${CMAKE_BINARY_DIR}/Neutrino.app/Contents/Frameworks" RENAME ${pythonqt_soname} COMPONENT binary)
                elseif (LINUX)
                    install(FILES ${pythonqt_location} DESTINATION lib RENAME ${pythonqt_soname} COMPONENT binary)
                elseif (WIN32)
                    install(FILES ${pythonqt_location} DESTINATION bin COMPONENT binary)
                endif()
            endif()
        endforeach()

        # the Python used by PythonQt: its home and standard library are shipped by the packaging scripts
        # (resources/macPackage/prepareapp.sh reads pyshell_python.sh from the build directory)
        if (DEFINED PythonQt_PYTHON_EXECUTABLE)
            set(PYSHELL_PYTHON_EXECUTABLE ${PythonQt_PYTHON_EXECUTABLE})
        else()
            set(PYSHELL_PYTHON_EXECUTABLE ${Python_EXECUTABLE})
        endif()
        execute_process(
            COMMAND ${PYSHELL_PYTHON_EXECUTABLE} -c "import sys, sysconfig; print(sys.base_prefix); print(sysconfig.get_path('stdlib')); print('%d.%d' % sys.version_info[:2])"
            OUTPUT_VARIABLE PYSHELL_PYTHON_PATHS
            RESULT_VARIABLE PYSHELL_PYTHON_RESULT
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (PYSHELL_PYTHON_RESULT EQUAL 0)
            string(REPLACE "\n" ";" PYSHELL_PYTHON_PATHS "${PYSHELL_PYTHON_PATHS}")
            list(GET PYSHELL_PYTHON_PATHS 0 PYSHELL_PYTHON_HOME)
            list(GET PYSHELL_PYTHON_PATHS 1 PYSHELL_PYTHON_STDLIB)
            list(GET PYSHELL_PYTHON_PATHS 2 PYSHELL_PYTHON_VERSION)
            message(STATUS "PyShell: Python ${PYSHELL_PYTHON_VERSION} home ${PYSHELL_PYTHON_HOME}")
            # checked before starting python when it is not packaged (see PyShell::startup)
            target_compile_definitions(PyShell PRIVATE PYSHELL_PYTHON_HOME="${PYSHELL_PYTHON_HOME}")
            # python modules shipped in the packages (installed with pip by the packaging scripts)
            set(PYSHELL_PIP_PACKAGES "numpy scipy" CACHE STRING "python modules shipped with PyShell (space separated, pip names)")
            file(WRITE ${CMAKE_BINARY_DIR}/pyshell_python.sh
                "PYSHELL_PYTHON_EXECUTABLE=\"${PYSHELL_PYTHON_EXECUTABLE}\"\n"
                "PYSHELL_PYTHON_HOME=\"${PYSHELL_PYTHON_HOME}\"\n"
                "PYSHELL_PYTHON_STDLIB=\"${PYSHELL_PYTHON_STDLIB}\"\n"
                "PYSHELL_PYTHON_VERSION=\"${PYSHELL_PYTHON_VERSION}\"\n"
                "PYSHELL_PIP_PACKAGES=\"${PYSHELL_PIP_PACKAGES}\"\n")

            if (WIN32)
                # windows: the python home is <bin>/python (see PyShell::startup), libpythonX.Y.dll and the
                # dlls of the modules are in <bin> with the other runtime dependencies (resources/cmake/Package.cmake)
                set(PYSHELL_PYTHON_DEST "bin/python/lib/python${PYSHELL_PYTHON_VERSION}")
                install(DIRECTORY ${PYSHELL_PYTHON_STDLIB}/ DESTINATION ${PYSHELL_PYTHON_DEST} COMPONENT binary
                    PATTERN "__pycache__" EXCLUDE
                    PATTERN "test" EXCLUDE
                    PATTERN "tests" EXCLUDE
                    PATTERN "idlelib" EXCLUDE
                    PATTERN "tkinter" EXCLUDE
                    PATTERN "turtledemo" EXCLUDE
                    PATTERN "ensurepip" EXCLUDE
                    PATTERN "site-packages" EXCLUDE
                    PATTERN "config-*" EXCLUDE)
                # no pip for the target: the modules are the ones installed with the python of the target
                # (e.g. mingw64-python3-numpy)
                separate_arguments(PYSHELL_MODULES UNIX_COMMAND "${PYSHELL_PIP_PACKAGES}")
                foreach(module ${PYSHELL_MODULES})
                    if (IS_DIRECTORY "${PYSHELL_PYTHON_STDLIB}/site-packages/${module}")
                        message(STATUS "PyShell: shipping python module ${module}")
                        file(GLOB module_info "${PYSHELL_PYTHON_STDLIB}/site-packages/${module}-*-info")
                        install(DIRECTORY "${PYSHELL_PYTHON_STDLIB}/site-packages/${module}" ${module_info}
                            DESTINATION ${PYSHELL_PYTHON_DEST}/site-packages COMPONENT binary
                            PATTERN "__pycache__" EXCLUDE
                            PATTERN "tests" EXCLUDE)
                    else()
                        message(STATUS "PyShell: python module ${module} not found in ${PYSHELL_PYTHON_STDLIB}/site-packages: not shipped")
                    endif()
                endforeach()
            endif()
        endif()

    else()
        set (PYTHONQT_FOUND_COMPLETE "FALSE")
        message(STATUS "PythonQt NOT FOUND: PyShell is DISABLED!")
    endif()

else()
    message(STATUS "No python libraries found: python subsystem is DISABLED!")
endif()
