# the neutrino python module (nanobind): Neutrino windows, images and pans in python, e.g. with PySide6.
# The python (-DPython_EXECUTABLE=...) must be the one of the PySide6 built on the same Qt as Neutrino
# (e.g. Homebrew python with Homebrew pyside and qt), or one with the pip PySide6 of the same Qt version
# (macOS, NEUTRINO_PYSIDE_QT below). The wheel for the users: pyproject.toml (NEUTRINO_WHEEL, Wheel.cmake).
# nanobind: the one installed for that python (pip install nanobind), otherwise downloaded (NEUTRINO_NANOBIND_TAG)
# (not NEUTRINO_PYTHON: that is the python of the dark icons, src/CMakeLists.txt)
# option(NEUTRINO_PYTHON_MODULE "Build the neutrino python module" ON)
# OFF: the app (AppImage, deb, rpm, dmg, installer) doesn't use it; ON for the wheels (pyproject.toml) and for
# development (cmake -DNEUTRINO_PYTHON_MODULE=ON, then PYTHONPATH=<build>/python)
option(NEUTRINO_PYTHON_MODULE "Build the neutrino python module" OFF)
# set(NEUTRINO_NANOBIND_TAG "v2.14.0" CACHE STRING "nanobind version downloaded when not installed")
# (the same major version as the wheels: pyproject.toml)
set(NEUTRINO_NANOBIND_TAG "v3.1.0" CACHE STRING "nanobind version downloaded when not installed")

if (NEUTRINO_PYTHON_MODULE)
    if (CMAKE_CROSSCOMPILING)
        message(STATUS "neutrino python module: not built when cross-compiling")
        return()
    endif()

    # find_package(Python 3.9 COMPONENTS Interpreter Development.Module)
    if (NEUTRINO_WHEEL)
        # stable ABI (python >= 3.12): one wheel for all the python versions
        find_package(Python 3.10 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
    else()
        # find_package(Python 3.9 COMPONENTS Interpreter Development.Module)
        # 3.10: the minimum of nanobind 3 (e.g. AlmaLinux 9 has 3.9: the app is built without the module)
        find_package(Python 3.10 COMPONENTS Interpreter Development.Module)
    endif()
    if (NOT Python_FOUND)
        # message(STATUS "neutrino python module: python development files not found, not built")
        message(STATUS "neutrino python module: python >= 3.10 with its development files not found, not built")
        return()
    endif()

    execute_process(COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
                    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if (NANOBIND_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
        find_package(nanobind CONFIG REQUIRED)
        message(STATUS "neutrino python module: nanobind from ${NANOBIND_CMAKE_DIR}")
    else()
        include(FetchContent)
        FetchContent_Declare(nanobind
            GIT_REPOSITORY https://github.com/wjakob/nanobind.git
            GIT_TAG ${NEUTRINO_NANOBIND_TAG}
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(nanobind)
        message(STATUS "neutrino python module: nanobind ${NEUTRINO_NANOBIND_TAG} downloaded")
    endif()

    # target name: "neutrino" would clash with "Neutrino" (the executable) on case-insensitive file systems
    # nanobind_add_module(neutrino_python NB_STATIC neutrino_python.cc)
    # set_target_properties(neutrino_python PROPERTIES OUTPUT_NAME neutrino)
    # target_link_libraries(neutrino_python PRIVATE NeutrinoLib)
    # the package neutrino: neutrino/__init__.py (imports PySide6 first) and the module neutrino._neutrino
    # (in the build directory too: PYTHONPATH=<build>/python)
    if (NEUTRINO_WHEEL)
        nanobind_add_module(neutrino_python NB_STATIC STABLE_ABI neutrino_python.cc)
    else()
        nanobind_add_module(neutrino_python NB_STATIC neutrino_python.cc)
    endif()
    set_target_properties(neutrino_python PROPERTIES OUTPUT_NAME _neutrino
                          LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/neutrino$<0:>)
    configure_file(neutrino/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/neutrino/__init__.py COPYONLY)
    if (NEUTRINO_WHEEL)
        # static libNeutrino: all its objects (the linker would drop the plugin registrations and the resources)
        target_link_libraries(neutrino_python PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,NeutrinoLib>")
    else()
        target_link_libraries(neutrino_python PRIVATE NeutrinoLib)
    endif()
    if (APPLE)
        target_compile_definitions(neutrino_python PRIVATE QT_NO_OPENGL)
    endif()
    message(STATUS "neutrino python module for ${Python_EXECUTABLE} (${Python_VERSION}): ${CMAKE_CURRENT_BINARY_DIR}")

    # PySide6 of that python with its own copy of Qt (e.g. the pip wheel, in Python/site-packages/PySide6/Qt/lib):
    # if it is the same Qt version, the module and a copy of libNeutrino next to it use that Qt (UsePySideQt.cmake).
    # Only macOS for now
    option(NEUTRINO_PYSIDE_QT "macOS: the python module uses the Qt of PySide6 if it has the same version" ON)
    if (NEUTRINO_WHEEL)
        include(${CMAKE_CURRENT_SOURCE_DIR}/Wheel.cmake)
    elseif (APPLE AND NEUTRINO_PYSIDE_QT)
        execute_process(COMMAND ${Python_EXECUTABLE} -c
            "import os, PySide6, PySide6.QtCore as c; print(c.qVersion()); print(os.path.join(os.path.dirname(PySide6.__file__), 'Qt', 'lib'))"
            OUTPUT_VARIABLE PYSIDE_INFO OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE PYSIDE_RESULT)
        if (PYSIDE_RESULT EQUAL 0)
            string(REPLACE "\n" ";" PYSIDE_INFO "${PYSIDE_INFO}")
            list(GET PYSIDE_INFO 0 PYSIDE_QT_VERSION)
            list(GET PYSIDE_INFO 1 PYSIDE_QT_LIB)
            find_package(Qt6 COMPONENTS Core REQUIRED)
            if (EXISTS "${PYSIDE_QT_LIB}/QtCore.framework")
                if (PYSIDE_QT_VERSION VERSION_EQUAL Qt6Core_VERSION)
                    add_custom_command(TARGET neutrino_python POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -DMODULE=$<TARGET_FILE:neutrino_python> -DLIBNEUTRINO=$<TARGET_FILE:NeutrinoLib>
                                -DPYSIDE_QT_LIB=${PYSIDE_QT_LIB} -DNPHYS_DIR=$<TARGET_FILE_DIR:nPhysImageF>
                                -P ${CMAKE_CURRENT_SOURCE_DIR}/UsePySideQt.cmake
                        VERBATIM)
                    message(STATUS "neutrino python module: will use the Qt ${PYSIDE_QT_VERSION} of PySide6 (${PYSIDE_QT_LIB})")
                else()
                    message(WARNING "neutrino python module: PySide6 has Qt ${PYSIDE_QT_VERSION}, Neutrino is compiled with Qt "
                                    "${Qt6Core_VERSION}: the module will not work with this PySide6 (it needs the same Qt version)")
                endif()
            endif()
        endif()
    endif()
endif()
