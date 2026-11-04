# NEUTRINO_WHEEL (pyproject.toml): the module neutrino._neutrino with nPhysImage, libNeutrino and the plugins linked in.
# No Qt in the wheel: the module uses the Qt of the PySide6 package (a dependency of the wheel), next to it in
# site-packages. Qt 6 is backward compatible: compiled with Qt 6.x it works with PySide6 >= 6.x (pyproject.toml);
# the build should use the official Qt (e.g. aqtinstall), the one of the PySide6 wheels.
# The other libraries (fftw, gsl, hdf5, ...) are put in the wheel by delocate/auditwheel/delvewheel (not Qt).

if (CMAKE_VERSION VERSION_LESS 3.26)
    message(FATAL_ERROR "neutrino wheel: cmake >= 3.26 needed (WHOLE_ARCHIVE, Development.SABIModule)")
endif()

# the Qt of the build must not be newer than the minimum PySide6 of pyproject.toml
find_package(Qt6 COMPONENTS Core REQUIRED)
file(READ ${PROJECT_SOURCE_DIR}/pyproject.toml NEUTRINO_PYPROJECT)
string(REGEX MATCH "\"PySide6>=([0-9.]+)" _ "${NEUTRINO_PYPROJECT}")
if (NOT CMAKE_MATCH_1)
    message(FATAL_ERROR "neutrino wheel: no \"PySide6>=x.y\" dependency in pyproject.toml")
elseif (Qt6Core_VERSION VERSION_GREATER_EQUAL "${CMAKE_MATCH_1}.99")
    message(FATAL_ERROR "neutrino wheel: compiled with Qt ${Qt6Core_VERSION}, it would not work with "
                        "PySide6 ${CMAKE_MATCH_1} (pyproject.toml): use an older Qt or require a newer PySide6")
endif()
message(STATUS "neutrino wheel: Qt ${Qt6Core_VERSION}, for PySide6 >= ${CMAKE_MATCH_1}")

# the Qt of PySide6: site-packages/neutrino/_neutrino.so -> site-packages/PySide6/Qt/lib
# (Windows: no run path, importing PySide6 adds its directory to the dll search)
if (APPLE)
    set(NEUTRINO_WHEEL_RPATH "@loader_path/../PySide6/Qt/lib")
elseif (LINUX)
    set(NEUTRINO_WHEEL_RPATH "$ORIGIN/../PySide6/Qt/lib")
endif()
set_target_properties(neutrino_python PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "${NEUTRINO_WHEEL_RPATH}")
if (APPLE)
    # Qt referenced by @rpath (Homebrew Qt has absolute install names, the official Qt has @rpath already)
    add_custom_command(TARGET neutrino_python POST_BUILD
        COMMAND ${CMAKE_COMMAND} -DMODULE=$<TARGET_FILE:neutrino_python> -DRPATHS=${NEUTRINO_WHEEL_RPATH}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/UsePySideQt.cmake
        VERBATIM)
endif()

# scikit-build-core installs only this component in the wheel (neutrino/__init__.py: wheel.packages)
install(TARGETS neutrino_python LIBRARY DESTINATION neutrino COMPONENT python)
if (APPLE)
    # scikit-build-core strips the installed module (install.strip): sign it again (arm64 needs a valid signature)
    install(CODE "execute_process(COMMAND codesign --force --sign - \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/neutrino/$<TARGET_FILE_NAME:neutrino_python>\")"
            COMPONENT python)
endif()
