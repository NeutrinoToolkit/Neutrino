#!/bin/bash
# cibuildwheel before-all on Windows (git bash; the MSVC environment comes from the workflow, msvc-dev-cmd):
# the dependencies of Neutrino as static libraries with the dynamic runtime (vcpkg x64-windows-static-md: no DLLs
# to put in the wheel but libomp.dll of LLVM, delvewheel) and the official Qt in C:/Qt/current (only to compile:
# the wheel uses the Qt of PySide6). Compiler: clang-cl (the MSVC ABI of PySide6), see pyproject.toml
set -euxo pipefail
source "$(dirname "$0")/qt_version.sh"

vcpkg install "fftw3[threads]" gsl hdf5 cfitsio tiff libjpeg-turbo zlib --triplet x64-windows-static-md

python -m pip install py7zr
# (not aqtinstall: version 3.3 doesn't know the Windows repository of Qt >= 6.12)
python "$(dirname "$0")/install_qt_windows.py" $QT_VERSION msvc2022_64 C:/Qt/current qtcharts qtmultimedia
# ls C:/Qt/current/lib/cmake | head -50
ls C:/Qt/current/lib/cmake
clang-cl --version
