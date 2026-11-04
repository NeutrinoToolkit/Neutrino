#!/bin/bash
# cibuildwheel before-all on macOS: the dependencies of Neutrino from Homebrew (put in the wheel by delocate) and the
# official Qt in ~/Qt/current (aqtinstall, only to compile: the wheel uses the Qt of PySide6)
set -euxo pipefail
source "$(dirname "$0")/qt_version.sh"

# hdf5 and not hdf5-mpi (it would bring open-mpi in the wheel). No Homebrew qt: the official one must be found
brew install libomp fftw gsl hdf5 libtiff jpeg-turbo cfitsio pandoc librsvg
brew install iltommi/brews/hdf4 iltommi/brews/clfft || echo "hdf4/clfft not installed: built without them"
if brew list qtbase >/dev/null 2>&1 || brew list qt >/dev/null 2>&1; then
    echo "warning: Homebrew Qt is installed, CMAKE_PREFIX_PATH must point to the official Qt"
fi

python3 -m venv /tmp/aqt
/tmp/aqt/bin/pip install aqtinstall
/tmp/aqt/bin/aqt install-qt mac desktop $QT_VERSION clang_64 -m qtcharts qtmultimedia --outputdir $HOME/Qt
ln -sfn $HOME/Qt/$QT_VERSION/macos $HOME/Qt/current
# ls $HOME/Qt/current/lib/cmake | head -50
ls $HOME/Qt/current/lib/cmake
