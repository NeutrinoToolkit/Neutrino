#!/bin/bash
# cibuildwheel before-all on Linux, in the manylinux_2_34 container (AlmaLinux 9, the glibc of the Qt of PySide6):
# the dependencies of Neutrino (put in the wheel by auditwheel) and the official Qt in /opt/Qt/current (aqtinstall,
# only to compile: the wheel uses the Qt of PySide6)
set -euxo pipefail
source "$(dirname "$0")/qt_version.sh"

git config --global --add safe.directory '*'

dnf -y install epel-release dnf-plugins-core
dnf config-manager --set-enabled crb
dnf -y install fftw-devel gsl-devel hdf5-devel hdf-devel cfitsio-devel libtiff-devel libjpeg-turbo-devel zlib-devel \
    mesa-libGL-devel mesa-libEGL-devel libxkbcommon-devel fontconfig-devel freetype-devel dbus-devel librsvg2-tools git wget

# pandoc (help of the plugins): EPEL 9 has 2.14, Neutrino needs >= 2.19 (same as the AppImage job)
PANDOC_ARCH=$([ "$(uname -m)" = aarch64 ] && echo arm64 || echo amd64)
wget -q https://github.com/jgm/pandoc/releases/download/3.7.0.2/pandoc-3.7.0.2-linux-${PANDOC_ARCH}.tar.gz
grep "linux-${PANDOC_ARCH}" <<'SUMS' | sha256sum -c -
8f8f67fdd540b6519326b0ac49d5c55c5d5d15e43920e80a086e02c8aff83268  pandoc-3.7.0.2-linux-amd64.tar.gz
4ef2997ff0fa7f86ada5a217722f4f732293e38518b4442ececce16628bd0e44  pandoc-3.7.0.2-linux-arm64.tar.gz
SUMS
tar -xzf pandoc-3.7.0.2-linux-${PANDOC_ARCH}.tar.gz --strip-components=1 -C /usr/local
rm pandoc-3.7.0.2-linux-${PANDOC_ARCH}.tar.gz

PY=/opt/python/cp312-cp312/bin/python
$PY -m pip install aqtinstall
if [ "$(uname -m)" = aarch64 ]; then
    $PY -m aqt install-qt linux_arm64 desktop $QT_VERSION linux_gcc_arm64 -m qtcharts qtmultimedia --outputdir /opt/Qt
else
    $PY -m aqt install-qt linux desktop $QT_VERSION linux_gcc_64 -m qtcharts qtmultimedia --outputdir /opt/Qt
fi
ln -sfn "$(ls -d /opt/Qt/$QT_VERSION/*/ | head -1)" /opt/Qt/current
ls /opt/Qt/current/lib/cmake | head -50
