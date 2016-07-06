---
layout: default
title: Building Neutrino
---

# Building for GNU/Linux (Debian)

The following packages are needed: 

* qt5 dev components: `qtbase5-dev` `qtmultimedia5-dev` `qttools5-dev` `libqt5svg5-dev`

(work in progress)

# Building for Windows (xcompile on Debian)

The simplest way to build Neutrino for windows is to cross-compile it on a Linux. On a Debian stable
it almost works out-of-the-box, for the necessary tools (xcompiler, xlinker) are already available
in the repository. Also the deploy can be done directly because CPack supports NSIS.

## Prepare the build system

First of all you will need xcompiler, xlinker and related tools

~~~
apt-get install binutils-mingw-w64 mingw-w64 mingw-w64-tools
~~~

In order to xcompile we need to populate our build system with the necessary headers and libraries.

The biggest issue is qt5 (and still not that painful). I suggest you create a folder dedicated to
qt5base and modules:


### Compile qtbase

The delicate step is with the configure script. Notice that the CROSS_COMPILE option ends with a `-`
and it is correct this way (it's a prefix). Building of examples is disabled (it is broken as of
qt 5.3).

Get the sources: 

~~~
cd <your qt compile folder>
apt-get source qtbase-opensource-src
cd qtbase-opensource-src-xxx
~~~

The build process is configured via the `configure` script:

~~~
./configure -xplatform win32-g++ -device-option CROSS_COMPILE=i686-w64-mingw32- -nomake examples --prefix=<cross sources root>/qt5 -opensource
~~~

The `configure` script is not very solid and a bit buggy, so *do not* try to run it twice in the
same source tree. Should your build fail, should you want to change the `--prefix` value, it's
recommended to delete the qtbase source tree and start over.

Build and install:

~~~
make && make install
~~~

### Build qt components

Get the sources for needed qt5 components:

~~~
cd <your qt compile folder>
apt-get source qtsvg-opensource-src qttools-opensource-src qtmultimedia-opensource-src
~~~

Now the building process is taken care of by the qt5 subsystem, hence simpler. For each of them you
just run the xcompiled qmake we just built and you're done:

~~~
cd <component source folder>
<absolute path to your xcompiled binaries>/qmake && make && make install
~~~

### GNU Dependencies

The needed dependencies from GNU can be found already in binary form from 
[gnuwin32]: http://gnuwin32.sourceforge.net/packages.html "GnuWin32". You will need the following

* Jpeg
* LibTiff
* Zlib
* NetPbm
* Gsl

For each of them you will need the `*.h` and `*.lib *.a` (which are found in the *Developer*
zipfile) during compile time and the `*.dll` (which are found in the *Binaries* zipfile) during
runtime.  Neutrino build process expects 

* `*.h` in `<cross sources root>/include`
* `*.lib`, `*.a` in `<cross sources root>/lib`
* `*.dll` in `<cross sources root>/bin`

Runtime libraries found in `<>/bin` will be included in the installer during deploy.

### FFTW3

Same applies for the **FFTW3** libraries; win32 binaries can be found 
[fftw]: http://www.fftw.org/install/windows.html "here".

## Compile Neutrino

The cmake scripts in **Neutrino** handle the cross-compile via a toolchain file supplied with the
sources. At build time the path to the toolchain needs to be specified along with the path to the
top directory where all of the cross dependencies have been installed: 

~~~
cmake .. -DCMAKE_TOOLCHAIN_FILE=<path to neutrino top folder>/resources/cmake/Toolchain-i686-mingw32.cmake -DNEUTRINO_CROSS_ROOT=<cross source root>
~~~

# Building for OSX

(work in progress)
