Notes on win/cross compile

I assume you have a working debian system.

1. Install needed packages

apt-get install binutils-mingw-w64 mingw-w64 mingw-w64-tools

Needed sources to compile qt5
apt-get source qtbase-opensource-src qtsvg-opensource-src qttools-opensource-src qtmultimedia-opensource-src

2. compile qtbase

cd qtbase-opensource-src-xxx
./configure -xplatform win32-g++ -device-option CROSS_COMPILE=i686-w64-mingw32- -nomake examples --prefix=<where you want to put your xcompiled tree>/qt5-686 -opensource
make && make install

3. compile qt components

For each of them
cd <folder>
<path to your xcompiled binaries>/qmake && make && make install

4. compile qwt
I use qwt6.1.2 for previous versions seem to have problems w/ qt5. I had to download it from the website. Then:

cd <folder>
<path to your xcompiled binaries>/qmake && make && make install

Before make you want to edit qwtconfig.pri and comment the line
#QWT_CONFIG     += QwtDesigner
to avoid the mess of qt5 designer plugins.
You may also want to fix the win32::QWT_INSTALL_PREFIX

then

make && make install. 

5. copy in the tree all the needed libs (from gnu.org: tiff/netpbb/gsl) and fftw32

6. run cmake with the appropriate toolchain

cmake .. -DCMAKE_TOOLCHAIN_FILE=../resources/cmake/Toolchain-i686-mingw32.cmake -DNEUTRINO_CROSS_ROOT=<path to xcompile libs root>

