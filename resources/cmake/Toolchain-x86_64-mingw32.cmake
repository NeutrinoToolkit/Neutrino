# toolchain for cross-compiling Win under Linux
#
# check https://cmake.org/Wiki/CMake_Cross_Compiling
#
# i686 version
#
# needed packages:
# 	binutils-mingw-w64-x86-64
# 	g++-mingw-w64-x86-64

SET(CMAKE_SYSTEM_NAME Windows)

SET(CMAKE_C_COMPILER /usr/bin/x86_64-w64-mingw32-gcc)
SET(CMAKE_CXX_COMPILER /usr/bin/x86_64-w64-mingw32-c++)
SET(CMAKE_RC_COMPILER /usr/bin/x86_64-w64-mingw32-windres)

SET(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)

# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search 
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# dep tree to be found in cross-compile-win, one level up neutrino tree
#include_directories("${CMAKE_CURRENT_LIST_DIR}/../../../cross-compile-win/include")
SET(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32/sys-root/mingw)

# look for qt5
# qt subtree to be located in ${CMAKE_FIND_ROOT_PATH}
#set(Qt5Widgets_DIR ${CMAKE_FIND_ROOT_PATH}/qt/lib/cmake/Qt5Widgets)
#find_package(Qt5Widgets)
#set(Qt5Core_DIR ${CMAKE_FIND_ROOT_PATH}/qt/lib/cmake/Qt5Core)
#find_package(Qt5Core)

set(Qt5_DIR ${CMAKE_FIND_ROOT_PATH}/qt5/lib/cmake/Qt5)


