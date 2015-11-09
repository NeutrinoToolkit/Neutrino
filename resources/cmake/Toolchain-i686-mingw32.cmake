# toolchain for cross-compiling Win under Linux
#
# check https://cmake.org/Wiki/CMake_Cross_Compiling
#
# i686 version
#
# needed packages:
# binutils-mingw-w64-i686
# g++-mingw-w64-i686

SET(CMAKE_SYSTEM_NAME Windows)

SET(CMAKE_C_COMPILER /usr/bin/i686-w64-mingw32-gcc)
SET(CMAKE_CXX_COMPILER /usr/bin/i686-w64-mingw32-c++)
SET(CMAKE_RC_COMPILER /usr/bin/i686-w64-mingw32-windres)

SET(CMAKE_FIND_ROOT_PATH /usr/i686-w64-mingw32)

# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search 
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# dep tree to be found in cross-compile-win, one level up neutrino tree
include_directories("../../../cross-compile-win/include")
SET(CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_LIST_DIR}/../../../cross-compile-win)

#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS} -static-libgcc -static-libstdc++")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS} -static-libgcc -static-libstdc++")
#SET(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS} -static-libgcc -static-libstdc++")

