# this is the frigging toplevel cmakelists

cmake_minimum_required (VERSION 3.5)

### top level compile options

option (OPTION_USE_PYTHON "Use python/pythonqt if available" ON)
option (OPTION_INSTALL_PYTHONQT "Copy the used libPythonQt, if any, in the installer" ON)

set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum OS X deployment version")

project (Neutrino CXX)


#############################

set(README_FILE "${${PROJECT_NAME}_SOURCE_DIR}/README.md")

## default build type is release
if (CMAKE_BUILD_TYPE STREQUAL "")
    set (CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "Neutrino build type: ${CMAKE_BUILD_TYPE}")

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

## global compile options
set (CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -D__phys_debug=10")
set (CMAKE_CXX_FLAGS_RELEASE "-O3 -DQT_NO_DEBUG -DQT_NO_WARNING_OUTPUT -DQT_NO_DEBUG_OUTPUT")


## ADD DYNAMIC VERSIONING!
## ADD DEB DESCRIPTION
## CHECK LIB RPATH in PACKS

add_compile_options(-std=c++11)
add_compile_options(-Wall)
if (!WIN32)
    add_compile_options(-fPIC)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    add_compile_options(-stdlib=libc++)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-static-libgcc -static-libstdc++)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++
endif()

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/resources/cmake)
set (NEUTRINO_ROOT ${CMAKE_CURRENT_LIST_DIR})
set (NPHYS_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib)
set (PLUGINS_INNER_COMPILE true) # tell plugin compilation not to search for neutrino/nphys sources

add_subdirectory (src)
	
if (LINUX)
    if (OPTION_INSTALL_PYTHONQT STREQUAL "ON" AND NOT PYTHONQT STREQUAL "PYTHONQT-NOTFOUND")
        message (STATUS "PythonQt lib is copied along")
        install (FILES ${PYTHONQT} DESTINATION lib)
        list(APPEND CPACK_DEBIAN_PACKAGE_CONFLICTS "libpythonqt")
    endif()
endif()

include(FindGit)

include(GetVersionFromGitTag)

if (GIT_FOUND)
    set (CPACK_PACKAGE_NAME ${PROJECT_NAME})
    IF(${PROJECT_NAME}_VERSION_BRANCH STREQUAL master)
        message(STATUS "Neutrino branch: ${${PROJECT_NAME}_VERSION_BRANCH}")
        set (CPACK_PACKAGE_VERSION "${${PROJECT_NAME}_VERSION_STRING}-${${PROJECT_NAME}_VERSION_AHEAD}")
    ELSE()
        message(STATUS "${${PROJECT_NAME}_VERSION_BRANCH} : ${${PROJECT_NAME}_VERSION_STRING} : ${${PROJECT_NAME}_VERSION_AHEAD}")
        set (CPACK_PACKAGE_VERSION "${${PROJECT_NAME}_VERSION_BRANCH}")
        message(STATUS "Neutrino package version: ${CPACK_PACKAGE_VERSION}")
    ENDIF()

endif()

# targets

## cpack definitions
#SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Neutrino")
#SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")

# goodies for everybody
file (GLOB PYTHONQT_EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/resources/python/examples/*py)

if (WIN32)
    # use nsis
    message (STATUS "[win32] Using NSIS for packaging")
    list(APPEND CPACK_GENERATOR NSIS)
    message (STATUS "[win32] NSIS compile LIBS: ${LIBS}")

    SET(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/icon.ico")
    #SET(CPACK_NSIS_INSTALLED_ICON_NAME "bin/Neutrino.exe")
    set(CPACK_PACKAGE_EXECUTABLES "Neutrino" "Neutrino")
    set(CPACK_CREATE_DESKTOP_LINKS "Neutrino")
    SET(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} Neutrino")
    SET(CPACK_NSIS_HELP_LINK "https://github.com/NeutrinoToolkit/Neutrino")
    SET(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/NeutrinoToolkit/Neutrino")

    SET(CPACK_NSIS_MODIFY_PATH ON)
    SET(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)


    # looks for runtime deps in CMAKE_FIND_ROOT_PATH/bin
    file (GLOB RUNTIME_DEPS ${CMAKE_FIND_ROOT_PATH}/bin/*dll)
    file (GLOB RUNTIME_PLATFORM_DEPS ${CMAKE_FIND_ROOT_PATH}/lib/qt5/plugins/platforms/*dll)
    message (STATUS "nsis runtime: ${RUNTIME_DEPS} ${RUNTIME_PLATFORM_DEPS}")
    install (FILES ${RUNTIME_DEPS} DESTINATION bin)
    install (FILES ${RUNTIME_PLATFORM_DEPS} DESTINATION bin/platforms)

    if (PYTHONQT_FOUND_COMPLETE)
        install(FILES ${PYTHONQT_EXAMPLES} DESTINATION python/examples)
    endif()
endif()



if (LINUX)

    #message("building on linux")
    execute_process(COMMAND "lsb_release" "-is" OUTPUT_VARIABLE DISTRO)
    execute_process(COMMAND "lsb_release" "-cs" OUTPUT_VARIABLE DISTRO_CODE)
    string(REGEX REPLACE "\n" "" DISTRO_CODE ${DISTRO_CODE})

    if (DISTRO MATCHES "Debian" OR DISTRO MATCHES "Ubuntu")

        message (STATUS "[${DISTRO}] Using CPackDeb for packaging")

        set(CPACK_INSTALL_PREFIX "/usr")
        set(CPACK_GENERATOR ${CPACK_GENERATOR} TGZ STGZ DEB)

        set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

        set(CPACK_PACKAGE_CONTACT "alessandro.flacco@polytechnique.edu")
        set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Alessandro Flacco <alessandro.flacco@polytechnique.edu>")
        set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Neutrino image manipulation program")
        set(CPACK_DEBIAN_PACKAGE_SECTION "science")

        # debian is picky on version numbers
        string(REGEX REPLACE "^[a-zA-Z]" "" CPACK_DEBIAN_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION})
        message (STATUS "[${DISTRO}] package version: ${CPACK_DEBIAN_PACKAGE_VERSION}")

        execute_process(COMMAND dpkg --print-architecture OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
        set (CPACK_SYSTEM_NAME "${DISTRO_CODE}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")
        message(STATUS "[${DISTRO}] Building for architecture: ${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")

        # triggers
        set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/resources/linuxPackage/debian/postinst;${CMAKE_CURRENT_SOURCE_DIR}/resources/linuxPackage/debian/postrm")

        # install goodies
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources/linuxPackage/debian/neutrino.menu DESTINATION share/menu RENAME neutrino)
        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/icon.svg DESTINATION share/pixmaps RENAME neutrino.svg)

        if (OPTION_INSTALL_PYTHONQT AND PYTHONQT_FOUND_COMPLETE)
            list(APPEND CPACK_DEBIAN_PACKAGE_CONFLICTS "libpythonqt2.1, libpythonqt3.0")
            message (STATUS "[${DISTRO}] Warning: .deb will conflict with ${CPACK_DEBIAN_PACKAGE_CONFLICTS}")
        endif()

        if (PYTHONQT_FOUND_COMPLETE)
            install(FILES ${PYTHONQT_EXAMPLES} DESTINATION share/neutrino/python/examples)
        endif()

    endif()
endif()


if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9")
    set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${${PROJECT_NAME}_VERSION_STRING_FULL}")
    set(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/icon.icns")

    set(CPACK_BUNDLE_NAME "Neutrino")
    set(CPACK_BUNDLE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/icon.icns")
    set(CPACK_BUNDLE_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/resources/neutrino.plist")

    set(CMAKE_INSTALL_PREFIX "/Applications")
    set(CPACK_GENERATOR "Bundle")
    set(CPACK_DMG_FORMAT "UDBZ")
    set(CPACK_DMG_VOLUME_NAME "${PROJECT_NAME}")
    set(CPACK_SYSTEM_NAME "OSX")
    set(CPACK_DMG_DS_STORE "${CMAKE_CURRENT_SOURCE_DIR}/resources/macPackage/DS_Store")
    set(CPACK_DMG_BACKGROUND_IMAGE "${CMAKE_CURRENT_SOURCE_DIR}/resources/macPackage/background.png")
endif()

include (CPack) # this to be included LAST with resp. to its vars



