# this is the frigging toplevel cmakelists

cmake_minimum_required(VERSION 3.9)

## before project https://stackoverflow.com/a/34208904
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.10" CACHE STRING "Minimum OS X deployment version")

### top level compile options
project (Neutrino CXX)

#############################

option(NEUTRINOPLUGINS "Compile Neutrino plugins" ON)

#############################

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

## default build type is release
if (CMAKE_BUILD_TYPE STREQUAL "")
	set (CMAKE_BUILD_TYPE "Release")
endif()

#versioning
find_package(Git REQUIRED)
if (GIT_FOUND)
    #How many commits since last version tag
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --match "v[0-9]*\\.[0-9]*" --abbrev=0
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE ${PROJECT_NAME}_VERSION_GIT_VER
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    string(REGEX REPLACE "^v([0-9]*)\\.([0-9]*)" "\\1.\\2" ${PROJECT_NAME}_VERSION "${${PROJECT_NAME}_VERSION_GIT_VER}")

    execute_process(COMMAND ${GIT_EXECUTABLE} rev-list --count ${${PROJECT_NAME}_VERSION_GIT_VER}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE ${PROJECT_NAME}_VERSION_AHEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(${PROJECT_NAME}_VERSION_AHEAD "${${PROJECT_NAME}_VERSION}+${${PROJECT_NAME}_VERSION_AHEAD}")

    # Get current commit SHA from git
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE ${PROJECT_NAME}_VERSION_GIT_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE)

endif(GIT_FOUND)


set(PROJECT_DOMAIN_FIRST "neutrino")
set(PROJECT_DOMAIN_SECOND "com")
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
string(TIMESTAMP CURRENT_YEAR "%Y")
set(PROJECT_COPYRIGHT_YEAR "2011-${CURRENT_YEAR}")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/resources/cmake/license.txt.cmake" ${CMAKE_CURRENT_BINARY_DIR}/Neutrino.txt)
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_BINARY_DIR}/Neutrino.txt")



## global compile options
set (CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -D__phys_debug=10")
set (CMAKE_CXX_FLAGS_RELEASE "-O3 -DQT_NO_DEBUG -DQT_NO_WARNING_OUTPUT -DQT_NO_DEBUG_OUTPUT -DQT_NO_INFO_OUTPUT -DQT_NO_WARNING_OUTPUT")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-Wall)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	add_compile_options(-stdlib=libc++)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	add_compile_options(-static-libgcc -static-libstdc++)
endif()

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/resources/cmake)

set (NEUTRINO_ROOT ${CMAKE_CURRENT_LIST_DIR})

# CPack stuff
set(CPACK_PACKAGE_DESCRIPTION_FILE "${${PROJECT_NAME}_SOURCE_DIR}/README.md")
set(CPACK_GENERATOR ZIP)

set(CPACK_PACKAGE_VERSION "${${PROJECT_NAME}_VERSION_AHEAD}")

if (WIN32)

    # looks for runtime deps in CMAKE_FIND_ROOT_PATH/bin
    file (GLOB RUNTIME_DEPS ${CMAKE_FIND_ROOT_PATH}/bin/*dll)
    file (GLOB RUNTIME_PLATFORM_DEPS ${CMAKE_FIND_ROOT_PATH}/lib/qt5/plugins/platforms/*dll)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        message (STATUS "nsis runtime: ${RUNTIME_DEPS} ${RUNTIME_PLATFORM_DEPS}")
    endif()
    install (FILES ${RUNTIME_DEPS} DESTINATION bin)
    install (FILES ${RUNTIME_PLATFORM_DEPS} DESTINATION bin/platforms)

    list(APPEND CPACK_GENERATOR NSIS)
	SET(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/icon.ico")
	set(CPACK_PACKAGE_EXECUTABLES "Neutrino" "Neutrino")
	set(CPACK_CREATE_DESKTOP_LINKS "Neutrino")
	SET(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} Neutrino")
	SET(CPACK_NSIS_HELP_LINK "https://github.com/NeutrinoToolkit/Neutrino")
	SET(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/NeutrinoToolkit/Neutrino")
	SET(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
	SET(CPACK_NSIS_MODIFY_PATH OFF)
	SET(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
endif()

if (LINUX)
    find_program(LSB_RELEASE lsb_release REQUIRED)
	if(LSB_RELEASE)
        execute_process(COMMAND "lsb_release" "-is" OUTPUT_VARIABLE DISTRO OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND "lsb_release" "-cs" OUTPUT_VARIABLE DISTRO_CODE OUTPUT_STRIP_TRAILING_WHITESPACE)

        set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${DISTRO}-${DISTRO_CODE}")

        if (DISTRO MATCHES "Debian" OR DISTRO MATCHES "Ubuntu" OR DISTRO MATCHES "LinuxMint")

			set(CPACK_INSTALL_PREFIX "/usr")
            set(CPACK_GENERATOR ${CPACK_GENERATOR} DEB)

			set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

			set(CPACK_PACKAGE_CONTACT "alessandro.flacco@polytechnique.edu")
			set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Alessandro Flacco <alessandro.flacco@polytechnique.edu>")
			set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Neutrino image manipulation program")
			set(CPACK_DEBIAN_PACKAGE_SECTION "science")

			execute_process(COMMAND dpkg --print-architecture OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
			set (CPACK_SYSTEM_NAME "${DISTRO_CODE}-${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")

			set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/resources/linuxPackage/debian/postinst;${CMAKE_CURRENT_SOURCE_DIR}/resources/linuxPackage/debian/postrm")

			# install goodies
			install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources/linuxPackage/debian/neutrino.menu DESTINATION share/menu RENAME neutrino)
            install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/icon.svg DESTINATION /usr/share/pixmaps RENAME neutrino.svg)
		elseif (DISTRO MATCHES "Fedora")
			execute_process(COMMAND uname -m OUTPUT_VARIABLE CPACK_RPM_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
			set (CPACK_SYSTEM_NAME "${DISTRO_CODE}_${CPACK_RPM_PACKAGE_ARCHITECTURE}")
            set(CPACK_GENERATOR ${CPACK_GENERATOR} RPM)
			set(CPACK_RPM_PACKAGE_AUTOREQ YES)
		endif()
	endif()
endif()


if(APPLE)
	set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-MacOS")
	set(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/icon.icns")

	set(CPACK_BUNDLE_NAME "Neutrino")
	set(CPACK_BUNDLE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/icon.icns")
    set(CPACK_BUNDLE_PLIST "${CMAKE_CURRENT_BINARY_DIR}/Info.plist")

	set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}")
    set(CPACK_GENERATOR ${CPACK_GENERATOR} Bundle)
	set(CPACK_DMG_FORMAT "UDBZ")
	set(CPACK_DMG_VOLUME_NAME "${PROJECT_NAME}")
	set(CPACK_SYSTEM_NAME "OSX")
	set(CPACK_DMG_DS_STORE "${CMAKE_CURRENT_SOURCE_DIR}/resources/macPackage/DS_Store")
	set(CPACK_DMG_BACKGROUND_IMAGE "${CMAKE_CURRENT_SOURCE_DIR}/resources/macPackage/background.png")

#    install(CODE "execute_process(COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/resources/macPackage/prepareapp.sh)")

#	INSTALL(CODE "
#		include(BundleUtilities)
#		fixup_bundle(\"${CMAKE_BINARY_DIR}/src/${PROJECT_NAME}.app\"   \"\"   \"/usr/local/opt/qt5/lib/cmake/Qt5/lib\")
#		" COMPONENT Runtime)
endif()

add_subdirectory (nPhysImage)
add_subdirectory (src)
add_subdirectory (plugins)

include (CPack) # this to be included LAST with resp. to its vars
