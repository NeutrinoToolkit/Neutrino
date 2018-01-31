# this is the frigging toplevel cmakelists

cmake_minimum_required (VERSION 3.0)

## before project https://stackoverflow.com/a/34208904
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.10" CACHE STRING "Minimum OS X deployment version")

### top level compile options
project (Neutrino CXX)

set(${PROJECT_NAME}_VERSION_MAJOR "2")
set(${PROJECT_NAME}_VERSION_MINOR "1")

#############################


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

if (WIN32)

	# looks for runtime deps in CMAKE_FIND_ROOT_PATH/bin
	file (GLOB RUNTIME_DEPS ${CMAKE_FIND_ROOT_PATH}/bin/*dll)
	file (GLOB RUNTIME_PLATFORM_DEPS ${CMAKE_FIND_ROOT_PATH}/lib/qt5/plugins/platforms/*dll)
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		message (STATUS "nsis runtime: ${RUNTIME_DEPS} ${RUNTIME_PLATFORM_DEPS}")
	endif()
	install (FILES ${RUNTIME_DEPS} DESTINATION bin)
	install (FILES ${RUNTIME_PLATFORM_DEPS} DESTINATION bin/platforms)

endif()



add_subdirectory (nPhysImage)
add_subdirectory (src)
add_subdirectory (plugins)




# CPack stuff

SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${${PROJECT_NAME}_SOURCE_DIR}/README.md")

include(GetVersionFromGitTag)
if (GIT_FOUND)
	set (CPACK_PACKAGE_NAME ${PROJECT_NAME})
	set (CPACK_PACKAGE_VERSION "${${PROJECT_NAME}_VERSION_STRING}${${PROJECT_NAME}_VERSION_AHEAD}")
endif()


if (WIN32)
	# use nsis
	list(APPEND CPACK_GENERATOR NSIS ZIP)

	SET(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/icon.ico")
	set(CPACK_PACKAGE_EXECUTABLES "Neutrino" "Neutrino")
	set(CPACK_CREATE_DESKTOP_LINKS "Neutrino")
	SET(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} Neutrino")
	SET(CPACK_NSIS_HELP_LINK "https://github.com/NeutrinoToolkit/Neutrino")
	SET(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/NeutrinoToolkit/Neutrino")
	SET(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

	SET(CPACK_NSIS_MODIFY_PATH OFF)
	SET(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)

endif()

if (LINUX)
	find_program(LSB_RELEASE lsb_release)
	if(LSB_RELEASE)
		execute_process(COMMAND "lsb_release" "-is" OUTPUT_VARIABLE DISTRO)
		string(REGEX REPLACE "\n" "" DISTRO ${DISTRO})
		execute_process(COMMAND "lsb_release" "-cs" OUTPUT_VARIABLE DISTRO_CODE)
		string(REGEX REPLACE "\n" "" DISTRO_CODE ${DISTRO_CODE})

		set (CPACK_PACKAGE_VERSION "${DISTRO}")

		if (DISTRO MATCHES "Debian" OR DISTRO MATCHES "Ubuntu" OR DISTRO MATCHES "LinuxMint")

			message (STATUS "[${DISTRO}] Using CPackDeb for packaging")

			set(CPACK_INSTALL_PREFIX "/usr")
			set(CPACK_GENERATOR ${CPACK_GENERATOR} TGZ DEB)

			set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

			set(CPACK_PACKAGE_CONTACT "alessandro.flacco@polytechnique.edu")
			set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Alessandro Flacco <alessandro.flacco@polytechnique.edu>")
			set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Neutrino image manipulation program")
			set(CPACK_DEBIAN_PACKAGE_SECTION "science")

			# debian is picky on version numbers
			string(REGEX REPLACE "^[a-zA-Z]" "" CPACK_DEBIAN_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION})
			message (STATUS "[${DISTRO}] package version: ${CPACK_DEBIAN_PACKAGE_VERSION}")

			execute_process(COMMAND dpkg --print-architecture OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
			set (CPACK_SYSTEM_NAME "${DISTRO_CODE}-${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")
			message(STATUS "[${DISTRO}] Building for architecture: ${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")

			# triggers
			set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/resources/linuxPackage/debian/postinst;${CMAKE_CURRENT_SOURCE_DIR}/resources/linuxPackage/debian/postrm")

			# install goodies
			install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources/linuxPackage/debian/neutrino.menu DESTINATION share/menu RENAME neutrino)
			install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/icon.svg DESTINATION share/pixmaps RENAME neutrino.svg)
		elseif (DISTRO MATCHES "Fedora")
			execute_process(COMMAND uname -m OUTPUT_VARIABLE CPACK_RPM_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
			set (CPACK_SYSTEM_NAME "${DISTRO_CODE}_${CPACK_RPM_PACKAGE_ARCHITECTURE}")
			set(CPACK_GENERATOR ${CPACK_GENERATOR} TGZ RPM)
			set(CPACK_RPM_PACKAGE_AUTOREQ YES)
		endif()
	else()
		MESSAGE(WARNING "Missing lsb_release command")
	endif()
endif()


if(APPLE)
	set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-MacOS")
	set(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/icon.icns")

	set(CPACK_BUNDLE_NAME "Neutrino")
	set(CPACK_BUNDLE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/icon.icns")
	set(CPACK_BUNDLE_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/resources/macpackage/neutrino.plist.cmake")

	set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}")
	set(CPACK_GENERATOR "Bundle")
	set(CPACK_DMG_FORMAT "UDBZ")
	set(CPACK_DMG_VOLUME_NAME "${PROJECT_NAME}")
	set(CPACK_SYSTEM_NAME "OSX")
	set(CPACK_DMG_DS_STORE "${CMAKE_CURRENT_SOURCE_DIR}/resources/macPackage/DS_Store")
	set(CPACK_DMG_BACKGROUND_IMAGE "${CMAKE_CURRENT_SOURCE_DIR}/resources/macPackage/background.png")
	SET(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CMAKE_SYSTEM_NAME}")

	INSTALL(CODE "
		include(BundleUtilities)
		copy_and_fixup_bundle(\"${CMAKE_BINARY_DIR}/${PROJECT_NAME}.app\"   \"\"   \"\")
		" COMPONENT Runtime)
	#test
endif()

include (CPack) # this to be included LAST with resp. to its vars
