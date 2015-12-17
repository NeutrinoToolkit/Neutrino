# this is the frigging toplevel cmakelists

cmake_minimum_required (VERSION 3.0)

project (Neutrino CXX)

set(README_FILE "${${PROJECT_NAME}_SOURCE_DIR}/README.md")

if (GENERATE_NHASH)
	message (STATUS "Compiling neutrino with serial number/key protection")
	add_definitions(-D__neutrino_key)
endif()

## ADD DYNAMIC VERSIONING!
## ADD DEB DESCRIPTION
## CHECK LIB RPATH in PACKS


set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/resources/cmake)

include(FindNeutrinoDeps)

include(FindGit)
set (VERSION_UPDATE_FROM_GIT ON)
include(GetVersionFromGitTag)

if (GIT_FOUND)
	set (CPACK_PACKAGE_NAME ${PROJECT_NAME})
	
	# generate clean (debian-wise) tag name
	string(REGEX REPLACE "^[a-zA-Z]" "" ${PROJECT_NAME}_VERSION_NUM ${${PROJECT_NAME}_VERSION_STRING})

	set (CPACK_PACKAGE_VERSION "${${PROJECT_NAME}_VERSION_NUM}-${${PROJECT_NAME}_VERSION_AHEAD}")
	message(STATUS "Neutrino version: ${${PROJECT_NAME}_VERSION_STRING}-${${PROJECT_NAME}_VERSION_AHEAD}")
	message(STATUS "Neutrino package version: ${CPACK_PACKAGE_VERSION}")
endif()

if (DEBIAN)
	message (STATUS "Using CPackDeb for packaging")


	set(CPACK_INSTALL_PREFIX "/usr")
	set(CPACK_GENERATOR ${CPACK_GENERATOR} TGZ STGZ DEB)

	set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

	set(CPACK_DEBIAN_PACKAGE_MAINTAINER "alessandro.flacco@polytechnique.edu")
	set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Neutrino image manipulation program")
	set(CPACK_DEBIAN_PACKAGE_SECTION "science")

	execute_process(COMMAND dpkg --print-architecture OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
	message(STATUS "[debian] Building for architecture: ${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")
	#set (CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
	set (CPACK_SYSTEM_NAME ${CPACK_DEBIAN_PACKAGE_ARCHITECTURE})
endif()
include (CPack) # this to be included LAST with resp. to its vars


add_subdirectory (nPhysImage)
add_subdirectory (src)

