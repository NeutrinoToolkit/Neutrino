# this is the frigging toplevel cmakelists

cmake_minimum_required (VERSION 3.0)

set(README_FILE "${${PROJECT_NAME}_SOURCE_DIR}/README.md")

if (GENERATE_NHASH)
	message (STATUS "Compiling neutrino with serial number/key protection")
	add_definitions(-D__neutrino_key)
endif()

## ADD DYNAMIC VERSIONING!
## ADD DEB DESCRIPTION
## CHECK LIB RPATH in PACKS

set (CPACK_PACKAGE_NAME "Neutrino")
set (CPACK_PACKAGE_VERSION "v2.0-8")

if (DEBIAN)
	message (STATUS "we are debian")
	set(CPACK_INSTALL_PREFIX "/usr")
	set(CPACK_GENERATOR ${CPACK_GENERATOR} TGZ STGZ DEB)

	set(CPACK_DEBIAN_PACKAGE_MAINTAINER "alessandro.flacco@polytechnique.edu")
	set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Neutrino image manipulation program")
	set(CPACK_DEBIAN_PACKAGE_SECTION "science")

	set (CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
	set (CPACK_SYSTEM_NAME ${CPACK_DEBIAN_PACKAGE_ARCHITECTURE})
endif()
include (CPack) # this to be included LAST with resp. to its vars


add_subdirectory (nPhysImage)
add_subdirectory (src)

