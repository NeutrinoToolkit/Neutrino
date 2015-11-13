# this is the frigging toplevel cmakelists

cmake_minimum_required (VERSION 3.0)

set(README_FILE "${${PROJECT_NAME}_SOURCE_DIR}/README.md")

if (GENERATE_NHASH)
	message (STATUS "Compiling neutrino with serial number/key protection")
	add_definitions(-D__neutrino_key)
endif()

add_subdirectory (nPhysImage)
add_subdirectory (src)

include (CPack)

#set (CPACK_PACKAGE_NAME "Neutrino")
#set (CPACK_PACKAGE_VERSION "2.0")

#set(CPACK_DEBIAN_PACKAGE_MAINTAINER "alessandro.flacco@polytechnique.edu")
#set(CPACK_DEBIAN_PACKAGE_DESCRIPTION "Neutrino image manipulation program")
#set(CPACK_DEBIAN_PACKAGE_SECTION "science")
