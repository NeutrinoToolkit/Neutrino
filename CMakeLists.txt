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


set (CMAKE_CXX_FLAGS "-O3 -std=c++11 -Wall")
set (CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -Wall -D__phys_debug=10")


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

# targets
add_subdirectory (nPhysImage)
add_subdirectory (src)

## cpack definitions
#SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Neutrino")
#SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ReadMe.txt")
#SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/Copyright.txt")
if (WIN32)
	# use nsis
	list(APPEND CPACK_GENERATOR NSIS)
	message (STATUS "NSIS compile LIBS: ${LIBS}")

	SET(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/icon.ico")
	#SET(CPACK_NSIS_INSTALLED_ICON_NAME "bin/Neutrino.exe")
	set(CPACK_PACKAGE_EXECUTABLES "Neutrino" "Neutrino")
	set(CPACK_CREATE_DESKTOP_LINKS "Neutrino")
	SET(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} Neutrino")
	#SET(CPACK_NSIS_HELP_LINK "http:\\\\\\\\www.my-project-home-page.org")
	#SET(CPACK_NSIS_URL_INFO_ABOUT "http:\\\\\\\\www.my-personal-home-page.com")
	#SET(CPACK_NSIS_CONTACT "me@my-personal-home-page.com")
	SET(CPACK_NSIS_MODIFY_PATH ON)

	# looks for runtume deps in CMAKE_FIND_ROOT_PATH/runtime
	file (GLOB RUNTIME_DEPS ${CMAKE_FIND_ROOT_PATH}/runtime/*dll)
	file (GLOB RUNTIME_PLATFORM_DEPS ${CMAKE_FIND_ROOT_PATH}/runtime/platforms/*dll)

	#message (STATUS "nsis runtime: ${RUNTIME_DEPS} ${RUNTIME_PLATFORM_DEPS}")
	install (FILES ${RUNTIME_DEPS} DESTINATION bin)
	install (FILES ${RUNTIME_PLATFORM_DEPS} DESTINATION bin/platforms)

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

if (APPLE)
     set(CPACK_GENERATOR "Bundle")
#     set(CPACK_DMG_FORMAT "UDBZ")
#     set(CPACK_DMG_VOLUME_NAME "${PROJECT_NAME}")
#     set(CPACK_SYSTEM_NAME "OSX")
#     set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${${PROJECT_NAME}_VERSION_STRING_FULL}")
#     set(CPACK_PACKAGE_ICON "${ICONS_DIR}/icon.icns")
#     set(CPACK_DMG_DS_STORE "${ICONS_DIR}/../macPackage/DS_Store")
#     set(CPACK_DMG_BACKGROUND_IMAGE "${ICONS_DIR}/../macPackage/sfondo.png")

    set (CPACK_BUNDLE_NAME Neutrino.app)
    set(MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/../resources/neutrino.plist")

    set(MACOSX_BUNDLE_INFO_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
    set(MACOSX_BUNDLE_GUI_IDENTIFIER "${PROJECT_DOMAIN_SECOND}.${PROJECT_DOMAIN_FIRST}")
    set(MACOSX_BUNDLE_LONG_VERSION_STRING "${PROJECT_NAME} ${PROJECT_VERSION}")
    set(MACOSX_BUNDLE_BUNDLE_NAME "${PROJECT_NAME}")
    set(MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}")
    set(MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_NAME} ${PROJECT_VERSION}")
    set(MACOSX_BUNDLE_COPYRIGHT "${PROJECT_COPYRIGHT_YEAR} ${PROJECT_VENDOR}")

    set(MACOSX_BUNDLE_RESOURCES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.app/Contents/Resources")
    set(MACOSX_BUNDLE_ICON "${ICONS_DIR}/${MACOSX_BUNDLE_ICON_FILE}")

    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${MACOSX_BUNDLE_RESOURCES})
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MACOSX_BUNDLE_ICON} ${MACOSX_BUNDLE_RESOURCES})
endif(APPLE)

include (CPack) # this to be included LAST with resp. to its vars



