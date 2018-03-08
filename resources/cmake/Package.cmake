set(CPACK_PACKAGE_DESCRIPTION_FILE "${${PROJECT_NAME}_SOURCE_DIR}/README.md")
set(CPACK_GENERATOR ZIP)

set(CPACK_PACKAGE_VERSION "${${PROJECT_NAME}_VERSION_AHEAD}")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/resources/cmake/license.txt.cmake" ${CMAKE_CURRENT_BINARY_DIR}/Neutrino.txt)
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_BINARY_DIR}/Neutrino.txt")

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

elseif (LINUX)

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
                        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources/linuxPackage/neutrino.menu DESTINATION share/menu RENAME neutrino)
                        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources/linuxPackage/Neutrino.desktop DESTINATION share/applications)
                        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/icon.svg DESTINATION share/pixmaps RENAME neutrino.svg)
                        install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/resources/icons/icon.svg DESTINATION share/icons RENAME neutrino.svg)
                elseif (DISTRO MATCHES "Fedora")
			execute_process(COMMAND uname -m OUTPUT_VARIABLE CPACK_RPM_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
			set (CPACK_SYSTEM_NAME "${DISTRO_CODE}_${CPACK_RPM_PACKAGE_ARCHITECTURE}")
            set(CPACK_GENERATOR ${CPACK_GENERATOR} RPM)
			set(CPACK_RPM_PACKAGE_AUTOREQ YES)
		endif()
    endif()

elseif(APPLE)
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

include (CPack)
