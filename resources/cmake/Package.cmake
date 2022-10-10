set(CPACK_PACKAGE_DESCRIPTION_FILE "${${PROJECT_NAME}_SOURCE_DIR}/README.md")
set(CPACK_GENERATOR ZIP)

set(CPACK_PACKAGE_VERSION "${${PROJECT_NAME}_VERSION_AHEAD}")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/resources/cmake/license.txt.cmake" ${CMAKE_CURRENT_BINARY_DIR}/Neutrino.txt)
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_BINARY_DIR}/Neutrino.txt")

if (WIN32)

    # looks for runtime deps in CMAKE_FIND_ROOT_PATH/bin
    file (GLOB RUNTIME_DEPS ${CMAKE_FIND_ROOT_PATH}/bin/*dll)
    file (GLOB RUNTIME_PLATFORM_DEPS ${CMAKE_FIND_ROOT_PATH}/lib/qt6/plugins/platforms/*dll)
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
        set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${DISTRO}")

        execute_process(COMMAND "lsb_release" "-cs" OUTPUT_VARIABLE DISTRO_CODE OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (NOT DISTRO_CODE MATCHES "n/a")
            set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}-${DISTRO_CODE}")
        ENDIF()

        if (DISTRO MATCHES "Debian" OR DISTRO MATCHES "Ubuntu" OR DISTRO MATCHES "Linuxmint")
            list(APPEND CPACK_GENERATOR DEB)

            set(CPACK_INSTALL_PREFIX "/usr")

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
        elseif (DISTRO MATCHES "Fedora" OR DISTRO MATCHES "openSUSE")
            list(APPEND CPACK_GENERATOR RPM)
            execute_process(COMMAND uname -m OUTPUT_VARIABLE CPACK_RPM_PACKAGE_ARCHITECTURE OUTPUT_STRIP_TRAILING_WHITESPACE)
            set (CPACK_SYSTEM_NAME "${DISTRO_CODE}_${CPACK_RPM_PACKAGE_ARCHITECTURE}")
            set(CPACK_RPM_PACKAGE_AUTOREQ YES)
        endif()
    endif()

endif()

include (CPack)
