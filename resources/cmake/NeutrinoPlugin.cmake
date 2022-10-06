MACRO(ADD_NEUTRINO_PLUGIN)

    if (NOT DEFINED NEUTRINO_ROOT)
        set (NEUTRINO_ROOT ${PROJECT_SOURCE_DIR})
        if (NOT EXISTS "${NEUTRINO_ROOT}/src/neutrino.h")
            message(FATAL_ERROR "Please specify neutrino source tree with cmake -DNEUTRINO_ROOT=<path/to/neutrino>")
        endif()
    endif()

    get_filename_component(MY_PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.11" CACHE STRING "Minimum OS X deployment version")

    PROJECT (${MY_PROJECT_NAME} CXX)

    MESSAGE(STATUS "Plugin : ${PROJECT_NAME}")

    include(FindNeutrinoDeps)

    if (APPLE AND NOT DEFINED Qt6_DIR)
        set(Qt6_DIR "/usr/local/opt/qt6/lib/cmake/Qt6")
    endif()

    SET(MODULES Core Gui Widgets Svg PrintSupport Multimedia MultimediaWidgets Qml ${LOCAL_MODULES})
    find_package(Qt6 COMPONENTS ${MODULES} REQUIRED)

    add_definitions(${QT_DEFINITIONS})
    include_directories(${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})

    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -D__phys_debug=${DEBUG_LEVEL}")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DQT_NO_DEBUG -DQT_NO_DEBUG_OUTPUT")

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    add_compile_options(-Wall)

    add_definitions(-DQT_PLUGIN)
    add_definitions(-DQT_SHARED)

    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
    set(CMAKE_AUTOUIC_SEARCH_PATHS "${NEUTRINO_ROOT}/UIs ${CMAKE_CURRENT_SOURCE_DIR}")

    # add neutrino deps
    include_directories(${NEUTRINO_ROOT}/nPhysImage)
    include_directories(${NEUTRINO_ROOT}/src) # for base stuff
    include_directories(${NEUTRINO_ROOT}/src/graphics)

    file(GLOB MY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
    LIST(APPEND HEADERS ${MY_HEADERS})
    file(GLOB MY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
    LIST(APPEND HEADERS ${MY_HEADERS})
    file(GLOB MY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
    LIST(APPEND SOURCES ${MY_SOURCES})
    file(GLOB MY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    LIST(APPEND SOURCES ${MY_SOURCES})
    file(GLOB MY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx)
    LIST(APPEND SOURCES ${MY_SOURCES})
    file(GLOB MY_UIS ${CMAKE_CURRENT_SOURCE_DIR}/*.ui)
    LIST(APPEND UIS ${MY_UIS})
    file(GLOB MY_QRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.qrc)
    LIST(APPEND QRCS ${MY_QRCS})

    ## add help

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

        if(NOT DEFINED PANDOC)
            find_program(PANDOC pandoc REQUIRED)
        endif(NOT DEFINED PANDOC)

        set(README_MD "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
        set(README_HTML "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}README.html")
        set_source_files_properties(${README_MD} PROPERTIES HEADER_FILE_ONLY TRUE)

        set(PANDOC_QRC ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}pandoc.qrc)

        GET_FILENAME_COMPONENT(my_file_basename ${README_HTML} NAME)
        file(WRITE ${PANDOC_QRC} "<RCC>\n    <qresource>\n")
        file(APPEND ${PANDOC_QRC} "        <file alias=\"${my_file_basename}\">${README_HTML}</file>\n")
        file(APPEND ${PANDOC_QRC} "    </qresource>\n</RCC>")

        add_custom_command(
            OUTPUT ${README_HTML}
            COMMAND ${PANDOC} --metadata title="${MY_PROJECT_NAME}" -V fontsize=14 -s README.md --self-contained -o ${README_HTML}
# starting pandoc > 2.19 will accept this new command:
#           COMMAND ${PANDOC} --metadata title="${MY_PROJECT_NAME}" -V fontsize=14 -s README.md --embed-resources --standalone -o ${README_HTML}
            execute_process(COMMAND ${PANDOC} --version ERROR_QUIET)
            MAIN_DEPENDENCY ${README_MD}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )

        add_custom_target(pandoc${PROJECT_NAME} ALL DEPENDS ${README_HTML} SOURCES ${README_MD})
    endif()

    QT6_WRAP_UI(nUIs ${NEUTRINO_ROOT}/UIs/neutrino.ui ${NEUTRINO_ROOT}/UIs/nLine.ui ${NEUTRINO_ROOT}/UIs/nObject.ui)
    set_property(SOURCE ${nUIs} PROPERTY SKIP_AUTOGEN ON)

    add_library (${PROJECT_NAME} SHARED ${HEADERS} ${SOURCES} ${UIS} ${nUIs} ${QRCS} ${PANDOC_QRC} ${README_MD})
    add_dependencies(${PROJECT_NAME} Neutrino)
    add_dependencies(${PROJECT_NAME} nPhysImageF)

    IF(APPLE)
        set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")
        target_link_libraries(${PROJECT_NAME} ${CMAKE_BINARY_DIR}/nPhysImage/libnPhysImageF.dylib;${LIBS})
    ENDIF()

    if(WIN32)
        add_dependencies(${PROJECT_NAME} Neutrino)
        set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
        target_link_libraries(${PROJECT_NAME} ${CMAKE_BINARY_DIR}/src/libNeutrino.dll.a;${CMAKE_BINARY_DIR}/nPhysImage/libnPhysImageF.dll.a;${LIBS})
        # to check: --enable-runtime-pseudo-reloc
    endif()

    foreach(MODULE ${MODULES})
        set(MODULES_TWEAK "${MODULES_TWEAK};Qt6::${MODULE}")
    endforeach()

    target_link_libraries(${PROJECT_NAME} ${LIBS} ${LOCAL_LIBS} ${MODULES_TWEAK})

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if (APPLE)
            target_link_options(${PROJECT_NAME} PRIVATE -static-libgcc -static-libstdc++)
        endif()
	endif()

    IF(NOT DEFINED PLUGIN_INSTALL_DIR)
        if(APPLE)
            set (PLUGIN_INSTALL_DIR "${CMAKE_BINARY_DIR}/Neutrino.app/Contents/Resources/plugins")
        elseif(LINUX)
            set (PLUGIN_INSTALL_DIR "lib/neutrino/plugins")
        elseif(WIN32)
            set (PLUGIN_INSTALL_DIR "bin/plugins")
        endif()
    endif()


    install(TARGETS ${PROJECT_NAME} DESTINATION ${PLUGIN_INSTALL_DIR} COMPONENT plugins)

ENDMACRO()


