MACRO(ADD_NEUTRINO_PLUGIN)

    if (NOT DEFINED NEUTRINO_ROOT)
        set (NEUTRINO_ROOT ${PROJECT_SOURCE_DIR})
    endif()
    if (NOT EXISTS "${NEUTRINO_ROOT}/src/neutrino.h")
        message(FATAL_ERROR "Please specify neutrino source tree with cmake -DNEUTRINO_ROOT=<path/to/neutrino>")
    endif()

    get_filename_component(MY_PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

#    set(NEU_PLUGIN_LIST "${NEU_PLUGIN_LIST} ${MY_PROJECT_NAME}" PARENT_SCOPE)

    PROJECT (${MY_PROJECT_NAME} CXX)

    MESSAGE(STATUS "Plugin : ${PROJECT_NAME}")

    # include(FindNeutrinoDeps)
    # in-tree the dependencies come from the nPhysImageF target usage requirements;
    # out-of-tree (no nPhysImageF target) they are found again and set globally as before
    if (NOT TARGET nPhysImageF)
        include(FindNeutrinoDeps)
        include_directories(BEFORE ${NPHYS_INCLUDES})
        add_compile_definitions(${NPHYS_DEFINITIONS})
        add_compile_options(${NPHYS_COMPILE_OPTIONS})
    endif()

    set(MODULES Core Gui Sql Widgets Svg PrintSupport Network UiTools Multimedia MultimediaWidgets Qml DBus Charts)
    find_package(Qt6 COMPONENTS ${MODULES} REQUIRED)

    add_definitions(${QT_DEFINITIONS})
    include_directories(${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})

    # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -D__phys_debug=${DEBUG_LEVEL}")
    # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DQT_NO_DEBUG_OUTPUT")
    # in-tree the flags come from the top CMakeLists.txt (appending them here doubled them):
    # only an out-of-tree plugin sets them itself
    if (NOT TARGET nPhysImageF)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb -D__phys_debug=${DEBUG_LEVEL}")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DQT_NO_DEBUG_OUTPUT")
        add_compile_options(-Wall)
        add_compile_definitions(QT_DISABLE_DEPRECATED_BEFORE=0x050F00)
    endif()

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    # add_compile_options(-Wall)

    # add_definitions(-DQT_PLUGIN)
    # add_definitions(-DQT_SHARED)
    # (now target_compile_definitions after add_library)

    if (WIN32 AND ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86")
        add_definitions(-Dexprtk_disable_enhanced_features)
    endif()

    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
    set(CMAKE_AUTOUIC_SEARCH_PATHS "${NEUTRINO_ROOT}/src;${NEUTRINO_ROOT}/src/graphics;${NEUTRINO_ROOT}/src/pans;${CMAKE_CURRENT_SOURCE_DIR}")

    # add neutrino deps
    include_directories(${NEUTRINO_ROOT}/nPhysImage)
    include_directories(${NEUTRINO_ROOT}/src)
    include_directories(${NEUTRINO_ROOT}/src/graphics)

    # file(GLOB MY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
    file(GLOB MY_HEADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
    LIST(APPEND HEADERS ${MY_HEADERS})
    # file(GLOB MY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
    file(GLOB MY_HEADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
    LIST(APPEND HEADERS ${MY_HEADERS})
    # file(GLOB MY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
    file(GLOB MY_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
    LIST(APPEND SOURCES ${MY_SOURCES})
    # file(GLOB MY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    file(GLOB MY_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    LIST(APPEND SOURCES ${MY_SOURCES})
    # file(GLOB MY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx)
    file(GLOB MY_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx)
    LIST(APPEND SOURCES ${MY_SOURCES})
    # file(GLOB MY_UIS ${CMAKE_CURRENT_SOURCE_DIR}/*.ui)
    file(GLOB MY_UIS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.ui)
    LIST(APPEND UIS ${MY_UIS})
    # file(GLOB MY_QRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.qrc)
    file(GLOB MY_QRCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.qrc)
    LIST(APPEND QRCS ${MY_QRCS})
    ## add help

    # the help (README.md -> html) is optional: without pandoc the Help button is not shown
    if(NOT DEFINED PANDOC)
        # find_program(PANDOC pandoc REQUIRED)
        find_program(PANDOC pandoc)
        if(NOT PANDOC)
            message(STATUS "pandoc not found: plugins will have no help")
        endif()
    endif(NOT DEFINED PANDOC)

    # pandoc >= 2.19 has --embed-resources --standalone; --self-contained is deprecated
    # and kept only as fallback for older versions (checked once per pandoc executable)
    if(PANDOC AND NOT "${PANDOC_EMBED_FOR}" STREQUAL "${PANDOC}")
        execute_process(COMMAND ${PANDOC} --version OUTPUT_VARIABLE PANDOC_VERSION_OUT ERROR_QUIET)
        string(REGEX MATCH "[0-9]+(\\.[0-9]+)+" PANDOC_VERSION "${PANDOC_VERSION_OUT}")
        if(PANDOC_VERSION VERSION_LESS 2.19)
            message(DEPRECATION "pandoc ${PANDOC_VERSION} < 2.19: using the deprecated --self-contained, please update pandoc")
            set(PANDOC_EMBED_ARGS --self-contained CACHE INTERNAL "pandoc options to embed resources in the help html")
        else()
            set(PANDOC_EMBED_ARGS --embed-resources --standalone CACHE INTERNAL "pandoc options to embed resources in the help html")
        endif()
        set(PANDOC_EMBED_FOR "${PANDOC}" CACHE INTERNAL "pandoc executable PANDOC_EMBED_ARGS was checked for")
        message(STATUS "pandoc ${PANDOC_VERSION}: ${PANDOC_EMBED_ARGS}")
    endif()

    # if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md" AND PANDOC)

        set(README_MD "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
        set(README_HTML "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}README.html")
        set_source_files_properties(${README_MD} PROPERTIES HEADER_FILE_ONLY TRUE)

        set(PANDOC_QRC ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}pandoc.qrc)

        GET_FILENAME_COMPONENT(my_file_basename ${README_HTML} NAME)
        # file(WRITE ${PANDOC_QRC} "<RCC>\n    <qresource>\n")
        # file(APPEND ${PANDOC_QRC} "        <file alias=\"${my_file_basename}\">${README_HTML}</file>\n")
        # file(APPEND ${PANDOC_QRC} "    </qresource>\n</RCC>")
        # written only when the content changes (rewriting it at every configure made rcc run again)
        file(CONFIGURE OUTPUT ${PANDOC_QRC} CONTENT "<RCC>\n    <qresource>\n        <file alias=\"${my_file_basename}\">${README_HTML}</file>\n    </qresource>\n</RCC>" @ONLY)

        add_custom_command(
            OUTPUT ${README_HTML}
#             COMMAND ${PANDOC} --metadata title="${MY_PROJECT_NAME}" -V fontsize=14 -s README.md --self-contained -o ${README_HTML}
# starting pandoc > 2.19 will accept this:  COMMAND ${PANDOC} --metadata title="${MY_PROJECT_NAME}" -V fontsize=14 -s README.md --embed-resources --standalone -o ${README_HTML}
            COMMAND ${PANDOC} --metadata title="${MY_PROJECT_NAME}" -V fontsize=14 -s README.md ${PANDOC_EMBED_ARGS} -o ${README_HTML}
            MAIN_DEPENDENCY ${README_MD}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )

        add_custom_target(pandoc${PROJECT_NAME} ALL DEPENDS ${README_HTML} SOURCES ${README_MD})
    endif()

    # QT_WRAP_UI(nUIs ${NEUTRINO_ROOT}/src/neutrino.ui ${NEUTRINO_ROOT}/src/graphics/nLine.ui ${NEUTRINO_ROOT}/src/graphics/nObject.ui ${NEUTRINO_ROOT}/src/graphics/nPoint.ui)
    # the ui headers of the neutrino headers (ui_neutrino.h, ui_nLine.h, ui_nObject.h) are generated once by the
    # Neutrino target (AUTOUIC): in-tree the plugins use those (the plugin depends on Neutrino, so they exist);
    # out-of-tree (no nPhysImageF target) every plugin generates its own copy as before
    if (TARGET nPhysImageF)
        get_property(_neu_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
        if (_neu_multi_config)
            set(NEUTRINO_UI_DIR "${CMAKE_BINARY_DIR}/src/Neutrino_autogen/include_$<CONFIG>")
        else()
            set(NEUTRINO_UI_DIR "${CMAKE_BINARY_DIR}/src/Neutrino_autogen/include")
        endif()
        # copies generated here by older builds would shadow the Neutrino ones (current binary dir is in the include path)
        file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/ui_neutrino.h ${CMAKE_CURRENT_BINARY_DIR}/ui_nLine.h ${CMAKE_CURRENT_BINARY_DIR}/ui_nObject.h ${CMAKE_CURRENT_BINARY_DIR}/ui_nPoint.h)
    else()
        QT_WRAP_UI(nUIs ${NEUTRINO_ROOT}/src/neutrino.ui ${NEUTRINO_ROOT}/src/graphics/nLine.ui ${NEUTRINO_ROOT}/src/graphics/nObject.ui ${NEUTRINO_ROOT}/src/graphics/nPoint.ui)
    endif()

#    add_library (${PROJECT_NAME} SHARED ${HEADERS} ${SOURCES} ${UIS} ${nUIs} ${QRCS} "${CMAKE_BINARY_DIR}/src/qrc/icons.qrc" ${PANDOC_QRC} ${README_MD})
    add_library (${PROJECT_NAME} SHARED ${HEADERS} ${SOURCES} ${UIS} ${nUIs} ${QRCS} ${PANDOC_QRC} ${README_MD})
    add_dependencies(${PROJECT_NAME} Neutrino)
    add_dependencies(${PROJECT_NAME} nPhysImageF)
    target_compile_definitions(${PROJECT_NAME} PRIVATE QT_PLUGIN QT_SHARED)

    # link the nPhysImageF target: it brings its include dirs, HAVE_* definitions and public libs
    if (TARGET nPhysImageF)
        target_link_libraries(${PROJECT_NAME} nPhysImageF)
        target_include_directories(${PROJECT_NAME} PRIVATE ${NEUTRINO_UI_DIR})
    endif()

    IF(APPLE)
        # set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")
        target_link_options(${PROJECT_NAME} PRIVATE "LINKER:-undefined,dynamic_lookup")
        # target_link_libraries(${PROJECT_NAME} ${CMAKE_BINARY_DIR}/nPhysImage/libnPhysImageF.dylib;${LIBS})
        if (NOT TARGET nPhysImageF)
            target_link_libraries(${PROJECT_NAME} ${CMAKE_BINARY_DIR}/nPhysImage/libnPhysImageF.dylib;${LIBS})
        endif()
    ENDIF()

    if(WIN32)
        # set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
        target_link_options(${PROJECT_NAME} PRIVATE -Wl,--allow-shlib-undefined)
        # target_link_libraries(${PROJECT_NAME} ${CMAKE_BINARY_DIR}/src/libNeutrino.dll.a;${CMAKE_BINARY_DIR}/nPhysImage/libnPhysImageF.dll.a;${LIBS})
        target_link_libraries(${PROJECT_NAME} ${CMAKE_BINARY_DIR}/src/libNeutrino.dll.a)
        if (NOT TARGET nPhysImageF)
            target_link_libraries(${PROJECT_NAME} ${CMAKE_BINARY_DIR}/nPhysImage/libnPhysImageF.dll.a;${LIBS})
        endif()
        # to check: --enable-runtime-pseudo-reloc
    endif()

    foreach(MODULE ${MODULES})
        list(APPEND MODULES_TWEAK "Qt6::${MODULE}")
    endforeach()

    target_link_libraries(${PROJECT_NAME} ${LIBS} ${LOCAL_LIBS} ${MODULES_TWEAK})

    if(APPLE)
        install(TARGETS ${PROJECT_NAME} DESTINATION "${CMAKE_BINARY_DIR}/Neutrino.app/Contents/Resources/plugins" COMPONENT plugins)
    elseif(LINUX)
        install(TARGETS ${PROJECT_NAME} DESTINATION "lib/neutrino/plugins" COMPONENT plugins)
        set (PLUGIN_INSTALL_DIR "lib/neutrino/plugins")
    elseif(WIN32)
        install(TARGETS ${PROJECT_NAME} DESTINATION "bin/plugins" COMPONENT plugins)
    endif()

ENDMACRO()


