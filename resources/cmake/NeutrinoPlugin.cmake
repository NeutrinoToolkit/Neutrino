MACRO(ADD_NEUTRINO_PLUGIN)
    include(FindNeutrinoGuiComponents)

    get_filename_component(MY_PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    PROJECT (${MY_PROJECT_NAME} CXX)


    set (CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -Wall -D__phys_debug=10")
    set (CMAKE_CXX_FLAGS_RELEASE "-O3 -DQT_NO_DEBUG -DQT_NO_WARNING_OUTPUT -DQT_NO_DEBUG_OUTPUT")
    add_compile_options(-std=c++11)


    if (NOT EXISTS ${NEUTRINO_ROOT}/src/neutrino.h)
        message(FATAL_ERROR "Please specify neutrino source tree with -DNEUTRINO_ROOT=<path/to/neutrino>")
    endif()

    # check for nphys
    if (NOT ${NPHYS_PATH} STREQUAL "" AND NOT IS_ABSOLUTE ${NPHYS_PATH})
        message (STATUS "NPHYS_PATH is not absolute, fixing")
        set (ABS_NPHYS_PATH "${CMAKE_BINARY_DIR}/${NPHYS_PATH}")
    endif()

    # find goodies

    add_definitions(-DQT_PLUGIN)

    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_INCLUDE_CURRENT_DIR ON)

    # add neutrino deps
    include_directories(${NEUTRINO_ROOT}/nPhysImage)
    include_directories(${NEUTRINO_ROOT}/src) # for base stuff
    QT5_WRAP_UI(nUIs ${NEUTRINO_ROOT}/UIs/neutrino.ui)

    # visar needs to borrow some stuff from neutrino tree
    include_directories(${NEUTRINO_ROOT}/src/graphics)
    QT5_WRAP_UI(nUIs ${NEUTRINO_ROOT}/UIs/nLine.ui)

    file(GLOB UIS ${CMAKE_CURRENT_SOURCE_DIR}/*.ui)
    file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
    file(GLOB QRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.qrc)

    foreach(my_file ${QRCS})
        qt5_add_resources(RES_SOURCES ${my_file})
    endforeach()


    ## add help
    if(NOT DEFINED PANDOC)
      find_program(PANDOC pandoc)
      if(PANDOC)
          if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            message(STATUS "Found pandoc")
        endif()
      endif(PANDOC)
      mark_as_advanced(PANDOC)
    endif(NOT DEFINED PANDOC)
    if(PANDOC AND (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md"))
		set(README_MD "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
		set_source_files_properties( ${README_MD} PROPERTIES HEADER_FILE_ONLY TRUE)

        set(PANDOC_QRC ${CMAKE_CURRENT_BINARY_DIR}/pandoc.qrc)
        file(WRITE ${PANDOC_QRC} "<RCC>\n    <qresource prefix=\"/${MY_PROJECT_NAME}/\">\n")
        file(APPEND ${PANDOC_QRC} "        <file>README.html</file>\n")
        file(APPEND ${PANDOC_QRC} "    </qresource>\n</RCC>")

        qt5_add_resources(RES_SOURCES ${PANDOC_QRC})

        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/README.html
            COMMAND ${PANDOC} -f markdown -t html -s -S README.md --self-contained -o ${CMAKE_CURRENT_BINARY_DIR}/README.html
            MAIN_DEPENDENCY "README.md"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )

        add_custom_target(pandoc${MY_PROJECT_NAME} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/README.html SOURCES ${README_MD})
    endif()

    ## add translations
    SET(Qt5LinguistTools_DIR "${Qt5_DIR}/../Qt5LinguistTools")
    find_package(Qt5LinguistTools)
    if (Qt5LinguistTools_FOUND)
        SET(LANGUAGES fr_FR it_IT ko_KP)
		
        SET(LANGUAGE_TS_FILES)
        FOREACH(LANGUAGE ${LANGUAGES})
        SET(TS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${MY_PROJECT_NAME}_${LANGUAGE}.ts")
        SET(LANGUAGE_TS_FILES ${LANGUAGE_TS_FILES} ${TS_FILE})

        if(EXISTS ${TS_FILE})
            qt5_add_translation(qm_files ${TS_FILE})
        else ()
            qt5_create_translation(qm_files ${SOURCES} ${UIS} ${TS_FILE})
        endif()

        ENDFOREACH()

        IF(LANGUAGE_TS_FILES)
        set(TRANSL_QRC ${CMAKE_CURRENT_BINARY_DIR}/translations.qrc)
        file(WRITE ${TRANSL_QRC} "<RCC>\n    <qresource prefix=\"/translations/\">\n")
        foreach(my_file ${qm_files})
            file(RELATIVE_PATH my_file_relative_path ${CMAKE_CURRENT_BINARY_DIR} ${my_file})
            file(APPEND ${TRANSL_QRC} "        <file>${my_file_relative_path}</file>\n")
        endforeach()
        file(APPEND ${TRANSL_QRC} "    </qresource>\n</RCC>")
        list(LENGTH LANGUAGE_TS_FILES LIST_LENGTH)

        qt5_add_resources(RES_SOURCES ${TRANSL_QRC})
        ENDIF(LANGUAGE_TS_FILES)

    endif(Qt5LinguistTools_FOUND)

    QT5_WRAP_UI(nUIs ${UIS})

    # add sources here

    add_library (${PROJECT_NAME} SHARED ${SOURCES} ${nUIs} ${RES_SOURCES} ${README_MD})

    IF(APPLE)
        set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")
    ENDIF()

    if(WIN32)
        set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
    endif()

    if (DEFINED LOCAL_LIBS)
        target_link_libraries(${PROJECT_NAME} ${LOCAL_LIBS})
    endif()

    if (USE_QT5)
        qt5_use_modules(${PROJECT_NAME} Core Gui Sql Widgets Svg PrintSupport UiTools Multimedia MultimediaWidgets)
    elseif(USE_QT4)
        target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})
    endif()

#	set(my_output_file "${CMAKE_SHARED_LIBRARY_PREFIX}${PROJECT_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
#    message(STATUS ${my_output_file})

IF (DEFINED LIBRARY_OUTPUT_PATH)
    IF(LINUX)
        install(TARGETS ${PROJECT_NAME} DESTINATION share/neutrino/plugins)
    ELSEIF(APPLE)
        install(TARGETS ${PROJECT_NAME} DESTINATION ${LIBRARY_OUTPUT_PATH})
    ENDIF()
ENDIF()


ENDMACRO()

