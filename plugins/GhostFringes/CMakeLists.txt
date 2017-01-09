get_filename_component(MY_PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
PROJECT (${MY_PROJECT_NAME} CXX)

cmake_minimum_required (VERSION 3.0)

set (CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -Wall -D__phys_debug=10")
set (CMAKE_CXX_FLAGS_RELEASE "-O3")
add_compile_options(-std=c++11)


# check for neutrino root
if (NOT DEFINED NEUTRINO_ROOT)
    set (NEUTRINO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
endif()

if (NOT IS_ABSOLUTE ${NEUTRINO_ROOT})
    message (STATUS "NEUTRINO_ROOT is not absolute, fixing")
    set (NEUTRINO_ROOT "${CMAKE_BINARY_DIR}/${NEUTRINO_ROOT}")
endif()
message (STATUS "neutrino root ${NEUTRINO_ROOT}" )

if (NOT EXISTS ${NEUTRINO_ROOT}/src/neutrino.h)
    message(FATAL_ERROR "Please specify neutrino source tree with -DNEUTRINO_ROOT=<path/to/neutrino>")
endif()
message (STATUS "Building with Neutrino root: ${NEUTRINO_ROOT}")

# check for nphys
if (NOT ${NPHYS_PATH} STREQUAL "" AND NOT IS_ABSOLUTE ${NPHYS_PATH})
    message (STATUS "NPHYS_PATH is not absolute, fixing")
    set (ABS_NPHYS_PATH "${CMAKE_BINARY_DIR}/${NPHYS_PATH}")
endif()

# find goodies
set (CMAKE_MODULE_PATH ${NEUTRINO_ROOT}/resources/cmake)
include(FindNeutrinoGuiComponents)
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


SET(Qt5LinguistTools_DIR "${Qt5_DIR}/../Qt5LinguistTools")

find_package(Qt5LinguistTools)

if (Qt5LinguistTools_FOUND)
    SET(LANGUAGES fr_FR it_IT ko_KP)

    SET(LANGUAGE_TS_FILES)
    FOREACH(LANGUAGE ${LANGUAGES})
    SET(TS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${MY_PROJECT_NAME}_${LANGUAGE}.ts")
    message(STATUS "Language : ${TS_FILE}")
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
    message(STATUS "${LIST_LENGTH} translation files: ${TRANSL_QRC}")

    qt5_add_resources(RES_SOURCES ${TRANSL_QRC})
    ENDIF(LANGUAGE_TS_FILES)

endif(Qt5LinguistTools_FOUND)

ADD_HELP()



QT5_WRAP_UI(nUIs ${UIS})

# add sources here
add_library (${PROJECT_NAME} SHARED ${SOURCES} ${nUIs} ${RES_SOURCES})

IF(APPLE)
set (CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
ENDIF()


if (USE_QT5)
    qt5_use_modules(${PROJECT_NAME} Core Gui Sql Widgets Svg PrintSupport UiTools Multimedia MultimediaWidgets)
elseif(USE_QT4)
    target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})
endif()



