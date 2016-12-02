# tp-plugin CMakeLists.txt

PROJECT (testPlugin CXX)

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
add_definitions(-DQT_SHARED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# add neutrino deps
include_directories(${NEUTRINO_ROOT}/nPhysImage)
include_directories(${NEUTRINO_ROOT}/src) # for base stuff
include_directories(${NEUTRINO_ROOT}/src/graphics) # for nLine
include_directories(${NEUTRINO_ROOT}/src/pans/colorbar)

# add needed UIs here
QT5_WRAP_UI(nUIs ${NEUTRINO_ROOT}/UIs/nPoint.ui)
QT5_WRAP_UI(nUIs ${NEUTRINO_ROOT}/UIs/nObject.ui)
QT5_WRAP_UI(nUIs ${NEUTRINO_ROOT}/UIs/nLine.ui)
QT5_WRAP_UI(nUIs ${NEUTRINO_ROOT}/UIs/nColorBarWin.ui)
QT5_WRAP_UI(nUIs ${NEUTRINO_ROOT}/UIs/nSbarra.ui)
QT5_WRAP_UI(nUIs ${NEUTRINO_ROOT}/UIs/nPanHelp.ui)
QT5_WRAP_UI(nUIs ${NEUTRINO_ROOT}/UIs/nAbout.ui)
QT5_WRAP_UI(nUIs ${NEUTRINO_ROOT}/UIs/neutrino.ui)

set (SOURCES test-plugin.cc test-plugin.h)

# add sources here
add_library (${PROJECT_NAME} SHARED ${SOURCES} ${nUIs})

IF(APPLE)
    set (CMAKE_SHARED_LINKER_FLAGS "-single_module -undefined dynamic_lookup")
ENDIF()


if (USE_QT5)
    qt5_use_modules(${PROJECT_NAME} Core Gui Sql Widgets Svg PrintSupport UiTools Multimedia MultimediaWidgets)
elseif(USE_QT4)
    target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})
endif()



message(STATUS ${QT_LIBRARIES})
