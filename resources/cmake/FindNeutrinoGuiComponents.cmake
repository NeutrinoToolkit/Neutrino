# find components specific to Neutrino GUI files here!

include(FindNeutrinoDeps)

if(DEFINED ENV{QTDIR})
    set(CMAKE_PREFIX_PATH $ENV{QTDIR} ${CMAKE_PREFIX_PATH})
endif()

if(DEFINED QTDIR)
    set(CMAKE_PREFIX_PATH ${QTDIR} ${CMAKE_PREFIX_PATH})
endif()

if (NOT DEFINED Qt5_DIR AND APPLE)
    set(Qt5_DIR "/usr/local/opt/qt5/lib/cmake/Qt5")
endif()

## find qt -- search for 5.x first, fallback to 4.x
find_package(Qt5 COMPONENTS Core Gui Sql Widgets Svg PrintSupport UiTools Multimedia MultimediaWidgets OpenGL REQUIRED)

add_definitions(${QT_DEFINITIONS})
include_directories(${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})


