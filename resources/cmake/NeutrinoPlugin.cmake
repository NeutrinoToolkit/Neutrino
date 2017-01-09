
MACRO(ADD_NEUTRINO_PLUGIN)
    include(FindNeutrinoGuiComponents)

    get_filename_component(MY_PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    message (STATUS "NeutrinoPlugin >>>>>>>>>>>> ${MY_PROJECT_NAME} : ${CMAKE_CURRENT_SOURCE_DIR}")
    PROJECT (${MY_PROJECT_NAME} CXX)


    set (CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -Wall -D__phys_debug=10")
    set (CMAKE_CXX_FLAGS_RELEASE "-O3")
    add_compile_options(-std=c++11)


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

    ADD_PLUGIN_HELP()
    ADD_PLUGIN_TRANSLATIONS()

    QT5_WRAP_UI(nUIs ${UIS})

    # add sources here
    add_library (${PROJECT_NAME} SHARED ${SOURCES} ${nUIs} ${RES_SOURCES})

    IF(APPLE)
    set (CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
    ENDIF()

    if (DEFINED LOCAL_LIBS)
        target_link_libraries(${PROJECT_NAME} ${LOCAL_LIBS})
    endif()

    if (USE_QT5)
        qt5_use_modules(${PROJECT_NAME} Core Gui Sql Widgets Svg PrintSupport UiTools Multimedia MultimediaWidgets)
    elseif(USE_QT4)
        target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})
    endif()

ENDMACRO()

