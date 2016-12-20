#libhdf5_hl

find_package(HDF5 COMPONENTS CXX)
if (HDF5_FOUND)
  # IF HDF5 is there, THEN look for hl...
  find_library(HDF5HL NAMES hdf5_hl PATHS ${HDF5_LIBRARY_DIRS})
  if (${HDF5HL} STREQUAL "HDF5HL-NOTFOUND")
          message (STATUS "Cannot find HDF5_HL: disabling HDF5 support")
          message (STATUS "Search dir: " ${HDF5_LIBRARY_DIRS})

  else()

          #hdf5 libs
          include_directories(${HDF5_INCLUDE_DIRS})
          set(LOCAL_LIBS ${LOCAL_LIBS} ${HDF5_LIBRARIES})
          add_definitions(-DHAVE_HDF5)

          # hdf5_hl
          message (STATUS "using libhdf5_hl: ${HDF5HL}")
          set(LOCAL_LIBS ${LOCAL_LIBS} ${HDF5HL})
          add_definitions(-DHAVE_LIBHDF5HL)

          set (HDF5_FOUND_COMPLETE "TRUE")
  endif()

endif (HDF5_FOUND)

if (HDF5_FOUND_COMPLETE)
    get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    PROJECT (${PROJECT_NAME}Plugin CXX)

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
    include_directories(${NEUTRINO_ROOT}/src/graphics)
    QT5_WRAP_UI(nUIs ${NEUTRINO_ROOT}/UIs/neutrino.ui)
    QT5_WRAP_UI(nUIs ${NEUTRINO_ROOT}/UIs/nLine.ui)


    file(GLOB UIS ${CMAKE_CURRENT_SOURCE_DIR}/*.ui)
    file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

    QT5_WRAP_UI(nUIs ${UIS})

    # add sources here
    add_library (${PROJECT_NAME} SHARED ${SOURCES} ${nUIs})

    IF(APPLE)
         set (CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
    ENDIF()

    target_link_libraries(${PROJECT_NAME} ${LOCAL_LIBS})


    if (USE_QT5)
        qt5_use_modules(${PROJECT_NAME} Core Gui Sql Widgets Svg PrintSupport UiTools Multimedia MultimediaWidgets)
    elseif(USE_QT4)
        target_link_libraries(${PROJECT_NAME} ${QT_LIBRARIES})
    endif()

endif()
