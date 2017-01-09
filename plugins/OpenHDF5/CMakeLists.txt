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
    # check for neutrino root
    if (NOT DEFINED NEUTRINO_ROOT)
        set (NEUTRINO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
    endif()

    if (NOT IS_ABSOLUTE ${NEUTRINO_ROOT})
        message (STATUS "NEUTRINO_ROOT is not absolute, fixing")
        set (NEUTRINO_ROOT "${CMAKE_BINARY_DIR}/${NEUTRINO_ROOT}")
    endif()
    message (STATUS "neutrino root ${NEUTRINO_ROOT}" )

    set (CMAKE_MODULE_PATH ${NEUTRINO_ROOT}/resources/cmake)
    include(NeutrinoPlugin)
    ADD_NEUTRINO_PLUGIN()

endif()
