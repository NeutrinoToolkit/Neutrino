find_package(HDF5 COMPONENTS HL QUIET)
#find_library(HDF5HL NAMES hdf5_hl PATHS ${HDF5_LIBRARY_DIRS}  QUIET)

if (HDF5_FOUND #[[AND HDF5_HL_LIBRARIES]])
    #hdf5 libs
    include_directories(${HDF5_INCLUDE_DIRS})
    set(LOCAL_LIBS ${LOCAL_LIBS} ${HDF5_LIBRARIES})
    add_definitions(-DHAVE_HDF5)

#    # hdf5_hl
#    set(LOCAL_LIBS ${LOCAL_LIBS} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
#    add_definitions(-DHAVE_LIBHDF5HL)

    ADD_NEUTRINO_PLUGIN()

else()
    message (STATUS "Cannot find HDF5: plugin discarded")
endif()


