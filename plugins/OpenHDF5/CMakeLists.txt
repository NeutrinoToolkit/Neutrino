# find_package(HDF5 COMPONENTS HL QUIET)
# find_library(HDF5HL NAMES hdf5_hl PATHS ${HDF5_LIBRARY_DIRS}  QUIET) # unused: HDF5_HL_LIBRARIES comes from find_package
# the high level library (H5LT) is not used any more (nHdf5io writes the attributes itself)
find_package(HDF5 COMPONENTS C QUIET)

# if (HDF5_FOUND AND HDF5_HL_LIBRARIES)
if (HDF5_FOUND)
    #hdf5 libs
    include_directories(${HDF5_INCLUDE_DIRS})
    set(LOCAL_LIBS ${LOCAL_LIBS} ${HDF5_LIBRARIES})
    add_definitions(-DHAVE_HDF5)
    add_definitions(${HDF5_DEFINITIONS})

    # hdf5_hl
    # set(LOCAL_LIBS ${LOCAL_LIBS} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
    # set(LOCAL_LIBS ${LOCAL_LIBS} ${HDF5_HL_LIBRARIES})
    # add_definitions(-DHAVE_LIBHDF5HL)

    ADD_NEUTRINO_PLUGIN()

else()
    message (STATUS "Cannot find HDF5: plugin discarded")
endif()


