# check for neutrino root
find_library(CFITS NAMES cfitsio)
if (NOT ${CFITS} STREQUAL "CFITS-NOTFOUND")
    # MUSE calls cfitsio directly (nPhysImageF links it privately)
    find_path(CFITS_INCLUDE_DIR fitsio.h PATH_SUFFIXES cfitsio)
    include_directories(${CFITS_INCLUDE_DIR})
    list(APPEND LOCAL_LIBS ${CFITS})
    ADD_NEUTRINO_PLUGIN()
else()
    message (STATUS "Cannot find FITS: plugin discarded")
endif()



