# check for neutrino root
find_library(CFITS NAMES cfitsio)
if (NOT ${CFITS} STREQUAL "CFITS-NOTFOUND")
    ADD_NEUTRINO_PLUGIN()
else()
    message (STATUS "Cannot find FITS: plugin discarded")
endif()



