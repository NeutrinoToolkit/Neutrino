# check for neutrino root
if (NOT DEFINED NEUTRINO_ROOT)
    set (NEUTRINO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
endif()

if (NOT IS_ABSOLUTE ${NEUTRINO_ROOT})
    set (NEUTRINO_ROOT "${CMAKE_BINARY_DIR}/${NEUTRINO_ROOT}")
endif()

set (CMAKE_MODULE_PATH ${NEUTRINO_ROOT}/resources/cmake)

set(HEADERS "../nLineout.h")
set(UIS "../nLineout.ui")

include(NeutrinoPlugin)
ADD_NEUTRINO_PLUGIN()

