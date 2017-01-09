# check for neutrino root
if (NOT DEFINED NEUTRINO_ROOT)
    set (NEUTRINO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
endif()

if (NOT IS_ABSOLUTE ${NEUTRINO_ROOT})
    message (STATUS "NEUTRINO_ROOT is not absolute, fixing")
    get_filename_component(${NEUTRINO_ROOT} ${NEUTRINO_ROOT} ABSOLUTE)
endif()
message (STATUS "neutrino root ${NEUTRINO_ROOT}" )

set (CMAKE_MODULE_PATH ${NEUTRINO_ROOT}/resources/cmake)
include(NeutrinoPlugin)

ADD_NEUTRINO_PLUGIN()
