# check for neutrino root
if (NOT DEFINED NEUTRINO_ROOT)
    set (NEUTRINO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../..")
endif()

if (NOT IS_ABSOLUTE ${NEUTRINO_ROOT})
    set (NEUTRINO_ROOT "${CMAKE_BINARY_DIR}/${NEUTRINO_ROOT}")
endif()

set (CMAKE_MODULE_PATH ${NEUTRINO_ROOT}/resources/cmake)

set(HEADERS "../Lineout.h")
set(UIS "../Lineout.ui")

include(NeutrinoPlugin)
set (PLUGIN_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../../Neutrino.app/Contents/Resources/plugins")
ADD_NEUTRINO_PLUGIN()

