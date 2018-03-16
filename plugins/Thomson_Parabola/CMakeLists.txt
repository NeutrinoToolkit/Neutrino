
include(FindOpenGL)
if (OPENGL_GLU_FOUND)
    list(APPEND LOCAL_LIBS ${OPENGL_GLU_LIBRARY})
    include_directories(BEFORE "${OPENGL_INCLUDE_DIR}")
else()
    message(WARNING "Cannot find GLU: Skipping plugin")
    return()
endif()

# check for neutrino root
if (NOT DEFINED NEUTRINO_ROOT)
    set (NEUTRINO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
endif()

if (NOT IS_ABSOLUTE ${NEUTRINO_ROOT})
    set (NEUTRINO_ROOT "${CMAKE_BINARY_DIR}/${NEUTRINO_ROOT}")
endif()

set(LOCAL_MODULES OpenGL)

set (CMAKE_MODULE_PATH ${NEUTRINO_ROOT}/resources/cmake)
include(NeutrinoPlugin)
ADD_NEUTRINO_PLUGIN()

