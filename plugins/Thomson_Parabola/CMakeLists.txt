

# GLU

find_package(OpenGL REQUIRED)

    if (OPENGL_FOUND)
        message("Found OpenGL for Mac OS")
        message("OpenGL for Mac OS, include dir: ${OPENGL_INCLUDE_DIR}")
        message("OpenGL for Mac OS, link libraries: ${OPENGL_gl_LIBRARY}")
        list(APPEND LOCAL_LIBS ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
    else (OPENGL_FOUND)
        message(FATAL_ERROR "OpenGL for Mac OS not found")
    endif()


#find_library(GLU NAMES GLU)
#if (NOT ${GLU} STREQUAL "GLU-NOTFOUND")
#        list(APPEND LOCAL_LIBS ${GLU})
#        message (STATUS "using GLU ${GLU}")
#else()
#        message (WARNING "Cannot find GLU: Skipping plugin")
#        return()
#endif()
set(QT_USE_QTOPENGL TRUE)


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

