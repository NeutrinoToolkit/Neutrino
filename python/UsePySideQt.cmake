# cmake -DMODULE=<neutrino module> -DLIBNEUTRINO=<libNeutrino.dylib> -DPYSIDE_QT_LIB=<PySide6/Qt/lib>
#       -DNPHYS_DIR=<dir of libnPhysImageF> -P UsePySideQt.cmake
#
# macOS: the python module and a copy of libNeutrino next to it use the Qt of PySide6 (e.g. the pip wheel)
# instead of the Qt Neutrino was compiled with. It must be the same Qt version (the headers are the same):
# two copies of Qt in the same process don't work (PySide's QApplication is not seen by the other copy).
# The executable and the original libNeutrino keep their Qt.
#
# cmake -DMODULE=<neutrino module> -DRPATHS=<run paths> -P UsePySideQt.cmake
#
# wheel (Wheel.cmake): only the module (libNeutrino is in it), Qt by @rpath and these run paths

# get_filename_component(OUT_DIR ${MODULE} DIRECTORY)
# get_filename_component(LIB_NAME ${LIBNEUTRINO} NAME)
# set(LIB_COPY ${OUT_DIR}/${LIB_NAME})
# file(COPY_FILE ${LIBNEUTRINO} ${LIB_COPY})
set(LIB_COPY)
if (LIBNEUTRINO)
    get_filename_component(OUT_DIR ${MODULE} DIRECTORY)
    get_filename_component(LIB_NAME ${LIBNEUTRINO} NAME)
    set(LIB_COPY ${OUT_DIR}/${LIB_NAME})
    file(COPY_FILE ${LIBNEUTRINO} ${LIB_COPY})
endif()
if (NOT DEFINED RPATHS)
    set(RPATHS @loader_path ${PYSIDE_QT_LIB} ${NPHYS_DIR})
endif()

foreach(bin ${MODULE} ${LIB_COPY})
    # Qt frameworks by absolute path (e.g. /opt/homebrew/opt/qtbase/lib/QtCore.framework/...) -> @rpath
    execute_process(COMMAND otool -L ${bin} OUTPUT_VARIABLE deps)
    string(REGEX MATCHALL "[^\n\t ]+/Qt[A-Za-z0-9]+\\.framework/Versions/A/Qt[A-Za-z0-9]+" qt_deps "${deps}")
    set(args)
    foreach(dep ${qt_deps})
        if (NOT dep MATCHES "^@rpath/")
            string(REGEX REPLACE ".*/(Qt[A-Za-z0-9]+)\\.framework/Versions/A/.*" "\\1" qt_name "${dep}")
            list(APPEND args -change ${dep} @rpath/${qt_name}.framework/Versions/A/${qt_name})
        endif()
    endforeach()

    # only these run paths: an old one (e.g. /opt/homebrew/lib, which has the Qt frameworks too) would come first
    # (install_name_tool refuses to delete and add the same one: those already there are kept)
    # set(wanted @loader_path ${PYSIDE_QT_LIB} ${NPHYS_DIR})
    set(wanted ${RPATHS})
    execute_process(COMMAND otool -l ${bin} OUTPUT_VARIABLE loads)
    string(REGEX MATCHALL "path [^\n]+ \\(offset" rpaths "${loads}")
    set(present)
    foreach(rpath ${rpaths})
        string(REGEX REPLACE "^path (.+) \\(offset$" "\\1" rpath "${rpath}")
        if (rpath IN_LIST wanted)
            list(APPEND present ${rpath})
        else()
            list(APPEND args -delete_rpath ${rpath})
        endif()
    endforeach()
    foreach(rpath ${wanted})
        if (NOT rpath IN_LIST present)
            list(APPEND args -add_rpath ${rpath})
        endif()
    endforeach()

    # nothing to change: e.g. the official Qt (@rpath already) with the run paths already there
    if (NOT args)
        continue()
    endif()
    execute_process(COMMAND install_name_tool ${args} ${bin} RESULT_VARIABLE res ERROR_VARIABLE err)
    if (NOT res EQUAL 0)
        message(FATAL_ERROR "install_name_tool ${bin}: ${err}")
    endif()
    # the changes invalidate the signature (arm64 needs one, ad hoc is enough)
    execute_process(COMMAND codesign --force --sign - ${bin} RESULT_VARIABLE res ERROR_QUIET)
endforeach()
# message(STATUS "neutrino python module: uses the Qt of PySide6 in ${PYSIDE_QT_LIB}")
message(STATUS "neutrino python module: Qt by @rpath, run paths ${RPATHS}")
