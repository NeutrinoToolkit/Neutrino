# this is the frigging nPhysImage cmakelists

# file(GLOB_RECURSE SOURCES *.cc *.cpp *.cxx *.h)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cc *.cpp *.cxx *.h)

add_library (nPhysImageF SHARED ${SOURCES})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include(FindNeutrinoDeps)

# target_link_libraries(nPhysImageF m ${LIBS})
# usage requirements: the app and the plugins just link nPhysImageF
target_include_directories(nPhysImageF BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${NPHYS_INCLUDES})
target_compile_definitions(nPhysImageF PUBLIC ${NPHYS_DEFINITIONS})
target_compile_options(nPhysImageF PUBLIC ${NPHYS_COMPILE_OPTIONS})
target_link_libraries(nPhysImageF PUBLIC ${NPHYS_PUBLIC_LIBS} PRIVATE m ${NPHYS_PRIVATE_LIBS})

# install
if (WIN32)
    install (TARGETS nPhysImageF COMPONENT binary DESTINATION "bin")
elseif(LINUX)
    install (TARGETS nPhysImageF COMPONENT binary DESTINATION lib)
elseif (APPLE)
	SET_TARGET_PROPERTIES(nPhysImageF PROPERTIES INSTALL_NAME_DIR "@executable_path/../Frameworks")
    install (TARGETS nPhysImageF COMPONENT binary DESTINATION "${PROJECT_BINARY_DIR}/Neutrino.app/Contents/Frameworks")
endif()

