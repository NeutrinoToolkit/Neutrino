# this is the frigging nPhysImage cmakelists
cmake_minimum_required (VERSION 3.0)


add_library (nPhysImageF SHARED NaNStream.cc  nPhysFormats.cc  nPhysImageF.cc  nPhysMaths.cc  nPhysWave.cc anymap.cc unwrapping/unwrap_goldstein.cc  unwrapping/unwrap_quality.cc  unwrapping/unwrap_simple.cc  unwrapping/unwrap_util.cc  )

target_include_directories (nPhysImageF PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
