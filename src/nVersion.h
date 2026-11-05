#ifndef __nVersion_h
#define __nVersion_h

// git sha and version (3.1.17: as the wheel, see python/ci/version.py) of the build: nVersion.cc is written at build time by resources/cmake/GitVersion.cmake
// (it was the -DSHAVERSION__ compile definition, which made every commit recompile all of src)
extern const char neutrinoGitSha[];
extern const char neutrinoVersion[];
#define SHAVERSION__ neutrinoGitSha

#endif
