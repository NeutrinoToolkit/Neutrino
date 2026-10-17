#ifndef __nVersion_h
#define __nVersion_h

// git sha of the build: nVersion.cc is written at build time by resources/cmake/GitVersion.cmake
// (it was the -DSHAVERSION__ compile definition, which made every commit recompile all of src)
extern const char neutrinoGitSha[];
#define SHAVERSION__ neutrinoGitSha

#endif
