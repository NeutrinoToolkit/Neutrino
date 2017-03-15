[![Build Status](https://travis-ci.org/NeutrinoToolkit/Neutrino.svg?branch=master)](https://travis-ci.org/NeutrinoToolkit/Neutrino)

neutrino
========

A light, expandable and full featured image analysis tool for research


Rationale
---------

*Neutrino* is an image viewer/analyser meant to be a research tool for manipulating experimental images. On a more general basis each bi-dimensional matrix of elements in the real or in the complex field can be seen as an image

Features
--------

Neutrino includes:

* multiple windows, multiple buffer support.
* common lossless input formats (including txt, netpbm, tiff, sif, fits, hdf, raw) plus common formats (as supported by QImage. New formats can be added runtime via C++ plugins or python scripting
* txt output format and proprietary binary format for image or session save
* vectorial export formats (pdf, svg)


Build
-----

To compile for osx, please follow the `.travis.yml` recipe file.

To compile for debian, please follow the docker recipe https://hub.docker.com/r/iltommi/debian-sid-neutrino/~/dockerfile/

To compile for Windows, we cross compile using this other Docker recipe: https://hub.docker.com/r/iltommi/neutrino-docker-cross/~/dockerfile/

