Neutrino
========

A light, expandable and full featured image analysis tool for research


Rationale
---------

*Neutrino* is an image viewer/analyser meant to be a research tool for manipulating experimental images. On a more general basis each bi-dimensional matrix of elements in the real or in the complex field can be seen as an image

Features
--------

Neutrino includes:

* multiple windows, multiple buffer support.
* common lossless input formats (including txt, tiff, sif, fits, hdf, raw) plus common formats (as supported by QImage. New formats can be added runtime via C++ plugins or python scripting
* txt output format and proprietary binary format for image or session save
* vectorial export formats (pdf, svg)


Build 
-----

We use Cirrus-ci for providing an update release executable for windows, linux and macos. Have a loot at the `.cyrrus.yml` if you want to recompile it.

Binary versions are available at this page: https://github.com/NeutrinoToolkit/Neutrino/releases/tag/latest

Status: [![Build Status](https://api.cirrus-ci.com/github/NeutrinoToolkit/Neutrino.svg?branch=master)](https://cirrus-ci.com/github/NeutrinoToolkit/Neutrino)
