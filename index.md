---
layout: default
title: Neutrino
---

*Neutrino* is an image viewer/analyser meant to be a research tool for manipulating experimental images. 
It has been written to: 

* be *fast*: it is entirely written in C++ and makes strong use of multi-core and gpu-assisted
  computation

* promote *interaction* of the researcher with the images (easy
access to colormaps, contrast/gamma/cutoffs, matrix operations, real-time lineouts)

* be *accurate*: buffers are loaded and manipulated with full dynamics, integers or real numbers

* let python do the work: the internal *python* interpreter can access the entire code structure
  in *runtime*, which can be used to **automate** repetitive tasks or **extend** Neutrino
capabilities.

Neutrino includes:

* multiple windows, multiple buffer support.
* internal **python** interpreter, able to access the entire code structure *runtime*
* common lossless input formats (including txt, netpbm, tiff, fits, hdf, raw), proprietary formats (Andor SIF, Pixelfly PCO, Fujitsu/Siemens FLA) and other common formats (as supported by QImage).
* **txt** output format and proprietary binary format for image or session save
* vectorial export formats (pdf, svg)

Image visualization/interaction:

* lineouts
* multiple colormaps
* powerful magnification tools

Advanced analysis tools:

* Interferometry (via 2D Wavelet analysis) ([tutorial](tutorials/wavelet/wavelet-tutorial))
* VISAR analysis
* Integral inversions (Abel)
* Fourier Analysis

# Getting Neutrino

The binary releases are available for GNU/Linux, Windows and OSX on [Neutrino Releases](https://github.com/NeutrinoToolkit/Neutrino/releases).

Sources are available from [GitHub Neutrino page](https://github.com/NeutrinoToolkit/Neutrino). For building instructions a tutorial is available [here](build)
