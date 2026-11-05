Neutrino
========

A light, expandable and full featured image analysis tool for research

[![build](https://github.com/NeutrinoToolkit/Neutrino/actions/workflows/main.yml/badge.svg)](https://github.com/NeutrinoToolkit/Neutrino/actions/workflows/main.yml)
[![PyPI](https://img.shields.io/pypi/v/neutrino-toolkit)](https://pypi.org/project/neutrino-toolkit/)

Installation
------------

Grab the file for your system from the [releases](https://github.com/NeutrinoToolkit/Neutrino/releases):
[`latest`](https://github.com/NeutrinoToolkit/Neutrino/releases/tag/latest) has the builds of the `master` branch,
`latest-<branch>` those of the other branches.

### macOS

Download `Neutrino-Darwin-arm64.dmg` (Apple Silicon) or `Neutrino-Darwin-x86_64.dmg` (Intel), open it and drag `Neutrino.app` into `Applications` (a `.zip` with the same app is also available).

The app is not signed with an Apple Developer ID nor notarized, so the first time macOS will refuse to open it ("Apple could not verify..." or "Neutrino is damaged"). To allow it, either:

* open `System Settings → Privacy & Security`, scroll down and click `Open Anyway` next to the Neutrino message, then confirm; or
* remove the quarantine flag from a terminal:
  ```
  xattr -dr com.apple.quarantine /Applications/Neutrino.app
  ```

This is needed only once (and again after each update).

### Windows

The installer `Neutrino-*.exe` (or the `.zip`, to be unpacked anywhere), 64 or 32 bit.

### Linux

* Ubuntu, Debian, Linux Mint: the `.deb` of your distribution (`sudo apt install ./Neutrino-*.deb`)
* Fedora, openSUSE: the `.rpm` of your distribution
* any other distribution: the `AppImage` (x86_64 or aarch64; the `olderLinux` ones for older systems), made executable
  with `chmod +x`

### Python

The `neutrino` module opens Neutrino windows from python, with the images as numpy arrays and the analysis windows
driven by script:

```
pip install neutrino-toolkit
```

```python
from PySide6.QtWidgets import QApplication
import neutrino, numpy as np

app = QApplication([])
w = neutrino.Window()
w.addShowPhys(np.random.rand(512, 512))
neutrino.run()
```

See [python/README.md](python/README.md), `help(neutrino)` and the [examples](python/examples).

Rationale
---------

*Neutrino* is an image viewer/analyser meant to be a research tool for manipulating experimental images. On a more general basis each bi-dimensional matrix of elements in the real or in the complex field can be seen as an image

Features
--------

Neutrino includes:

* multiple windows, multiple buffer support
* common lossless input formats (including txt, tiff, fits, hdf4, hdf5, sif, spe, pcoraw, b16, raw) plus the common
  image formats (as supported by QImage)
* txt output format and proprietary binary format for image or session save
* vectorial export formats (pdf, svg)
* analysis windows (pans), among them: interferometry (phase retrieval, unwrapping, Abel inversion), VISAR and SOP,
  lineouts, contours, focal spot, spectral analysis, XRD, math operations and functions, affine transformations,
  MUSE cubes, camera acquisition. Each one has its help (the Help button of its toolbar)
* scripting in python (see above): a script can open images, set and run the pans and read their results

Build
-----

Needs CMake >= 3.18, a C++17 compiler and:

* Qt 6 (Widgets, Svg, PrintSupport, Network, Multimedia)
* FFTW 3, GSL, libtiff, libjpeg, zlib, OpenMP
* optional: HDF4, HDF5, CFITSIO, pandoc (the help of the pans)

```
mkdir build && cd build
cmake ..
make -j8            # make package: the .dmg, .deb, .rpm or installer
```

The CI ([`main.yml`](.github/workflows/main.yml)) has the packages and commands for each system, and the build of the
python wheels ([python/README.md](python/README.md)).
