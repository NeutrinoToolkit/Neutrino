Statistics of a focal spot (e.g. a laser focus imaged on a camera): background, centre, contour at a fraction of
the peak and the energy inside it, FWHM, second-moment widths (D4σ), radial profile and encircled energy.

The pan works on the image shown and updates at each change of the image or of a parameter. The lengths are in the
units of the image scale (px if the scale is 1), the energies in counts above the background.

Settings
--------

- **Centre**: the centre of the spot, used for the contour, the FWHM, the radial profile and the encircled energy:
  - **Maximum**: the maximum of the blurred image.
  - **Contour centroid**: the centroid of the area inside the contour (the contour is traced around the maximum).
  - **Manual (double click)**: double click on the spot in the image (e.g. to measure a secondary spot).
- **Set origin**: moves the origin of the image to the centre (e.g. to read the distances from the spot on the
  rulers). The pan does not change the image otherwise.
- **Background**: the level subtracted from all the pixels:
  - **Auto (border)**: the median of a frame at the border of the image (5% of its size). The border must be free
    of signal.
  - **Manual**: the value on the right.
  - **Manual (old method)**: the computation of the previous versions, for comparison: the pixels between the
    background and the contour level counted as the background and the ones below it as themselves. The noise of
    the background was rectified and summed over the whole image: with noise, the energy fractions were too low
    (e.g. 30% instead of 50%). Its two values are on the line **Old method**.
- **Contour**: the level of the contour in % of the peak above the background (50%: the FWHM contour).
- **Blur** [px]: gaussian blur of the image used for the maximum, the contour and the FWHM (0: none), against the
  noise. The energies, the moments and the radial profile use the image itself.

The contour is drawn on the image (a closed line).

Results
-------

One line per group of results, in world units (the unit at the end of the line, or after each value):

| line | meaning |
|---|---|
| Background | the level subtracted |
| Centre x, y | the centre [px] |
| Peak blurred, pixel/total | the value of the blurred image at the centre minus the background, and the brightest pixel in % of the total energy |
| Energy total, in contour | the sum of all the pixels minus the background, and the part of the pixels whose centre is inside the contour [%] |
| Old method threshold, contour | only with **Manual (old method)**: the two energies of the previous versions [%] |
| Contour level, area | the level of the contour [counts] and the area inside it |
| Contour radius √(area/π), min, max | the radius of the disc of the same area, and the distances of the contour from its centroid |
| Centroid x, y | the first moments [px] |
| D4σ x, y, major, minor | 4 times the standard deviations (second moments) of the spot |
| FWHM x, y, major, minor | the full widths at half maximum of the blurred image through the centre, along x, y and the axes of the moments |
| Major axis angle, ellipticity | the orientation of the major axis of the moments, from x toward y (down on the image), and FWHM minor / major |
| Encircled radius 50%, 86.5% | the radii with 50% and 86.5% of the energy (86.5%: the 1/e² radius of a gaussian) |

The names in the exported file and in `pan.results()` are the line and the value, e.g. `FWHM major`.

For a gaussian of standard deviation σ: FWHM = 2.355 σ, D4σ = 4σ (the 1/e² diameter), the 50% contour holds 50% of
the energy, and for a round spot the radii with 50% and 86.5% of the energy are 1.177 σ and 2 σ.

The moments are computed as in ISO 11146: iterated in a rectangle of 3 times the D4σ widths around the centroid, so
that the noise far from the spot does not count. The energy fractions are relative to the whole image: a small
error of the background is multiplied by the number of pixels (e.g. 0.04 counts on 200000 pixels): if the
encircled energy levels off below 1, the background is slightly overestimated (above 1: underestimated).

Radial
------

The mean value (minus the background) in rings around the centre (red, left axis) and the encircled energy, the
fraction of the total energy within a radius (blue, right axis), up to 4 times the FWHM. The mouse on the image
shows its distance from the centre on the plot.

Toolbar
-------

- <img src="../../resources/icons/saveClipboard.png" width="16" />: copies the results (lines starting with #)
  and the radial profile (radius, profile, encircled energy) to the clipboard.
- <img src="../../resources/icons/saveTxt.png" width="16" />: exports them to a text file.

From Python, `pan.results()` gives the results as a dict (with the radial profile as arrays).
