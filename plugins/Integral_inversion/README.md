Abel inversion of an image of a cylindrically symmetric object: from a quantity integrated along the line of
sight (e.g. the phase of an interferogram) it gives the local quantity per unit length (e.g. the refractive index
or the density) as a function of the distance from the symmetry axis.

The inversion is done independently for each point of the axis, on the profile perpendicular to the axis:

    f(r) = -1/π ∫ F'(y) / √(y² - r²) dy      (y from r to ∞)

where F(y) is the image value at distance y (in pixels) from the axis and f(r) the result (in units of the
image per pixel).

Typical use
-----------

1. Choose the **image** (e.g. the `phase_2pi` of the Interferometry pan) and, if needed, a reference to
   **Subtract** (e.g. the phase without the plasma).
2. Show the axis (**line** button of the toolbar) and put it on the symmetry axis of the object.
3. Choose the algorithm and the physics (tab), then **Calculate**.

Axis
----

The axis is a line on the image (toolbar: **line** shows its window with the points). It is *monotone*: each column
(horizontal axis) or each row (vertical axis) crosses it once, and the profiles are taken perpendicular to it
(vertical profiles for a horizontal axis). The axis can be a polyline or a **bezier** curve, to follow an object
that is not straight. **flipline** switches between horizontal and vertical.

Each profile starts on the axis and goes to the border of the image; the two sides of the axis are inverted
separately, and the value on the axis is their average. Beyond the border the profile is continued with its last
value: only the derivative of the profile is used, so a constant background does not matter, but the profile
should be **flat at the border** of the image (a profile still decreasing at the border is cut, and the result
near the border is wrong).

Image
-----

The operations on the image, in order:

- **Subtract**: the reference image (same size) is subtracted.
- **Blur** [px]: gaussian blur before the inversion (off at 0). The inversion uses the derivative of the
  profile, which amplifies the noise, mostly near the axis: a blur of a few pixels is often needed.
- **Multiply**: the image is multiplied by this factor (nothing at 1). E.g. -1 if the phase of the plasma is
  positive.
- **Phase in fringes (×2π)**: the image is a phase in fringes (as from Interferometry), converted to radians
  before applying the physics (only for the **Neutral** and **Plasma** tabs).

Inversion Algorithm
-------------------

- **Abel**: direct integration of the formula above, exact for a profile linear between the pixels.
- **Abel-HF**: Hankel-Fourier method (experimental).
- **Symmetric**: the two sides of the axis are averaged (over the length both sides have inside the image) and
  inverted once: the result is symmetric about the axis. It removes the asymmetries of the image (e.g. a tilt of
  the background) and reduces the noise, at the cost of the true asymmetries of the object. Beyond the length of
  the shorter side the result is 0.
- **Probe** [nm]: wavelength of the probe beam (Neutral and Plasma).
- **Scale** [µm/px]: size of the pixel in the object plane (all the physics).

Physics
-------

The tab chosen when calculating sets the conversion of the result f (phase per pixel, in radians), with the
wavenumber k = 2π/λ of the probe and the pixel size Δx:

- **None**: the inverted image, in units of the image per pixel.
- **Neutral**: density of a neutral gas [m⁻³] from its **Molar Refractivity** K (Lorentz-Lorenz), with the
  refractive index n = 1 + f/(k Δx):

        N = N_A (n² - 1) / (3K)

  K[H2]=2.01e-6, K[He]=5.23e-7, K[Ar]=4.2e-6 [m³/mol].
- **Plasma**: electron density [m⁻³] from the cold plasma dispersion law, for a density much lower than the
  critical density n_c = ε₀ mₑ ω²/e²:

        n_e = -2 n_c f / (k Δx)

  The phase of a plasma is negative (n < 1): a positive phase gives a negative density (use **Multiply** -1).
- **Protons**: charge density [C m⁻³] for the proton radiography with protons of **Energy** [MeV], at
  **Distance** [cm] from the detector with **Magnification** M: the inverted image times
  2ε₀ M E / (d Δx), with E in eV (i.e. the voltage E/e). The scale of the result is set in cm.

Output
------

- **Min** / **Max**: the result is clipped to these values (empty: no limit).
- **Erase previous**: the new result replaces the previous one, otherwise a new image is added.
- **AutoUpdate**: the inversion runs again each time the axis is moved. If the axis moves during an inversion, one
  more inversion runs at its end with the last position. On large images an inversion takes some time (≈1.6 s for
  1500×1500 pixels): the result follows the axis with a delay.

During the calculation the progress is shown in the status bar of the pan, with a **Stop** button.
