Spectral analysis
=================

Fourier transform of an image (or of a complex image, with the imaginary part from a second image).

- **transform**: 1D horizontal (each row), 1D vertical (each column) or 2D.
- **direction**: forward or backward (the arrow).
- **output**: polar (modulus and phase), rectangular (real and imaginary parts), power spectrum (|F|^2) or its log10.
- **shift**: the zero frequency in the centre (after a forward transform; before a backward one it is moved back). With shift, the results of a forward transform have frequency axes: origin at the zero frequency and scale 1/(N dx), where dx is the scale of the image.
- **normalize**: divides by the square root of the number of points of the transform (unitary transform: the energy is the same).

Filter
------

Band-pass on the forward transform: only the frequencies with |f| between **min** and **max** are kept, horizontally (H) and vertically (V); a max of 0 means no upper limit. The frequencies are in the units of the frequency axes, 1/(N dx): for an image with scale 1, cycles per pixel (0.5 is the highest). For a 1D transform only its direction is filtered.

Besides the filtered spectrum, the **filtered image** is given (backward transform of the filtered spectrum), with the origin and scale of the source.
