Phase of an interferogram (e.g. of a plasma or a gas jet) by a wavelet analysis of the fringes, with the reference
interferogram (without the object) subtracted, then unwrapped and cleaned.

The pan is a chain of steps, each with its **...** button. Each step works on the result of the previous one and,
by default, runs the following ones: change a parameter and click the button of its step.

    wavelet (reference, shot) → unwrap → zero, sign, rotation → blur, mask, quality → interpolate → cutoff, crop

The results are images named after their step (`phase_2pi_wrap`, `phase_2pi_unwrap`, `phase_2pi`, `phaseMask`, ...),
replaced at each run.

Typical use
-----------

1. Choose the **Reference** and the **Shot** images (tabs).
2. Show the region (**rect** button of the toolbar) and put it around the interesting part of the shot.
3. Guess the carrier (**...** of **Main interfringe**): the interfringe and the orientation of the fringes.
4. **doAll** (toolbar): the wavelets of both images and all the steps after.
5. Put the zero of the phase on a point without the object (**Zero point**), clean the phase (mask, quality,
   interpolation).

The result is `interpPhase_2piMaskCutoff`, a phase in fringes (1 = 2π), ready for the **Integral inversion** pan
(with **Phase in fringes** checked).

Main interfringe
----------------

- **...**: guess of the carrier from the Fourier transform of the region of the reference: the peak of
  |FFT|·r<sup>w</sup>, with r the frequency and w the weight (the spin box on the left, guessing again when changed):
  w < 0 prefers the low frequencies, w > 0 the high ones.
- Interfringe [px] and orientation [˚] of the carrier: the centre of the scan of the wavelets.
- **Thick**: half width of the analysis window along the fringes, in interfringes of the carrier. A thin window
  follows the fast changes of the fringes, a thick one averages the noise.
- **Correlation**: length of the window across the fringes, in interfringes (the damping of the wavelet). A long
  window is selective in interfringe and angle (less noise) but smooths the steep gradients of the phase.
- **Pad**: the image around the region (2 analysis windows on each side, inside the image) is used in the
  computation: less error at the borders of the region.
- **K**: regularization of the interfringe and the angle: each pixel keeps its K best filters and takes the one
  nearest to the interfringe and angle maps smoothed over the neighbours (weighted by the contrast), against the
  isolated wrong choices of the filter in the noisy regions. 0: off (the best filter of each pixel).
- **log λ**: the interfringes of the scan are spaced geometrically (the same relative step everywhere), otherwise
  linearly.

Reference and Shot
------------------

Each tab sets the image and its scan of the wavelet, around the carrier:

- Angle min : max [˚] and number: the orientations, relative to the carrier. Number 0: the carrier only.
- Stretch min : max and number: the interfringes, in units of the carrier interfringe. Number 0: the carrier only.
- **...**: the wavelet of this image only (without the following steps).

The scan should cover the interfringes and orientations of the fringes in the region: a small range for the
reference, a wider one for the shot, where the object bends the fringes. If the reference and the shot are the same
image, only the shot is analysed and the carrier is subtracted instead of the reference.

The wavelet gives, for each image (suffix `_ref` or `_shot`): `phase_2pi` (wrapped phase, in fringes), `contrast`
(amplitude of the fringes), `intensity` (the image without the fringes), `visibility`, `lambda` and `angle` (the
interfringe and the orientation of the best filter), and a `synthetic` interferogram rebuilt from them.

Unwrap
------

The wrapped phase `phase_2pi_wrap` (shot − reference, in fringes, modulo 1) is unwrapped into `phase_2pi_unwrap`.
The quality of the phase, `phase_quality`, is √(contrast<sub>shot</sub>·contrast<sub>ref</sub>).

- **Simple H+V**, **Simple V+H**: along the rows, then the columns (or the opposite). Fast, but an error propagates
  along the whole row or column.
- **Goldstein**: residues and branch cuts, the flood fill ordered by the quality.
- **Miguel**, **Miguel+Quality**: sorting by reliability (Herráez et al.), without or with the quality.
- **Quality**: flood fill guided by the quality, the best pixels first.
- **angle**: where the fringes turn by more than 90˚, the best filter can point in the opposite direction, which
  flips the sign of the phase. The angle of the shot is unwrapped and the phase of the shot flipped where needed
  (images `unwrap angles` and `phase2`).
- Barrier (check box and **...**): a line that the unwrap does not cross (e.g. the edge of a target). Not for the
  Simple methods.

Zero point - sign - rotation
----------------------------

- x, y: the point of the shot where the phase is set to 0 (a point without the object). **...**, then a double
  click on the image picks it.
- **±**: changes the sign of the phase.
- Rotation [˚] of the results (`phase_2pi`, `contrast`, `angle`, `lambda`, `quality`): e.g. to put the axis of the
  object horizontal for the Abel inversion.

Here `contrast` is the ratio of the intensities shot/reference (the absorption), `lambda` the ratio of the
interfringes and `angle` the difference of the orientations.

Blur, mask & quality cutoff
---------------------------

- Blur [px]: gaussian blur of the phase (0: off), before the mask.
- Mask (check box and **...**): only the phase inside the closed line is kept, 0 outside.
- Quality cutoff [%]: the pixels of quality below this fraction of its range (from min to max) are set to 0.

The result is `phaseMask`.

Interpolate
-----------

With the check box, the phase inside each shape (**+** adds one, its button shows it) is replaced by a harmonic
(Laplace) interpolation of the phase around it: to fill a hole, or a region where the unwrap failed. The result is
`interpPhase_2piMask`.

Cutoff & Rectangular crop
-------------------------

- **99%**: the values are clipped to keep 99% of the pixels.
- min : max: the values are clipped to these limits (empty: no limit).
- Crop (check box and **...**): only the rectangle is kept.

The result is `interpPhase_2piMaskCutoff`.

Toolbar
-------

- **rect**: shows the region of the wavelet.
- **Automatic workflow**: when checked, each step does only itself: the following steps are not run.
- **doAll**: the wavelets of the reference and the shot, and all the steps after.
- **Duplicate**: the next run creates new images instead of replacing the current results (to compare two settings).
- **Delete**: closes the results of the pan, except the image shown.

On the lines and rectangles of the pan (region, mask, barrier, shapes) the **.** key runs their step.

During the wavelet the progress is shown in the status bar of the pan, with a **Stop** button.
