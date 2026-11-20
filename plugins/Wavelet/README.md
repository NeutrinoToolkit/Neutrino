Phase of the fringes of a single interferogram by a wavelet analysis, then unwrapped and with the carrier (or a
reference phase) subtracted. For a shot and its reference together, with masks, interpolation and cutoffs, use the
**Interferometry** pan; for the density from the phase, the **Integral inversion** pan.

How it works
------------

The image is filtered with a set of complex wavelets (gaussian filters in the frequency domain around a carrier):
each filter is tuned to an interfringe and an orientation, and is as long across the fringes as **Damp** interfringes
and as long along them as **thick**. Each pixel keeps the filter with the largest response: its phase is the phase of
the fringes, its amplitude their contrast, its interfringe and orientation the local ones.

Typical use
-----------

1. Choose the **Image** (Wavelet tab).
2. Show the region (**rect** in the toolbar) and put it around the part of the interferogram to analyse. The region is
   shown only on its image.
3. Guess the carrier (**Carrier** in the toolbar): the interfringe and orientation of the fringes, shown next to
   **Rel. to carrier** and set in the Carrier tab.
4. Check **Rel. to carrier** and set the scan around the carrier (e.g. stretch 0.8 : 1.2 with 5 values, angle
   -10 : 10 with 5 values). Start with a coarse scan and refine it.
5. **Calculate**, then look at `phase_2pi`: its fringes should follow those of the interferogram.
6. **Unwrap** (Unwrap tab), then **Subtract** (Carrier tab) the carrier or the reference.

**doAll** in the toolbar runs the wavelet and the unwrap.

Wavelet
-------

- **Image**: the interferogram.
- **Stretch & rotation**: the scan of the filters, Min, Max and number of values for the interfringe (first row)
  and the angle (second row).
  - With **Rel. to carrier** the interfringes are in units of the carrier interfringe and the angles are relative to
    the carrier angle; otherwise in pixels and degrees.
  - Number 0: no scan, only the carrier.
  - The scan has to cover the interfringes and orientations of the fringes in the region. More values cost more time
    (one FFT per filter) but follow the fringes better.
- **Damp**: length of the filter across the fringes, in interfringes. A long filter is selective (less noise, about
  1/(π·damp) of relative mismatch of interfringe accepted) but smooths the steep gradients of the phase; a short one
  follows them but is noisier and lets the background of the image leak into the fringes.
- **log λ**: the interfringes of the scan are spaced geometrically (the same relative step everywhere), otherwise
  linearly.
- **thick**: half width of the filter along the fringes (pixels, or carrier interfringes with **Rel. to carrier**).
  Long averages more noise away but is biased where the fringes bend. 0: the carrier interfringe.
- **crop region**: also shows the part of the image analysed (`wavelet source`).
- **synthetic**: also rebuilds an interferogram from the results (`synthetic`), to compare with the original.
- **Erase previous**: the new results replace those of the previous run, otherwise they are added.

The results:

- `phase_2pi`: the wrapped phase, in fringes (1 = 2π), between -0.5 and 0.5.
- `contrast`: the amplitude of the fringes (the quality map for the unwrap).
- `intensity`: the image without the fringes.
- `visibility`: (Imax-Imin)/(Imax+Imin) of the fringes.
- `lambda`, `angle`: the interfringe and the orientation of the best filter (only when they are scanned), refined
  between the values of the scan.

Unwrap
------

- **FringeShift & Quality**: the wrapped phase (`phase_2pi`) and its quality (`contrast`), chosen automatically
  after the wavelet.
- **Method**:
  - **Simple H+V**, **Simple V+H**: along the rows, then the columns (or the opposite). Fast, but an error propagates
    along the whole row or column.
  - **Goldstein**: residues and branch cuts, the flood fill ordered by the quality.
  - **Miguel**, **Miguel+Quality**: sorting by reliability (Herráez et al.), without or with the quality.
  - **Quality**: flood fill guided by the quality, the best pixels first.
- Barrier (check box and **...**): a line that the unwrap does not cross (e.g. the edge of a target).
- **Unwrap angles**: where the fringes turn by more than 90˚ the best filter can point the opposite way, which flips
  the sign of the phase. The angle map is unwrapped and the phase flipped where needed (needs an angle scan).
- **remove carrier**: subtracts the carrier from the result.

The result is `unwrap`, in fringes.

Carrier
-------

The unwrapped phase still contains the carrier (the tilt of the unperturbed fringes): it is removed here.

- **Phase/2pi**: the unwrapped phase, and an **Offset** subtracted from the result.
- Carrier: the interfringe [px], the angle [˚] and the weight used by the guess of the carrier: the peak of
  |FFT|·r<sup>w</sup> in the region, r the frequency and w the weight (w < 0 prefers the low frequencies, w > 0 the
  high ones). Changing the weight guesses again. With the check box the subtraction is updated at each change of the
  values: adjust them until the phase is flat (or symmetric) where there is no object.
- Reference: subtracts the unwrapped phase of a reference interferogram (same size) instead of the carrier.
- **Subtract**: the result is `No carrier` or `Ref removed`, in fringes.

The phase is then ready for the **Integral inversion** pan (with the phase in fringes).
