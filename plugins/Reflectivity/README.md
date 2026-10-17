Reflectivity
============

Ratio between a shot image and a reference image, pixel by pixel:

R = (blur(shot) - offset) / (multiplier * (blur(reference) - offset))

then clipped between min and max.

- **shot**, **reference**: the two images, with the same size. They are divided pixel by pixel: their origins are not used to align them.
- **blur**: radius of the gaussian blur applied to both images before the ratio (0: no blur).
- **offset**: subtracted from both images (e.g. the background level of the camera).
- **multiplier**: multiplies the reference (e.g. a different energy or filter between the shots).
- **min**, **max**: the ratio is clipped to this range.

**erase previous**: the new result replaces the previous one. In this case the result is also updated when a parameter changes; otherwise press the button.
