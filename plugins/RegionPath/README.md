RegionPath
==========

Keeps the part of an image inside a region (the closed line drawn on the image) and replaces the rest.

- **image**: the image to process.
- **value**: what replaces the pixels: NaN, the minimum, the maximum, the mid-range (min+max)/2, zero, or a typed value.
- **inverse**: replace the inside of the region instead of the outside.
- **crop**: the result is cropped to the rectangle around the region (inside the image).
- **erase previous**: the new result replaces the previous one.

A pixel belongs to the region when its centre is inside the line. The toolbar edits the region (points, bezier).
