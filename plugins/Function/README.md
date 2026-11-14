Creates new images from a formula of the pixel coordinates and of the open images, e.g. to combine or correct images,
or to make synthetic ones. The formulas use the [ExprTk](https://www.partow.net/programming/exprtk/) syntax.

Use
---

1. Set the size of the new image (**w**, **h** in the toolbar), or copy it from the shown image (**Copy Size**).
2. Write the formula: it is computed for each pixel, `x` from 0 to w-1 and `y` from 0 to h-1.
3. **Create** (or Ctrl+Return / ⌘+Return in the text). The image is named `Function` and shown.

With **Erase previous** the new images replace those of the previous run (only if the formula is correct: an error
keeps them). On an error the status bar shows the message and the cursor goes to it.

Several images: separate their formulas with a line containing only `<br>`.

Variables and functions
-----------------------

- `x`, `y`: the coordinates of the pixel (in pixels, from the top left corner, without the origin of the images).
- `width`, `height`: the size of the new image.
- `phys(x,y)`: the value of the shown image.
- `physN(x,y)`: the value of image number N, in the order of the image list (starting from 0). N can be written plain
  (`phys3`) or zero-padded to the number of digits of the last image (with 11 images `phys03` ... `phys10`).
- `n_phys(N,x,y)`: the value of image number N, with N a variable (e.g. in a loop).
- `num_phys`: the number of open images when **Create** is clicked (including the results of a previous run, if they
  are still open).
- `rand(m)`: a random number between 0 and m.
- `pi`, `epsilon`, `inf`.

The coordinates can be non-integer (the value is interpolated between the pixels). Outside the image, or for an image
that does not exist, the value is NaN.

Examples
--------

The shown image minus its value at the pixel (100,100):

```
phys(x,y)-phys(100,100)
```

The difference of the first two images:

```
phys0(x,y)-phys1(x,y)
```

The image shifted by 10.5 pixels to the right (the pixels with nothing to their left are NaN):

```
phys(x-10.5,y)
```

The average of all the images (several statements separated by `;`: the value of the last one is the result):

```
var s:=0;
for (var i:=0; i<num_phys; i+=1) {
    s+=n_phys(i,x,y);
};
s/num_phys
```

A mask: 1 where the shown image is above 100, 0 elsewhere:

```
phys(x,y)>100 ? 1 : 0
```

Three interferograms (a carrier, a tilted carrier, and their sum) in a super-gaussian window:

```
exp(-(2.5*(x-width/2)/width)^8)*exp(-(2.5*(y-height/2)/height)^8)*cos(pi*50*x/width)
<br>
exp(-(2.5*(x-width/2)/width)^8)*exp(-(2.5*(y-height/2)/height)^8)*cos(pi*50*x/width+10*pi*y/height)
<br>
exp(-(2.5*(x-width/2)/width)^8)*exp(-(2.5*(y-height/2)/height)^8)*(cos(pi*50*x/width+10*pi*y/height)+0.3*cos(pi*50*x/width))
```

The syntax
----------

- Operators: `+ - * / % ^`, comparisons `== != < <= > >=`, logical `and or not xor`.
- Functions: `abs sgn sqrt exp log log10 pow min max clamp floor ceil round trunc frac erf`, `sin cos tan asin acos
  atan atan2 sinh cosh tanh hypot`, `deg2rad rad2deg`, and the others of ExprTk.
- Conditions: `if (c, a, b)`, `c ? a : b`, `if (c) { ... } else { ... }`, `switch`.
- Variables `var a:=1;`, assignments `:= += -= *= /=`, loops `for`, `while`, `repeat ... until`.
- Comments: `#` or `//` to the end of the line, `/* ... */`.
