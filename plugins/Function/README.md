We implemented these functions: 
 
 - `phys(x,y)` will return the current image value of pixel `(x,y)`
 - `physN(x,y)` will return the value of pixel `(x,y)` of the Nth image (where N is a number)
 - `n_phys(N,x,y)` will return the value of pixel `(x,y)` of the Nth image
 
There are also 2 values:

 - `width`    : width of the new image
 - `height`   : height of the new image
 - `num_phys` : number of images in the stack (this might contain also a previous function image)
 
It is possible to create multiple images by separing them with a `<br>`

Examples:
---------

 * the actual image divided by the value of the pixel (100,100):

```
phys(x,y)-phys(100,100)
```

 * same as above but of the first image:

```
phys0(x,y)-phys0(100,100)
```

 * average of all images (the last line is the return value of the function):

```
var my_sum:=0;
for(var i:=0; i<num_phys; i+=1) {
my_sum+=n_phys(i,x,y)/num_phys;
}
```




# Features

 * Mathematical operators (+, -, *, /, %, ^)
 
 * Functions (min, max, avg, sum, abs, ceil, floor, round, roundn, exp, log, log10, logn, pow, root, sqrt, clamp, inrange, swap)
 
 * Trigonometry (sin, cos, tan, acos, asin, atan, atan2, cosh, cot, csc, sec, sinh, tanh, d2r, r2d, d2g, g2d, hyp)
 
 * Equalities & Inequalities (=, ==, <>, !=, <, <=, >, >=)
 
 * Assignment (:=, +=, -=, *=, /=, %=)
 
 * Logical operators (and, nand, nor, not, or, xor, xnor, mand, mor)
 
 * Control structures (if-then-else, ternary conditional, switch case, return-statement)
 
 * Loop structures (while loop, for loop, repeat until loop, break, continue)
 
 * Optimization of expressions (constant folding, strength reduction, operator coupling, special functions and dead code elimination)
 
 * String operations (equalities, inequalities, logical operators, concatenation and sub-ranges)
 
 * Expression local variables, vectors and strings
 
 * User defined variables, vectors, strings, constants and function support
 
 * Multivariate function composition
 
 * Multiple sequence point and sub expression support
 
 * Numeric integration and differentiation
 
 * Vector Processing: BLAS-L1 (axpy, axpby, axpb), all/any-true/false, count, rotate-left/right, shift-left/right, sort, nth_element, iota, sum, kahan-sum, dot-product, copy

