We implemented these functions: 
 
 - `phys(x,y)` will return the current image value of pixel `(x,y)`
 - `physN(x,y)` will return the value of pixel `(x,y)` of the Nth image (where N is a number)
 - `n_phys(N,x,y)` will return the value of pixel `(x,y)` of the Nth image
 
There are also 2 values:

 - `width`    : width of the new image
 - `height`   : height of the new image
 - `num_phys` : number of images in the stack (this might contain also a previous function image)

Examples:
---------

the actual image divided by the value of the pixel (100,100):

```
phys(x,y)-phys(100,100)
```

same as above but of the first image:

```
phys0(x,y)-phys0(100,100)
```

average of all images (the last line is the return value of the function):

```
var my_sum:=0;
for(var i:=0; i<num_phys; i+=1) {
my_sum+=n_phys(i,x,y)/num_phys;
}
```
