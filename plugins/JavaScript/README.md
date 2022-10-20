`neu` is the Viewer
`nApp` is the application

Example:
--------

This line:
```
nApp.neus()[0].openPan("Function")
```

is equivalent to the first line of:

```
v=neu.openPan("Function")
v.set("sb_width",1000)
v.set("sb_height",1000)
v.set("function","1000*sin(10*pi*x/width +2*pi*y/1000)")
v.button("doIt")
```


