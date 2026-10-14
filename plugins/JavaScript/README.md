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



Alternatively, you can put the name of a file in the "command" section to run it.

Output and long scripts:
------------------------

`print(...)` (and `console.log(...)`) writes its arguments in the output box; the value of the last expression is
written at the end. Errors are reported with their line number.

A script runs in the GUI thread: the Stop button can interrupt it only while the script gives control back to
the interface, which happens at every `print(...)` or with an explicit `processEvents()` call. Long loops should
call one of them from time to time:

```
for (i=0; i<1000; i++) {
    // ... work ...
    processEvents()
}
```
