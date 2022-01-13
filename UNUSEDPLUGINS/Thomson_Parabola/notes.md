
# notes on TP

## transverse tracking

Having tracked the particle trajectories, it is necessary to integrate the transversal extent of the
track (and to integrate it..). This is necessary for cases when the track divergence changes with
energy (which is true only in special cases, like with gas diffusion or heavy ions/molecules)

1. find maximum

2. find boundaries, given threshold


## associate nLine to a given line in the table (not to be forced to redo the entire calculation
upon change)

This is trickier; probably trying with deriving some of the classes within a tab widget (eg: cell
widgets) or adding a property

## GUI

* connect table modification signal to updateTracks 
* lock nLines
 
