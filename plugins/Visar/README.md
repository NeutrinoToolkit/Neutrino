<!-- 
Visar analysis
Tommaso Vinci 
tommaso.vinci@polytechnique.edu
 -->

VISAR analysis
==============

In the Window there are 3 tabs on the left : *Settings*, *Velocity* and
*SOP*

Settings :
----------

Select the *Visar 1* or *Visar 2* tab to set the initial parameters of
each VISAR:

  * Select the *Reference* and the *Shot* images
  * With your mouse, select the ROI (the phase is taken from the middle
    of the ROI, while the intensity is averaged over the entire ROI)
  * Get the *Carrier* (orientation and interfringe): hit on the button <img src="../../resources/icons/refresh2.png" width="20" />
    (adjusting the *weight* if the values are not correct, to remove
    either the lower frequencies or the higher frequencies 

    
  * set the *Time resolution* parameter (in px) is corresponding to the
    width of the Gaussian part of the Morlet function in the direction
    of the fringes, i.e. the time direction.
  * Fit the interference fringes by hitting <img src="../../resources/icons/refresh.png" width="20" />
  * Adjust the *intensity parameters*: 
      o adjust the background *offset* of the ref and of the shot. To
        estimate its value, read on the image directly on an appropriate
        area (long time after the shock for example or on the side where
        fringes are usually not present).
      o adjust the *Pixel delay* between the reference and the shot.
      o adjust the *Refence multiplication factor* to have the ref
        (dashed line) and the shot (full line) at the same level, to
        correct from the shot-to-shot fluctuations of the probe laser
        energy.


Velocity:
---------

Select the *Visar 1* or *Visar 2* tab to set the velocity parameters of
each VISAR:

 1. For each VISAR, specify the sensitivity (Velocity/fringe)
 2. Specify the Time 0 window
     1. Pixel position of the synchronization (see synchronization shot)
     2. Delay when compared with the initial synchronization shot (if
        changed) 
 3. **Sweep Time** is the temporal calibration of the streak camera. The is
    a list of values as defined by th Hamamtsu camera (i.e $T(X)=A_1
    x+A_2 X^2 /2+A_3 X^3 /3+...+A_n X^n /n$). One value is linear response.
 4. The offset window is corresponding to a fine adjustment between the
    reference and the shot. This is adjusted automatically. The
    automatic value is displayed. If needed (when the value of the
    velocity at negative time is not at 0), an additional offset can be
    set manually.
 5. The Reflectivity window allows to precise the real value of the
    reflectivity.
     1. To set the final reflectivity at long time delay, after the
        shock, in order to obtain 0
     2. To set the real value of the reflectivity of the used material
 6. To correct the jumps of fringes In case of a jump, both Visars
    won’t have the same velocity. Here the objective is to apply a
    correction and obtain the 2 same velocities.
     1. first look manually to find the number of jumps of each VISAR.
        After this manual look, set this parameter to 0
     2. Three parameters need to be adjusted:
         1. t = time for the beginning of the velocity change (in pixel).
         2. i = number of jumps to correct for (defined by step 6.1)
         3. [n0] = index of refraction. This is optional.
     3. This set of parameters is set for each jump. As an example
        `0 3; 109 5 1.7` in case of a first change of velocity at the
        time `0` with a jump of `3` followed by a second change of velocity
        at the time `3.5` with a jump of `5` in a medium with refractive
        index of `1.7`.


Temperaure:
-----------

 1. Select *Reference* and *Shot* images. In case only shot image is
    present use the same on bot combo-box and use the *background*
    spin-box to adjust the background level
 2. set the pixel *time zero  *and the delay *delta t*
 3. Choose the *refelectiviy*:
     1. *V1*: Visar1
     2. *V2*: Visar2
     3. *Mean* of Visar1 and Visar2
     4. *Zero*: no reflectivity (Blackbody temperature)
 4. **Sweep Time**: conversion px to time (same as above)
 5. Direction: **H**orizontal or **V**ertical
 6. SOP calibration: **$T_0$** and **A** : calibration values ($T_0$ is
    the color temperature of the light collected by optical system and
    "A" is the efficiency of the streak).
 7. *min max* : restrain the temperature scale (right axis) between
    minimum and maximum


