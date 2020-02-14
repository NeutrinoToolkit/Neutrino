/*
 *
 *    Copyright (C) 2013 Alessandro Flacco, Tommaso Vinci All Rights Reserved
 *
 *    This file is part of neutrino.
 *
 *    Neutrino is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU Lesser General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    Neutrino is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public License
 *    along with neutrino.  If not, see <http://www.gnu.org/licenses/>.
 *
 *    Contact Information:
 *	Alessandro Flacco <alessandro.flacco@polytechnique.edu>
 *	Tommaso Vinci <tommaso.vinci@polytechnique.edu>
 *
 */
#include "Reflectivity.h"
#include "neutrino.h"

// physGhosts

Reflectivity::Reflectivity(neutrino *nparent) : nGenericPan(nparent)
{
    my_w.setupUi(this);


    connect(my_w.doIt, SIGNAL(pressed()), this, SLOT(doIt()));
    connect(my_w.offset, SIGNAL(valueChanged(double)), this, SLOT(doIt()));
    connect(my_w.multiplier, SIGNAL(valueChanged(double)), this, SLOT(doIt()));
    connect(my_w.min_val, SIGNAL(valueChanged(double)), this, SLOT(doIt()));
    connect(my_w.max_val, SIGNAL(valueChanged(double)), this, SLOT(doIt()));
    connect(my_w.blur, SIGNAL(valueChanged(int)), this, SLOT(doIt()));
    show();

}

void Reflectivity::doIt () {
    nPhysD *imageShot=getPhysFromCombo(my_w.shot);
    nPhysD *imageRef=getPhysFromCombo(my_w.ref);
    if (imageShot && imageRef) {
        saveDefaults();

        nPhysD *shot=new nPhysD(*imageShot);
        physMath::phys_fast_gaussian_blur(*shot,my_w.blur->value());
        physMath::phys_subtract(*shot,my_w.offset->value());

        nPhysD ref(*imageRef);
        physMath::phys_fast_gaussian_blur(ref,my_w.blur->value());
        physMath::phys_subtract(ref,my_w.offset->value());
        physMath::phys_multiply(ref,my_w.multiplier->value());

        physMath::phys_point_divide(*shot,ref);

        physMath::cutoff(*shot,my_w.min_val->value(),my_w.max_val->value());

        shot->prop["display_range"]=shot->get_min_max();

        if (my_w.erasePrevious->isChecked()) {
            Refle=nparent->replacePhys(shot,Refle,true);
        } else {
            nparent->addShowPhys(shot);
            Refle=shot;
        }
        my_w.erasePrevious->setEnabled(true);
    }
}


