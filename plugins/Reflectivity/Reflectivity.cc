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
    setupUi(this);


    connect(doItB, SIGNAL(pressed()), this, SLOT(doIt()));
    connect(offset, SIGNAL(valueChanged(double)), this, SLOT(doIt()));
    connect(multiplier, SIGNAL(valueChanged(double)), this, SLOT(doIt()));
    connect(min_val, SIGNAL(valueChanged(double)), this, SLOT(doIt()));
    connect(max_val, SIGNAL(valueChanged(double)), this, SLOT(doIt()));
    connect(blur, SIGNAL(valueChanged(int)), this, SLOT(doIt()));
    show();

}

void Reflectivity::doIt () {
    nPhysD *imageShot=getPhysFromCombo(shot);
    nPhysD *imageRef=getPhysFromCombo(ref);
    if (imageShot && imageRef) {
        saveDefaults();

        nPhysD *shot=new nPhysD(*imageShot);
        physMath::phys_fast_gaussian_blur(*shot,blur->value());
        physMath::phys_subtract(*shot,offset->value());

        nPhysD ref(*imageRef);
        physMath::phys_fast_gaussian_blur(ref,blur->value());
        physMath::phys_subtract(ref,offset->value());
        physMath::phys_multiply(ref,multiplier->value());

        physMath::phys_point_divide(*shot,ref);

        physMath::cutoff(*shot,min_val->value(),max_val->value());

        shot->prop["display_range"]=shot->get_min_max();

        if (erasePrevious->isChecked()) {
            Refle=nparent->replacePhys(shot,Refle,true);
        } else {
            nparent->addShowPhys(shot);
            Refle=shot;
        }
        erasePrevious->setEnabled(true);
    }
}


