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
#include "Rotate.h"
#include "neutrino.h"

Rotate::Rotate(neutrino *nparent) : nGenericPan(nparent)
{
    setupUi(this);

    show();

    // connect(valueAngle, SIGNAL(valueChanged(double)), this, SLOT(doRotateLive()));
    // connect(image,SIGNAL(activated(int)),this,SLOT(doRotateLive()));
    // (every step of the angle rotated again, and without "erase previous" added a new image;
    // the fill value and "same size" were not used until the angle changed)
    // live (300 ms after the last change) with "erase previous", otherwise the angle on Enter
    updateTimer.setSingleShot(true);
    updateTimer.setInterval(300);
    connect(&updateTimer, SIGNAL(timeout()), this, SLOT(doRotateLive()));
    connect(valueAngle, SIGNAL(valueChanged(double)), this, SLOT(angleChanged()));
    connect(valueAngle, SIGNAL(editingFinished()), this, SLOT(doRotateLive()));
    connect(image,SIGNAL(activated(int)),this,SLOT(optionChanged()));
    connect(defaultValue,SIGNAL(currentIndexChanged(int)),this,SLOT(optionChanged()));
    connect(sameSize,SIGNAL(toggled(bool)),this,SLOT(optionChanged()));
    rotated=nullptr;
	// doRotateLive();
	// (at 0 degrees opening the plugin added a copy of the image)
	if (valueAngle->value()!=0.0) doRotateLive();
}

void Rotate::angleChanged() {
    if (erasePrevious->isChecked()) updateTimer.start();
}

void Rotate::optionChanged() {
    updateTimer.start();
}

void Rotate::doRotateLive () {
    updateTimer.stop();
    double alpha=valueAngle->value();
    nPhysD *my_phys=getPhysFromCombo(image);
    if (my_phys) {
        // same parameters as the result still shown: nothing to do (e.g. Enter without changes)
        if (nparent->nPhysExists(rotated) && my_phys==lastSource && alpha==lastAngle &&
            defaultValue->currentIndex()==lastFill && sameSize->isChecked()==lastSameSize) {
            return;
        }
        if (my_phys!=rotated) {
            nPhysD *my_rot;
            if (sameSize->isChecked()) {
                my_rot=new nPhysD(my_phys->fast_rotated(alpha,getReplaceVal(my_phys)));
            } else {
                my_rot=new nPhysD(my_phys->rotated(alpha,getReplaceVal(my_phys)));
            }
            erasePrevious->setEnabled(true);
            if (erasePrevious->isChecked()) {
                rotated=nparent->replacePhys(my_rot,rotated, true);
            } else {
                nparent->addShowPhys(my_rot);
                rotated=my_rot;
            }
            lastSource=my_phys;
            lastAngle=alpha;
            lastFill=defaultValue->currentIndex();
            lastSameSize=sameSize->isChecked();
        } else {
            statusbar->showMessage("Can't work on this image",5000);
		}
	}
}

double Rotate::getReplaceVal(nPhysD* image) {
	double val=0.0;
    switch (defaultValue->currentIndex()) {
        case 0:
            val=std::numeric_limits<double>::quiet_NaN();
            break;
        case 1:
            val=image->get_min();
            break;
        case 2:
            val=image->get_max();
            break;
        case 3:
            val=0.5*(image->get_min()+image->get_max());
            break;
        case 4:
            val=0.0;
            break;
        default:
            WARNING("something is broken here");
            break;
    }
    return val;
}
