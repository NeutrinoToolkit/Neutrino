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
#include "nRotate.h"
#include "neutrino.h"

// physWavelets

nRotate::nRotate(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname)
{
	my_w.setupUi(this);

	decorate();

	connect(my_w.valueAngle, SIGNAL(valueChanged(double)), this, SLOT(doRotateLive()));
	connect(my_w.doIt,SIGNAL(pressed()),this,SLOT(doRotate()));
	connect(my_w.image,SIGNAL(activated(int)),this,SLOT(doRotateLive()));
	rotated=NULL;
	doRotateLive();
}

void nRotate::doRotateLive () {
	double alpha=my_w.valueAngle->value();
	nPhysD *image=getPhysFromCombo(my_w.image);
	if (image) {
		if (image!=rotated) {
			rotated=nparent->replacePhys(image->rotated(alpha),rotated, true);
		} else {
			my_w.statusbar->showMessage("Can't work on this image",5000);
		}
	}
}

void nRotate::doRotate () {
	doRotateLive();
	if (rotated) {
		nPhysD *newRotated=new nPhysD(*rotated);
		nparent->addPhys(newRotated);
	}
}
