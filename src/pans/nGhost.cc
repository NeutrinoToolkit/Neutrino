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
#include "nGhost.h"
#include "neutrino.h"

nGhost::nGhost(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname), ghostBusted(NULL)
{
	my_w.setupUi(this);

	region =  new nRect(nparent);
	region->setParentPan(panName,1);
	region->setRect(QRectF(100,100,100,100));
	
	decorate();

	connect(my_w.actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
	connect(my_w.actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));
    connect(my_w.actionRect, SIGNAL(triggered()), region, SLOT(togglePadella()));
    connect(this, SIGNAL(changeCombo(QComboBox *)), this, SLOT(checkChangeCombo(QComboBox *)));
    connect(region, SIGNAL(sceneChanged()), this, SLOT(doGhost()));

    checkChangeCombo(my_w.shot);
}

void nGhost::checkChangeCombo(QComboBox *combo) {
	if (combo==my_w.shot) {
        nPhysD *imageShot=getPhysFromCombo(my_w.shot);

        if (imageShot) {
            size_t dx=imageShot->getW();
            size_t dy=imageShot->getH();
            deghosted.resize(dx,dy);

            nPhysC imageFFT = imageShot->ft2(PHYS_FORWARD);
            for (size_t x=1;x<dx;x++) {
                imageFFT.Timg_matrix[0][x]=0.0;
            }
            imageFFT = imageFFT.ft2(PHYS_BACKWARD);
            for (size_t i=0;i<imageFFT.getSurf();i++) {
                deghosted.Timg_buffer[i]=(imageFFT.Timg_buffer[i]).real()/imageFFT.getSurf();
            }
            doGhost();
        }
    }
}

void nGhost::doGhost () {
	nPhysD *imageShot=getPhysFromCombo(my_w.shot);
	if (imageShot) {
		saveDefaults();
        
        QRect geom=QRect(0,0,imageShot->getW(),imageShot->getH()).intersected(region->getRect());
        
        nPhysD *deepcopy=new nPhysD(*imageShot);
        deepcopy->setShortName("deghost");
        deepcopy->setName("deghost("+imageShot->getName()+")");
        
        for(int i=geom.left();i<geom.right(); i++) {
            for(int j=geom.top();j<geom.bottom(); j++) {
                deepcopy->set(i,j, deghosted.point(i,j));
            }
        }
        deepcopy->TscanBrightness();
        
        if (!my_w.erasePrevious->isChecked()) {
            ghostBusted=NULL;
        }
        ghostBusted=nparent->replacePhys(deepcopy,ghostBusted,true);

        my_w.erasePrevious->setEnabled(true);        
	}
}


