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

            // 1. allocation
            fftw_complex *t = fftw_alloc_complex(dx*(dy/2+1));

            fftw_plan plan_t_fw = fftw_plan_dft_r2c_2d(dx, dy, imageShot->Timg_buffer, t, FFTW_ESTIMATE);

            fftw_execute(plan_t_fw);
            fftw_destroy_plan(plan_t_fw);

            for (size_t x=1;x<dx;x++) {
                t[x][0]=0.0;
                t[x][1]=0.0;
            }

            fftw_plan plan_t_bw = fftw_plan_dft_c2r_2d(dx, dy, t, deghosted.Timg_buffer, FFTW_ESTIMATE);
            fftw_execute(plan_t_bw);
            fftw_destroy_plan(plan_t_bw);
            fftw_free(t);

            phys_divide(deghosted,dx*dy);
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


