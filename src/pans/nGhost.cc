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

// physGhosts

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
	connect(my_w.actionCarrier, SIGNAL(triggered()), this, SLOT(guessCarrier()));
	connect(my_w.actionRect, SIGNAL(triggered()), region, SLOT(togglePadella()));
	connect(my_w.doGhost, SIGNAL(pressed()), this, SLOT(doGhost()));
	connect(my_w.weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(guessCarrier()));
	connect(this, SIGNAL(changeCombo(QComboBox *)), this, SLOT(checkChangeCombo(QComboBox *)));
    
}

void nGhost::checkChangeCombo(QComboBox *combo) {
	if (combo==my_w.shot) {
        imageFFT.resize(0,0);
	}
}

void nGhost::guessCarrier() {
	nPhysD *image=getPhysFromCombo(my_w.shot);
	if (image) {
		QRect geom2=region->getRect();
		nPhysD datamatrix;
		datamatrix = image->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height());

		vec2f vecCarr=phys_guess_carrier(datamatrix, my_w.weightCarrier->value());
		if (vecCarr.first()==0) {
			my_w.statusbar->showMessage(tr("ERROR: Problem finding the carrier"), 5000);
		} else {
			my_w.statusbar->showMessage(tr("Carrier: ")+QString::number(vecCarr.first())+"px "+QString::number(vecCarr.second())+"deg", 5000);
			my_w.widthCarrier->setValue(vecCarr.first());
			my_w.angleCarrier->setValue(vecCarr.second());
		}
	}
}

void nGhost::doGhost () {
	nPhysD *imageShot=getPhysFromCombo(my_w.shot);
	if (imageShot) {
		saveDefaults();
        
        QRect geom=QRect(0,0,imageShot->getW(),imageShot->getH()).intersect(region->getRect());
        
        QTime timer;
        timer.start();

        size_t dx=imageShot->getW();
        size_t dy=imageShot->getH();
        
        if (imageFFT.getSurf() == 0) {
            imageFFT = imageShot->ft2(PHYS_FORWARD);
            xx.resize(dx);
            yy.resize(dy);
            for (size_t i=0;i<dx;i++) xx[i]=(i+(dx+1)/2)%dx-(dx+1)/2; // swap and center
            for (size_t i=0;i<dy;i++) yy[i]=(i+(dy+1)/2)%dy-(dy+1)/2;
            morlet.resize(dx,dy);
        }

        double cr_ghost = cos((my_w.angleCarrier->value()) * _phys_deg); 
        double sr_ghost = sin((my_w.angleCarrier->value()) * _phys_deg);
        double thick_ghost=M_PI;
        double lambda_ghost=my_w.widthCarrier->value()/sqrt(pow(cr_ghost*dx,2)+pow(sr_ghost*dy,2));
        
        double cr_norm = cos((my_w.angleCarrier->value()+my_w.rotation->value()) * _phys_deg); 
        double sr_norm = sin((my_w.angleCarrier->value()+my_w.rotation->value()) * _phys_deg);
        double thick_norm=M_PI/sqrt(pow(sr_norm*dx,2)+pow(cr_norm*dy,2));
        double lambda_norm=my_w.widthCarrier->value()/sqrt(pow(cr_norm*dx,2)+pow(sr_norm*dx,2));
#pragma omp parallel for collapse(2)
        for (size_t x=0;x<dx;x++) {
            for (size_t y=0;y<dy;y++) {
                double xr_ghost = xx[x]*cr_ghost - yy[y]*sr_ghost;
                double yr_ghost = xx[x]*sr_ghost + yy[y]*cr_ghost;
                
                double ex_ghost = -pow(M_PI*(xr_ghost*lambda_ghost-1.0), 2);
                double ey_ghost = -pow(yr_ghost*thick_ghost, 2);
                
                double xr_norm = xx[x]*cr_norm - yy[y]*sr_norm;
                double yr_norm = xx[x]*sr_norm + yy[y]*cr_norm;
                
                double ex_norm = -pow(M_PI*(xr_norm*lambda_norm-1.0), 2);
                double ey_norm = -pow(yr_norm*thick_norm, 2);
                
                double e_tot=exp(ey_norm)*exp(ex_norm) - exp(ey_ghost)*exp(ex_ghost);
                
                morlet.Timg_matrix[y][x]=imageFFT.Timg_matrix[y][x] * e_tot; 

            }
        }

        morlet = morlet.ft2(PHYS_BACKWARD);
        
        nPhysD *deepcopy=new nPhysD(*imageShot);
        deepcopy->setShortName("deghost");
        deepcopy->setName("deghost("+imageShot->getName()+")");
        
        for(int i=geom.left();i<geom.right(); i++) {
            for(int j=geom.top();j<geom.bottom(); j++) {
                double val=deepcopy->point(i,j);
                double valNorm= 2.0*morlet.point(i,j).mod()/(dx*dy)*cos(-morlet.point(i,j).arg());
                deepcopy->set(i,j, val+valNorm); 
            }
        }
        deepcopy->TscanBrightness();
        
        if (my_w.erasePrevious->isChecked()) {
            ghostBusted=nparent->replacePhys(deepcopy,ghostBusted,true);
        } else {
            nparent->addShowPhys(deepcopy);
            ghostBusted=deepcopy;
        }

        my_w.erasePrevious->setEnabled(true);
        QString out;
        out.sprintf("%d msec",timer.elapsed());
        my_w.statusbar->showMessage(out);
        
	}
}


