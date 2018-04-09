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
#include "Ghost_fringes.h"
#include "neutrino.h"

// physGhosts

Ghost_fringes::Ghost_fringes(neutrino *nparent) : nGenericPan(nparent),
  ghostBusted(NULL)
{
	my_w.setupUi(this);

    maskRegion =  new nLine(this,1);
    maskRegion->changeToolTip("MaskLine");
    QPolygonF poly;
    poly << QPointF(50,50) << QPointF(50,150) << QPointF(150,150) << QPointF(150,50);
    maskRegion->setPoints(poly);
    maskRegion->toggleClosedLine(true);


    show();

    connect(my_w.actionCarrier, SIGNAL(triggered()), this, SLOT(guessCarrier()));
    connect(my_w.actionRegion, SIGNAL(triggered()), maskRegion, SLOT(togglePadella()));
	connect(my_w.doGhost, SIGNAL(pressed()), this, SLOT(doGhost()));
    connect(my_w.weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(guessCarrier()));

}

void Ghost_fringes::guessCarrier() {
    nPhysD *image=getPhysFromCombo(my_w.ref);
	if (image) {
        QRect geom2=maskRegion->path().boundingRect().toRect();
		nPhysD datamatrix;
        datamatrix = image->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height());

        vec2f vecCarr=physWave::phys_guess_carrier(datamatrix, my_w.weightCarrier->value());
		if (vecCarr.first()==0) {
			my_w.statusbar->showMessage(tr("ERROR: Problem finding the carrier"), 5000);
		} else {
			my_w.widthCarrier->setValue(vecCarr.first());
			my_w.angleCarrier->setValue(vecCarr.second());
		}
	}
}

void Ghost_fringes::doGhost () {
	nPhysD *imageShot=getPhysFromCombo(my_w.shot);
    if (imageShot) {
        saveDefaults();

        QTime timer;
        timer.start();

        size_t dx=imageShot->getW();
        size_t dy=imageShot->getH();
        
        nPhysC imageFFT = imageShot->ft2(PHYS_FORWARD);
        std::vector<int> xx(dx), yy(dy);

        for (size_t i=0;i<dx;i++)
            xx[i]=(i+(dx+1)/2)%dx-(dx+1)/2; // swap and center
        for (size_t i=0;i<dy;i++)
            yy[i]=(i+(dy+1)/2)%dy-(dy+1)/2;

        double cr = cos((my_w.angleCarrier->value()) * _phys_deg);
        double sr = sin((my_w.angleCarrier->value()) * _phys_deg);

        double lambda=sqrt(pow(cr*dx,2)+pow(sr*dy,2))/(M_PI*my_w.widthCarrier->value());

        DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << lambda);


        for (size_t x=0;x<dx;x++) {
            for (size_t y=0;y<dy;y++) {
                double xr = xx[x]*cr - yy[y]*sr;
                double yr = xx[x]*sr + yy[y]*cr;
                double e_tot = 1.0-exp(-pow(yr,2))/(1.0+exp(lambda-std::abs(xr)));
                imageFFT.set(x,y,imageFFT.point(x,y) * e_tot);
            }
        }

        imageFFT = imageFFT.ft2(PHYS_BACKWARD);

        nPhysD *deepcopy=new nPhysD(*imageShot);
        deepcopy->setShortName("deghost");
        deepcopy->setName("deghost("+imageShot->getName()+")");
        
        QRect geom=maskRegion->path().boundingRect().toRect();



        QPolygonF regionPoly=maskRegion->poly(1);
        regionPoly=regionPoly.translated(imageShot->get_origin().x(),imageShot->get_origin().y());
        std::vector<vec2f> vecPoints(regionPoly.size());
        for(int k=0;k<regionPoly.size();k++) {
            vecPoints[k]=vec2f(regionPoly[k].x(),regionPoly[k].y());
        }

        for(int i=geom.left();i<geom.right(); i++) {
            for(int j=geom.top();j<geom.bottom(); j++) {
                vec2f pp(i,j);
                if (point_inside_poly(pp,vecPoints)==true) {
                    deepcopy->set(i,j, imageFFT.point(i,j).mod()/(dx*dy));
                }
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
        out.sprintf("Time: %d msec",timer.elapsed());
        my_w.statusbar->showMessage(out);

	}
}


