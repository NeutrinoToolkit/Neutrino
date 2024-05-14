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
    ghostBusted(nullptr)
  , filter(nullptr), spectrum(nullptr)
{
    setupUi(this);

    maskRegion =  new nLine(this,1);
    maskRegion->changeToolTip("MaskLine");
    QPolygonF poly;
    poly << QPointF(50,50) << QPointF(50,150) << QPointF(150,150) << QPointF(150,50);
    maskRegion->setPoints(poly);
    maskRegion->toggleClosedLine(true);


    show();

    connect(actionCarrier, SIGNAL(triggered()), this, SLOT(guessCarrier()));
    connect(actionRegion, SIGNAL(triggered()), maskRegion, SLOT(togglePadella()));
    connect(doGhostB, SIGNAL(pressed()), this, SLOT(doGhost()));
    connect(weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(guessCarrier()));

}

void Ghost_fringes::guessCarrier() {
    nPhysD *image=getPhysFromCombo(ref);
	if (image) {
        QRect geom2=maskRegion->path().boundingRect().toRect();
		nPhysD datamatrix;
        datamatrix = image->sub(geom2);

        std::vector<vec2f> vecCarr=physWave::phys_guess_carrier(datamatrix, weightCarrier->value());
        if (vecCarr.size()==0) {
            statusbar->showMessage(tr("ERROR: Problem finding the carrier"), 5000);
		} else {
            widthCarrier->setValue(vecCarr[0].first());
            angleCarrier->setValue(vecCarr[0].second());
		}
	}
}

void Ghost_fringes::doGhost () {
    nPhysD *imageShot=getPhysFromCombo(shot);
    if (imageShot) {
        saveDefaults();

        QElapsedTimer timer;
        timer.start();

        unsigned int dx=imageShot->getW();
        unsigned int dy=imageShot->getH();
        
        physC imageFFT = imageShot->ft2(PHYS_FORWARD);
        std::vector<int> xx(dx), yy(dy);

        for (unsigned int i=0;i<dx;i++)
            xx[i]=(i+(dx+1)/2)%dx-(dx+1)/2; // swap and center
        for (unsigned int i=0;i<dy;i++)
            yy[i]=(i+(dy+1)/2)%dy-(dy+1)/2;

        double cr = cos((angleCarrier->value()) * _phys_deg);
        double sr = sin((angleCarrier->value()) * _phys_deg);

        // double lambda=sqrt(pow(cr*dx,2)+pow(sr*dy,2))/(M_PI*widthCarrier->value());
        double thick_norm= resolution->value()/M_PI;
        double lambda_norm=M_PI*widthCarrier->value()/sqrt(pow(cr*dx,2)+pow(sr*dy,2));
        // DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " << lambda_norm << " " << thick_norm << " : " << sqrt(pow(cr*dx,2)+pow(sr*dy,2)));

        nPhysD *myfilter=new nPhysD(dx,dy,0.0,"Filter");
        nPhysD *myspectrum=new nPhysD(dx,dy,0.0,"Spectrum");
        for (unsigned int x=0;x<dx;x++) {
            for (unsigned int y=0;y<dy;y++) {
                double xr = xx[x]*cr - yy[y]*sr;
                double yr = xx[x]*sr + yy[y]*cr;
                // double e_tot = 1.0-exp(-pow(yr,2)/lambda)/(1.0+exp(lambda-std::abs(xr)));
                double e_tot = 1.0-exp(-pow(yr/thick_norm,2))*exp(-pow(std::abs(xr)*lambda_norm-M_PI, 2));
                myfilter->set(x,y,e_tot);
                myspectrum->set(x,y,imageFFT.point(x,y).mod());
                imageFFT.set(x,y,imageFFT.point(x,y) * e_tot);
            }
        }
        myfilter->TscanBrightness();
        myspectrum->TscanBrightness();
        myfilter->fftshift();
        myspectrum->fftshift();

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
//                    deepcopy->set(i,j, imageFFT.point(i,j).real()/(dx*dy));
                }
            }
        }
        deepcopy->TscanBrightness();
        
        if (erasePrevious->isChecked()) {
            filter=nparent->replacePhys(myfilter,filter,true);
            spectrum=nparent->replacePhys(myspectrum,spectrum,true);
            ghostBusted=nparent->replacePhys(deepcopy,ghostBusted,true);
        } else {
            filter=myfilter;
            nparent->addShowPhys(filter);
            spectrum=myspectrum;
            nparent->addShowPhys(spectrum);
            ghostBusted=deepcopy;
            nparent->addShowPhys(ghostBusted);
        }

        erasePrevious->setEnabled(true);
        statusbar->showMessage(QString(tr("Time: %1 msec")).arg(timer.elapsed()));

	}
}


