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
    connect(doGhostB, SIGNAL(clicked()), this, SLOT(doGhost()));
    // the result was closed by the user
    connect(nparent, &neutrino::physDel, this, [this](nPhysD *buf) { if (buf==ghostBusted) ghostBusted=nullptr; });
    connect(weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(guessCarrier()));

}

void Ghost_fringes::guessCarrier() {
    nPhysD *image=getPhysFromCombo(ref);
	if (image) {
        // mask region in the pixels of this image: line points are relative to the image origin
        QRect geom2=maskRegion->poly(1).translated(image->get_origin().x(),image->get_origin().y()).boundingRect().toRect().intersected(QRect(0,0,image->getW(),image->getH()));
        if (geom2.width()<2 || geom2.height()<2) {
            statusbar->showMessage(tr("The mask region is outside the image"), 5000);
            return;
        }
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

// void Ghost_fringes::doGhost () {
//     nPhysD *imageShot=getPhysFromCombo(shot);
//     if (imageShot) {
//         saveDefaults();

//         QElapsedTimer timer;
//         timer.start();

//         unsigned int dx=imageShot->getW();
//         unsigned int dy=imageShot->getH();
        
//         physC imageFFT = imageShot->ft2(PHYS_FORWARD);
//         std::vector<int> xx(dx), yy(dy);

//         for (unsigned int i=0;i<dx;i++)
//             xx[i]=(i+(dx+1)/2)%dx-(dx+1)/2; // swap and center
//         for (unsigned int i=0;i<dy;i++)
//             yy[i]=(i+(dy+1)/2)%dy-(dy+1)/2;

//         double cr = cos((angleCarrier->value()) * _phys_deg);
//         double sr = sin((angleCarrier->value()) * _phys_deg);

//         // double lambda=sqrt(pow(cr*dx,2)+pow(sr*dy,2))/(M_PI*widthCarrier->value());
//         double thick_norm= resolution->value()/M_PI;
//         double lambda_norm=M_PI*widthCarrier->value()/sqrt(pow(cr*dx,2)+pow(sr*dy,2));
//         // DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " << lambda_norm << " " << thick_norm << " : " << sqrt(pow(cr*dx,2)+pow(sr*dy,2)));

//         nPhysD *myfilter=new nPhysD(dx,dy,0.0,"Filter");
//         nPhysD *myspectrum=new nPhysD(dx,dy,0.0,"Spectrum");
//         for (unsigned int x=0;x<dx;x++) {
//             for (unsigned int y=0;y<dy;y++) {
//                 double xr = xx[x]*cr - yy[y]*sr;
//                 double yr = xx[x]*sr + yy[y]*cr;
//                 // double e_tot = 1.0-exp(-pow(yr,2)/lambda)/(1.0+exp(lambda-std::abs(xr)));
//                 double e_tot = 1.0-exp(-pow(yr/thick_norm,2))*exp(-pow(std::abs(xr)*lambda_norm-M_PI, 2));
//                 myfilter->set(x,y,e_tot);
//                 myspectrum->set(x,y,imageFFT.point(x,y).mod());
//                 imageFFT.set(x,y,imageFFT.point(x,y) * e_tot);
//             }
//         }
//         myfilter->TscanBrightness();
//         myspectrum->TscanBrightness();
//         myfilter->fftshift();
//         myspectrum->fftshift();

//         imageFFT = imageFFT.ft2(PHYS_BACKWARD);

//         nPhysD *deepcopy=new nPhysD(*imageShot);
//         deepcopy->setShortName("deghost");
//         deepcopy->setName("deghost("+imageShot->getName()+")");
        
//         QRect geom=maskRegion->path().boundingRect().toRect();



//         QPolygonF regionPoly=maskRegion->poly(1);
//         regionPoly=regionPoly.translated(imageShot->get_origin().x(),imageShot->get_origin().y());
//         std::vector<vec2f> vecPoints(regionPoly.size());
//         for(int k=0;k<regionPoly.size();k++) {
//             vecPoints[k]=vec2f(regionPoly[k].x(),regionPoly[k].y());
//         }

//         for(int i=geom.left();i<geom.right(); i++) {
//             for(int j=geom.top();j<geom.bottom(); j++) {
//                 vec2f pp(i,j);
//                 if (point_inside_poly(pp,vecPoints)==true) {
//                     deepcopy->set(i,j, imageFFT.point(i,j).mod()/(dx*dy));
// //                    deepcopy->set(i,j, imageFFT.point(i,j).real()/(dx*dy));
//                 }
//             }
//         }
//         deepcopy->TscanBrightness();
        
//         if (erasePrevious->isChecked()) {
//             filter=nparent->replacePhys(myfilter,filter,true);
//             spectrum=nparent->replacePhys(myspectrum,spectrum,true);
//             ghostBusted=nparent->replacePhys(deepcopy,ghostBusted,true);
//         } else {
//             filter=myfilter;
//             nparent->addShowPhys(filter);
//             spectrum=myspectrum;
//             nparent->addShowPhys(spectrum);
//             ghostBusted=deepcopy;
//             nparent->addShowPhys(ghostBusted);
//         }

//         erasePrevious->setEnabled(true);
//         statusbar->showMessage(QString(tr("Time: %1 msec")).arg(timer.elapsed()));

// 	}
// }

void Ghost_fringes::doGhost () {
    nPhysD *imageShot=getPhysFromCombo(shot);
    if (imageShot) {
        saveDefaults();

        QElapsedTimer timer;
        timer.start();
        size_t dx=imageShot->getW();
        size_t dy=imageShot->getH();

        physC imageFFT = imageShot->ft2(PHYS_FORWARD);
        std::vector<int> xx(dx), yy(dy);

        for (size_t i=0;i<dx;i++)
            xx[i]=(i+(dx+1)/2)%dx-(dx+1)/2; // swap and center
        for (size_t i=0;i<dy;i++)
            yy[i]=(i+(dy+1)/2)%dy-(dy+1)/2;

        double cr = cos((angleCarrier->value()) * _phys_deg);
        double sr = sin((angleCarrier->value()) * _phys_deg);

        // the filter is built on true spatial frequencies (cycles/px) before rotating:
        // rotating the FFT indices directly misplaces the notch on non-square images (unless angle is 0 or 90)
        double wc=widthCarrier->value();
        double res=resolution->value();
        double depth=parameter->value();
        double sharpness=sqrt(pow(cr*dx,2)+pow(sr*dy,2)); // model 1 edge, in frequency bins along the carrier

        // DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << wc);

        const bool model1 = filter_model->currentText()=="1";
        const bool model2 = filter_model->currentText()=="2";
        if (model1 || model2) {
            for (size_t x=0;x<dx;x++) {
                for (size_t y=0;y<dy;y++) {
                    double fx = xx[x]/(double)dx;
                    double fy = yy[y]/(double)dy;
                    double u = fx*cr - fy*sr; // along the carrier
                    double v = fx*sr + fy*cr; // across the carrier
                    double across = exp(-pow(v*M_PI*res,2));
                    double e_tot = model1 ?
                                1.0-depth*across/(1.0+exp(sharpness*(1.0/(M_PI*wc)-std::abs(u)))) :
                                1.0-depth*across*exp(-pow(M_PI*(std::abs(u)*wc-1.0), 2));
                    imageFFT.set(x,y,imageFFT.point(x,y) * e_tot);
                }
            }
        }
        imageFFT = imageFFT.ft2(PHYS_BACKWARD);

        nPhysD *deepcopy=new nPhysD(*imageShot);
        deepcopy->setShortName("deghost");
        deepcopy->setName("deghost("+imageShot->getName()+","+filter_model->currentText().toStdString()+","+QString::number(resolution->value()).toStdString()+")");

        // line points are relative to the image origin: pixels of the shot image = points + its origin
        // (the bounding box must be shifted too, not only the polygon)
        QPolygonF regionPoly=maskRegion->poly(1).translated(imageShot->get_origin().x(),imageShot->get_origin().y());
        QRect geom=regionPoly.boundingRect().toRect().intersected(QRect(0,0,dx,dy));
        std::vector<vec2f> vecPoints(regionPoly.size());
        for(int k=0;k<regionPoly.size();k++) {
            vecPoints[k]=vec2f(regionPoly[k].x(),regionPoly[k].y());
        }

        for(int i=geom.left();i<=geom.right(); i++) {
            for(int j=geom.top();j<=geom.bottom(); j++) {
                vec2f pp(i,j);
                if (point_inside_poly(pp,vecPoints)==true) {
                    // the filter is symmetric: the result is real (the modulus would flip negative ringing)
                    deepcopy->set(i,j, imageFFT.point(i,j).real()/(dx*dy));
                }
            }
        }
        deepcopy->TscanBrightness();

        if (erasePrevious->isChecked()) {
            ghostBusted=nparent->replacePhys(deepcopy,ghostBusted,true);
        } else {
            nparent->addShowPhys(deepcopy);
            ghostBusted=deepcopy;
        }

        erasePrevious->setEnabled(true);
        statusbar->showMessage(QString(tr("Time: %1 msec")).arg(timer.elapsed()));

    }
}
