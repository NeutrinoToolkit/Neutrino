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
#include "Affine_transformation.h"
#include "neutrino.h"

// physWavelets

Affine_transformation::Affine_transformation(neutrino *nparent) : nGenericPan(nparent),
    affined(nullptr),
    l1(this,1),
    l2(this,1),
    region(this,1)
{
    qDebug() << "here";
    region.setRect(QRectF(100,100,100,100));

    setupUi(this);
    forwardLine = {A00,A01,A02,A10,A11,A12};
    backwardLine ={B00,B01,B02,B10,B11,B12};

    l1.changeToolTip(panName()+"Line 1");
    l1.changeColorHolder("red");
	QPolygonF poly;
    poly << QPointF(100,0) << QPointF(0,0) << QPointF(0,100);
    l1.setPoints(poly);
    qDebug() << "here";

    l2.changeToolTip(panName()+"Line 2");
    l2.changeColorHolder("blue");
    poly.translate(50,50);
    l2.setPoints(poly);
	


    qDebug() << "here";
    connect(line1, SIGNAL(released()), &l1, SLOT(togglePadella()));
    connect(line2, SIGNAL(released()), &l2, SLOT(togglePadella()));

    connect(actionRegion, SIGNAL(triggered()), &region, SLOT(togglePadella()));

    connect(&l1, SIGNAL(sceneChanged()), this, SLOT(apply()));
    connect(&l2, SIGNAL(sceneChanged()), this, SLOT(apply()));

    connect(first,SIGNAL(pressed()),this,SLOT(affine()));
    connect(second,SIGNAL(pressed()),this,SLOT(affine()));
	
    connect(actionReset, SIGNAL(triggered()), this, SLOT(resetPoints()));

    connect(actionShift, &QAction::triggered, this, &Affine_transformation::findshift);

    qDebug() << "here";
    show();
    qDebug() << "here";
    apply();
    qDebug() << "here";


}

void Affine_transformation::resetPoints() {
    QPolygonF poly;
    poly << QPointF(100,0) << QPointF(0,0) << QPointF(0,100);
    l1.setPoints(poly);
    poly.translate(50,50);
    l2.setPoints(poly);
    region.setRect(QRectF(100,100,100,100));

    apply();
}

void Affine_transformation::bufferChanged(nPhysD* buf) {
    nGenericPan::bufferChanged(buf);
	if (buf) {
        if (buf==getPhysFromCombo(image1)) {
            l1.show();
		} else {
            l1.hide();
		}
        if (buf==getPhysFromCombo(image2)) {
            l2.show();
		} else {
            l2.hide();
		}
    } else {
        l1.hide();
        l2.hide();
    }
}

void Affine_transformation::affine() {
    nPhysD *my_phys=nullptr;
    nPhysD *my_phys_other=nullptr;

    std::array<double,6> vecForward,vecBackward;
    if (sender()==first) {
        my_phys=getPhysFromCombo(image1);
        my_phys_other=getPhysFromCombo(image2);
		vecForward=forward;
		vecBackward=backward;
    } else if (sender()==second) {
        my_phys=getPhysFromCombo(image2);
        my_phys_other=getPhysFromCombo(image1);
		vecForward=backward;
		vecBackward=forward;
	}
	if (my_phys) {
		
		double replaceVal=0.0;
        switch (defaultValue->currentIndex()) {
			case 0:
				replaceVal=std::numeric_limits<double>::quiet_NaN();
				break;
			case 1:
				replaceVal=my_phys->get_min();
				break;
			case 2:
				replaceVal=my_phys->get_max();
				break;
			case 3:
				replaceVal=0.5*(my_phys->get_min()+my_phys->get_max());
				break;
			case 4:
				replaceVal=0.0;
				break;
			default:
				WARNING("something is broken here");
				break;
		}

		unsigned int dx=my_phys_other->getW();
		unsigned int dy=my_phys_other->getH();
		
		double minx=0.0;
		double miny=0.0;
        std::vector<vec2f> corners(4); //clockwise...
        corners[0]=affine(vec2f(0,0),vecForward);
        corners[1]=affine(vec2f(my_phys->getW(),0),vecForward);
        corners[2]=affine(vec2f(my_phys->getW(),my_phys->getH()),vecForward);
        corners[3]=affine(vec2f(0,my_phys->getH()),vecForward);
        minx=corners[0].x();
        double maxx=corners[0].x();
        miny=corners[0].y();
        double maxy=corners[0].y();
        for (unsigned int i=1;i<4;i++) {
            if (minx>corners[i].x()) minx=corners[i].x();
            if (maxx<corners[i].x()) maxx=corners[i].x();
            if (miny>corners[i].y()) miny=corners[i].y();
            if (maxy<corners[i].y()) maxy=corners[i].y();
        }
        dx=(unsigned int) (maxx-minx);
        dy=(unsigned int) (maxy-miny);
		
        DEBUG(affine(vec2f(0,0),vecForward).x() << " " << affine(vec2f(0,0),vecForward).y());
        DEBUG(affine(vec2f(0,0),vecBackward).x() << " " << affine(vec2f(0,0),vecBackward).y());

        nPhysD affinePhys(dx,dy,0.0,"affine");
//        affinePhys->set_origin(affine(my_phys_other->get_origin(),vecForward)-vec2f(minx,miny));

        QProgressDialog progress("", "Cancel", 0, dx, this);
        progress.setCancelButton(nullptr);
        progress.setWindowModality(Qt::WindowModal);
        progress.setValue(0);
        progress.show();

		for (unsigned int i=0; i<dx; i++) {
            progress.setValue(i);
			for (unsigned int j=0; j<dy; j++) {
                affinePhys.set(i,j,my_phys->getPoint(affine(vec2f(i,j),vecBackward),replaceVal));
			}
		}

        if (crop->isChecked()){
            QRectF reg=region.getRect();
            nPhysD mycopy(affinePhys.sub(reg));
            affinePhys=mycopy;
        }

        affinePhys.TscanBrightness();

        if (erasePrevious->isChecked()) {
            affined=nparent->replacePhys(new nPhysD(affinePhys),affined,true);
        } else {
            affined=new nPhysD(affinePhys);
            nparent->addShowPhys(affined);
        }
	}
}

vec2f Affine_transformation::affine(vec2f in, std::array<double,6>& vec){
	return vec2f(in.x()*vec[0]+in.y()*vec[1]+vec[2],in.x()*vec[3]+in.y()*vec[4]+vec[5]);
}

void Affine_transformation::apply() {
	
    forward=physMath::getAffine(l1.getPointsVec2f(),l2.getPointsVec2f());
    qDebug() << "here";
    for (unsigned int i=0; i<forward.size(); i++) {
        qDebug() << "here" << forward[i];
    }
    for (unsigned int i=0; i<forward.size(); i++) {
        qDebug() << "here" << forward[i] << forwardLine[i];
        forwardLine[i]->setText(QLocale().toString(forward[i]));
    }
    qDebug() << "here";

    backward=physMath::getAffine(l2.getPointsVec2f(),l1.getPointsVec2f());
    for (unsigned int i=0; i<backward.size(); i++) {
        qDebug() << "here";
        backwardLine[i]->setText(QLocale().toString(backward[i]));
    }
    qDebug() << "here";

}

void Affine_transformation::findshift() {
    qDebug() << "here in";
    nPhysD *my_phys=nullptr;
    nPhysD *my_phys_other=nullptr;

    std::array<double,6> vecForward,vecBackward;
    my_phys=getPhysFromCombo(image1);
    my_phys_other=getPhysFromCombo(image2);
    vecForward=forward;
    vecBackward=backward;
    if (my_phys) {
        physC imageFFT = my_phys->ft2(PHYS_FORWARD);
        physC imageFFTother = my_phys_other->ft2(PHYS_FORWARD);
        size_t dx=my_phys->getW();
        size_t dy=my_phys->getH();

        for (int i=0; i<dx*dy;i++) {
            mcomplex A=imageFFT.point(i);
            mcomplex B=imageFFTother.point(i);
            double vreal=A.real()*B.real()+A.imag()*B.imag();
            double vimag=A.imag()*B.real()-A.real()*B.imag();
            mcomplex val = mcomplex(vreal, vimag);
            imageFFT.set(i,val/(A.mod()*B.mod()));
        }
        imageFFTother = imageFFT.ft2(PHYS_BACKWARD);
        // nPhysD *magphys=new nPhysD(dx,dy,0.0,"mag");
        double max_val = 0.0;
        int maxx = 0;
        int maxy = 0;
        for (int i = 0; i < dy; i++) {
            for (int j = 0; j < dx; j++) {
                if (i!=0 and j!=0) {
                    int k=i * dx + j;
                    double magnitude = imageFFTother.point(k).mcabs();
                    // magphys->set(j,i,magnitude);
                    if (magnitude > max_val) {
                        max_val = magnitude;
                        maxx = j;
                        maxy = i;
                    }
                }
            }
        }
        QPolygonF poly;
        poly << QPointF(100,0) << QPointF(0,0) << QPointF(0,100);
        l1.setPoints(poly);
        poly.translate(-maxx,-maxy);
        l2.setPoints(poly);
        qDebug() << maxx << maxy;
        apply();
        // magphys->TscanBrightness();
        // nparent->addShowPhys(magphys);

    }
    qDebug() << "here out";

}

