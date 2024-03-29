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
        if (!crop->isChecked()){
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
        }
		
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
                affinePhys.set(i,j,my_phys->getPoint(affine(vec2f(i,j)+vec2f(minx,miny),vecBackward),replaceVal));
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

