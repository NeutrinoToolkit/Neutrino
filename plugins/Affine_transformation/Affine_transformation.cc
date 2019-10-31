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
    l1(this,1),
    l2(this,1),
    region(this,1)
{

    region.setRect(QRectF(100,100,100,100));

	my_w.setupUi(this);
    l1.changeToolTip(panName()+"Line 1");
    l1.changeColorHolder("red");
	QPolygonF poly;
    poly << QPointF(100,0) << QPointF(0,0) << QPointF(0,100);
    l1.setPoints(poly);
	
    l2.changeToolTip(panName()+"Line 2");
    l2.changeColorHolder("blue");
    poly.translate(50,50);
    l2.setPoints(poly);
	


    connect(my_w.line1, SIGNAL(released()), &l1, SLOT(togglePadella()));
    connect(my_w.line2, SIGNAL(released()), &l2, SLOT(togglePadella()));

    connect(my_w.actionRegion, SIGNAL(triggered()), &region, SLOT(togglePadella()));

    connect(&l1, SIGNAL(sceneChanged()), this, SLOT(apply()));
    connect(&l2, SIGNAL(sceneChanged()), this, SLOT(apply()));

    connect(my_w.first,SIGNAL(pressed()),this,SLOT(affine()));
    connect(my_w.second,SIGNAL(pressed()),this,SLOT(affine()));
	
    connect(my_w.actionReset, SIGNAL(triggered()), this, SLOT(resetPoints()));

    affined=NULL;
    show();
	apply();


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
		if (buf==getPhysFromCombo(my_w.image1)) {
            l1.show();
		} else {
            l1.hide();
		}
		if (buf==getPhysFromCombo(my_w.image2)) {
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
	nPhysD *my_phys=NULL;
	nPhysD *my_phys_other=NULL;

    std::array<double,6> vecForward,vecBackward;
	if (sender()==my_w.first) {
		my_phys=getPhysFromCombo(my_w.image1);
		my_phys_other=getPhysFromCombo(my_w.image2);
		vecForward=forward;
		vecBackward=backward;
	} else if (sender()==my_w.second) {
		my_phys=getPhysFromCombo(my_w.image2);
		my_phys_other=getPhysFromCombo(my_w.image1);
		vecForward=backward;
		vecBackward=forward;
	}
	if (my_phys) {
		
		double replaceVal=0.0;
		switch (my_w.defaultValue->currentIndex()) {
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
        if (!my_w.crop->isChecked()){
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
        progress.setCancelButton(0);
        progress.setWindowModality(Qt::WindowModal);
        progress.setValue(0);
        progress.show();

		for (unsigned int i=0; i<dx; i++) {
            progress.setValue(i);
			for (unsigned int j=0; j<dy; j++) {
                affinePhys.set(i,j,my_phys->getPoint(affine(vec2f(i,j)+vec2f(minx,miny),vecBackward),replaceVal));
			}
		}
        affinePhys.TscanBrightness();

        if (my_w.crop->isChecked()){
            QRectF reg=region.getRect();
            qDebug() << "----------------------------------------------------";
            qDebug() << "----------------------------------------------------";
            qDebug() << "----------------------------------------------------";
            qDebug() << "----------------------------------------------------";
            qDebug() << reg;
            qDebug() << "----------------------------------------------------";
            qDebug() << "----------------------------------------------------";
            qDebug() << "----------------------------------------------------";
            qDebug() << "----------------------------------------------------";
            qDebug() << "----------------------------------------------------";

            nPhysD mycopy(affinePhys.sub(reg.x(),reg.y(),reg.width(),reg.height()));
            affinePhys=mycopy;
        }

        if (my_w.erasePrevious->isChecked()) {
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

std::array<double,6> Affine_transformation::getAffine(QPolygonF poly1, QPolygonF poly2) {
    std::array<double,6>ret;
	poly1.resize(3);
	poly2.resize(3);

    std::array<double,9> p1, p2, mat, inva;
	
	p1[0] = poly1[0].x(); p1[1] = poly1[1].x(); p1[2] = poly1[2].x();
	p1[3] = poly1[0].y(); p1[4] = poly1[1].y(); p1[5] = poly1[2].y();
	p1[6] = 1.0;          p1[7] = 1.0;          p1[8] = 1.0;
	
	
	p2[0] = poly2[0].x(); p2[1] = poly2[1].x(); p2[2] = poly2[2].x();
	p2[3] = poly2[0].y(); p2[4] = poly2[1].y(); p2[5] = poly2[2].y();
	p2[6] = 1.0;          p2[7] = 1.0;          p2[8] = 1.0;
	
	gsl_matrix_view m1 = gsl_matrix_view_array(&p1[0], 3, 3);
	gsl_matrix_view m2 = gsl_matrix_view_array(&p2[0], 3, 3);
    gsl_matrix_view affineMat = gsl_matrix_view_array(&mat[0], 3, 3);
	
	gsl_matrix_view inv = gsl_matrix_view_array(&inva[0],3,3);
	gsl_permutation *p = gsl_permutation_alloc (3);
	
	int s;
	gsl_linalg_LU_decomp (&m1.matrix, p, &s);  
	
	gsl_linalg_LU_invert (&m1.matrix, p, &inv.matrix);
	
	gsl_permutation_free (p);
	
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &m2.matrix, &inv.matrix, 0.0, &affineMat.matrix);
	
    ret[0]=gsl_matrix_get(&affineMat.matrix,0,0);
    ret[1]=gsl_matrix_get(&affineMat.matrix,0,1);
    ret[2]=gsl_matrix_get(&affineMat.matrix,0,2);
    ret[3]=gsl_matrix_get(&affineMat.matrix,1,0);
    ret[4]=gsl_matrix_get(&affineMat.matrix,1,1);
    ret[5]=gsl_matrix_get(&affineMat.matrix,1,2);
	return ret;
}

void Affine_transformation::apply() {
	
	
    forward=getAffine(l1.getPoints(),l2.getPoints());
	
	
	my_w.A00->setText(QLocale().toString(forward[0]));
	my_w.A01->setText(QLocale().toString(forward[1]));
	my_w.A02->setText(QLocale().toString(forward[2]));
	my_w.A10->setText(QLocale().toString(forward[3]));
	my_w.A11->setText(QLocale().toString(forward[4]));
	my_w.A12->setText(QLocale().toString(forward[5]));
	
    backward=getAffine(l2.getPoints(),l1.getPoints());

	my_w.B00->setText(QLocale().toString(backward[0]));
	my_w.B01->setText(QLocale().toString(backward[1]));
	my_w.B02->setText(QLocale().toString(backward[2]));
	my_w.B10->setText(QLocale().toString(backward[3]));
	my_w.B11->setText(QLocale().toString(backward[4]));
	my_w.B12->setText(QLocale().toString(backward[5]));	
	
}

