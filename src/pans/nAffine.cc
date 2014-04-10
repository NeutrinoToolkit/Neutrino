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
#include "nAffine.h"
#include "neutrino.h"

// physWavelets

nAffine::nAffine(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname)
{
	my_w.setupUi(this);
	l1 =  new nLine(nparent);
	l1->setParentPan(panName,1);
	l1->changeToolTip(panName+"Line 1");
	l1->changeColorHolder("red");
	QPolygonF poly;
	poly << QPointF(100,0) << QPointF(0,0) << QPointF(0,100);
	l1->setPoints(poly);
	

	l2 =  new nLine(nparent);
	l2->setParentPan(panName,1);
	l2->changeToolTip(panName+"Line 2");
	l2->changeColorHolder("blue");
	poly.clear();
	poly << QPointF(50,50) << QPointF(50,150) << QPointF(150,150);
	l2->setPoints(poly);
	

	connect(my_w.line1, SIGNAL(released()), l1, SLOT(togglePadella()));
	connect(my_w.line2, SIGNAL(released()), l2, SLOT(togglePadella()));

	connect(l1, SIGNAL(sceneChanged()), this, SLOT(apply()));
	connect(l2, SIGNAL(sceneChanged()), this, SLOT(apply()));

	connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(bufferChanged(nPhysD*)));

	connect(my_w.first,SIGNAL(pressed()),this,SLOT(affine()));
	connect(my_w.second,SIGNAL(pressed()),this,SLOT(affine()));
	
	
	Affined=NULL;
	decorate();
	apply();
	
}

void nAffine::bufferChanged(nPhysD* buf) {
	if (buf) {
		DEBUG(buf->getName());
		if (buf==getPhysFromCombo(my_w.image1)) {
			l1->show();
		} else {
			l1->hide();
		}
		if (buf==getPhysFromCombo(my_w.image2)) {
			l2->show();
		} else {
			l2->hide();
		}
	}
}

void nAffine::affine() {
	nPhysD *my_phys=NULL;
	if (sender()==my_w.first) {
		my_phys=getPhysFromCombo(my_w.image1);
	} else if (sender()==my_w.second) {
		my_phys=getPhysFromCombo(my_w.image2);
	}
	if (my_phys) {
//		vector<vec2f> corners(4); //clockwise...
//		corners[0]=affine(vec2f(0,0));
//		corners[1]=affine(vec2f(my_phys->getW(),0));
//		corners[2]=affine(vec2f(my_phys->getW(),my_phys->getH()));
//		corners[3]=affine(vec2f(0,my_phys->getH()));		
//
//	
//		double minx=corners[0].x();
//		double maxx=corners[0].x();
//		double miny=corners[0].y();
//		double maxy=corners[0].y();
//		for (unsigned int i=1;i<3;i++) {
//			if (minx>corners[i].x()) minx=corners[i].x();
//			if (maxx<corners[i].x()) maxx=corners[i].x();
//			if (miny>corners[i].y()) miny=corners[i].y();
//			if (maxy<corners[i].y()) maxy=corners[i].y();
//		}
//		unsigned int dx=maxx-minx+1;
//		unsigned int dy=maxy-miny+1;
//		
//		nPhysD *affinePhys=new nPhysD(dx,dy,0.0,"affine");

		unsigned int dx=my_phys->getW();
		unsigned int dy=my_phys->getH();
		
		nPhysD *affinePhys=new nPhysD(dx,dy,0.0,"affine");
		
		for (unsigned int i=0; i<dx; i++) {
			for (unsigned int j=0; j<dy; j++) {
				vec2f affPoint=affine(vec2f(i,j));
				affinePhys->set(i,j,my_phys->getPoint(affPoint.x(),affPoint.y()));
			}
		}
		affinePhys->TscanBrightness();
		
		if (my_w.erasePrevious->isChecked()) {
			Affined=nparent->replacePhys(affinePhys,Affined,true);
		} else {
			nparent->addPhys(affinePhys);
			Affined=affinePhys;
		}
		
	}
}

vec2f nAffine::affine(vec2f in){
	return vec2f(in.x()*a00+in.y()*a01+a02,in.x()*a10+in.y()*a11+a12);
}

void nAffine::apply() {
	
	vector<double> p1(9), p2(9), mat(9), inva(9);
	
	QPolygonF poly1=l1->getPoints();
	poly1.resize(3);
	
		
	p1[0] = poly1[0].x(); p1[1] = poly1[1].x(); p1[2] = poly1[2].x();
	p1[3] = poly1[0].y(); p1[4] = poly1[1].y(); p1[5] = poly1[2].y();
	p1[6] = 1.0;          p1[7] = 1.0;          p1[8] = 1.0;

	QPolygonF poly2=l2->getPoints();
	poly2.resize(3);

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

	DEBUG("\nDa capire " << s << " Control (should be 0 0 1) : " << gsl_matrix_get(&affineMat.matrix,2,0) << " " << gsl_matrix_get(&affineMat.matrix,2,1) << " " << gsl_matrix_get(&affineMat.matrix,2,2));

	a00=gsl_matrix_get(&affineMat.matrix,0,0);
	a01=gsl_matrix_get(&affineMat.matrix,0,1);
	a02=gsl_matrix_get(&affineMat.matrix,0,2);
	a10=gsl_matrix_get(&affineMat.matrix,1,0);
	a11=gsl_matrix_get(&affineMat.matrix,1,1);
	a12=gsl_matrix_get(&affineMat.matrix,1,2);
	
	my_w.A00->setText(QString::number(a00));
	my_w.A01->setText(QString::number(a01));
	my_w.A02->setText(QString::number(a02));
	my_w.A10->setText(QString::number(a10));
	my_w.A11->setText(QString::number(a11));
	my_w.A12->setText(QString::number(a12));
	
}

