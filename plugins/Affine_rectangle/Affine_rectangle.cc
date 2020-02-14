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
#include "Affine_rectangle.h"
#include "neutrino.h"

Affine_rectangle::Affine_rectangle(neutrino *nparent) : nGenericPan(nparent),
    affined(nullptr), l1(this,1)
{
	my_w.setupUi(this);
    l1.changeToolTip(panName()+"Points");
    l1.changeColorHolder("red");
    resetPoints();

    connect(my_w.actionReset, SIGNAL(triggered()), this, SLOT(resetPoints()));
    connect(my_w.actionLine, SIGNAL(triggered()), &l1, SLOT(togglePadella()));
    connect(my_w.transform,SIGNAL(pressed()),this,SLOT(affine()));

    show();
}

void Affine_rectangle::resetPoints() {
    QPolygonF poly;
    poly << QPointF(100,0) << QPointF(0,0) << QPointF(0,100);
    l1.setPoints(poly);
}

void Affine_rectangle::bufferChanged(nPhysD* buf) {
    nGenericPan::bufferChanged(buf);
    if (buf && buf==getPhysFromCombo(my_w.image1)) {
        l1.show();
    } else {
        l1.hide();
    }
}

void Affine_rectangle::affine() {
    QPolygonF rectPoly;

    rectPoly << QPointF(my_w.newWidth->value(),0) << QPointF(0,0) << QPointF(0,my_w.newHeight->value());

    std::array<double,6> transVect=getAffine(rectPoly,l1.getPoints());

    nPhysD *my_phys=getPhysFromCombo(my_w.image1);
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

        unsigned int dx=my_w.newWidth->value();
        unsigned int dy=my_w.newHeight->value();
		
        DEBUG(affine(vec2f(0,0),transVect).x() << " " << affine(vec2f(0,0),transVect).y());

        nPhysD affinePhys(dx,dy,0.0,"affine");

        QProgressDialog progress("", "Cancel", 0, dx, this);
        progress.setCancelButton(0);
        progress.setWindowModality(Qt::WindowModal);
        progress.setValue(0);
        progress.show();

		for (unsigned int i=0; i<dx; i++) {
            progress.setValue(i);
			for (unsigned int j=0; j<dy; j++) {
                affinePhys.set(i,j,my_phys->getPoint(affine(vec2f(i,j),transVect),replaceVal));
			}
		}
        affinePhys.TscanBrightness();

        if (my_w.erasePrevious->isChecked()) {
            affined=nparent->replacePhys(new nPhysD(affinePhys),affined,true);
        } else {
            affined=new nPhysD(affinePhys);
            nparent->addShowPhys(affined);
        }
	}
}

vec2f Affine_rectangle::affine(vec2f in, std::array<double,6>& vec){
	return vec2f(in.x()*vec[0]+in.y()*vec[1]+vec[2],in.x()*vec[3]+in.y()*vec[4]+vec[5]);
}

std::array<double,6> Affine_rectangle::getAffine(QPolygonF poly1, QPolygonF poly2) {
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

