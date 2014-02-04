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
#include "nAutoAlign.h"
#include "neutrino.h"

nAutoAlign::nAutoAlign(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname)
{
	my_w.setupUi(this);

	decorate();

	connect(my_w.doIt,SIGNAL(pressed()),this,SLOT(doOperation()));
}

void nAutoAlign::doOperation_old () {
	nPhysD *img1=getPhysFromCombo(my_w.image1);
	nPhysD *img2=getPhysFromCombo(my_w.image2);
	
	qDebug() << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SIZES" << img1->get_size().x() << img1->get_size().y();
	qDebug() << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SIZES" << img2->get_size().x() << img2->get_size().y();
	if (img1==NULL || img2==NULL) return;
	
	size_t diag=1;
	while (diag < std::max(img1->get_size().mod(),img2->get_size().mod())) {
//	while (diag < std::max(max(img1->get_size().x(),img2->get_size().x()),max(img1->get_size().y(),img2->get_size().y()))) {
		diag*=2;
	}
	qDebug() << "DIAGONAL" << diag << img1->get_size().mod() << img2->get_size().mod();

	nPhysD *img1pad =img1->padding(diag,diag,0.0);
	nPhysD *img2pad =img2->padding(diag,diag,0.0);
	phys_divide(*img1pad,img1pad->sum());
	phys_divide(*img2pad,img2pad->sum());
	img1pad->setName("img1pad");
	img2pad->setName("img2pad");
	
	nparent->addShowPhys(img1pad);
	nparent->addShowPhys(img2pad);

	nPhysD *rPhys=new nPhysD(diag,diag,0.0,"Result");
	nparent->addShowPhys(rPhys);
	fftw_complex *cPhys1=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*diag*(diag/2+1));
	fftw_complex *cPhys2=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*diag*(diag/2+1));
	fftw_complex *cPhys3=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*diag*(diag/2+1));
	fftw_complex *cPhys4=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*diag*(diag/2+1));

	fftw_plan plan1R2C=fftw_plan_dft_r2c_2d(diag,diag, img1pad->Timg_buffer, cPhys1, FFTW_ESTIMATE);
	fftw_plan plan2R2C=fftw_plan_dft_r2c_2d(diag,diag, img2pad->Timg_buffer, cPhys2, FFTW_ESTIMATE);
	fftw_plan plan3C2R=fftw_plan_dft_c2r_2d(diag,diag, cPhys4, rPhys->Timg_buffer, FFTW_ESTIMATE);
	
	fftw_execute(plan1R2C);
	fftw_execute(plan2R2C);	

//	for (size_t j=0;j<diag;j++) {
//		cPhys2[j*(diag/2+1)][0]=0;
//		cPhys2[j*(diag/2+1)][1]=0;
//	}
//	
//	for (size_t i=0;i<diag/2+1;i++) {
//		cPhys2[i][0]=0;
//		cPhys2[i][1]=0;
//	}
	
	int iterMax=0;
	QVector <pair<double, bidimvec<int> > > resCorr;

	ofstream ostr("autoCorr.dat");

	for (int iter=0;iter<my_w.aN->value(); iter++) {		
		double alpha=M_PI/180.*(my_w.aMin->value()+iter*(my_w.aMax->value()-my_w.aMin->value())/my_w.aN->value());
		qDebug() << "> > > > > > > > > > > > > > > >" << alpha;
		for (size_t j=0;j<diag;j++) {
			for (size_t i=0;i<diag/2+1;i++) {
				size_t ir= (diag/2+1+(int)(cos(alpha)*i-sin(alpha)*j))%(diag/2+1);
				size_t jr= (diag+(int)(sin(alpha)*i+cos(alpha)*j))%diag;
				size_t ij=i+j*(diag/2+1);
				size_t ijr=ir+jr*(diag/2+1);
				cPhys4[ij][0]=cPhys2[ijr][0];
				cPhys4[ij][1]=cPhys2[ijr][1];
			}
		}

		
		nPhysD *fftFirst=new nPhysD(diag/2+1,diag,0.0,"fftFirst");
		nPhysD *fftOriginal=new nPhysD(diag/2+1,diag,0.0,"fftOriginal");
		nPhysD *fftRotated=new nPhysD(diag/2+1,diag,0.0,"fftRotated");
		
		for (size_t i=0;i<diag*(diag/2+1);i++) {
//			cPhys3[i][0]=(cPhys1[i][0]*cPhys4[i][0] + cPhys1[i][1]*cPhys4[i][1]);
//			cPhys3[i][1]=(cPhys1[i][1]*cPhys4[i][0] - cPhys1[i][0]*cPhys4[i][1]);		
			
			fftFirst->set(i,log10(sqrt(pow(cPhys1[i][0],2)+pow(cPhys1[i][1],2))));
			fftOriginal->set(i,log10(sqrt(pow(cPhys2[i][0],2)+pow(cPhys2[i][1],2))));
			fftRotated->set(i,log10(sqrt(pow(cPhys4[i][0],2)+pow(cPhys4[i][1],2))));
			//		cPhys3[i][0]=(cPhys1[i][0]*cPhys2[i][0] + cPhys1[i][1]*cPhys2[i][1]);
			//		cPhys3[i][1]=(cPhys1[i][1]*cPhys2[i][0] - cPhys1[i][0]*cPhys2[i][1]);		
		}
		
		fftw_execute(plan3C2R);
		
		nparent->addShowPhys(fftFirst);
		nparent->addShowPhys(fftOriginal);
		nparent->addShowPhys(fftRotated);
		

		bidimvec<int> maxP=phys_max_p(*rPhys);
		double maxValue=rPhys->point(maxP.x(),maxP.y());
		bidimvec<int> shift(diag/2+1,diag/2+1);
		maxP+=shift;
		maxP=vec2f(maxP.x()%diag,maxP.y()%diag)-shift;
		
		resCorr.append(make_pair(maxValue, maxP));
		if (resCorr.last().first>=resCorr[iterMax].first) {
			iterMax=iter;
		}
		
		ostr << alpha << " " << maxValue << " " << maxP.x() << " " << maxP.y() << endl;
		cerr  << __FUNCTION__ << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " << maxP.x() << " " << maxP.y() << " " << maxValue << endl; 

	}
	rPhys->TscanBrightness();
	
	
//	delete rPhys;
	
	fftw_free(cPhys1);
	fftw_free(cPhys2);
	fftw_free(cPhys3);
	fftw_free(cPhys4);

	fftw_destroy_plan(plan1R2C);
	fftw_destroy_plan(plan2R2C);
	fftw_destroy_plan(plan3C2R);
	
	return;	
}

void nAutoAlign::doOperation () {
	nPhysD *img1=getPhysFromCombo(my_w.image1);
	nPhysD *img2=getPhysFromCombo(my_w.image2);
	
	if (img1==NULL || img2==NULL) return;
	
	//! we need to calculate this to have a good match for the triangles formed outside the rotation
	double mean2=0.0;
	for (size_t i=0; i<img2->getSurf();i++) {
		mean2+=img2->point(i);
	}
	mean2/=img2->getSurf();
	
	QVector <pair<double, bidimvec<int> > > resCorr;
	int iterMax=0;
//	ofstream ostr("autoCorr.dat");
	double amin=my_w.aMin->value();
	double amax=my_w.aMax->value();
	int an=my_w.aN->value();
	nPhysD bestMatch;
	for (int iter=0;iter<an; iter++) {		
		double alpha=amin+(iter+0.5)*(amax-amin)/an;
		nPhysD *imgr=img2->rotated(alpha,mean2);
		bidimvec<int> dim(min(img1->getW(),imgr->getW()),min(img1->getH(),imgr->getH()));
		DEBUG("DIM "<<dim);
		nPhysD image1 =img1->sub(0.5*(img1->getW()-dim.x()),0.5*(img1->getH()-dim.y()),dim.x(),dim.y());
		nPhysD image2 =imgr->sub(0.5*(imgr->getW()-dim.x()),0.5*(imgr->getH()-dim.y()),dim.x(),dim.y());
		
		delete imgr;

		resCorr.append(phys_cross_correlate(&image1, &image2));
		
		if (resCorr.last().first>=resCorr[iterMax].first) {
			iterMax=iter;
			bestMatch=image2;
			bestMatch.set_origin(img1->get_origin()-resCorr[iter].second);
		}

//		ostr << alpha << " " << resCorr[iter].first << " " << resCorr[iter].second.x() << " " << resCorr[iter].second.y() << endl;
		DEBUG(alpha << " " << resCorr[iter].first << " " << resCorr[iter].second.x() << " " << resCorr[iter].second.y());
	}	
	if (bestMatch.getSurf()==0) {
		statusBar()->showMessage("Problem", 5000);
	} else {
		nparent->addShowPhys(bestMatch); 
	}
	my_w.statusbar->showMessage(QString::number(amin+iterMax*(amax-amin)/an)+" "+QString::number(resCorr[iterMax].second.x())+" "+QString::number(resCorr[iterMax].second.y()));	
}


