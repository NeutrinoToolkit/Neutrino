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
#include "nOperator.h"
#include "neutrino.h"

// physWavelets

nOperator::nOperator(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname)
{
	my_w.setupUi(this);
	connect(my_w.calculate, SIGNAL(pressed()), this, SLOT(doOperation()));
	connect(my_w.copyButton, SIGNAL(pressed()), this, SLOT(copyResult()));
	connect(my_w.operation, SIGNAL(currentIndexChanged(int)), this, SLOT(enableGroups(int)));
	operatorResult=NULL;
	
	// separator represents the difference between operations with 2 operands or 1 modify it in .h
	separator[0]=7;
	separator[1]=13;
	
	my_w.operation->insertSeparator(separator[0]);
	my_w.operation->insertSeparator(separator[1]);
	decorate();
	enableGroups(my_w.operation->currentIndex());
}

void nOperator::copyResult () {
	if (!operatorResult) doOperation();
	if (operatorResult) {
		nPhysD *newoperatorResult=new nPhysD(*operatorResult);
		nparent->addPhys(newoperatorResult);
	}
}

void nOperator::enableGroups (int num) {
	if (num < separator[0]) {
		my_w.second->setEnabled(true);
		my_w.radioImage2->setEnabled(true);
		my_w.image2->setEnabled(true);
		my_w.radioNumber1->setEnabled(true);
		my_w.num1->setEnabled(true);
	} else if (num < separator[1]) {
		my_w.second->setEnabled(true);
		my_w.radioImage1->setChecked(true);
		my_w.radioNumber1->setEnabled(false);
		
		my_w.num1->setEnabled(false);
		my_w.radioNumber2->setChecked(true);
		my_w.radioImage2->setEnabled(false);
		my_w.image2->setEnabled(false);
	} else {
		my_w.radioNumber1->setEnabled(true);
		my_w.radioImage1->setChecked(true);
		my_w.second->setEnabled(false);
	}
}

void nOperator::doOperation () {
	nPhysD *image1=getPhysFromCombo(my_w.image1);
	nPhysD *image2=getPhysFromCombo(my_w.image2);
	
	saveDefaults();
	qDebug() << "Operator " << my_w.operation->currentIndex() << separator[0] << separator[1];
	nPhysD *myresult=NULL;	
	if (my_w.operation->currentIndex() < separator[0]) { // two values neeeded
		nPhysD *operand1=NULL;
		nPhysD *operand2=NULL;		
		
		if (my_w.radioNumber1->isChecked() && my_w.radioNumber2->isChecked()) {
			my_w.statusbar->showMessage("One operand must be an image",5000);
			return;
		}
		
		if (my_w.radioNumber1->isChecked()) {
			operand1=new nPhysD(image2->getW(),image2->getH(),my_w.num1->text().toDouble(),my_w.num1->text().toStdString());
			operand1->set_origin(image2->get_origin());
			operand1->set_scale(image2->get_scale());
			operand2=new nPhysD();
			*operand2=*image2;
		} else if (my_w.radioNumber2->isChecked()) {
			operand2=new nPhysD(image1->getW(),image1->getH(),my_w.num2->text().toDouble(),my_w.num2->text().toStdString());
			operand2->set_origin(image1->get_origin());
			operand2->set_scale(image1->get_scale());
			operand1=new nPhysD();
			*operand1=*image1;
		} else {
			operand1=new nPhysD();
			*operand1=*image1;
			operand2=new nPhysD();
			*operand2=*image2;
		}
		
		if (operand1->get_scale() != operand2->get_scale())  {
			QMessageBox::question(this,tr("Scale problem"),tr("The images have not the same scale"),QMessageBox::Ok); 
		}
		QRectF r1=QRectF(-operand1->get_origin().x(),-operand1->get_origin().y(),operand1->getW(),operand1->getH());
		QRectF r2=QRectF(-operand2->get_origin().x(),-operand2->get_origin().y(),operand2->getW(),operand2->getH());
		
		QRectF rTot=r1.intersect(r2);;
		
		myresult=new nPhysD(rTot.width(),rTot.height(), 1.0);
		myresult->setName(my_w.operation->currentText().toStdString()+" "+operand1->getName()+" "+operand2->getName());
		myresult->setFromName(operand1->getFromName()+" "+operand2->getFromName());
		myresult->set_origin(-rTot.left(),-rTot.top());
		vec2f shift1=-(myresult->get_origin() - operand1->get_origin());
		vec2f shift2=-(myresult->get_origin() - operand2->get_origin());
		for (size_t j=0; j<myresult->getH(); j++) {
			for (size_t i=0; i<myresult->getW(); i++) {
				double val1=operand1->point(i+shift1.x(),j+shift1.y());
				double val2=operand2->point(i+shift2.x(),j+shift2.y());
				if (isfinite(val1) && isfinite(val2)) {
					switch (my_w.operation->currentIndex()) {
						case 0:
							myresult->set(i,j,val1+val2);
							break;
						case 1:
							myresult->set(i,j,val1-val2);
							break;
						case 2:
							myresult->set(i,j,val1*val2);
							break;
						case 3:
							myresult->set(i,j,val1/val2);
							break;
						case 4:
							myresult->set(i,j,0.5*(val1+val2));
							break;
						case 5:
							myresult->set(i,j,std::min(val1,val2));
							break;
						case 6:
							myresult->set(i,j,std::max(val1,val2));
							break;
					}
				} else {
					myresult->set(i,j,isfinite(val1)?val1:val2);
				}
				
			}
		}
		delete operand1;
		delete operand2;
	} else 	if (my_w.operation->currentIndex() < separator[1]) { // 1 image and 1(or more) scalars
		if (image1) {
			if (my_w.operation->currentIndex()==separator[0]+1) {
				bool ok;
				double scalar=my_w.num2->text().toDouble(&ok);
				if (ok) {
					myresult=new nPhysD(*image1);
					phys_pow(*myresult, scalar);				
				}
			} else if (my_w.operation->currentIndex()==separator[0]+2) {
				bool ok;
				double scalar=my_w.num2->text().toDouble(&ok);
				if (ok) {
					myresult=new nPhysD(*image1);
					phys_fast_gaussian_blur(*myresult, scalar);				
				}
			} else if (my_w.operation->currentIndex()==separator[0]+3) {
				bool ok;
				double scalar=my_w.num2->text().toDouble(&ok);
				if (ok) {
					myresult=image1->rotated(scalar);
				}
			} else if (my_w.operation->currentIndex()==separator[0]+4) {
				QStringList scalars=my_w.num2->text().split(" ");
				if (scalars.size()==2) {
					bool ok1, ok2;
					double scalar1=scalars.at(0).toDouble(&ok1);
					double scalar2=scalars.at(1).toDouble(&ok2);
					if(ok1 && ok2) {
						myresult=new nPhysD(*image1);
						phys_gaussian_subtraction(*myresult, scalar1, scalar2);				
					}
				}
			} else if (my_w.operation->currentIndex()==separator[0]+5) { //test!!!!
				QStringList scalars=my_w.num2->text().split(" ");
				if (scalars.size()==3) {
					bool ok1, ok2, ok3;
					double scalar1=scalars.at(0).toDouble(&ok1);
					double scalar2=scalars.at(1).toDouble(&ok2)*M_PI/180.0;
					double scalar3=scalars.at(2).toDouble(&ok3);
					if(ok1 && ok2 && ok3) {
						myresult=new nPhysD(*image1);
						nPhysC physfft=myresult->ft2(PHYS_FORWARD);
						int dx=physfft.getW();
						int dy=physfft.getH();
						
						double xx[dx], yy[dy];
						
						nPhysImageF<mcomplex> zz_morlet("morlet");
						zz_morlet.resize(dx,dy);
						
						for (int i=0;i<dx;i++) xx[i]=(double) ((i+(dx+1)/2)%dx-(dx+1)/2); // swap and center
						for (int i=0;i<dy;i++) yy[i]=(double) ((i+(dy+1)/2)%dy-(dy+1)/2);
						
						double lambda_norm=M_PI*dx/scalar1;
						
						
						qDebug() << lambda_norm;
						for (int x=0;x<dx;x++) {
							for (int y=0;y<dy;y++) {
								double radius=sqrt(pow(xx[x]/dx, 2) + pow(yy[y]/dy,2));
								double angle=atan2(yy[y], xx[x]);
								//	exp (-abs(sqrt(x**2/4+y**2*4)-5))   
								
								
								//					double gauss = exp(-fabs(radius-lambda_norm));
								double gauss = exp(-pow(radius*lambda_norm-1.0, 2.)) * (1.0-exp(-pow((angle-scalar2)/scalar3,2)));
								//						myresult->set(x,y, gauss);
								zz_morlet.set(x,y, physfft.point(x,y)*gauss/(dx*dy));
								
							}
						}
						physfft = zz_morlet.ft2(PHYS_BACKWARD);
						for (register size_t ii=0; ii<physfft.getSurf(); ii++) {
							mcomplex pt = physfft.point(ii);
							myresult->set(ii, pt.real() );
						}			
					}
				}
			}
			
		}
	} else { // one value needed
		myresult=new nPhysD(*image1);
		myresult->setName(my_w.operation->currentText().toStdString()+" "+image1->getName());
		if (my_w.operation->currentIndex()==separator[1]+1) {
			phys_log10(*myresult);
		} else if (my_w.operation->currentIndex()==separator[1]+2) {
			phys_log(*myresult);
		} else if (my_w.operation->currentIndex()==separator[1]+3) {
			phys_abs(*myresult);
		} else if (my_w.operation->currentIndex()==separator[1]+4) {
			phys_square(*myresult);
		} else if (my_w.operation->currentIndex()==separator[1]+5) {
			phys_sqrt(*myresult);
		} else if (my_w.operation->currentIndex()==separator[1]+6) {
			phys_sin(*myresult);
		} else if (my_w.operation->currentIndex()==separator[1]+7) {
			phys_cos(*myresult);
		} else if (my_w.operation->currentIndex()==separator[1]+8) {
			phys_tan(*myresult);
		}
	}
	
	if (myresult) {
		myresult->TscanBrightness();
		myresult->setShortName(my_w.operation->currentText().toStdString());
		if (myresult->getSurf()>0) operatorResult=nparent->replacePhys(myresult,operatorResult);
	}		
}

