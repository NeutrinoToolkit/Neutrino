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
#include "nCutoffMask.h"
#include "neutrino.h"
#include "nPhysMaths.h"

// physWavelets

nCutoffMask::nCutoffMask(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname)
{
	my_w.setupUi(this);

	decorate();

	connect(my_w.cutValue,SIGNAL(editingFinished()), this, SLOT(doOperation()));
	connect(my_w.doIt,SIGNAL(pressed()),this,SLOT(doOperation()));
	connect(my_w.slider,SIGNAL(valueChanged(int)),this,SLOT(sliderChanged(int)));
	connect(my_w.image2, SIGNAL(currentIndexChanged(int)), this, SLOT(updateMiniMaxi()));
	connect(my_w.replaceVal, SIGNAL(currentIndexChanged(int)), this, SLOT(doOperation()));

	nPhysD *image2=getPhysFromCombo(my_w.image2);
	if (image2) {
		my_w.cutValue->setText(QString::number(image2->Tminimum_value));
	}
	updateMiniMaxi();
	cutoffPhys=NULL;
}


void nCutoffMask::updateMiniMaxi () {
	nPhysD *image2=getPhysFromCombo(my_w.image2);
	if (image2) {
		my_w.mini->setText(QString::number(image2->Tminimum_value));
		my_w.maxi->setText(QString::number(image2->Tmaximum_value));
	}
}

void nCutoffMask::sliderChanged(int val) {
	nPhysD *image2=getPhysFromCombo(my_w.image2);
	if (image2) {
		double valDouble=(val-my_w.slider->minimum())*(image2->Tmaximum_value-image2->Tminimum_value)/(my_w.slider->maximum()-my_w.slider->minimum());
		my_w.cutValue->setText(QString::number(valDouble));
		doOperation();
	}
}

void nCutoffMask::doOperation () {
	bool ok;
	double val=my_w.cutValue->text().toDouble(&ok);
	if (ok) {
		nPhysD *image1=getPhysFromCombo(my_w.image1);
		nPhysD *image2=getPhysFromCombo(my_w.image2);
		if (image1 && image2 && image1->getW() == image2->getW() && image1->getH() == image2->getH()) {
			double replaceVal=std::numeric_limits<double>::quiet_NaN();
			if (my_w.replaceVal->currentText().toLower() == "min") {
				replaceVal=image1->Tminimum_value;
			} else if (my_w.replaceVal->currentText().toLower() == "max") {
				replaceVal=image1->Tmaximum_value;
			} else if (my_w.replaceVal->currentText().toLower() == "mean") {
				replaceVal=0.5*(image1->Tminimum_value+image1->Tmaximum_value);
			} else if (my_w.replaceVal->currentText().toLower() == "zero") {
				replaceVal=0.0;
			}
			nPhysImageF<double> *masked = NULL;
			if (image2->getSurf() == image1->getSurf()) {
				masked = new nPhysImageF<double> (image1->getW(),image1->getH(), replaceVal);
				masked->set_origin(image1->get_origin());
				masked->set_scale(image1->get_scale());
				for (size_t k=0; k<image1->getSurf(); k++) if (image2->Timg_buffer[k] >= val) masked->Timg_buffer[k]=image1->Timg_buffer[k];
				std::ostringstream my_name;
				my_name << "mask(" << image2->getName() << "," << val << ")";
				masked->setName(my_name.str());
				masked->setShortName("mask");
				masked->setFromName(image1->getFromName());
				masked->TscanBrightness();
				
				cutoffPhys=nparent->replacePhys(masked,cutoffPhys);
			}

		}
	} else {
		statusBar()->showMessage("Error "+my_w.cutValue->text(), 5000);
	}
}


