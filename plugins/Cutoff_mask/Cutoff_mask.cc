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
#include "Cutoff_mask.h"
#include "neutrino.h"
#include "nPhysMaths.h"

// physWavelets

Cutoff_mask::Cutoff_mask(neutrino *nparent) : nGenericPan(nparent)
{
    setupUi(this);

    show();

    connect(cutValue,SIGNAL(editingFinished()), this, SLOT(doOperation()));
    connect(doIt,SIGNAL(pressed()),this,SLOT(doOperation()));
    connect(slider,SIGNAL(valueChanged(int)),this,SLOT(sliderChanged(int)));
    connect(image2, SIGNAL(currentIndexChanged(int)), this, SLOT(updateMiniMaxi()));
    connect(replaceVal, SIGNAL(currentIndexChanged(int)), this, SLOT(doOperation()));

    nPhysD *my_phys2=getPhysFromCombo(image2);
    if (my_phys2) {
        cutValue->setText(QLocale().toString(my_phys2->get_min()));
    }
    updateMiniMaxi();
    cutoffPhys=nullptr;
}


void Cutoff_mask::updateMiniMaxi () {
    nPhysD *my_phys2=getPhysFromCombo(image2);
    if (my_phys2) {
        mini->setText(QLocale().toString(my_phys2->get_min()));
        maxi->setText(QLocale().toString(my_phys2->get_max()));
    }
}

void Cutoff_mask::sliderChanged(int val) {
    nPhysD *my_phys2=getPhysFromCombo(image2);
    if (my_phys2) {
        double valDouble=(val-slider->minimum())*(my_phys2->get_max()-my_phys2->get_min())/(slider->maximum()-slider->minimum());
        cutValue->setText(QLocale().toString(valDouble));
        doOperation();
    }
}

void Cutoff_mask::doOperation () {
    bool ok;
    double val=locale().toDouble(cutValue->text(),&ok);
    if (ok) {
        nPhysD *my_phys1=getPhysFromCombo(image1);
        nPhysD *my_phys2=getPhysFromCombo(image2);
        if (my_phys1 && my_phys2 && my_phys1->getW() == my_phys2->getW() && my_phys1->getH() == my_phys2->getH()) {
            double replaceDbl=std::numeric_limits<double>::quiet_NaN();
            if (replaceVal->currentText().toLower() == "min") {
                replaceDbl=my_phys1->get_min();
            } else if (replaceVal->currentText().toLower() == "max") {
                replaceDbl=my_phys1->get_max();
            } else if (replaceVal->currentText().toLower() == "mean") {
                replaceDbl=0.5*(my_phys1->get_min()+my_phys1->get_max());
            } else if (replaceVal->currentText().toLower() == "zero") {
                replaceDbl=0.0;
            }
            nPhysD *masked = new nPhysD(my_phys1->getW(),my_phys1->getH(), replaceDbl);
            masked->set_origin(my_phys1->get_origin());
            masked->set_scale(my_phys1->get_scale());
            size_t k;
            if (!opposite->isChecked()) {
                for (k=0; k<my_phys1->getSurf(); k++)
                    if (my_phys2->Timg_buffer[k] >= val)
                        masked->Timg_buffer[k]=my_phys1->Timg_buffer[k];
            } else {
                for (k=0; k<my_phys1->getSurf(); k++)
                    if (my_phys2->Timg_buffer[k] <= val)
                        masked->Timg_buffer[k]=my_phys1->Timg_buffer[k];
            }

            std::ostringstream my_name;
            my_name << "mask(" << my_phys2->getName() << "," << val << ")";
            masked->setName(my_name.str());
            masked->setShortName("mask");
            masked->setFromName(my_phys1->getFromName());
            masked->TscanBrightness();

            cutoffPhys=nparent->replacePhys(masked,cutoffPhys);
        } else {
            statusBar()->showMessage("Error image size do not match", 5000);
        }
    } else {
        statusBar()->showMessage("Error "+cutValue->text(), 5000);
    }
}


