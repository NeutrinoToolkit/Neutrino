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

    // the slider applies on release (keyboard/wheel steps immediately); while dragging only the value is shown
    slider->setTracking(false);

    connect(cutValue,SIGNAL(editingFinished()), this, SLOT(doOperation()));
    connect(cutValue,SIGNAL(editingFinished()), this, SLOT(syncSlider()));
    connect(doIt,SIGNAL(clicked()),this,SLOT(doOperation()));
    connect(slider,SIGNAL(valueChanged(int)),this,SLOT(sliderChanged(int)));
    connect(slider,SIGNAL(sliderMoved(int)),this,SLOT(sliderMoved(int)));
    connect(image2, SIGNAL(currentIndexChanged(int)), this, SLOT(updateMiniMaxi()));
    connect(replaceVal, SIGNAL(currentIndexChanged(int)), this, SLOT(doOperation()));
    // the result was closed by the user
    connect(nparent, &neutrino::physDel, this, [this](nPhysD *buf) { if (buf==cutoffPhys) cutoffPhys=nullptr; });

    nPhysD *my_phys2=getPhysFromCombo(image2);
    if (my_phys2) {
        cutValue->setText(QLocale().toString(my_phys2->get_min()));
    }
    updateMiniMaxi();
}


void Cutoff_mask::updateMiniMaxi () {
    nPhysD *my_phys2=getPhysFromCombo(image2);
    if (my_phys2) {
        mini->setText(QLocale().toString(my_phys2->get_min()));
        maxi->setText(QLocale().toString(my_phys2->get_max()));
    }
    syncSlider();
}

// slider position -> value in [min, max] of the mask image
double Cutoff_mask::sliderToValue(int val) {
    nPhysD *my_phys2=getPhysFromCombo(image2);
    if (!my_phys2) return 0;
    return my_phys2->get_min()+(val-slider->minimum())*(my_phys2->get_max()-my_phys2->get_min())/(slider->maximum()-slider->minimum());
}

void Cutoff_mask::sliderMoved(int val) {
    if (getPhysFromCombo(image2)) cutValue->setText(QLocale().toString(sliderToValue(val)));
}

void Cutoff_mask::sliderChanged(int val) {
    if (getPhysFromCombo(image2)) {
        cutValue->setText(QLocale().toString(sliderToValue(val)));
        doOperation();
    }
}

// slider position from the typed value
void Cutoff_mask::syncSlider() {
    nPhysD *my_phys2=getPhysFromCombo(image2);
    bool ok;
    double val=locale().toDouble(cutValue->text(),&ok);
    double range=my_phys2 ? my_phys2->get_max()-my_phys2->get_min() : 0;
    if (ok && range>0 && std::isfinite(range)) {
        double f=std::min(1.0,std::max(0.0,(val-my_phys2->get_min())/range));
        const QSignalBlocker block(slider);
        slider->setValue(slider->minimum()+int(f*(slider->maximum()-slider->minimum())+0.5));
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
                // mean of the finite pixels of the source
                double sum=0; size_t n=0;
                for (size_t k=0; k<my_phys1->getSurf(); k++) {
                    if (std::isfinite(my_phys1->Timg_buffer[k])) { sum+=my_phys1->Timg_buffer[k]; n++; }
                }
                if (n) replaceDbl=sum/n;
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

            // never overwrite one of the inputs (the previous result used as source or mask)
            if (cutoffPhys==my_phys1 || cutoffPhys==my_phys2) cutoffPhys=nullptr;
            cutoffPhys=nparent->replacePhys(masked,cutoffPhys);
        } else {
            statusBar()->showMessage("Error image size do not match", 5000);
        }
    } else {
        statusBar()->showMessage("Error "+cutValue->text(), 5000);
    }
}


