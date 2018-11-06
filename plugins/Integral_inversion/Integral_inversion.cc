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
#include "Integral_inversion.h"
#include "neutrino.h"

Integral_inversion::Integral_inversion(neutrino *nparent)
: nGenericPan(nparent),
  invertedPhys(nullptr)
{

	my_w.setupUi(this);

    axis =  new nLine(this,1);
	axis->forceMonotone = true;
	axis->setPoints(QPolygonF()<<QPointF(10, 10)<<QPointF(50, 50));

	connect(axis, SIGNAL(sceneChanged()), this, SLOT(sceneChanged()));

	QDoubleValidator *dVal = new QDoubleValidator(this);
	dVal->setNotation(QDoubleValidator::ScientificNotation);
	my_w.molarRefr_le->setValidator(dVal);
    my_w.molarRefr_le->setText(QString::number(5.23e-7));

	connect(my_w.actionLine, SIGNAL(triggered()), axis, SLOT(togglePadella()));
    connect(my_w.actionFlipline, SIGNAL(triggered()), axis, SLOT(switchOrdering()));
	connect(my_w.actionBezier, SIGNAL(triggered()), axis, SLOT(toggleBezier()));
	connect(my_w.refphase_checkb, SIGNAL(stateChanged(int)), this, SLOT(refphase_checkbChanged(int)));

	connect(my_w.doInversion, SIGNAL(clicked()), SLOT(doInversion()));

    connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));

	my_w.invAlgo_cb->addItem("Abel", QVariant::fromValue(10));
	my_w.invAlgo_cb->addItem("Abel-HF (experimental!)", QVariant::fromValue(20));
//	my_w.invAlgo_cb->addItem("Abel-derived (experimental!)", QVariant::fromValue(30));
    
	refphase_checkbChanged(my_w.refphase_checkb->checkState());
    show();
}

void Integral_inversion::physDel(nPhysD* buf) {
    if (buf==invertedPhys) {
        invertedPhys=NULL;
    }
}


void Integral_inversion::refphase_checkbChanged(int val) {
	my_w.refphase_cb->setEnabled(val==2);
}

void
Integral_inversion::sceneChanged()
{
	if (sender()==axis && my_w.autoUpdate->isChecked()) {
		doInversion();
	}
}

void Integral_inversion::doInversion() {
	saveDefaults();
	nPhysD *image=getPhysFromCombo(my_w.image);
	if (image) {
		//axis->rearrange_monotone();
		bool isHorizontal = axis->getHMonotone();

		QPolygonF axis_poly;
		QPolygon axis_clean;
        std::vector<vec2i> inv_axis;

		int npoints=2.0*((axis->ref.last()->pos()-axis->ref.first()->pos()).manhattanLength());
		axis_poly = axis->getLine(npoints);
		
		//!fixme we should cut the line when it goes outside and not move the point
        int xpos= std::max<int>(0lu,std::min((unsigned int)axis_poly.first().x(),image->getW()-1));
        int ypos=std::max<int>(0lu,std::min((unsigned int)axis_poly.first().y(),image->getH()-1));
		axis_clean << QPoint(xpos,ypos);

		for (int i=1; i<axis_poly.size(); i++) {
            int xpos=std::max<int>(0lu,std::min((unsigned int)axis_poly.at(i).x(),image->getW()-1));
            int ypos=std::max<int>(0lu,std::min((unsigned int)axis_poly.at(i).y(),image->getH()-1));
			if (isHorizontal) {
				if (xpos != axis_clean.last().x()) axis_clean << QPoint(xpos,ypos);
			} else {
				if (ypos != axis_clean.last().y()) axis_clean << QPoint(xpos,ypos);
			}
		}

		inv_axis.resize(axis_clean.size());
		for (int ii = 0; ii<axis_clean.size(); ii++) {
            inv_axis[ii] = vec2i(axis_clean.at(ii).x(),axis_clean.at(ii).y());
		}

		// launch inversion
        physD iimage(static_cast<physD*>(image)->copy());
        // deep copy and perform operations
        if (my_w.refphase_checkb->isChecked()) {
            physD *ref=static_cast<physD*>(getPhysFromCombo(my_w.refphase_cb));
            if (ref && iimage.getW() == ref->getW() && iimage.getH() == ref->getH()) {
                iimage = iimage - (*ref);
            } else {
                statusBar()->showMessage("Problem in removing reference", 5000);
            }
        }
        iimage.setFromName(iimage.getName());
        iimage.setName("inverted");

        if (my_w.blurRadius_checkb->isChecked()) {	// blur
            physMath::phys_fast_gaussian_blur(iimage, my_w.blurRadius_sb->value());
        }

        if (my_w.multiply_checkb->isChecked()) {	// multiply
            physMath::phys_multiply(iimage, my_w.multiply_sb->value());
        }


		enum phys_direction inv_axis_dir = (isHorizontal) ? PHYS_X : PHYS_Y;
		
		//inv_image = phys_invert_abel(*iimage, inv_axis, inv_axis_dir, ABEL, ABEL_NONE);
		// launching thread

		my_abel_params.iaxis = inv_axis;
		my_abel_params.idir = inv_axis_dir;
		//my_abel_params.ialgo = ABEL;


		// questo sara' da risolvere ma se bestemmio ancora un po' poi non eleggono piu' il papa
		// (workaround)

		int cb_idx = my_w.invAlgo_cb->currentIndex();
        my_abel_params.ialgo = (enum physWave::inversion_algo) my_w.invAlgo_cb->itemData(cb_idx).value<int>();

        my_abel_params.iphysics = physWave::ABEL_NONE;

		DEBUG(10,"algo value is: "<<my_abel_params.ialgo);

        my_abel_params.iimage= &iimage;

        runThread(&my_abel_params, phys_invert_abel_transl, "Abel inversion..." , inv_axis.size());

		

		DEBUG(5,"about to launch thread");

			// apply physics
		QApplication::processEvents();		

        if (my_abel_params.oimage) {
            nPhysD inv_image(my_abel_params.oimage->copy());

            switch (my_w.physTabs->currentIndex()) {
            case 0:
                DEBUG("Inversions: no physics applied");
                break;
            case 1:
                physWave::phys_apply_inversion_gas(inv_image, my_w.probeLambda_sb->value()*1e-9, my_w.imgRes_sb->value()*1e-6, locale().toDouble(my_w.molarRefr_le->text()));
                break;
            case 2:
                DEBUG("Inversions: applying plasma physics");
                physWave::phys_apply_inversion_plasma(inv_image, my_w.probeLambda_sb->value()*1e-9, my_w.imgRes_sb->value()*1e-6);
                break;
            case 3: {
                DEBUG("Inversions: applying proton  physics");
                physWave::phys_apply_inversion_protons(inv_image, my_w.energy->value()*1e6, my_w.imgRes_sb->value()*1e-6, my_w.distance->value()*1e-2, my_w.magnificaton->value());
                break;
            }
            default:
                break;
            }
            inv_image.setShortName(my_w.invAlgo_cb->currentText().toUtf8().constData());

            bool ok1,ok2;
            double mini=locale().toDouble(my_w.minCut->text(),&ok1);
            double maxi=locale().toDouble(my_w.maxCut->text(),&ok2);
            if (ok1 || ok2) {
                physMath::cutoff(inv_image,
                            ok1?mini:inv_image.get_min(),
                            ok2?maxi:inv_image.get_max());
            }
            
            nPhysD *invert = new nPhysD(inv_image);
			if (my_w.erasePrevious->isChecked()) {
                invertedPhys=nparent->replacePhys(invert,invertedPhys);
			} else {
                invertedPhys=invert;
                nparent->addPhys(invert);
			}
        } else {
            statusBar()->showMessage("Problem in inversion", 5000);
        }
		
	}
}

void phys_invert_abel_transl(void *params, int& iter) {
    ((physWave::abel_params *)params)->iter_ptr = &iter;
    phys_invert_abel(*((physWave::abel_params *)params));
}


