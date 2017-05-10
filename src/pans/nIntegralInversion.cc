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
#include "nIntegralInversion.h"
#include "neutrino.h"

nIntegralInversion::nIntegralInversion(neutrino *nparent)
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

	connect(my_w.actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
	connect(my_w.actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));
	connect(my_w.actionLine, SIGNAL(triggered()), axis, SLOT(togglePadella()));
	connect(my_w.actionFlipline, SIGNAL(triggered()), axis, SLOT(switchOrdering()));
	connect(my_w.actionBezier, SIGNAL(triggered()), axis, SLOT(toggleBezier()));
	connect(my_w.refphase_checkb, SIGNAL(stateChanged(int)), this, SLOT(refphase_checkbChanged(int)));

	connect(my_w.doInversion, SIGNAL(clicked()), SLOT(doInversion()));

    connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));

	my_w.invAlgo_cb->addItem("Abel", QVariant::fromValue(10));
	my_w.invAlgo_cb->addItem("Abel-HF (experimental!)", QVariant::fromValue(20));
	my_w.invAlgo_cb->addItem("Abel-derived (experimental!)", QVariant::fromValue(30));
    
	refphase_checkbChanged(my_w.refphase_checkb->checkState());
    show();
}

void nIntegralInversion::physDel(nPhysD* buf) {
    if (buf==invertedPhys) {
        invertedPhys=NULL;
    }
}


void nIntegralInversion::refphase_checkbChanged(int val) {
	my_w.refphase_cb->setEnabled(val==2);
}

void
nIntegralInversion::sceneChanged()
{
	if (sender()==axis && my_w.autoUpdate->isChecked()) {
		doInversion();
	}
}

QVariant nIntegralInversion::doInversion() {
	saveDefaults();
	QVariant retVar;
	nPhysD *image=getPhysFromCombo(my_w.image);
	if (image) {
		//axis->rearrange_monotone();
		bool isHorizontal = axis->getHMonotone();

		QPolygonF axis_poly;
		QPolygon axis_clean;
		std::vector<vec2> inv_axis;

		int npoints=2.0*((axis->ref.last()->pos()-axis->ref.first()->pos()).manhattanLength());
		axis_poly = axis->getLine(npoints);
		
		//!fixme we should cut the line when it goes outside and not move the point
        int xpos= std::max<int>(0lu,std::min((size_t)axis_poly.first().x(),image->getW()-1));
        int ypos=std::max<int>(0lu,std::min((size_t)axis_poly.first().y(),image->getH()-1));
		axis_clean << QPoint(xpos,ypos);

		for (int i=1; i<axis_poly.size(); i++) {
            int xpos=std::max<int>(0lu,std::min((size_t)axis_poly.at(i).x(),image->getW()-1));
            int ypos=std::max<int>(0lu,std::min((size_t)axis_poly.at(i).y(),image->getH()-1));
			if (isHorizontal) {
				if (xpos != axis_clean.last().x()) axis_clean << QPoint(xpos,ypos);
			} else {
				if (ypos != axis_clean.last().y()) axis_clean << QPoint(xpos,ypos);
			}
		}

		inv_axis.resize(axis_clean.size());
		for (int ii = 0; ii<axis_clean.size(); ii++) {
			inv_axis[ii] = vec2(axis_clean.at(ii).x(),axis_clean.at(ii).y());
		}

		// launch inversion
		nPhysD *iimage=NULL;
		// do we need a copy?
		if ((my_w.refphase_checkb->isChecked() && getPhysFromCombo(my_w.refphase_cb)) || my_w.blurRadius_checkb->isChecked() || my_w.multiply_checkb->isChecked()) {
			// deep copy and perform operations
			if (my_w.refphase_checkb->isChecked()) {
				nPhysD *ref=getPhysFromCombo(my_w.refphase_cb);
				if (ref && image->getW() == ref->getW() && image->getH() == ref->getH()) {
					iimage = new nPhysD();
					*iimage=(*image)-(*ref);
				} else {
					statusBar()->showMessage("Problem in removing reference", 5000);
				}
			}
			if (iimage==NULL) iimage = new nPhysD(*image);
			iimage->setFromName(iimage->getName());
			iimage->setName("inverted");

			if (my_w.blurRadius_checkb->isChecked()) {	// blur
				phys_fast_gaussian_blur(*iimage, my_w.blurRadius_sb->value());
                std::ostringstream oss; oss<<iimage->getName()<<" (blur"<<my_w.blurRadius_sb->value()<<")";
				iimage->setName(oss.str());
			}

			if (my_w.multiply_checkb->isChecked()) {	// multiply
				phys_multiply(*iimage, my_w.multiply_sb->value());
                std::ostringstream oss; oss<<iimage->getName()<<" *( "<<my_w.multiply_sb->value()<<")";
				iimage->setName(oss.str());
			}


		} else {
			// move pointer
			iimage = image;
		}

		nPhysD *inv_image=NULL;
		enum phys_direction inv_axis_dir = (isHorizontal) ? PHYS_X : PHYS_Y;
		
		//inv_image = phys_invert_abel(*iimage, inv_axis, inv_axis_dir, ABEL, ABEL_NONE);
		// launching thread

		my_abel_params.iaxis = inv_axis;
		my_abel_params.idir = inv_axis_dir;
		//my_abel_params.ialgo = ABEL;


		// questo sara' da risolvere ma se bestemmio ancora un po' poi non eleggono piu' il papa
		// (workaround)

		int cb_idx = my_w.invAlgo_cb->currentIndex();
		//if (cb_idx == 1)
		//	my_abel_params.ialgo = ABEL_HF;
		//else
		//	my_abel_params.ialgo = ABEL;
		my_abel_params.ialgo = (enum inversion_algo) my_w.invAlgo_cb->itemData(cb_idx).value<int>();

		my_abel_params.iphysics = ABEL_NONE;

		DEBUG(10,"algo value is: "<<my_abel_params.ialgo);

        my_abel_params.iimage=iimage;

        runThread(&my_abel_params, phys_invert_abel_transl, "Abel inversion..." , inv_axis.size());

        inv_image = my_abel_params.oimage;
		

		DEBUG(5,"about to launch thread");

			// apply physics
		QApplication::processEvents();		

        if (inv_image) {
            switch (my_w.physTabs->currentIndex()) {
            case 0:
                DEBUG("Inversions: no physics applied");
                break;
            case 1:
                phys_apply_inversion_gas(*inv_image, my_w.probeLambda_sb->value()*1e-9, my_w.imgRes_sb->value()*1e-6, QLocale().toDouble(my_w.molarRefr_le->text()));
                break;
            case 2:
                DEBUG("Inversions: applying plasma physics");
                phys_apply_inversion_plasma(*inv_image, my_w.probeLambda_sb->value()*1e-9, my_w.imgRes_sb->value()*1e-6);
                break;
            case 3: {
                DEBUG("Inversions: applying proton  physics");
                phys_apply_inversion_protons(*inv_image, my_w.energy->value()*1e6, my_w.imgRes_sb->value()*1e-6, my_w.distance->value()*1e-2, my_w.magnificaton->value());
                //                    nPhysD *pippo= new nPhysD(my_abel_params.rimage);
                //                    phys_point_multiply(*pippo, *inv_image);
                //                    phys_multiply(*pippo, my_w.imgRes_sb->value()*1e-6/(2.0*_phys_vacuum_eps));
                //                    nparent->addPhys(pippo);
                break;
            }
            default:
                break;
            }
            inv_image->setShortName(my_w.invAlgo_cb->currentText().toUtf8().constData());

			//		if (my_w.blurRadius_checkb->isChecked()) {	// blur
			//			phys_fast_gaussian_blur(*inv_image, my_w.blurRadius_sb->value());
			//		}

            bool ok1,ok2;
            double mini=QLocale().toDouble(my_w.minCut->text(),&ok1);
            double maxi=QLocale().toDouble(my_w.maxCut->text(),&ok2);
            if (ok1 || ok2) {
                phys_cutoff(*inv_image, 
                            ok1?mini:inv_image->get_min(), 
                            ok2?maxi:inv_image->get_max());
            }
            
            
			if (my_w.erasePrevious->isChecked()) {
				invertedPhys=nparent->replacePhys(inv_image,invertedPhys);
			} else {
				invertedPhys=inv_image;
				nparent->addPhys(inv_image);
			}
			retVar=qVariantFromValue(*invertedPhys);
		} else {
			DEBUG("[nIntegralInversion] Error: inversion returned NULL");
		}
		
	}

	return retVar;
}

void phys_invert_abel_transl(void *params, int& iter) {
	((abel_params *)params)->iter_ptr = &iter;
	phys_invert_abel(*((abel_params *)params));
}


