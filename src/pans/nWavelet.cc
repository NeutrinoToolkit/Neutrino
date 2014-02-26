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
#include "nWavelet.h"
#include "neutrino.h"

// physWavelets

nWavelet::nWavelet(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname)
{
	my_w.setupUi(this);

	region =  new nRect(nparent);
	region->setParentPan(panName,1);
	region->setRect(QRectF(100,100,100,100));

	decorate();

	connect(my_w.actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
	connect(my_w.actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));
	connect(my_w.actionCarrier, SIGNAL(triggered()), this, SLOT(guessCarrier()));
	connect(my_w.actionDoAll, SIGNAL(triggered()), this, SLOT(doAll()));

	connect(my_w.actionRect, SIGNAL(triggered()), region, SLOT(togglePadella()));

	connect(my_w.doWavelet, SIGNAL(pressed()), this, SLOT(doWavelet()));
	connect(my_w.doUnwrap, SIGNAL(pressed()), this, SLOT(doUnwrap()));

	connect(my_w.weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(guessCarrier()));

	connect(my_w.doRemove, SIGNAL(pressed()), this, SLOT(doRemove()));

	my_w.widthCarrierLabel->setText(QString::number(my_w.widthCarrier->value())+my_w.widthCarrier->suffix());
	my_w.angleCarrierLabel->setText(QString::number(my_w.angleCarrier->value())+my_w.angleCarrier->suffix());
	connect(my_w.widthCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
	connect(my_w.angleCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
	connect(my_w.weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));

	origSubmatrix=unwrapPhys=referencePhys=carrierPhys=NULL;
	waveletPhys.resize(5);
	for (unsigned int i=0;i<waveletPhys.size();i++) {
		waveletPhys.at(i)=NULL;
	}
}


void nWavelet::guessCarrier() {
	nPhysD *image=getPhysFromCombo(my_w.image);
	if (image) {
		QRect geom2=region->getRect();
		QPoint offset=geom2.topLeft();
		nPhysD datamatrix;
		datamatrix = image->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height());

		vec2f vecCarr=phys_guess_carrier(datamatrix, my_w.weightCarrier->value());
		qDebug() << datamatrix.getH();

		if (vecCarr.first()==0) {
			my_w.statusbar->showMessage(tr("ERROR: Problem finding the carrier"), 5000);
		} else {
			my_w.statusbar->showMessage(tr("Carrier: ")+QString::number(vecCarr.first())+"px "+QString::number(vecCarr.second())+"deg", 5000);
			disconnect(my_w.widthCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
			disconnect(my_w.angleCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
			disconnect(my_w.weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
			my_w.widthCarrier->setValue(vecCarr.first());
			my_w.angleCarrier->setValue(vecCarr.second());
			my_w.widthCarrierLabel->setText(QString::number(my_w.widthCarrier->value())+my_w.widthCarrier->suffix());
			my_w.angleCarrierLabel->setText(QString::number(my_w.angleCarrier->value())+my_w.angleCarrier->suffix());
			connect(my_w.widthCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
			connect(my_w.angleCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
			connect(my_w.weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
		}
	}
}

void nWavelet::doWavelet () {
	nPhysD *image=getPhysFromCombo(my_w.image);
	if (image) {
		
		nPhysD *bufferDisplayed=currentBuffer;
		QTime timer;
		timer.start();

		saveDefaults();
		QRect geom2=region->getRect();
		QPoint offset=geom2.topLeft();
		nPhysD *datamatrix = new nPhysD();
		
		*datamatrix=image->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height());
		datamatrix->setShortName("wavelet source");

		if (my_w.erasePrevious->isChecked()) {
			origSubmatrix=nparent->replacePhys(datamatrix,origSubmatrix,false);
		} else {
			nparent->addPhys(datamatrix);
			origSubmatrix=datamatrix;
		}

		double conversionAngle=0.0;
		double conversionStretch=1.0;
		if (my_w.relative->isChecked()) {
			conversionAngle=my_w.angleCarrier->value();
			conversionStretch=my_w.widthCarrier->value();
		}

		//qCalculation_th my_qt;

		QSettings settings("neutrino","");
		settings.beginGroup("Preferences");


		//my_qt.useCuda=settings.value("useCuda").toBool();

		wavelet_params my_params;
		if (my_w.numAngle->value()==0) {
			my_params.init_angle=my_w.angleCarrier->value();
			my_params.end_angle=my_w.angleCarrier->value();
			my_params.n_angles=1;
		} else {
			my_params.init_angle=my_w.minAngle->value()+conversionAngle;
			my_params.end_angle=my_w.maxAngle->value()+conversionAngle;
			my_params.n_angles=my_w.numAngle->value();
		}
		if (my_w.numStretch->value()==0) {
			my_params.init_lambda=my_w.widthCarrier->value();
			my_params.end_lambda=my_w.widthCarrier->value();
			my_params.n_lambdas=1;
		} else {
			my_params.init_lambda=my_w.minStretch->value()*conversionStretch;
			my_params.end_lambda=my_w.maxStretch->value()*conversionStretch;
			my_params.n_lambdas=my_w.numStretch->value();
		}
		my_params.thickness=my_w.thickness->value();
		my_params.damp=my_w.damp->value();
		
		if (settings.value("useCuda").toBool() && cudaEnabled()) {
			// use cuda
			nThread.setThread(origSubmatrix, &my_params, phys_wavelet_trasl_cuda);
		} else {
			// don't use cuda
			nThread.setThread(origSubmatrix, &my_params, phys_wavelet_trasl_nocuda);
		}
		
		nThread.setTitle("Wavelet...");
		progressRun(my_params.n_angles*my_params.n_lambdas);

		std::list<nPhysD *>::const_iterator itr;
		my_w.erasePrevious->setEnabled(true);
		unsigned int position=0;
		if (nThread.n_iter==0) {
			QMessageBox::critical(this, tr("Neutrino Wavelet"),tr("CUDA didn't work.\nDisable from preferences window"),QMessageBox::Ok);
		}
		for(itr = nThread.odata.begin(); itr != nThread.odata.end() && position<waveletPhys.size(); ++itr, ++position) {
			nPhysD *mat=(*itr);
			WARNING(nThread.odata.size() << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   " << mat->getShortName()<< " : " << position);
			if ((mat->getShortName()=="angle" && my_params.n_angles==1) ||
				mat->getShortName()=="lambda" && my_params.n_lambdas==1) {
				delete mat;
				mat=NULL;
			} else {
				if (my_w.erasePrevious->isChecked()) {
					waveletPhys.at(position)=nparent->replacePhys(mat,waveletPhys.at(position),false);
				} else {
					nparent->addPhys(mat);
					waveletPhys.at(position)=mat;
				}
				if (waveletPhys.at(position)->getShortName()=="phase") {
					my_w.imageUnwrap->setCurrentIndex(my_w.imageUnwrap->findData(qVariantFromValue((void*)(waveletPhys.at(position)))));
				} else if (waveletPhys.at(position)->getShortName()=="quality") {
					my_w.qualityUnwrap->setCurrentIndex(my_w.imageUnwrap->findData(qVariantFromValue((void*)(waveletPhys.at(position)))));
				}
			}
		}
		//delete my_qt.risultati;
		nparent->showPhys(bufferDisplayed);
		QString out;
		out.sprintf(": %.2f sec, %.2f Mpx/s",1.0e-3*timer.elapsed(), 1.0e-3*my_params.n_angles*my_params.n_lambdas*geom2.width()*geom2.height()/timer.elapsed());
		if (settings.value("useCuda").toBool()) {
			out.prepend("GPU");
		} else {
			out.prepend("CPU");
		}
		my_w.statusbar->showMessage(out);

	}
}

// ---------------------- thread transport functions ------------------------

std::list<nPhysD *>
phys_wavelet_trasl_cuda(nPhysD *iimage, void *params, int &iter) {
	((wavelet_params *)params)->iter_ptr = &iter;
	std::list<nPhysD *> retList = *phys_wavelet_field_2D_morlet_cuda(*iimage, *((wavelet_params *)params));
	return retList;
}

std::list<nPhysD *>
phys_wavelet_trasl_nocuda(nPhysD *iimage, void *params, int &iter) {
	((wavelet_params *)params)->iter_ptr = &iter;
	std::list<nPhysD *> retList =  *phys_wavelet_field_2D_morlet(*iimage, *((wavelet_params *)params));
	return retList;
}

// --------------------------------------------------------------------------

void nWavelet::doUnwrap () {
	nPhysD *phase=getPhysFromCombo(my_w.imageUnwrap);
	nPhysD *qual=getPhysFromCombo(my_w.qualityUnwrap);
	if (qual && phase) {
		nPhysD *uphase=NULL;
		QString methodName=my_w.method->currentText();
		// esistono sicuramente dei metodi piu' intelligenti
		if (methodName=="Simple H+V") {
			uphase = phys_phase_unwrap(*phase, *qual, SIMPLE_HV);
		} else if (methodName=="Simple V+H") {
			uphase = phys_phase_unwrap(*phase, *qual, SIMPLE_VH);
		} else if (methodName=="Goldstein") {
			uphase = phys_phase_unwrap(*phase, *qual, GOLDSTEIN);
		} else if (methodName=="Miguel") {
			uphase = phys_phase_unwrap(*phase, *qual, MIGUEL);
		} else if (methodName=="Quality") {
			uphase = phys_phase_unwrap(*phase, *qual, QUALITY);
		}
		
		if (uphase) {
			uphase->setShortName("unwrap");
			uphase->setName(uphase->getShortName()+"-"+methodName.toStdString()+" "+QFileInfo(QString::fromStdString(phase->getFromName())).fileName().toStdString());
			uphase->setFromName(phase->getFromName());
			my_w.erasePreviousUnwrap->setEnabled(true);

			uphase->TscanBrightness();
			if (my_w.erasePreviousUnwrap->isChecked()) {
				unwrapPhys=nparent->replacePhys(uphase,unwrapPhys);
			} else {
				unwrapPhys=uphase;
				nparent->addShowPhys(unwrapPhys);
			}
			my_w.unwrapped->setCurrentIndex(my_w.unwrapped->findData(qVariantFromValue((void*)unwrapPhys)));
		}
	}
}

void nWavelet::doAll () {
	doWavelet();
	doUnwrap();
	doRemove();
}

void nWavelet::doRemove () {
	if (my_w.carrier->isChecked()) {
		doRemoveCarrier();
	}
	if (my_w.reference->isChecked()) {
		doRemoveReference();
	}
}

void nWavelet::doRemoveCarrier () {
	my_w.widthCarrierLabel->setText(QString::number(my_w.widthCarrier->value())+my_w.widthCarrier->suffix());
	my_w.angleCarrierLabel->setText(QString::number(my_w.angleCarrier->value())+my_w.angleCarrier->suffix());
	if (sender() && (sender()==my_w.widthCarrier || sender()==my_w.angleCarrier || sender()==my_w.weightCarrier) && (!my_w.liveView->isChecked())) {
		return;
	}
	my_w.carrier->setChecked(true);
	// check normalize
	nPhysD *unwrapped=getPhysFromCombo(my_w.unwrapped);
	if (unwrapped) {
		nPhysD *unwrappedSubtracted;
		unwrappedSubtracted = new nPhysD(*unwrapped);

		double alpha=my_w.angleCarrier->value();
		double lambda=my_w.widthCarrier->value();

		unwrappedSubtracted->setName("No carrier ("+QString::number(lambda).toStdString()+","+
																 QString::number(alpha).toStdString()+") "+
																 QFileInfo(QString::fromStdString(unwrapped->getFromName())).fileName().toStdString());
		unwrappedSubtracted->setShortName("No carrier");
		unwrappedSubtracted->setFromName(unwrapped->getFromName());

		double kx = cos(alpha*_phys_deg)/lambda;
		double ky = -sin(alpha*_phys_deg)/lambda;
		phys_subtract_carrier(*unwrappedSubtracted, kx, ky);
		phys_subtract(*unwrappedSubtracted, my_w.phaseOffset->value());
		unwrappedSubtracted->TscanBrightness();
		my_w.erasePreviuos->setEnabled(true);
		if (my_w.erasePreviuos->isChecked()) {
			carrierPhys=nparent->replacePhys(unwrappedSubtracted,carrierPhys);
		} else {
			nparent->addShowPhys(unwrappedSubtracted);
		}
	}
}

void nWavelet::doRemoveReference () {
	// check normalize
	nPhysD *ref=getPhysFromCombo(my_w.refUnwrap);
	nPhysD *unwrapped=getPhysFromCombo(my_w.unwrapped);
	if (unwrapped && ref) {
		if (unwrapped->getW() == ref->getW() && unwrapped->getH() == ref->getH()) {
			nPhysD *unwrappedSubtracted;
			unwrappedSubtracted = new nPhysD();
			*unwrappedSubtracted=(*unwrapped)-(*ref);
			phys_subtract(*unwrappedSubtracted, my_w.phaseOffset->value());
			unwrappedSubtracted->setName(unwrapped->getName()+" - "+ref->getName());
			unwrappedSubtracted->setName("Reference removed : "+ref->getName());
			unwrappedSubtracted->setShortName("Ref removed");
			if (unwrapped->getFromName()==ref->getFromName()) {
				unwrappedSubtracted->setFromName(unwrapped->getFromName());
			} else {
				unwrappedSubtracted->setFromName(unwrapped->getFromName()+" "+ref->getFromName());
			}
			unwrappedSubtracted->TscanBrightness();
			my_w.erasePreviuos->setEnabled(true);
			if (my_w.erasePreviuos->isChecked()) {
				referencePhys=nparent->replacePhys(unwrappedSubtracted,referencePhys);
			} else {
				nparent->addShowPhys(unwrappedSubtracted);
			}
		}
	}
}

