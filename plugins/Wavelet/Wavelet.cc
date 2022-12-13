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
#include "Wavelet.h"
#include "neutrino.h"

// physWavelets

Wavelet::Wavelet(neutrino *nparent) : nGenericPan(nparent), region(this,1), linebarrier(this,1)
{
    setupUi(this);

    region.setRect(QRectF(100,100,100,100));
	
	QPolygonF poly;
	poly << QPointF(0,0) << QPointF(100,100);
    linebarrier.setPoints(poly);
    show();

    connect(actionCarrier, SIGNAL(triggered()), this, SLOT(guessCarrier()));
    connect(actionDoAll, SIGNAL(triggered()), this, SLOT(doAll()));

    connect(actionRect, SIGNAL(triggered()), &region, SLOT(togglePadella()));
	
    connect(doWaveletB, SIGNAL(pressed()), this, SLOT(doWavelet()));
    connect(doUnwrapB, SIGNAL(pressed()), this, SLOT(doUnwrap()));

    connect(weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(guessCarrier()));

    connect(doRemoveB, SIGNAL(pressed()), this, SLOT(doRemove()));
	
    connect(useBarrier, SIGNAL(toggled(bool)), this, SLOT(useBarrierToggled(bool)));
    connect(lineBarrier, SIGNAL(released()), &linebarrier, SLOT(togglePadella()));
    useBarrierToggled(useBarrier->isChecked());

    widthCarrierLabel->setText(QLocale().toString(widthCarrier->value())+widthCarrier->suffix());
    angleCarrierLabel->setText(QLocale().toString(angleCarrier->value())+angleCarrier->suffix());
    connect(widthCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
    connect(angleCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
    connect(weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
	
    origSubmatrix=unwrapPhys=referencePhys=carrierPhys=syntheticPhys=nullptr;
}

void Wavelet::on_relative_toggled(bool tog) {
    minStretch->setSuffix(tog?" X":" [px]");
    maxStretch->setSuffix(tog?" X":" [px]");
    minThick->setSuffix(tog?" X":" [px]");
    maxThick->setSuffix(tog?" X":" [px]");
}

void Wavelet::useBarrierToggled(bool val) {
	if (val) {
        linebarrier.show();
	} else {
        linebarrier.hide();
	}
}

void Wavelet::bufferChanged(nPhysD* buf) {
    nGenericPan::bufferChanged(buf);
    if (buf) {
        if (buf==getPhysFromCombo(image)) {
            region.show();
		} else {
            region.hide();
		}
	}
}

void Wavelet::physDel(nPhysD* buf) {
    std::vector<nPhysD *> localPhys;
    localPhys.push_back(origSubmatrix);
    localPhys.push_back(unwrapPhys);
    localPhys.push_back(referencePhys);
    localPhys.push_back(carrierPhys);
    localPhys.push_back(syntheticPhys);
    for (std::vector<nPhysD *>::iterator itr=localPhys.begin(); itr!=localPhys.end(); itr++) {
        if (buf==*itr) {
            *itr=nullptr;
        }
    }
}


void Wavelet::guessCarrier() {
    QApplication::processEvents();
    nPhysD *my_phys=getPhysFromCombo(image);
    if (my_phys) {
        QRect geom2=region.getRect(my_phys);
        qDebug() << geom2;

		nPhysD datamatrix;
        DEBUG(my_phys);
        DEBUG("1");
        datamatrix = my_phys->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height());
        DEBUG("2");

        vec2f vecCarr=physWave::phys_guess_carrier(datamatrix, weightCarrier->value());
		
		if (vecCarr.first()==0) {
            statusbar->showMessage(tr("ERROR: Problem finding the carrier"), 5000);
		} else {
            statusbar->showMessage(tr("Carrier: ")+QLocale().toString(vecCarr.first())+"px "+QLocale().toString(vecCarr.second())+"deg", 5000);
            disconnect(widthCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
            disconnect(angleCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
            disconnect(weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
            widthCarrier->setValue(vecCarr.first());
            angleCarrier->setValue(vecCarr.second());
            widthCarrierLabel->setText(QLocale().toString(widthCarrier->value())+widthCarrier->suffix());
            angleCarrierLabel->setText(QLocale().toString(angleCarrier->value())+angleCarrier->suffix());
            connect(widthCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
            connect(angleCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
            connect(weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(doRemoveCarrier()));
		}
	}
}

void Wavelet::doWavelet () {
    setEnabled(false);
    nPhysD *my_phys=getPhysFromCombo(image);
    if (my_phys) {
        QElapsedTimer timer;
		timer.start();

		saveDefaults();
        QRect geom2=region.getRect();

        nPhysD datamatrix = my_phys->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height());
		
		double conversionAngle=0.0;
		double conversionStretch=1.0;
        if (relative->isChecked()) {
            conversionAngle=angleCarrier->value();
            conversionStretch=widthCarrier->value();
		}

		//qCalculation_th my_qt;

		QSettings settings("neutrino","");
        settings.beginGroup("nPreferences");

        if (numAngle->value()==0) {
            my_params.init_angle=angleCarrier->value();
            my_params.end_angle=angleCarrier->value();
			my_params.n_angles=1;
		} else {
            my_params.init_angle=minAngle->value()+conversionAngle;
            my_params.end_angle=maxAngle->value()+conversionAngle;
            my_params.n_angles=numAngle->value();
		}
        if (numStretch->value()==0) {
            my_params.init_lambda=widthCarrier->value();
            my_params.end_lambda=widthCarrier->value();
            my_params.n_lambdas=1;
        } else {
            my_params.init_lambda=minStretch->value()*conversionStretch;
            my_params.end_lambda=maxStretch->value()*conversionStretch;
            my_params.n_lambdas=numStretch->value();
        }
        if (numThick->value()==0) {
            my_params.init_thick=widthCarrier->value();
            my_params.end_thick=widthCarrier->value();
            my_params.n_thicks=1;
        } else {
            my_params.init_thick=minThick->value()*conversionStretch;
            my_params.end_thick=maxThick->value()*conversionStretch;
            my_params.n_thicks=numThick->value();
        }
        my_params.damp=damp->value();
        my_params.data=&datamatrix;

        QString out;

        qDebug() << physWave::openclEnabled() << settings.value("openclUnit").toInt();

        if (physWave::openclEnabled()>0 && settings.value("openclUnit").toInt()>0) {
            out="OpenCL: ";
            my_params.opencl_unit=settings.value("openclUnit").toInt();
            runThread(&my_params, physWave::phys_wavelet_trasl_opencl, "OpenCL wavelet", my_params.n_angles*my_params.n_lambdas*my_params.n_thicks);
        } else {
            out="CPU: ";
            runThread(&my_params, physWave::phys_wavelet_trasl_cpu, "CPU wavelet", my_params.n_angles*my_params.n_lambdas*my_params.n_thicks);
        }


        erasePrevious->setEnabled(true);
        for(auto &itr : my_params.olist) {
            if (!(itr.first=="angle"  && my_params.n_angles ==1) &&
                !(itr.first=="lambda" && my_params.n_lambdas==1) &&
                !(itr.first=="thick"  && my_params.n_thicks ==1)) {
                nPhysD *this_phys = new nPhysD(*itr.second);
                if (erasePrevious->isChecked()) {
                    waveletPhys[itr.first]=nparent->replacePhys(this_phys,waveletPhys[itr.first],false);
                } else {
                    nparent->addPhys(this_phys);
                    waveletPhys[itr.first]=this_phys;
                }
            }
            delete itr.second;
        }
        QApplication::processEvents();
        
        for(auto &itr: waveletPhys) {
            if (itr.first=="phase_2pi") {
                imageUnwrap->setCurrentIndex(imageUnwrap->findData(QVariant::fromValue(itr.second)));
            } else if (itr.first=="contrast") {
                qualityUnwrap->setCurrentIndex(qualityUnwrap->findData(QVariant::fromValue(itr.second)));
            }
        }
        
        if (showSource->isChecked()) {
			datamatrix.setShortName("wavelet source");
            nPhysD *deepcopy=new nPhysD(datamatrix);
            if (erasePrevious->isChecked()) {
				origSubmatrix=nparent->replacePhys(deepcopy,origSubmatrix,false);
			} else {
				nparent->addPhys(deepcopy);
				origSubmatrix=deepcopy;
			}
		}
        if (synthetic->isChecked()) {
            nPhysD *tmpSynthetic = new nPhysD(physWave::phys_synthetic_interferogram(waveletPhys["phase_2pi"],waveletPhys["contrast"]));
            if (erasePrevious->isChecked()) {
                syntheticPhys=nparent->replacePhys(tmpSynthetic,syntheticPhys,false);
            } else {
                nparent->addPhys(tmpSynthetic);
                syntheticPhys=tmpSynthetic;
            }
        }
        
        QString status_bar_measure=QString("%1 sec, %2 Mpx/s").arg(1.0e-3*timer.elapsed(),0,' ',1).arg(1.0e-3*my_params.n_angles*my_params.n_lambdas*my_params.n_thicks*geom2.width()*geom2.height()/timer.elapsed(),0,' ',1);
        statusbar->showMessage(out+status_bar_measure, 50000);
        DEBUG(status_bar_measure.toStdString());
    } else {
        statusbar->showMessage("Canceled", 5000);
    }
    QApplication::processEvents();
    setEnabled(true);
}


// --------------------------------------------------------------------------

void Wavelet::doUnwrap () {
    physD *phase=static_cast<physD*>(getPhysFromCombo(imageUnwrap));
    physD *qual=static_cast<physD*>(getPhysFromCombo(qualityUnwrap));
	nPhysD barrierPhys;
	
    QElapsedTimer timer;
	timer.start();

	if (qual && phase) {
        physD uphase;

        QString methodName=method->currentText();

        if (useBarrier->isChecked()) {
			barrierPhys = nPhysD(phase->getW(),phase->getH(),1.0,"barrier");
            QPolygonF my_poly=linebarrier.poly(phase->getW()+phase->getH());
            my_poly.translate(qual->get_origin().x(),qual->get_origin().y());
			foreach(QPointF p, my_poly) {
				barrierPhys.set(p.x()-1,p.y()-1,0.0);
				barrierPhys.set(p.x()-1,p.y()  ,0.0);
				barrierPhys.set(p.x()-1,p.y()+1,0.0);
				barrierPhys.set(p.x()  ,p.y()-1,0.0);
				barrierPhys.set(p.x()  ,p.y()  ,0.0);
				barrierPhys.set(p.x()  ,p.y()+1,0.0);
				barrierPhys.set(p.x()+1,p.y()-1,0.0);
				barrierPhys.set(p.x()+1,p.y()  ,0.0);
				barrierPhys.set(p.x()+1,p.y()+1,0.0);
			}
			if (methodName=="Simple H+V") {
                physWave::phys_phase_unwrap(*phase, barrierPhys, physWave::SIMPLE_HV, uphase);
			} else if (methodName=="Simple V+H") {
                physWave::phys_phase_unwrap(*phase, barrierPhys, physWave::SIMPLE_VH, uphase);
			} else if (methodName=="Goldstein") {
                physWave::phys_phase_unwrap(*phase, barrierPhys, physWave::GOLDSTEIN, uphase);
			} else if (methodName=="Miguel") {
                phys_phase_unwrap(*phase, barrierPhys, physWave::MIGUEL_QUALITY, uphase);
			} else if (methodName=="Miguel+Quality") {
                physMath::phys_point_multiply(barrierPhys,*qual);
                physWave::phys_phase_unwrap(*phase, barrierPhys, physWave::MIGUEL_QUALITY, uphase);
			} else if (methodName=="Quality") {
                physMath::phys_point_multiply(barrierPhys,*qual);
                physWave::phys_phase_unwrap(*phase, barrierPhys, physWave::QUALITY, uphase);
			}
		} else {
			if (methodName=="Simple H+V") {
                physWave::phys_phase_unwrap(*phase, *qual, physWave::SIMPLE_HV, uphase);
			} else if (methodName=="Simple V+H") {
                physWave::phys_phase_unwrap(*phase, *qual, physWave::SIMPLE_VH, uphase);
			} else if (methodName=="Goldstein") {
                physWave::phys_phase_unwrap(*phase, *qual, physWave::GOLDSTEIN, uphase);
			} else if (methodName=="Miguel") {
                physWave::phys_phase_unwrap(*phase, *qual, physWave::MIGUEL, uphase);
			} else if (methodName=="Miguel+Quality") {
                physWave::phys_phase_unwrap(*phase, *qual, physWave::MIGUEL_QUALITY, uphase);
			} else if (methodName=="Quality") {
                physWave::phys_phase_unwrap(*phase, *qual, physWave::QUALITY, uphase);
			}
		}

        uphase.setShortName("unwrap");
        uphase.setName(uphase.getShortName()+"-"+methodName.toStdString()+" "+QFileInfo(QString::fromUtf8(phase->getFromName().c_str())).fileName().toStdString());
        uphase.setFromName(phase->getFromName());
        erasePreviousUnwrap->setEnabled(true);

        if (removeCarrierAfterUnwrap->isChecked()) {
            physWave::phys_subtract_carrier(uphase, angleCarrier->value(), widthCarrier->value());
        }

        if (erasePreviousUnwrap->isChecked()) {
            unwrapPhys=nparent->replacePhys(new nPhysD(uphase),unwrapPhys);
        } else {
            unwrapPhys=new nPhysD(uphase);
            nparent->addShowPhys(unwrapPhys);
        }
        unwrapped->setCurrentIndex(unwrapped->findData(QVariant::fromValue(unwrapPhys)));

	}
    statusbar->showMessage(QString("%1 msec").arg(timer.elapsed()));
}

void Wavelet::doAll () {
	doWavelet();
	doUnwrap();
}

void Wavelet::doRemove () {
    if (carrier->isChecked()) {
		doRemoveCarrier();
	}
    if (reference->isChecked()) {
		doRemoveReference();
	}
}

void Wavelet::doRemoveCarrier () {
    widthCarrierLabel->setText(QLocale().toString(widthCarrier->value())+widthCarrier->suffix());
    angleCarrierLabel->setText(QLocale().toString(angleCarrier->value())+angleCarrier->suffix());
    if (sender() && (sender()==widthCarrier || sender()==angleCarrier || sender()==weightCarrier) && (!liveView->isChecked())) {
		return;
	}
    carrier->setChecked(true);
	// check normalize
    nPhysD *my_phys=getPhysFromCombo(unwrapped);
    if (my_phys) {
		nPhysD *unwrappedSubtracted;
        unwrappedSubtracted = new nPhysD(*my_phys);

        double alpha=angleCarrier->value();
        double lambda=widthCarrier->value();

		unwrappedSubtracted->setName("No carrier ("+QLocale().toString(lambda).toStdString()+","+
																 QLocale().toString(alpha).toStdString()+") "+
                                                                 QFileInfo(QString::fromUtf8(my_phys->getFromName().c_str())).fileName().toStdString());
		unwrappedSubtracted->setShortName("No carrier");
        unwrappedSubtracted->setFromName(my_phys->getFromName());

        physWave::phys_subtract_carrier(*unwrappedSubtracted, alpha, lambda);
        physMath::phys_subtract(*unwrappedSubtracted, phaseOffset->value());
        erasePreviuos->setEnabled(true);
        if (erasePreviuos->isChecked()) {
			carrierPhys=nparent->replacePhys(unwrappedSubtracted,carrierPhys);
		} else {
			nparent->addShowPhys(unwrappedSubtracted);
		}
	}
}

void Wavelet::doRemoveReference () {
	// check normalize
    nPhysD *ref=getPhysFromCombo(refUnwrap);
    nPhysD *my_phys=getPhysFromCombo(unwrapped);
    if (my_phys && ref) {
        if (my_phys->getW() == ref->getW() && my_phys->getH() == ref->getH()) {
			nPhysD *unwrappedSubtracted;
			unwrappedSubtracted = new nPhysD();
            *unwrappedSubtracted=(*my_phys)-(*ref);
            physMath::phys_subtract(*unwrappedSubtracted, phaseOffset->value());
            unwrappedSubtracted->setName(my_phys->getName()+" - "+ref->getName());
			unwrappedSubtracted->setName("Reference removed : "+ref->getName());
			unwrappedSubtracted->setShortName("Ref removed");
            if (my_phys->getFromName()==ref->getFromName()) {
                unwrappedSubtracted->setFromName(my_phys->getFromName());
			} else {
                unwrappedSubtracted->setFromName(my_phys->getFromName()+" "+ref->getFromName());
			}
            erasePreviuos->setEnabled(true);
            if (erasePreviuos->isChecked()) {
				referencePhys=nparent->replacePhys(unwrappedSubtracted,referencePhys);
			} else {
				nparent->addShowPhys(unwrappedSubtracted);
			}
		}
	}
}

