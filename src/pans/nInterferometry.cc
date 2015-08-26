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
#include "nInterferometry.h"
#include "neutrino.h"

// physWavelets

nInterferometry::nInterferometry(neutrino *nparent, QString winname)
: nGenericPan(nparent, winname), my_image(2)
{
	my_w.setupUi(this);
	
	region =  new nRect(nparent);
	region->setParentPan(panName,1);
	region->setRect(QRectF(100,100,100,100));
	
    maskRegion =  new nLine(nparent);
    maskRegion->setParentPan(panName,1);
    maskRegion->changeToolTip("MaskLine");
	QPolygonF poly;
	poly << QPointF(50,50) << QPointF(50,150) << QPointF(150,150) << QPointF(150,50);
    maskRegion->setPoints(poly);
    maskRegion->toggleClosedLine(true);

	unwrapBarrier =  new nLine(nparent);
	unwrapBarrier->setParentPan(panName,1);
    unwrapBarrier->changeToolTip("BarrierLine");
    poly.clear();
	poly << QPointF(0,0) << QPointF(100,100);
	unwrapBarrier->setPoints(poly);

	for (unsigned int k=0;k<my_image.size();k++){
		my_image[k].setupUi(my_w.images->widget(k));
        my_w.images->setCurrentIndex(k);
		foreach (QWidget *obj, my_w.images->widget(k)->findChildren<QWidget*>()) {
			obj->setObjectName(obj->objectName()+"Interf"+QString::number(k));
		}

        connect(my_image[k].doit, SIGNAL(released()), this, SLOT(doWavelet()));
    }

	decorate();

    connect(region, SIGNAL(key_pressed(int)), this, SLOT(line_key_pressed(int)));
    connect(maskRegion, SIGNAL(key_pressed(int)), this, SLOT(line_key_pressed(int)));
    connect(unwrapBarrier, SIGNAL(key_pressed(int)), this, SLOT(line_key_pressed(int)));

//    connect(nparent->my_w.my_view, SIGNAL(objectRemoved(QGraphicsObject*)), this, SLOT(removeShape(QGraphicsObject*)));
	connect(my_w.actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
	connect(my_w.actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));
	connect(my_w.actionDoWavelet, SIGNAL(triggered()), this, SLOT(doWavelet()));
	connect(my_w.actionTrash, SIGNAL(triggered()), this, SLOT(doTrash()));
	connect(my_w.actionDuplicate, SIGNAL(triggered()), this, SLOT(duplicate()));

	connect(my_w.actionRect, SIGNAL(triggered()), region, SLOT(togglePadella()));
	connect(my_w.lineBarrier, SIGNAL(released()), unwrapBarrier, SLOT(togglePadella()));
	connect(my_w.maskRegion, SIGNAL(released()), maskRegion, SLOT(togglePadella()));
	
	connect(my_w.doCarrier, SIGNAL(released()), this, SLOT(guessCarrier()));
	connect(my_w.doUnwrap, SIGNAL(released()), this, SLOT(doUnwrap()));
	connect(my_w.doSubtract, SIGNAL(released()), this, SLOT(doSubtract()));
	connect(my_w.doMaskCutoff, SIGNAL(released()), this, SLOT(doMaskCutoff()));
	connect(my_w.doInterpolate, SIGNAL(released()), this, SLOT(doShape()));
	connect(my_w.doPlasma, SIGNAL(released()), this, SLOT(doPlasma()));

    connect(my_w.weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(guessCarrier()));
	
	connect(my_w.useBarrier, SIGNAL(toggled(bool)), this, SLOT(useBarrierToggled(bool)));
	connect(my_w.useMask, SIGNAL(toggled(bool)), this, SLOT(maskRegionToggled(bool)));
    connect(my_w.useInterpolate, SIGNAL(toggled(bool)), this, SLOT(interpolateToggled(bool)));
    
    useBarrierToggled(my_w.useBarrier->isChecked());
	maskRegionToggled(my_w.useMask->isChecked());
	interpolateToggled(my_w.useInterpolate->isChecked());

	connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(bufferChanged(nPhysD*)));
	connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));

	connect(this, SIGNAL(changeCombo(QComboBox *)), this, SLOT(checkChangeCombo(QComboBox *)));
	
    connect(my_w.rotAngle,SIGNAL(valueChanged(double)), this, SLOT(doSubtract()));

    connect(my_w.posZeroButton,SIGNAL(toggled(bool)), this, SLOT(getPosZero(bool)));

    connect(my_w.cutoffValue, SIGNAL(valueChanged(double)), this, SLOT(doMaskCutoff()));

    connect(my_w.addShape, SIGNAL(released()), this, SLOT(addShape()));

    vector<string> localnames=localPhysNames();
    for (vector<string>::const_iterator itr=localnames.begin(); itr!=localnames.end(); itr++) {
        localPhys[*itr]=NULL;
    }

}

void nInterferometry::duplicate() {
    vector<string> localnames=localPhysNames();
    for (vector<string>::const_iterator itr=localnames.begin(); itr!=localnames.end(); itr++) {
        localPhys[*itr]=NULL;
    }
    for (unsigned int iimage=0;iimage<2;iimage++) {
        for(map<string, nPhysD *>::iterator itr = waveletPhys[iimage].begin(); itr != waveletPhys[iimage].end(); ++itr) {
            itr->second=NULL;
        }
    }
}

vector<string> nInterferometry::localPhysNames() {
    vector<string> localnames;
    localnames.push_back("phase");
    localnames.push_back("contrast");
    localnames.push_back("quality");
    localnames.push_back("phaseMask");
    localnames.push_back("angle");
    localnames.push_back("lambda");
    localnames.push_back("interpPhaseMask");
    localnames.push_back("integratedPlasma");
    return localnames;
}


void nInterferometry::line_key_pressed(int key) {
    if (key==Qt::Key_Period) {
        if (sender()==maskRegion) {
            doMaskCutoff();
        } else if (sender()==unwrapBarrier) {
            doUnwrap();
        } else if (sender()==region) {
            doWavelet();
        } else {
            for (map<QToolButton*,nLine*>::iterator it = my_shapes.begin(); it != my_shapes.end(); it++) {
                if (sender()==(*it).second) {
                    doShape();
                }
            }
        }
        nparent->activateWindow();
    }
}

void nInterferometry::physDel(nPhysD* buf) {
    vector<string> localnames=localPhysNames();
    for (vector<string>::const_iterator itr=localnames.begin(); itr!=localnames.end(); itr++) {
        if (buf==localPhys[*itr]) {
            localPhys[*itr]=NULL;
        }
    }
    for (unsigned int iimage=0;iimage<2;iimage++) {
        for(map<string, nPhysD *>::iterator itr = waveletPhys[iimage].begin(); itr != waveletPhys[iimage].end(); ++itr) {
            if (buf==itr->second) itr->second=NULL;
        }
    }
}

void nInterferometry::getPosZero(bool check) {
    if (check) {
        nparent->showPhys(getPhysFromCombo(my_image[1].image));
        connect(nparent->my_w.my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setPosZero(QPointF)));
    } else {
        disconnect(nparent->my_w.my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setPosZero(QPointF)));
    }
}

void nInterferometry::setPosZero(QPointF point) {
    if(currentBuffer) {
        disconnect(my_w.posZeroX,SIGNAL(valueChanged(int)), this, SLOT(doSubtract()));
        disconnect(my_w.posZeroY,SIGNAL(valueChanged(int)), this, SLOT(doSubtract()));

        vec2f my_pos=currentBuffer->to_real(vec2f(point.x(),point.y()));
        my_w.posZeroX->setValue(my_pos.x());
        my_w.posZeroY->setValue(my_pos.y());
        my_w.Zero->setChecked(true);
        doSubtract();

        connect(my_w.posZeroX,SIGNAL(valueChanged(int)), this, SLOT(doSubtract()));
        connect(my_w.posZeroY,SIGNAL(valueChanged(int)), this, SLOT(doSubtract()));
    }
}

void nInterferometry::useBarrierToggled(bool val) {
    unwrapBarrier->setVisible(val);
}

void nInterferometry::maskRegionToggled(bool val) {
    maskRegion->setVisible(val);
}

void nInterferometry::interpolateToggled(bool val) {
    for (map<QToolButton*,nLine*>::iterator it = my_shapes.begin(); it != my_shapes.end(); it++) {
        it->second->setVisible(val);
    }
}

void nInterferometry::checkChangeCombo(QComboBox *combo) {
	if (combo==my_image[0].image || combo==my_image[1].image) {
		region->show();
	}
}

void nInterferometry::bufferChanged(nPhysD* buf) {
	if (buf) {
		if (buf==getPhysFromCombo(my_image[0].image) || buf==getPhysFromCombo(my_image[1].image)) {
			region->show();
		} else {
			region->hide();
		}
		if (buf==localPhys["phase"] ||
            buf==localPhys["phaseMask"] ||
            buf==getPhysFromCombo(my_image[0].image) ||
            buf==getPhysFromCombo(my_image[1].image)) {
            maskRegion->setVisible(my_w.useMask->isChecked());
		} else {
			maskRegion->hide();
		}
	}
    nGenericPan::bufferChanged(buf);
}

void nInterferometry::guessCarrier() {
	nPhysD *image=getPhysFromCombo(my_image[0].image);
	if (image) {
		QRect geom2=region->getRect(image);
		nPhysD datamatrix;
		datamatrix = image->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height());

		vec2f vecCarr=phys_guess_carrier(datamatrix, my_w.weightCarrier->value());
		
		if (vecCarr.first()==0) {
			my_w.statusbar->showMessage(tr("ERROR: Problem finding the carrier"), 5000);
		} else {
			my_w.statusbar->showMessage(tr("Carrier: ")+QString::number(vecCarr.first())+"px "+QString::number(vecCarr.second())+"deg", 5000);
			my_w.widthCarrier->setValue(vecCarr.first());
			my_w.angleCarrier->setValue(vecCarr.second());
		}
	}
}

void nInterferometry::doTrash () {
    for (unsigned int iimage=0;iimage<2;iimage++) {
        for(map<string, nPhysD *>::iterator itr = waveletPhys[iimage].begin(); itr != waveletPhys[iimage].end(); ++itr) {
            nparent->removePhys(itr->second);
            itr->second=NULL;
        }
    }
    vector<string> localnames=localPhysNames();
    for (vector<string>::const_iterator itr=localnames.begin(); itr!=localnames.end(); itr++) {
        if (*itr != "interpPhaseMask") {
            nparent->removePhys(localPhys[*itr]);
        }
    }
}

void nInterferometry::doWavelet () {
    QTime timer;
    timer.start();
    if (sender()==my_w.actionDoWavelet || sender()==region) {
        doWavelet(0);
        doWavelet(1);
    } else {
        for (int iimage=0;iimage<2;iimage++) {
            if(sender()==my_image[iimage].doit) {
                doWavelet(iimage);
            }
        }
    }
    my_w.statusbar->showMessage(QString::number(timer.elapsed())+" msec");
    if (!my_w.chained->isChecked()) doSubtract();
}

void nInterferometry::doWavelet (int iimage) {
    qDebug() << sender();
    string suffix=iimage==0?"ref":"shot";
    nPhysD *image=getPhysFromCombo(my_image[iimage].image);
    if (image) {
        saveDefaults();

        if (my_image[iimage].numAngle->value()==0) {
            my_params.init_angle=my_w.angleCarrier->value();
            my_params.end_angle=my_w.angleCarrier->value();
            my_params.n_angles=1;
        } else {
            my_params.init_angle=my_image[iimage].minAngle->value()+my_w.angleCarrier->value();
            my_params.end_angle=my_image[iimage].maxAngle->value()+my_w.angleCarrier->value();
            my_params.n_angles=my_image[iimage].numAngle->value();
        }
        if (my_image[iimage].numStretch->value()==0) {
            my_params.init_lambda=my_w.widthCarrier->value();
            my_params.end_lambda=my_w.widthCarrier->value();
            my_params.n_lambdas=1;
        } else {
            my_params.init_lambda=my_image[iimage].minStretch->value()*my_w.widthCarrier->value();
            my_params.end_lambda=my_image[iimage].maxStretch->value()*my_w.widthCarrier->value();
            my_params.n_lambdas=my_image[iimage].numStretch->value();
        }
        my_params.thickness=my_w.widthCarrier->value()*my_w.thickness->value();
        my_params.damp=my_w.correlation->value();
        my_params.dosynthetic=true;
        my_params.docropregion=true;
        my_params.trimimages=false;

        QRect geom2=region->getRect(image);
        
        nPhysD datamatrix = image->sub(geom2.left(),geom2.top(),geom2.width(),geom2.height());		
        my_params.data=&datamatrix;

        int niter=my_params.n_angles*my_params.n_lambdas+1;

        QSettings settings("neutrino","");
        settings.beginGroup("Preferences");
        if (settings.value("useCuda").toBool() && cudaEnabled()) {
//            phys_wavelet_field_2D_morlet_cuda(my_params);
            runThread(&my_params, phys_wavelet_trasl_cuda, "Wavelet "+QString(suffix.c_str()), niter);
        } else {
//            phys_wavelet_field_2D_morlet(my_params);
            runThread(&my_params, phys_wavelet_trasl_nocuda, "Wavelet "+QString(suffix.c_str()), niter);
        }

        map<string,nPhysD *> retList = my_params.olist;

        for(map<string, nPhysD *>::const_iterator itr = retList.begin(); itr != retList.end(); ++itr) {
            itr->second->setShortName(suffix+itr->second->getShortName());
            waveletPhys[iimage][itr->first]=nparent->replacePhys(itr->second,waveletPhys[iimage][itr->first],false);
        }
        doUnwrap(iimage);
    }    
}

void nInterferometry::doUnwrap () {
    QProgressDialog progress("Unwrap",QString(), 0, 2, this);
    progress.setWindowModality(Qt::WindowModal);
    for (unsigned int iimage=0;iimage<2;iimage++) {
        progress.setValue(iimage);
        QApplication::processEvents();
        doUnwrap(iimage);
    }
    progress.close();
    if (!my_w.chained->isChecked()) doSubtract();
}

void nInterferometry::doUnwrap (int iimage) {
    string suffix=iimage==0?"ref":"shot";
    
    nPhysD *phase=waveletPhys[iimage]["phase_2pi"];
    nPhysD *qual=waveletPhys[iimage]["contrast"];
    
    if (phase && qual && nPhysExists(phase) && nPhysExists(qual)) {
        
        QTime timer;
        timer.start();
        
        if (qual && phase) {
            nPhysD *unwrap=NULL;
            
            QString methodName=my_w.method->currentText();
            
            nPhysD barrierPhys = nPhysD(phase->getW(),phase->getH(),1.0,"barrier");
            if (my_w.useBarrier->isChecked()) {
                QPolygon my_poly=unwrapBarrier->poly(phase->getW()+phase->getH()).translated(phase->get_origin().x(),phase->get_origin().y()).toPolygon();                    
                for(int ip=0; ip<my_poly.size(); ip++) {
                    QPoint p=my_poly[ip];
                    if (ip>0 && p!=my_poly[ip-1]) {
                        barrierPhys.set(p.x()-1,p.y()-1,0*barrierPhys.point(p.x()-1,p.y()-1));
                        barrierPhys.set(p.x()-1,p.y()  ,0*barrierPhys.point(p.x()-1,p.y()  ));
                        barrierPhys.set(p.x()-1,p.y()+1,0*barrierPhys.point(p.x()-1,p.y()+1));
                        barrierPhys.set(p.x()  ,p.y()-1,0*barrierPhys.point(p.x()  ,p.y()-1));
                        barrierPhys.set(p.x()  ,p.y()  ,0*barrierPhys.point(p.x()  ,p.y()  ));
                        barrierPhys.set(p.x()  ,p.y()+1,0*barrierPhys.point(p.x()  ,p.y()+1));
                        barrierPhys.set(p.x()+1,p.y()-1,0*barrierPhys.point(p.x()+1,p.y()-1));
                        barrierPhys.set(p.x()+1,p.y()  ,0*barrierPhys.point(p.x()+1,p.y()  ));
                        barrierPhys.set(p.x()+1,p.y()+1,0*barrierPhys.point(p.x()+1,p.y()+1));
                    }
                }
            }
            if (methodName=="Simple H+V") {
                unwrap = phys_phase_unwrap(*phase, barrierPhys, SIMPLE_HV);
            } else if (methodName=="Simple V+H") {
                unwrap = phys_phase_unwrap(*phase, barrierPhys, SIMPLE_VH);
            } else if (methodName=="Goldstein") {
                unwrap = phys_phase_unwrap(*phase, barrierPhys, GOLDSTEIN);
            } else if (methodName=="Miguel") {
                unwrap = phys_phase_unwrap(*phase, barrierPhys, MIGUEL_QUALITY);
            } else if (methodName=="Miguel+Quality") {
                phys_point_multiply(barrierPhys,*qual);
                unwrap = phys_phase_unwrap(*phase, barrierPhys, MIGUEL_QUALITY);
            } else if (methodName=="Quality") {
                phys_point_multiply(barrierPhys,*qual);
                unwrap = phys_phase_unwrap(*phase, barrierPhys, QUALITY);
            }
            //                nparent->addPhys(new nPhysD(barrierPhys));
            
            if (unwrap) {
                
                double alpha=my_w.angleCarrier->value();
                double lambda=my_w.widthCarrier->value();
                double kx = cos(alpha*_phys_deg)/lambda;
                double ky = -sin(alpha*_phys_deg)/lambda;
                phys_subtract_carrier(*unwrap, kx, ky);				
                
                unwrap->setShortName(suffix+"unwrap");
                unwrap->setName(unwrap->getShortName()+"-"+methodName.toStdString()+" "+QFileInfo(QString::fromUtf8(phase->getFromName().c_str())).fileName().toStdString());
                unwrap->setFromName(phase->getFromName());
                
                waveletPhys[iimage]["unwrap"]=nparent->replacePhys(unwrap,waveletPhys[iimage]["unwrap"]);
                
            }
        }
    }
}

void nInterferometry::doSubtract () {

    if (waveletPhys[0]["intensity"] && waveletPhys[1]["intensity"]) {
        nPhysD contrast_loc= *waveletPhys[1]["intensity"] / *waveletPhys[0]["intensity"];
        contrast_loc.TscanBrightness();
        localPhys["contrast"]=nparent->replacePhys(contrast_loc.fast_rotated(my_w.rotAngle->value()),localPhys["contrast"]);
        localPhys["contrast"]->setShortName("contrast");
    }

    if (waveletPhys[0]["contrast"] && waveletPhys[1]["contrast"]) {
        nPhysD quality_loc= *waveletPhys[1]["contrast"] * *waveletPhys[0]["contrast"];
        phys_sqrt(quality_loc);
        localPhys["quality"]=nparent->replacePhys(quality_loc.fast_rotated(my_w.rotAngle->value()),localPhys["quality"]);
        localPhys["quality"]->setShortName("quality");
    }

    if (waveletPhys[0]["angle"] && waveletPhys[1]["angle"]) {
        nPhysD angle_loc= *waveletPhys[1]["angle"] - *waveletPhys[0]["angle"];
        angle_loc.TscanBrightness();
        localPhys["angle"]=nparent->replacePhys(angle_loc.fast_rotated(my_w.rotAngle->value()),localPhys["angle"]);
        localPhys["angle"]->setShortName("angle");
    }

    if (waveletPhys[0]["lambda"] && waveletPhys[1]["lambda"]) {
        nPhysD lambda_loc= *waveletPhys[1]["lambda"] / *waveletPhys[0]["lambda"];
        lambda_loc.TscanBrightness();
        localPhys["lambda"]=nparent->replacePhys(lambda_loc.fast_rotated(my_w.rotAngle->value()),localPhys["lambda"]);
        localPhys["lambda"]->setShortName("lambda");
    }

    if (waveletPhys[0]["unwrap"] && waveletPhys[1]["unwrap"]) {

        nPhysD phase;

        if (my_w.opposite->isChecked()) {
            phase = *waveletPhys[0]["unwrap"] - *waveletPhys[1]["unwrap"];
        } else {
            phase = *waveletPhys[1]["unwrap"] - *waveletPhys[0]["unwrap"];
        }
        double offset=phase.point(waveletPhys[1]["unwrap"]->to_pixel(vec2f(my_w.posZeroX->value(),my_w.posZeroY->value())));
        if (std::isfinite(offset)) phys_subtract(phase,offset);

        localPhys["phase"]=nparent->replacePhys(phase.fast_rotated(my_w.rotAngle->value()),localPhys["phase"]);
        localPhys["phase"]->setShortName("phase");

    }
    if (!my_w.chained->isChecked()) doMaskCutoff();
}


void nInterferometry::doMaskCutoff() {
    nPhysD* phase=localPhys["phase"];
    nPhysD *phaseMask=NULL;
    if (nPhysExists(phase)) {
        nparent->showPhys(phase);
        QApplication::processEvents();
        if (my_w.useMask->isChecked()) {
            maskRegion->show();
            nPhysD* phase=localPhys["phase"];
            if (nPhysExists(phase)) {
                phaseMask =new nPhysD(phase->getW(),phase->getH(),0.0);
//                phaseMask =new nPhysD(phase->getW(),phase->getH(),numeric_limits<double>::quiet_NaN());
                phaseMask->property=phase->property;
                //                 _____ _                     
                //                |  ___(_)_  ___ __ ___   ___ 
                //                | |_  | \ \/ / '_ ` _ \ / _ \
                //                |  _| | |>  <| | | | | |  __/
                //                |_|   |_/_/\_\_| |_| |_|\___|
                //                
                QPolygon regionPoly=maskRegion->getLine().toPolygon();
                QRect imageRect=QRect(0,0,phase->getW(),phase->getH());
                QRect rectRegion=regionPoly.boundingRect().intersected(imageRect);
                QProgressDialog progress("Extracting",QString(), 0, rectRegion.width(), this);
                progress.setWindowModality(Qt::WindowModal);
                progress.show();
                for (int i=rectRegion.left(); i<rectRegion.right(); i++) {
                    for (int j=rectRegion.top(); j<rectRegion.bottom(); j++) {
                        if (regionPoly.containsPoint(QPoint(i,j),Qt::OddEvenFill)) {
                            phaseMask->set(i,j,phase->point(i,j));
                        }
                    }
                    progress.setValue(i-rectRegion.left());
                    QApplication::processEvents();
                }
                progress.close();
            }
        } else {
            maskRegion->hide();
        }

        if (phaseMask==NULL) {
            phaseMask=new nPhysD(*phase);
        }

        if (my_w.cutoffValue->value()!=0.0) {
            nPhysD *quality=localPhys["quality"];
            if (nPhysExists(quality)) {
                nPhysD loc_qual(*quality);
                if (my_w.cutoffLog->isChecked()) {
                    phys_log10(loc_qual);
                }
                double mini=loc_qual.get_min();
                double maxi=loc_qual.get_max();
                double valDouble=mini+my_w.cutoffValue->value()*(maxi-mini)/100.0;
                for (size_t k=0; k<loc_qual.getSurf(); k++)
                    if (loc_qual.Timg_buffer[k] < valDouble)
                        phaseMask->Timg_buffer[k]=0.0;
            }
        }

        if (phaseMask==NULL) {
            phaseMask=new nPhysD(*phase);
        }

        phaseMask->TscanBrightness();
        localPhys["phaseMask"]=nparent->replacePhys(phaseMask,localPhys["phaseMask"]);

    }
    if (!my_w.chained->isChecked()) doShape();
}

void nInterferometry::doShape(){
    nPhysD *image=localPhys["phaseMask"];
    
    if (nPhysExists(image)) {
        nPhysD *regionPath = new nPhysD(*image);
        regionPath->setShortName("phaseInterp");
        regionPath->setName("phase interpolated");
        
        if (my_w.useInterpolate->isChecked()) {
            QProgressDialog progress("Interpolate", "Stop", 0, my_shapes.size(), this);            
            progress.setWindowModality(Qt::WindowModal);
            progress.show();
            int prog=0;
            for (map<QToolButton*,nLine*>::iterator it = my_shapes.begin(); it != my_shapes.end(); it++, prog++) {
                progress.setValue(prog);
                if (progress.wasCanceled()) break;
                QApplication::processEvents();

                nLine* region = (*it).second;
                
                region->toggleClosedLine(true); // ensure no mess
                
                QPolygonF regionPoly=region->poly(1).translated(image->get_origin().x(),image->get_origin().y());
                
                // convert QPolygonF to vector<vec2f>
                vector<vec2f> vecPoints(regionPoly.size());
                for(int k=0;k<regionPoly.size();k++) {
                    QPointF pp=regionPoly[k];
                    vecPoints[k]=vec2f(pp.x(),pp.y());
                }
                
                // these are the points on wich we will calculate the weighted mean
                const unsigned int npoints=20;
                vector<pair<vec2f, double> > vals;
                for(unsigned int k=0;k<vecPoints.size()-1;k++) {
                    vec2f p1=vecPoints[k];
                    vec2f p2=vecPoints[k+1];
                    for(unsigned int kk=0;kk<npoints;kk++) {
                        vec2f pp=p1+kk*(p2-p1)/npoints;
                        double val=regionPath->point(pp);
                        if (isfinite(val)) {
                            vals.push_back(make_pair(pp, val));
                        }
                    }
                }
                
                QRect rectRegion=regionPoly.boundingRect().toRect().adjusted(-1,-1,+1,+1);
                
                for (int i=rectRegion.left(); i<=rectRegion.right(); i++) {
                    for (int j=rectRegion.top(); j<=rectRegion.bottom(); j++) {
                        vec2f pp(i,j);
                        if (point_inside_poly(pp,vecPoints)) {
                            double mean=0;
                            double weight=0;
                            int m=0;
                            for(vector<pair<vec2f,double> >::iterator it=vals.begin();it!=vals.end();++it){
                                vec2f p=it->first;
                                double pval=it->second;
                                double wi=1.0/(pp-p).mod2();
                                mean+=wi*pval;
                                weight+=wi; 
                                m++;
                            }
                            double ppval=mean/weight;
                            regionPath->set(pp,ppval);
                        }
                    }
                }          
            }            
        }
        localPhys["interpPhaseMask"]=nparent->replacePhys(regionPath,localPhys["interpPhaseMask"]);
    }
    if (!my_w.chained->isChecked()) doPlasma();
}

void nInterferometry::addShape(){
    int num=0;
    while (1) {
        num++;
        bool found =false;
        foreach (QObject* widget, nparent->children()) {
            nLine *line=qobject_cast<nLine *>(widget);
            if (line && line->property("parentPan").toString()==panName) {
                if (line->toolTip()=="interpolateShape"+QString::number(num)) {
                    found=true;
                }
            }
        }
        if (!found) {
            addShape("interpolateShape"+QString::number(num));
            break;
        }
    }
}

void nInterferometry::addShape(QString name){
    nLine *my_l=new nLine(nparent);
    my_l->setParentPan(panName,0);
	QPolygonF poly;
    if (my_shapes.size()==0){
        poly << QPointF(50,50) << QPointF(50,150) << QPointF(150,150) << QPointF(150,50);
    } else {
        poly=my_shapes.begin()->second->getPoints();
    }
    my_l->setPoints(poly);
    my_l->changeToolTip(name);
    my_l->toggleClosedLine(true);
    QToolButton *my_b=new QToolButton(this);
    my_b->setIcon(QIcon(":icons/line"));
    my_b->setToolTip(name+my_b->toolTip());
    my_shapes[my_b]=my_l;
    my_w.shapes->layout()->addWidget(my_b);
    
    connect(my_l, SIGNAL(key_pressed(int)), this, SLOT(line_key_pressed(int)));
    connect(my_l, SIGNAL(destroyed(QObject*)), this, SLOT(removeShape(QObject*)));
    connect(my_b, SIGNAL(released()), my_l, SLOT(togglePadella()));
}

void nInterferometry::removeShape(QObject *obj){
    QToolButton *found=NULL;
    for (map<QToolButton*,nLine*>::iterator it = my_shapes.begin(); it != my_shapes.end(); it++) {
        if ((*it).second==obj) {
            (*it).first->deleteLater(); 
            found=(*it).first;
        }
    }
    if (found) my_shapes.erase(my_shapes.find(found));
}

void nInterferometry::doPlasma(){
    if (my_w.useInterpolate->isChecked()) {
        nPhysD *image=localPhys["interpPhaseMask"];
        if (nPhysExists(image)) {
            nPhysD *intNe = new nPhysD(*image);
            
            if (my_w.usePlasma->isChecked()) {
                
                double lambda_m=my_w.probeLambda->value()*1e-9; // nm to m
                double scale_cm=my_w.imgRes->value()*1e-4; // convert micron/px to cm/px
                
                double toNe = -1.0e-4*8.0*M_PI*M_PI*_phys_emass*_phys_vacuum_eps*_phys_cspeed*_phys_cspeed/(_phys_echarge*_phys_echarge*lambda_m);
                
                phys_multiply(*intNe, toNe); 
                intNe->setShortName("intergratedNe");
                intNe->set_scale(scale_cm,scale_cm);
                intNe->property["unitsX"]="cm";
                intNe->property["unitsY"]="cm";
                intNe->property["unitsCB"]="cm-2";
            }
            
            bool ok1,ok2;
            double mini=my_w.cutoffMin->text().toDouble(&ok1);
            double maxi=my_w.cutoffMax->text().toDouble(&ok2);
            if (!ok1) mini=intNe->get_min();
            if (!ok2) maxi=intNe->get_max();
            phys_cutoff(*intNe,mini,maxi);
            intNe->TscanBrightness();
            localPhys["integratedPlasma"]=nparent->replacePhys(intNe,localPhys["integratedPlasma"]);
        }        
    }
}


//////////////////////////////////////////////////////////

void nInterferometry::loadSettings(QSettings *settings){
    QStringList valu=settings->value("interpolateShape").toStringList();
    foreach (QString name, valu) {
        bool found=false;
        foreach (QObject* widget, nparent->children()) {
            nLine *line=qobject_cast<nLine *>(widget);
            if (line && line->property("parentPan").toString()==panName) {
                if (line->toolTip()==name) found=true;
            }
        }            
        if (!found) addShape(name);
    }    
    nGenericPan::loadSettings(settings);
}

void nInterferometry::saveSettings(QSettings *settings){
    QStringList names;
    for (map<QToolButton*,nLine*>::iterator it = my_shapes.begin(); it != my_shapes.end(); it++) {
        names.append((*it).second->toolTip());
    }
    settings->setValue("interpolateShape", names);

    nGenericPan::saveSettings(settings);
}



