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
: nGenericPan(nparent, winname)
{
	my_w.setupUi(this);
    
	region =  new nRect(nparent);
	region->setParentPan(panName,1);
	region->setRect(QRectF(100,100,100,100));
	
    lineRegion =  new nLine(nparent);
    lineRegion->setParentPan(panName,1);
    lineRegion->changeToolTip(winname+"Line1");
    
	linebarrier =  new nLine(nparent);
	linebarrier->setParentPan(panName,1);
	QPolygonF poly;
	poly << QPointF(0,0) << QPointF(100,100);
	linebarrier->setPoints(poly);
    
    QList<QWidget*> father1;
	father1<< my_w.image1 <<my_w.image2;
	for (int k=0;k<2;k++){
		my_image[k].setupUi(father1.at(k));
		father1.at(k)->show();
		//hack to save diffrent uis!!!
		foreach (QWidget *obj, father1.at(k)->findChildren<QWidget*>()) {
			obj->setObjectName(obj->objectName()+"Interf"+QString::number(k));
		}
    }        
    
    
	decorate();
    
	connect(my_w.actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
	connect(my_w.actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));
	connect(my_w.actionCarrier, SIGNAL(triggered()), this, SLOT(guessCarrier()));
	connect(my_w.actionDoAll, SIGNAL(triggered()), this, SLOT(doWavelet()));
    
	connect(my_w.actionRect, SIGNAL(triggered()), region, SLOT(togglePadella()));
	
    connect(my_w.weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(guessCarrier()));
	
	connect(my_w.useBarrier, SIGNAL(toggled(bool)), this, SLOT(useBarrierToggled(bool)));
	connect(my_w.lineBarrier, SIGNAL(released()), linebarrier, SLOT(togglePadella()));
	useBarrierToggled(my_w.useBarrier->isChecked());
    
	connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(bufferChanged(nPhysD*)));
	connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));
    
	connect(this, SIGNAL(changeCombo(QComboBox *)), this, SLOT(checkChangeCombo(QComboBox *)));
	
    connect(my_w.method,SIGNAL(currentIndexChanged(int)), this, SLOT(doUnwrap()));
        
    connect(my_w.posZero,SIGNAL(stateChanged(int)), this, SLOT(doSubtract()));
    connect(my_w.rotAngle,SIGNAL(valueChanged(double)), this, SLOT(doSubtract()));
    
    connect(my_w.posZeroX,SIGNAL(valueChanged(int)), this, SLOT(doSubtract()));
    connect(my_w.posZeroY,SIGNAL(valueChanged(int)), this, SLOT(doSubtract()));
    connect(my_w.posZeroButton,SIGNAL(released()), this, SLOT(getPosZero()));

//	connect(my_w.cutoff,SIGNAL(valueChanged(double)),this,SLOT(doCutoff()));
	connect(my_w.useRegion, SIGNAL(toggled(bool)), this, SLOT(useRegionToggled(bool)));
	connect(my_w.lineBarrier, SIGNAL(released()), linebarrier, SLOT(togglePadella()));
	useRegionToggled(my_w.useRegion->isChecked());

    connect(my_w.abel,SIGNAL(stateChanged(int)), this, SLOT(doAbel()));
    connect(my_w.symmetric,SIGNAL(released()), this, SLOT(doAbel()));
    connect(my_w.abelY,SIGNAL(valueChanged(int)), this, SLOT(doAbel()));
    connect(my_w.abelButton,SIGNAL(released()), this, SLOT(getPosAbel()));

    vector<string> localnames=localPhysNames();
    for (vector<string>::const_iterator itr=localnames.begin(); itr!=localnames.end(); itr++) {
        localPhys[*itr]=NULL;
    }
    
}

vector<string> nInterferometry::localPhysNames() {
    vector<string> localnames;
    localnames.push_back("phase");
    localnames.push_back("contrast");
    localnames.push_back("quality");
    localnames.push_back("phaseCutoff");
    localnames.push_back("abel");
    return localnames;
}

void nInterferometry::physDel(nPhysD* buf) {
    vector<string> localnames=localPhysNames();
    for (vector<string>::const_iterator itr=localnames.begin(); itr!=localnames.end(); itr++) {
        if (buf==localPhys[*itr]) {
            localPhys[*itr]=NULL;
        }
    }
}

void nInterferometry::getPosZero() {
    my_w.posZeroButton->setChecked(true);
    connect(nparent->my_w.my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(setPosZero(QPointF)));
    nparent->showPhys(getPhysFromCombo(my_image[1].image));
}

void nInterferometry::setPosZero(QPointF point) {
    if(currentBuffer) {
        disconnect(my_w.posZeroX,SIGNAL(valueChanged(int)), this, SLOT(doSubtract()));
        disconnect(my_w.posZeroY,SIGNAL(valueChanged(int)), this, SLOT(doSubtract()));
        vec2f my_pos=currentBuffer->to_real(vec2f(point.x(),point.y()));
        my_w.posZeroX->setValue(my_pos.x());
        my_w.posZeroY->setValue(my_pos.y());
        my_w.posZeroButton->setChecked(false);
        my_w.Zero->setChecked(true);
        disconnect(nparent->my_w.my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(setPosZero(QPointF)));
        doSubtract();
        connect(my_w.posZeroX,SIGNAL(valueChanged(int)), this, SLOT(doSubtract()));
        connect(my_w.posZeroY,SIGNAL(valueChanged(int)), this, SLOT(doSubtract()));
    }    
}


void nInterferometry::getPosAbel() {
    my_w.abelButton->setChecked(true);
    connect(nparent->my_w.my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(setPosAbel(QPointF)));
    nparent->showPhys(localPhys["phase"]);
}

void nInterferometry::setPosAbel(QPointF point) {
    if(currentBuffer) {
        disconnect(my_w.abelY,SIGNAL(valueChanged(int)), this, SLOT(doAbel()));
        vec2f my_pos=currentBuffer->to_real(vec2f(point.x(),point.y()));
        my_w.abelY->setValue(my_pos.y());
        my_w.abelButton->setChecked(false);
        my_w.abel->setChecked(true);
        disconnect(nparent->my_w.my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(setPosAbel(QPointF)));
        doAbel();
        connect(my_w.abelY,SIGNAL(valueChanged(int)), this, SLOT(doAbel()));
    }
}

void nInterferometry::useBarrierToggled(bool valDouble) {
	if (valDouble) {
		linebarrier->show();
	} else {
		linebarrier->hide();
	}
}

void nInterferometry::useRegionToggled(bool valDouble) {
	if (valDouble) {
		lineRegion->show();
	} else {
		lineRegion->hide();
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
		if (buf==localPhys["phase"]) {
			lineRegion->show();
		} else {
			lineRegion->hide();
		}
	}
    nGenericPan::bufferChanged(buf);
}

void nInterferometry::guessCarrier() {
	nPhysD *image=getPhysFromCombo(my_image[0].image);
	if (image) {
		QRect geom2=region->getRect();
		QPoint offset=geom2.topLeft();
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

void nInterferometry::doWavelet () {
    QTime timer;
    timer.start();
    
    for (int iimage=0;iimage<2;iimage++) {
        string suffix=iimage==0?"ref":"shot";
        nPhysD *image=getPhysFromCombo(my_image[iimage].image);
        if (image) {
            saveDefaults();
            QRect geom2=region->getRect();
            QPoint offset=geom2.topLeft();
            
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
            my_params.thickness=my_w.widthCarrier->value()*my_w.correlation->value();
            my_params.damp=my_w.correlation->value();
            my_params.dosynthetic=true;
            my_params.docropregion=true;
            my_params.trimimages=true;
            
            nPhysD datamatrix = image->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height());		
            my_params.data=&datamatrix;
            
            int niter=my_params.n_angles*my_params.n_lambdas;
            
            QSettings settings("neutrino","");
            settings.beginGroup("Preferences");
            if (settings.value("useCuda").toBool() && cudaEnabled()) {
                runThread(&my_params, phys_wavelet_trasl_cuda, "Interferometry...", niter);
            } else {
                runThread(&my_params, phys_wavelet_trasl_nocuda, "Interferometry...", niter);
            }
            
            map<string,nPhysD *> retList = my_params.olist;
            
            for(map<string, nPhysD *>::const_iterator itr = retList.begin(); itr != retList.end(); ++itr) {
                itr->second->setShortName(suffix+itr->second->getShortName());
                waveletPhys[iimage][itr->first]=nparent->replacePhys(itr->second,waveletPhys[iimage][itr->first],false);                
            }
        }
    }
        
    doUnwrap();
    
    my_w.statusbar->showMessage(QString::number(timer.elapsed())+" msec");

}

// --------------------------------------------------------------------------

void nInterferometry::doUnwrap () {
    
    for (unsigned int iimage=0;iimage<2;iimage++) {
        string suffix=iimage==0?"ref":"shot";

        nPhysD *phase=waveletPhys[iimage]["phase_2pi"];
        nPhysD *qual=waveletPhys[iimage]["contrast"];
        
        if (phase && qual) {
            nPhysD barrierPhys;
            
            QTime timer;
            timer.start();
            
            if (qual && phase) {
                nPhysD *unwrap=NULL;
                
                QString methodName=my_w.method->currentText();
                
                if (my_w.useBarrier->isChecked()) {
                    barrierPhys = nPhysD(phase->getW(),phase->getH(),1.0,"barrier");
                    foreach(QPointF p, linebarrier->poly(phase->getW()+phase->getH())) {
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
                } else {
                    if (methodName=="Simple H+V") {
                        unwrap = phys_phase_unwrap(*phase, *qual, SIMPLE_HV);
                    } else if (methodName=="Simple V+H") {
                        unwrap = phys_phase_unwrap(*phase, *qual, SIMPLE_VH);
                    } else if (methodName=="Goldstein") {
                        unwrap = phys_phase_unwrap(*phase, *qual, GOLDSTEIN);
                    } else if (methodName=="Miguel") {
                        unwrap = phys_phase_unwrap(*phase, *qual, MIGUEL);
                    } else if (methodName=="Miguel+Quality") {
                        unwrap = phys_phase_unwrap(*phase, *qual, MIGUEL_QUALITY);
                    } else if (methodName=="Quality") {
                        unwrap = phys_phase_unwrap(*phase, *qual, QUALITY);
                    }
                }
                
                if (unwrap) {
                    unwrap->setShortName(suffix+"unwrap");
                    unwrap->setName(unwrap->getShortName()+"-"+methodName.toStdString()+" "+QFileInfo(QString::fromUtf8(phase->getFromName().c_str())).fileName().toStdString());
                    unwrap->setFromName(phase->getFromName());
                    waveletPhys[iimage]["unwrap"]=nparent->replacePhys(unwrap,waveletPhys[iimage]["unwrap"]);
                    
                }
            }
        }
    }
    
    doSubtract();    
}

void nInterferometry::doSubtract () {
    if (waveletPhys[0]["intensity"] && waveletPhys[1]["intensity"]) {
        nPhysD contrast_loc= *waveletPhys[1]["intensity"] / *waveletPhys[0]["intensity"];
        localPhys["contrast"]=nparent->replacePhys(contrast_loc.fast_rotated(my_w.rotAngle->value()),localPhys["contrast"]);
        localPhys["contrast"]->setShortName("contrast");
    }
    
    if (waveletPhys[0]["contrast"] && waveletPhys[1]["contrast"]) {
        nPhysD quality_loc= *waveletPhys[1]["contrast"] * *waveletPhys[0]["contrast"];
        phys_sqrt(quality_loc);
        quality_loc.setShortName("quality");
        localPhys["quality"]=nparent->replacePhys(quality_loc.fast_rotated(my_w.rotAngle->value()),localPhys["quality"]);
        localPhys["quality"]->setShortName("quality");
    }
    if (waveletPhys[0]["unwrap"] && waveletPhys[1]["unwrap"]) {
        
        double offset=0.0;
        
        if (my_w.posZero->isChecked()) {
            vec2f absPoint=waveletPhys[1]["unwrap"]->to_pixel(vec2f(my_w.posZeroX->value(),my_w.posZeroY->value()));
            offset=waveletPhys[0]["unwrap"]->point(absPoint)-waveletPhys[1]["unwrap"]->point(absPoint);
            if (!std::isfinite(offset)) {
                my_w.statusbar->showMessage("Point outside");
                offset=0.0;
            }
            my_w.offset->setText(QString::number((int)(100.0*(offset-((int) offset))))+"%");
        }
        nPhysD phase= *waveletPhys[1]["unwrap"] - *waveletPhys[0]["unwrap"];
        if (offset!=0.0) phys_add(phase, offset);  
        localPhys["phase"]=nparent->replacePhys(phase.fast_rotated(my_w.rotAngle->value()),localPhys["phase"]);
        localPhys["phase"]->setShortName("phase");
    }
    doCutoff();
}


void nInterferometry::doCutoff () {
    nPhysD *regionPath = NULL;
    if (nPhysExists(localPhys["phase"])) {
        nPhysD* image=localPhys["phase"];
        if (my_w.useRegion->isChecked()) {
            regionPath = new nPhysD(image->getW(),image->getH(),0.0);
            vec2 offset=image->get_origin();
            offset=vec2(0,0);
            
            
            QPolygon regionPoly=lineRegion->poly(1).toPolygon();
            //            QRect rectRegion=lineRegion->boundingRect().toRect().intersected(QRect(0,0,image->getW(),image->getH()));
            
            QRect imageRect=QRect(-offset.x(),-offset.y(),image->getW(),image->getH());
            QRect rectRegion=regionPoly.boundingRect().intersected(imageRect);
            
            qDebug() << ">>>>>>>>>>>>>>>>>>>>>> " << imageRect << regionPoly.boundingRect();
            qDebug() << ">>>>>>>>>>>>>>>>>>>>>> " << rectRegion;
            qDebug() << ">>>>>>>>>>>>>>>>>>>>>> " << offset.x() << offset.y();
            regionPath->property=image->property;
            regionPath->setShortName("Region mask");
            QProgressDialog progress("Extracting", "Stop", 0, rectRegion.width(), this);
            progress.setWindowModality(Qt::WindowModal);
            progress.show();
            for (int i=rectRegion.left(); i<rectRegion.right(); i++) {
                if (progress.wasCanceled()) break;
                QApplication::processEvents();
                for (int j=rectRegion.top(); j<rectRegion.bottom(); j++) {
                    if (regionPoly.containsPoint(QPoint(i,j),Qt::OddEvenFill)) {
                        regionPath->set(i+offset.x(),j+offset.y(),image->point(i+offset.x(),j+offset.y()));
                    }
                    
                }
                progress.setValue(i-rectRegion.left());
            }
            regionPath->TscanBrightness();
            localPhys["phaseCutoff"]=nparent->replacePhys(regionPath,localPhys["phaseCutoff"]);
        } else {
            localPhys["phaseCutoff"] = image;        
        }
    }        
    
//    if(nPhysExists(localPhys["phase"]) && nPhysExists(localPhys["quality"])) {
//        nPhysD *phase=localPhys["phase"];
//        nPhysD *quality=localPhys["quality"];
//
//        double mini=quality->Tminimum_value;
//        double maxi=quality->Tmaximum_value;
//        
//        double valDouble=mini+my_w.cutoff->value()*(maxi-mini)/100.0;
//        
//        DEBUG(mini << " " << valDouble << " " << my_w.cutoff->value());
//        
//        if (phase->getW() == quality->getW() && phase->getH() == quality->getH()) {
//            nPhysD *phaseCutoff = new nPhysD(phase->getW(),phase->getH(), numeric_limits<double>::quiet_NaN());
//            phaseCutoff->set_origin(phase->get_origin());
//            phaseCutoff->set_scale(phase->get_scale());
//            for (size_t k=0; k<phase->getSurf(); k++) 
//                if (quality->Timg_buffer[k] >= valDouble) 
//                    phaseCutoff->Timg_buffer[k]=phase->Timg_buffer[k];
//            
//            std::ostringstream my_name;
//            my_name << "mask(" << quality->getName() << "," << valDouble << ")";
//            phaseCutoff->setName(my_name.str());
//            phaseCutoff->setShortName("mask");
//            phaseCutoff->setFromName(phase->getFromName());
//            phaseCutoff->TscanBrightness();
//            
//            localPhys["phaseCutoff"]=nparent->replacePhys(phaseCutoff,localPhys["phaseCutoff"]);
//        }    
//    }
    doAbel();    
}


void nInterferometry::doAbel() {
    if (my_w.abel->isChecked() && nPhysExists(localPhys["phaseCutoff"])) {
        int dx=localPhys["phaseCutoff"]->getW();
        int dy=localPhys["phaseCutoff"]->getH();
        nPhysD* data = localPhys["phaseCutoff"];

        nPhysD* abel= new nPhysD(dx,dy,0.0,"Abel");
        abel->property=data->property;
        int y0=(abel->to_pixel(vec2f(0,my_w.abelY->value()))).y();
        if (my_w.symmetric->isChecked()) {
        	int dy2=min(y0,dy-y0);
        	for (int i=0;i<dx;i++){
        		for (int j=0;j<dy2;j++){
        			for (int k=j+1; k<dy2;k++) {
        				double val=0.5*(data->point(i,y0+k) + data->point(i,y0-k) - data->point(i,y0+k-1) - data->point(i,y0-(k-1)));
                        if (isfinite(val)){
                            abel->set(i,j+y0,abel->point(i,j+y0)-val/sqrt(k*k-j*j)/M_PI);
                        } else {
                            break;
                        }

        			}
        			abel->set(i,y0-j,abel->point(i,j+y0));	
        		}
        	}
        } else {
        	for (int i=0;i<dx;i++){
        		for (int j=y0;j>=0;j--){ // lower part
        			for (int k=j-1; k>0;k--) {
                        double val=(data->point(i,k-1)-data->point(i,k))/sqrt((k-y0)*(k-y0)-(j-y0)*(j-y0))/M_PI;
                        if (!isfinite(val)) break;
        				abel->set(i,j,abel->point(i,j)-val);
        			}
        		}
        		for (int j=y0;j<dy;j++){ // upper part
        			for (int k=j+1; k<dy;k++) {
                        double val=(data->point(i,k)-data->point(i,k-1))/sqrt((k-y0)*(k-y0)-(j-y0)*(j-y0))/M_PI;
                        if (!isfinite(val)) break;
        				abel->set(i,j,abel->point(i,j)-val);
        			}
        		}
        		abel->set(i,y0,abel->point(i,y0)/2.0);
        	}
        }        
        phys_apply_inversion_plasma(*abel, my_w.probeLambda->text().toDouble()*1e-9, -my_w.imgRes->text().toDouble()/2*M_PI);
        phys_fast_gaussian_blur(*abel,0.2*my_w.widthCarrier->value()*my_w.correlation->value());

        localPhys["abel"]=nparent->replacePhys(abel,localPhys["abel"], true);
    }
}
