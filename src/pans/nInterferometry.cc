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
	
    maskRegion =  new nLine(nparent);
    maskRegion->setParentPan(panName,1);
    maskRegion->changeToolTip(winname+"Line1");
    
    
	unwrapBarrier =  new nLine(nparent);
	unwrapBarrier->setParentPan(panName,1);
	QPolygonF poly;
	poly << QPointF(0,0) << QPointF(100,100);
	unwrapBarrier->setPoints(poly);

    QList<QWidget*> father1;
	father1<< my_w.image1 <<my_w.image2;
	for (int k=0;k<2;k++){
		my_image[k].setupUi(father1.at(k));
		father1.at(k)->show();
		//hack to save diffrent uis!!!
		foreach (QWidget *obj, father1.at(k)->findChildren<QWidget*>()) {
			obj->setObjectName(obj->objectName()+"Interf"+QString::number(k));
		}
        
        connect(my_image[k].doit, SIGNAL(released()), this, SLOT(doWavelet()));
    }        
    
    
	decorate();
    
    connect(region, SIGNAL(key_pressed(int)), this, SLOT(line_key_pressed(int)));
    connect(maskRegion, SIGNAL(key_pressed(int)), this, SLOT(line_key_pressed(int)));
    connect(unwrapBarrier, SIGNAL(key_pressed(int)), this, SLOT(line_key_pressed(int)));
    
    
	connect(my_w.actionLoadPref, SIGNAL(triggered()), this, SLOT(loadSettings()));
	connect(my_w.actionSavePref, SIGNAL(triggered()), this, SLOT(saveSettings()));
	connect(my_w.actionCarrier, SIGNAL(triggered()), this, SLOT(guessCarrier()));
	connect(my_w.actionDoAll, SIGNAL(triggered()), this, SLOT(doWavelet()));
    
	connect(my_w.actionRect, SIGNAL(triggered()), region, SLOT(togglePadella()));
	
    connect(my_w.weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(guessCarrier()));
	
	connect(my_w.useBarrier, SIGNAL(toggled(bool)), this, SLOT(useBarrierToggled(bool)));
	connect(my_w.lineBarrier, SIGNAL(released()), unwrapBarrier, SLOT(togglePadella()));
	useBarrierToggled(my_w.useBarrier->isChecked());
    
	connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(bufferChanged(nPhysD*)));
	connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));
    
	connect(this, SIGNAL(changeCombo(QComboBox *)), this, SLOT(checkChangeCombo(QComboBox *)));
	
    connect(my_w.method,SIGNAL(currentIndexChanged(int)), this, SLOT(doUnwrap()));
        
    connect(my_w.posZero,SIGNAL(stateChanged(int)), this, SLOT(doSubtract()));
    connect(my_w.rotAngle,SIGNAL(valueChanged(double)), this, SLOT(doSubtract()));
    
    connect(my_w.posZeroX,SIGNAL(valueChanged(int)), this, SLOT(doSubtract()));
    connect(my_w.posZeroY,SIGNAL(valueChanged(int)), this, SLOT(doSubtract()));
    connect(my_w.posZeroButton,SIGNAL(toggled(bool)), this, SLOT(getPosZero(bool)));

//	connect(my_w.Mask,SIGNAL(valueChanged(double)),this,SLOT(doMask()));
	connect(my_w.useMask, SIGNAL(stateChanged(int)), this, SLOT(doMask()));
	connect(my_w.maskRegion, SIGNAL(released()), maskRegion, SLOT(togglePadella()));

    connect(my_w.abel,SIGNAL(stateChanged(int)), this, SLOT(doAbel()));
    connect(my_w.symmetric,SIGNAL(released()), this, SLOT(doAbel()));
    connect(my_w.abelY,SIGNAL(editingFinished()), this, SLOT(doAbel()));
    connect(my_w.abelButton,SIGNAL(toggled(bool)), this, SLOT(getPosAbel(bool)));

    vector<string> localnames=localPhysNames();
    for (vector<string>::const_iterator itr=localnames.begin(); itr!=localnames.end(); itr++) {
        localPhys[*itr]=NULL;
    }
    
}

void nInterferometry::line_key_pressed(int key) {
    if (key==Qt::Key_Period) {
        if (sender()==maskRegion) {
            doMask();
        } else if (sender()==unwrapBarrier) {
            doUnwrap();
        } else if (sender()==region) {
            doWavelet();
        }
    }
}

vector<string> nInterferometry::localPhysNames() {
    vector<string> localnames;
    localnames.push_back("phase");
    localnames.push_back("contrast");
    localnames.push_back("quality");
    localnames.push_back("intergratedNe");
    localnames.push_back("intergratedNeMask");
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
        
        if (nPhysExists(waveletPhys[0]["unwrap"]) && nPhysExists(waveletPhys[1]["unwrap"])) {
            vec2f my_pos=waveletPhys[0]["unwrap"]->to_real(vec2f(point.x(),point.y()));
            my_w.posZeroX->setValue(my_pos.x());
            my_w.posZeroY->setValue(my_pos.y());
            my_w.Zero->setChecked(true);
            doSubtract();
        }
        
        connect(my_w.posZeroX,SIGNAL(valueChanged(int)), this, SLOT(doSubtract()));
        connect(my_w.posZeroY,SIGNAL(valueChanged(int)), this, SLOT(doSubtract()));
    }    
}


void nInterferometry::getPosAbel(bool check) {
    if (check) {
        nparent->showPhys(localPhys["intergratedNe"]);
        connect(nparent->my_w.my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setPosAbel(QPointF)));
    } else {
        disconnect(nparent->my_w.my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setPosAbel(QPointF)));
    }
}

void nInterferometry::setPosAbel(QPointF point) {
    if(currentBuffer) {
        disconnect(my_w.abelY,SIGNAL(valueChanged(int)), this, SLOT(doAbel()));
        vec2f my_pos=currentBuffer->to_real(vec2f(point.x(),point.y()));
        my_w.abelY->setValue(my_pos.y());
        my_w.abel->setChecked(true);
        doAbel();
        connect(my_w.abelY,SIGNAL(valueChanged(int)), this, SLOT(doAbel()));
    }
}

void nInterferometry::useBarrierToggled(bool valDouble) {
	if (valDouble) {
		unwrapBarrier->show();
	} else {
		unwrapBarrier->hide();
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
		if (buf==localPhys["phase"] || buf==localPhys["intergratedNeMask"]) {
			maskRegion->show();
		} else {
			maskRegion->hide();
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
        if(sender()==my_image[iimage].doit) {
            doWavelet(iimage);
        }
    }
    if (sender()==my_w.actionDoAll || sender()==region) {
        doWavelet(0);
        doWavelet(1);
    }    
    doUnwrap ();        
    my_w.statusbar->showMessage(QString::number(timer.elapsed())+" msec");
}

void nInterferometry::doWavelet (int iimage) {
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
        my_params.thickness=my_w.widthCarrier->value()*my_w.thickness->value();
        my_params.damp=my_w.correlation->value();
        my_params.dosynthetic=true;
        my_params.docropregion=true;
        my_params.trimimages=false;
        
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

void nInterferometry::doUnwrap () {    
    QProgressDialog progress("Extracting",QString(), 0, 2, this);
    progress.show();
    for (unsigned int iimage=0;iimage<2;iimage++) {
        progress.setValue(iimage);
        QApplication::processEvents();
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
                    foreach(QPointF p, unwrapBarrier->poly(phase->getW()+phase->getH())) {
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
        }
        nPhysD phase= *waveletPhys[1]["unwrap"] - *waveletPhys[0]["unwrap"];
        if (offset!=0.0) phys_add(phase, offset);  
        
        localPhys["phase"]=nparent->replacePhys(phase.fast_rotated(my_w.rotAngle->value()),localPhys["phase"]);
        localPhys["phase"]->setShortName("phase");

        
        nPhysD intNe(phase);
        double lambda_m=my_w.probeLambda->text().toDouble()*1e-9;
        double mult = 8.0*M_PI*M_PI*_phys_emass*_phys_vacuum_eps*_phys_cspeed*_phys_cspeed/(_phys_echarge*_phys_echarge*lambda_m);         
                
        phys_multiply(intNe, mult/1.0e4); // convert m-2 to cm-2
        
        localPhys["intergratedNe"]=nparent->replacePhys(intNe.fast_rotated(my_w.rotAngle->value()),localPhys["intergratedNe"], true);
        localPhys["intergratedNe"]->setShortName("intergratedNe");
    }
    doMask();
}


void nInterferometry::doMask () {
    if (my_w.useMask->isChecked()) {
        maskRegion->show();
        nPhysD* image=localPhys["intergratedNe"];
        nPhysD *regionPhys= NULL;
        if (nPhysExists(image)) {
            regionPhys = new nPhysD(image->getW(),image->getH(),0.0);
            QPolygon regionPoly=maskRegion->getLine().toPolygon();
            QRect imageRect=QRect(0,0,image->getW(),image->getH());
            QRect rectRegion=regionPoly.boundingRect().intersected(imageRect);
            regionPhys->property=image->property;
            regionPhys->setShortName("intergratedNeMask");
            QProgressDialog progress("Extracting",QString(), 0, rectRegion.width(), this);
            progress.setWindowModality(Qt::WindowModal);
            progress.show();
            for (int i=rectRegion.left(); i<rectRegion.right(); i++) {
                for (int j=rectRegion.top(); j<rectRegion.bottom(); j++) {
                    if (regionPoly.containsPoint(QPoint(i,j),Qt::OddEvenFill)) {
                        regionPhys->set(i,j,image->point(i,j));
                    }
                }
                progress.setValue(i-rectRegion.left());
                QApplication::processEvents();
            }
            regionPhys->TscanBrightness();
            phys_fast_gaussian_blur(*regionPhys,0.5*my_w.widthCarrier->value()*my_w.thickness->value());
            
            localPhys["intergratedNeMask"]=nparent->replacePhys(regionPhys,localPhys["intergratedNeMask"]);
        }
    } else {
        maskRegion->hide();
    }
    doAbel();    
}


void nInterferometry::doAbel() {
    if (my_w.abel->isChecked() && nPhysExists(localPhys["intergratedNeMask"])) {
        
        int dx=localPhys["intergratedNeMask"]->getW();
        int dy=localPhys["intergratedNeMask"]->getH();
        nPhysD* data = localPhys["intergratedNeMask"];

        nPhysD* abel= new nPhysD(dx,dy,0.0,"Abel");
        abel->property=data->property;
        int y0=(abel->to_pixel(vec2f(0,my_w.abelY->value()))).y();
        
        QProgressDialog progress("Abel", QString(), 0, dx, this);
        progress.setWindowModality(Qt::WindowModal);
        progress.show();

        if (my_w.symmetric->isChecked()) {
        	int dy2=min(y0,dy-y0);
        	for (int i=0;i<dx;i++){
                progress.setValue(i);
                QApplication::processEvents();
        		for (int j=0;j<dy2;j++){
        			for (int k=j+1; k<dy2;k++) {
        				double val=0.5*(data->point(i,y0+k) - data->point(i,y0+k-1) + data->point(i,y0-k) - data->point(i,y0-(k-1)))/M_PI;
                        abel->set(i,j+y0,abel->point(i,j+y0)-val/sqrt(k*k-j*j));
                    }
        			abel->set(i,y0-j,abel->point(i,j+y0));	
        		}
        	}
        } else {
        	for (int i=0;i<dx;i++){
                progress.setValue(i);
                QApplication::processEvents();
        		for (int j=y0;j>=0;j--){ // lower part
        			for (int k=j-1; k>0;k--) {
                        double val=(data->point(i,k-1)-data->point(i,k))/sqrt((k-y0)*(k-y0)-(j-y0)*(j-y0))/M_PI;
        				abel->set(i,j,abel->point(i,j)-val);
        			}
        		}
        		for (int j=y0;j<dy;j++){ // upper part
        			for (int k=j+1; k<dy;k++) {
                        double val=(data->point(i,k)-data->point(i,k-1))/sqrt((k-y0)*(k-y0)-(j-y0)*(j-y0))/M_PI;
        				abel->set(i,j,abel->point(i,j)-val);
        			}
        		}
        		abel->set(i,y0,abel->point(i,y0)/2.0);
        	}
        }        
        double mult=my_w.imgRes->text().toDouble()*1e-4; // convert micron/px to cm/px
        abel->set_scale(mult,mult);
        phys_divide(*abel,mult);


        if (my_w.symmetric->isChecked()) {
            abel->setShortName("AbelSymm");
            abel->setName("AbelSymm("+data->getName()+")");
        } else {
            abel->setShortName("Abel");
            abel->setName("Abel("+data->getName()+")");
        }
        localPhys["abel"]=nparent->replacePhys(abel,localPhys["abel"], true);
    }
}
