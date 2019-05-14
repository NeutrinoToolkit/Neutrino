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
#include "Interferometry.h"
#include "neutrino.h"

// physWavelets

Interferometry::Interferometry(neutrino *nparent) : nGenericPan(nparent) {
    my_w.setupUi(this);

    region =  new nRect(this,1);
    region->setRect(QRectF(100,100,100,100));

    maskRegion =  new nLine(this,1);
    maskRegion->changeToolTip("MaskLine");
    QPolygonF poly;
    poly << QPointF(50,50) << QPointF(50,150) << QPointF(150,150) << QPointF(150,50);
    maskRegion->setPoints(poly);
    maskRegion->toggleClosedLine(true);

    unwrapBarrier =  new nLine(this,1);
    unwrapBarrier->changeToolTip("BarrierLine");
    poly.clear();
    poly << QPointF(0,0) << QPointF(100,100);
    unwrapBarrier->setPoints(poly);

    for (unsigned int k=0;k<my_image.size();k++){
        my_w.images->setCurrentIndex(k);
        my_image[k].setupUi(my_w.images->widget(k));
        foreach (QWidget *my_obj,  my_w.images->widget(k)->findChildren<QWidget *>()) {
            if (!my_obj->objectName().isEmpty()) {
                my_obj->setObjectName(my_obj->objectName()+panName()+QLocale().toString(k));
                my_obj->setProperty("id",k);
            }
        }
        connect(my_image[k].doit, SIGNAL(released()), this, SLOT(doWavelet()));
    }

    show();

    connect(my_w.images, SIGNAL(currentChanged(int)), this, SLOT(imagesTabBarClicked(int)));
    connect(region, SIGNAL(key_pressed(int)), this, SLOT(line_key_pressed(int)));
    connect(maskRegion, SIGNAL(key_pressed(int)), this, SLOT(line_key_pressed(int)));
    connect(unwrapBarrier, SIGNAL(key_pressed(int)), this, SLOT(line_key_pressed(int)));

    connect(my_w.actionDoWavelet, SIGNAL(triggered()), this, SLOT(doWavelet()));

    connect(my_w.actionRect, SIGNAL(triggered()), region, SLOT(togglePadella()));
    connect(my_w.lineBarrier, SIGNAL(released()), unwrapBarrier, SLOT(togglePadella()));
    connect(my_w.maskRegion, SIGNAL(released()), maskRegion, SLOT(togglePadella()));

    connect(my_w.doCarrier, SIGNAL(released()), this, SLOT(guessCarrier()));
    connect(my_w.doUnwrap, SIGNAL(released()), this, SLOT(doUnwrap()));
    connect(my_w.doSubtract, SIGNAL(released()), this, SLOT(doSubtract()));
    connect(my_w.doMaskCutoff, SIGNAL(released()), this, SLOT(doMaskCutoff()));
    connect(my_w.doInterpolate, SIGNAL(released()), this, SLOT(doShape()));
    connect(my_w.doCutoff, SIGNAL(released()), this, SLOT(doCutoff()));

    connect(my_w.weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(guessCarrier()));

    connect(my_w.useBarrier, SIGNAL(toggled(bool)), this, SLOT(useBarrierToggled(bool)));
    connect(my_w.useMask, SIGNAL(toggled(bool)), this, SLOT(maskRegionToggled(bool)));
    connect(my_w.useInterpolate, SIGNAL(toggled(bool)), this, SLOT(interpolateToggled(bool)));

    useBarrierToggled(my_w.useBarrier->isChecked());
    maskRegionToggled(my_w.useMask->isChecked());
    interpolateToggled(my_w.useInterpolate->isChecked());

    connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));

    connect(my_w.rotAngle,SIGNAL(valueChanged(double)), this, SLOT(doSubtract()));

    connect(my_w.posZeroButton,SIGNAL(toggled(bool)), this, SLOT(getPosZero(bool)));

    connect(my_w.cutoffValue, SIGNAL(valueChanged(double)), this, SLOT(doMaskCutoff()));

    connect(my_w.addShape, SIGNAL(released()), this, SLOT(addShape()));
    connect(my_w.cutoffMin, SIGNAL(editingFinished()), this, SLOT(doCutoff()));
    connect(my_w.cutoffMax, SIGNAL(editingFinished()), this, SLOT(doCutoff()));

}

void Interferometry::imagesTabBarClicked(int num) {
    nparent->showPhys(getPhysFromCombo(my_image[num].image));
}

void Interferometry::on_actionDuplicate_triggered() {
    localPhys.clear();
}

void Interferometry::on_actionDelete_triggered() {
    std::map<std::string, nPhysD *> oldPhys=localPhys;
    localPhys.clear();
    nPhysD *c_buf=currentBuffer;
    for(std::map<std::string, nPhysD *>::const_iterator itr = oldPhys.begin(); itr != oldPhys.end(); ++itr) {
        if (itr->second != c_buf)
            nparent->removePhys(itr->second);
    }
    DEBUG(localPhys.size());
}

void Interferometry::line_key_pressed(int key) {
    if (key==Qt::Key_Period) {
        if (sender()==maskRegion) {
            doMaskCutoff();
        } else if (sender()==unwrapBarrier) {
            doUnwrap();
        } else if (sender()==region) {
            doWavelet();
        } else {
            for (std::map<QToolButton*,nLine*>::iterator it = my_shapes.begin(); it != my_shapes.end(); it++) {
                if (sender()==(*it).second) {
                    doShape();
                }
            }
        }
        nparent->activateWindow();
    }
}

void Interferometry::physDel(nPhysD* buf) {
    for (auto it = localPhys.cbegin(); it != localPhys.cend(); ) // no "++"!
    {
        if (it->second == buf) {
            localPhys.erase(it++);
        } else {
            ++it;
        }
    }
}

void Interferometry::getPosZero(bool check) {
    if (check) {
        nparent->showPhys(getPhysFromCombo(my_image[1].image));
        connect(nparent->my_w->my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setPosZero(QPointF)));
    } else {
        disconnect(nparent->my_w->my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setPosZero(QPointF)));
    }
}

void Interferometry::setPosZero(QPointF point) {
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

void Interferometry::useBarrierToggled(bool val) {
    unwrapBarrier->setVisible(val);
}

void Interferometry::maskRegionToggled(bool val) {
    maskRegion->setVisible(val);
}

void Interferometry::interpolateToggled(bool val) {
    for (std::map<QToolButton*,nLine*>::iterator it = my_shapes.begin(); it != my_shapes.end(); it++) {
        it->second->setVisible(val);
    }
}

void Interferometry::bufferChanged(nPhysD* buf) {
    nGenericPan::bufferChanged(buf);
    if (buf) {
        if (buf==getPhysFromCombo(my_image[0].image) || buf==getPhysFromCombo(my_image[1].image)) {
            region->show();
        } else {
            region->hide();
        }
        if (buf==localPhys["phase_2pi"] ||
                buf==localPhys["phaseMask"] ||
                buf==getPhysFromCombo(my_image[0].image) ||
                buf==getPhysFromCombo(my_image[1].image)) {
            maskRegion->setVisible(my_w.useMask->isChecked());
        } else {
            maskRegion->hide();
        }
    }
}

void Interferometry::guessCarrier() {
    nPhysD *image=getPhysFromCombo(my_image[0].image);
    if (image) {
        QRect geom2=region->getRect(image);
        nPhysD datamatrix;
        datamatrix = image->sub(geom2.x(),geom2.y(),geom2.width(),geom2.height());

        vec2f vecCarr=physWave::phys_guess_carrier(datamatrix, my_w.weightCarrier->value());

        if (vecCarr.first()==0) {
            my_w.statusbar->showMessage(tr("ERROR: Problem finding the carrier"), 5000);
        } else {
            my_w.statusbar->showMessage(tr("Carrier: ")+QLocale().toString(vecCarr.first())+"px "+QLocale().toString(vecCarr.second())+"deg", 5000);
            my_w.widthCarrier->setValue(vecCarr.first());
            my_w.angleCarrier->setValue(vecCarr.second());
        }
    }
}

void Interferometry::doWavelet () {
    QTime timer;
    timer.start();
    if (sender()->property("id").isValid()) {
        doWavelet(sender()->property("id").toInt());
    } else {
        if (getPhysFromCombo(my_image[0].image) != getPhysFromCombo(my_image[1].image)) {
            doWavelet(0);
        }
        doWavelet(1);
    }
    my_w.statusbar->showMessage(QLocale().toString(timer.elapsed())+" msec", 5000);
    if (!my_w.chained->isChecked()) doUnwrap();
}

void Interferometry::doWavelet (int iimage) {
    std::string suffix=iimage==0?"_ref":"_shot";
    nPhysD *image=getPhysFromCombo(my_image[iimage].image);
    physWave::wavelet_params my_params;
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
        double thick=my_w.widthCarrier->value()*my_w.thickness->value();
        my_params.thickness=thick;
        my_params.damp=my_w.correlation->value();

        QRect geom2=region->getRect(image);

        nPhysD datamatrix = image->sub(geom2.left(),geom2.top(),geom2.width(),geom2.height(),my_w.padding->isChecked()?thick:0);

        std::ostringstream my_name;

        my_params.data=&datamatrix;

        int niter=my_params.n_angles*my_params.n_lambdas+1;

        QSettings settings("neutrino","");
        settings.beginGroup("nPreferences");
        if (physWave::openclEnabled()>0 && settings.value("openclUnit").toInt()>0) {
            DEBUG("Ready to run on OpenCL");
            my_params.opencl_unit=settings.value("openclUnit").toInt();
            runThread(&my_params, physWave::phys_wavelet_trasl_opencl, "Wavelet"+QString(suffix.c_str()), niter);
        } else {
            runThread(&my_params, physWave::phys_wavelet_trasl_cpu, "Wavelet"+QString(suffix.c_str()), niter);
        }

        std::map<std::string,nPhysD *> retList;
        for (auto& it : my_params.olist) {
            nPhysD my_phys;
            if (my_w.padding->isChecked()) {
                my_phys = it.second->sub(thick,thick,geom2.width(),geom2.height());
            } else {
                my_phys = *it.second;
            }
            DEBUG(my_phys);
            retList[it.first] = new nPhysD(my_phys);
            delete it.second;
        }

        if (retList["phase_2pi"] && retList["contrast"]) {
            retList["synthetic"] = new nPhysD(physWave::phys_synthetic_interferogram(retList["phase_2pi"], retList["contrast"]));
        }
        qDebug() << retList.size();
        for(auto & itr : retList) {
            if (itr.second) {
                itr.second->setShortName(itr.second->getShortName()+suffix);
                localPhys[itr.first+suffix]=nparent->replacePhys(itr.second,localPhys[itr.first+suffix],false);
            }
        }
        my_params.data=nullptr;
    }
    qDebug() << "exit here";
}

void Interferometry::doUnwrap () {
    my_w.statusbar->showMessage("Unwrapping");
    if (localPhys["phase_2pi_shot"]) {
        nPhysD diff=localPhys["phase_2pi_shot"]->copy();
        if ( !localPhys["phase_2pi_ref"] || getPhysFromCombo(my_image[0].image) == getPhysFromCombo(my_image[1].image)) {
            physWave::phys_subtract_carrier(diff, my_w.angleCarrier->value(), my_w.widthCarrier->value());
        } else if (localPhys["phase_2pi_ref"]) {
            nPhysD ref_phase=localPhys["phase_2pi_ref"]->copy();
            physMath::phys_point_subtract(diff,ref_phase);
        }
        physMath::phys_remainder(diff,1.0);
        localPhys["phase_2pi_wrap"]=nparent->replacePhys(new nPhysD(diff),localPhys["phase_2pi_wrap"]);
        localPhys["phase_2pi_wrap"]->setShortName("phase_2pi_wrap");
    }
    qDebug() << "here";

    if (localPhys["contrast_shot"]) {
        nPhysD quality_loc= localPhys["contrast_shot"]->copy();
        if (localPhys["contrast_ref"] && (getPhysFromCombo(my_image[0].image) != getPhysFromCombo(my_image[1].image))) {
            physMath::phys_point_multiply(quality_loc,localPhys["contrast_ref"]->copy());
        }
        localPhys["phase_quality"]=nparent->replacePhys(new nPhysD(quality_loc),localPhys["phase_quality"]);
        localPhys["phase_quality"]->setShortName("phase_quality");
    }

    if (nPhysExists(localPhys["phase_2pi_wrap"]) && nPhysExists(localPhys["phase_quality"])) {

        physD *phase = static_cast<physD*>(localPhys["phase_2pi_wrap"]);
        physD *qual = static_cast<physD*>(localPhys["phase_quality"]);

        QTime timer;
        timer.start();

        qDebug() << "here";
        if (qual && phase) {

            physD loc_qual(*qual);
            double mini=loc_qual.get_min();
            double maxi=loc_qual.get_max();
            double valDouble=mini+my_w.cutoffValue->value()*(maxi-mini)/100.0;
#pragma omp parallel for
            for (size_t k=0; k<loc_qual.getSurf(); k++)
                if (loc_qual.Timg_buffer[k] < valDouble)
                    loc_qual.Timg_buffer[k]=0.0;
            qDebug() << "here";
            physD unwrap;
            QString methodName=my_w.method->currentText();

            physD barrierPhys = physD(phase->getW(),phase->getH(),1.0,"barrier");
            if (my_w.useBarrier->isChecked()) {
                QPolygon my_poly=unwrapBarrier->poly(phase->getW()+phase->getH()).translated(phase->get_origin().x(),phase->get_origin().y()).toPolygon();
                std::vector<vec2u> vec_poly;
                for(auto &p : my_poly) {
                    vec_poly.push_back(vec2u(p.x(),p.y()));
                }
                for(auto &p: vec_poly) {
                    barrierPhys.set(p.x()-1,p.y()-1,0);
                    barrierPhys.set(p.x()-1,p.y()  ,0);
                    barrierPhys.set(p.x()-1,p.y()+1,0);
                    barrierPhys.set(p.x()  ,p.y()-1,0);
                    barrierPhys.set(p.x()  ,p.y()  ,0);
                    barrierPhys.set(p.x()  ,p.y()+1,0);
                    barrierPhys.set(p.x()+1,p.y()-1,0);
                    barrierPhys.set(p.x()+1,p.y()  ,0);
                    barrierPhys.set(p.x()+1,p.y()+1,0);
                }
            }
            if (methodName=="Simple H+V") {
                physWave::phys_phase_unwrap(*phase, barrierPhys, physWave::SIMPLE_HV, unwrap);
            } else if (methodName=="Simple V+H") {
                physWave::phys_phase_unwrap(*phase, barrierPhys, physWave::SIMPLE_VH, unwrap);
            } else if (methodName=="Goldstein") {
                physWave::phys_phase_unwrap(*phase, barrierPhys, physWave::GOLDSTEIN, unwrap);
            } else if (methodName=="Miguel") {
                physWave::phys_phase_unwrap(*phase, barrierPhys, physWave::MIGUEL_QUALITY, unwrap);
            } else if (methodName=="Miguel+Quality") {
                physMath::phys_point_multiply(barrierPhys,loc_qual);
                physWave::phys_phase_unwrap(*phase, barrierPhys, physWave::MIGUEL_QUALITY, unwrap);
            } else if (methodName=="Quality") {
                physMath::phys_point_multiply(barrierPhys,loc_qual);
                physWave::phys_phase_unwrap(*phase, barrierPhys, physWave::QUALITY, unwrap);
            }
#ifdef __phys_debug
            if (my_w.useBarrier->isChecked()) {
                localPhys["barrier"]=nparent->replacePhys(new nPhysD(barrierPhys),localPhys["barrier"]);
            }
#endif

            unwrap.setShortName("phase_2pi_unwrap");
            unwrap.setName(unwrap.getShortName()+"(\""+methodName.toStdString()+"\") "+QFileInfo(QString::fromUtf8(phase->getFromName().c_str())).fileName().toStdString());
            unwrap.setFromName(phase->getFromName());

            nPhysD *unwrapcopy= new nPhysD(unwrap);
            localPhys["phase_2pi_unwrap"]=nparent->replacePhys(unwrapcopy,localPhys["phase_2pi_unwrap"]);

        }
    }
    my_w.statusbar->clearMessage();
    if (!my_w.chained->isChecked()) doSubtract();
}

void Interferometry::doSubtract () {
    my_w.statusbar->showMessage("Subtract and rotate");

    if (localPhys["intensity_ref"] && localPhys["intensity_shot"]) {
        nPhysD contrast_loc= *localPhys["intensity_shot"] / *localPhys["intensity_ref"];
        contrast_loc.TscanBrightness();
        localPhys["contrast"]=nparent->replacePhys(new nPhysD(contrast_loc.fast_rotated(my_w.rotAngle->value())),localPhys["contrast"]);
        localPhys["contrast"]->setShortName("contrast");
    }

    if (localPhys["angle_ref"] && localPhys["angle_shot"]) {
        nPhysD angle_loc= *localPhys["angle_shot"] - *localPhys["angle_ref"];
        angle_loc.TscanBrightness();
        localPhys["angle"]=nparent->replacePhys(new nPhysD(angle_loc.fast_rotated(my_w.rotAngle->value())),localPhys["angle"]);
        localPhys["angle"]->setShortName("angle");
    }

    if (localPhys["lambda_ref"] && localPhys["lambda_shot"]) {
        nPhysD lambda_loc= *localPhys["lambda_shot"] / *localPhys["lambda_ref"];
        lambda_loc.TscanBrightness();
        localPhys["lambda"]=nparent->replacePhys(new nPhysD(lambda_loc.fast_rotated(my_w.rotAngle->value())),localPhys["lambda"]);
        localPhys["lambda"]->setShortName("lambda");
    }

    if (localPhys["phase_quality"]) {
        localPhys["quality"]=nparent->replacePhys(new nPhysD(localPhys["phase_quality"]->fast_rotated(my_w.rotAngle->value())),localPhys["quality"]);
        localPhys["quality"]->setShortName("quality");
    }

    if (localPhys["phase_2pi_unwrap"]) {

        nPhysD phase=localPhys["phase_2pi_unwrap"]->copy();

        if (my_w.opposite->isChecked()) {
            physMath::phys_multiply(phase,-1.0);
        }

        double offset=phase.point(localPhys["phase_2pi_unwrap"]->to_pixel(vec2f(my_w.posZeroX->value(),my_w.posZeroY->value())));
        qDebug() << "Ofest phase " << offset;
        if (std::isfinite(offset)) {
            physMath::phys_subtract(phase,offset);
        } else {
            my_w.statusbar->showMessage("Can't subtract point " + QLocale().toString(my_w.posZeroX->value()) + " , " + QLocale().toString(my_w.posZeroY->value()) + " is not finite", 5000);
        }

        localPhys["phase_2pi"]=nparent->replacePhys(new nPhysD(phase.fast_rotated(my_w.rotAngle->value())),localPhys["phase_2pi"]);
        localPhys["phase_2pi"]->setShortName("phase_2pi");

    }
    my_w.statusbar->clearMessage();

    if (!my_w.chained->isChecked()) doMaskCutoff();
}


void Interferometry::doMaskCutoff() {
    my_w.statusbar->showMessage("Mask");
    nPhysD *phase=localPhys["phase_2pi"];
    nPhysD *phaseMask=NULL;
    if (nPhysExists(phase)) {
        if (my_w.useMask->isChecked()) {
            maskRegion->show();
            phaseMask =new nPhysD(phase->getW(),phase->getH(),0.0,"Mask");
            phaseMask->prop=phase->prop;
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
        } else {
            maskRegion->hide();
        }

        if (phaseMask==NULL) {
            phaseMask=new nPhysD(*phase);
            phaseMask->setShortName("Mask");
        }
        nPhysD *quality=localPhys["quality"];
        if (nPhysExists(quality)) {
            nPhysD loc_qual(*quality);
            double mini=loc_qual.get_min();
            double maxi=loc_qual.get_max();
            double valDouble=mini+my_w.cutoffValue->value()*(maxi-mini)/100.0;
            for (size_t k=0; k<loc_qual.getSurf(); k++)
                if (loc_qual.Timg_buffer[k] < valDouble)
                    phaseMask->Timg_buffer[k]=0.0;
        }
        phaseMask->TscanBrightness();
        localPhys["phaseMask"]=nparent->replacePhys(phaseMask,localPhys["phaseMask"]);
    }
    my_w.statusbar->clearMessage();

    if (!my_w.chained->isChecked()) doShape();
}

void Interferometry::doShape(){
    my_w.statusbar->showMessage("Interpolating");

    nPhysD *image=localPhys["phaseMask"];

    if (nPhysExists(image)) {
        if (my_w.useInterpolate->isChecked()) {
            nPhysD *regionPath = new nPhysD(*image);
            regionPath->setShortName("phaseInterp");
            QProgressDialog progress("Interpolate", "Stop", 0, my_shapes.size(), this);
            progress.setWindowModality(Qt::WindowModal);
            progress.show();
            int prog=0;
            for (std::map<QToolButton*,nLine*>::iterator it = my_shapes.begin(); it != my_shapes.end(); it++, prog++) {
                progress.setValue(prog);
                if (progress.wasCanceled()) break;
                QApplication::processEvents();

                nLine* region = (*it).second;

                region->toggleClosedLine(true); // ensure no mess

                QPolygonF regionPoly=region->poly(1).translated(image->get_origin().x(),image->get_origin().y());

                // convert QPolygonF to vector<vec2f>
                std::vector<vec2f> vecPoints(regionPoly.size());
                for(int k=0;k<regionPoly.size();k++) {
                    QPointF pp=regionPoly[k];
                    vecPoints[k]=vec2f(pp.x(),pp.y());
                }

                // these are the points on wich we will calculate the weighted mean
                const unsigned int npoints=20;
                std::vector<std::pair<vec2f, double> > vals;
                for(unsigned int k=0;k<vecPoints.size()-1;k++) {
                    vec2f p1=vecPoints[k];
                    vec2f p2=vecPoints[k+1];
                    for(unsigned int kk=0;kk<npoints;kk++) {
                        vec2f pp=p1+kk*(p2-p1)/npoints;
                        double val=regionPath->point(pp);
                        if (std::isfinite(val)) {
                            vals.push_back(std::make_pair(pp, val));
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
                            for(std::vector<std::pair<vec2f,double> >::iterator it=vals.begin();it!=vals.end();++it){
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
            localPhys["interpPhase_2piMask"]=nparent->replacePhys(regionPath,localPhys["interpPhase_2piMask"]);
        } else {
            localPhys["interpPhase_2piMask"]=image;
        }
    }
    my_w.statusbar->clearMessage();

    if (!my_w.chained->isChecked()) doCutoff();
}

void Interferometry::addShape(){
    int num=0;
    while (1) {
        num++;
        QString tooltipStr="interpolateShape"+QLocale().toString(num);
        bool found =false;
        foreach (QObject* widget, nparent->children()) {
            nLine *line=qobject_cast<nLine *>(widget);
            if (line && line->parent()==this) {
                qDebug() << line->toolTip();
                if (line->toolTip()==tooltipStr) {
                    found=true;
                }
            }
        }
        if (!found) {
            addShape(tooltipStr);
            break;
        }
    }
}

void Interferometry::addShape(QString name){
    nLine *my_l=new nLine(this,0);
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
    my_b->setIcon(QIcon(":icons/region"));
    my_b->setToolTip(name+my_b->toolTip());
    my_shapes[my_b]=my_l;
    my_w.shapes->layout()->addWidget(my_b);

    connect(my_l, SIGNAL(key_pressed(int)), this, SLOT(line_key_pressed(int)));
    connect(my_l, SIGNAL(destroyed(QObject*)), this, SLOT(removeShape(QObject*)));
    connect(my_b, SIGNAL(released()), my_l, SLOT(togglePadella()));
}

void Interferometry::removeShape(QObject *obj){
    QToolButton *found=NULL;
    for (std::map<QToolButton*,nLine*>::iterator it = my_shapes.begin(); it != my_shapes.end(); it++) {
        if ((*it).second==obj) {
            (*it).first->deleteLater();
            found=(*it).first;
        }
    }
    if (found) my_shapes.erase(my_shapes.find(found));
}

void Interferometry::doCutoff(){
    my_w.statusbar->showMessage("Cutoff");
    nPhysD *image=localPhys["interpPhase_2piMask"];
    if (nPhysExists(image)) {
        nPhysD *intNe = new nPhysD(*image);
        if (my_w.display99->isChecked()) {
            physMath::cutoff(*intNe,physMath::getColorPrecentPixels(*intNe,99));
        }
        bool ok1,ok2;
        double mini=locale().toDouble(my_w.cutoffMin->text(),&ok1);
        double maxi=locale().toDouble(my_w.cutoffMax->text(),&ok2);
        if (!ok1) mini=intNe->get_min();
        if (!ok2) maxi=intNe->get_max();
        if (ok1||ok2) {
            physMath::cutoff(*intNe,mini,maxi);
            intNe->prop["display_range"]=intNe->get_min_max();

            if (localPhys["interpPhase_2piMaskCutoff"]) {
                localPhys["interpPhase_2piMaskCutoff"]->prop["display_range"]=intNe->get_min_max();
            }
            localPhys["interpPhase_2piMaskCutoff"]=nparent->replacePhys(intNe,localPhys["interpPhase_2piMaskCutoff"],true);
        } else {
            localPhys["interpPhase_2piMaskCutoff"]=image;
        }
    }
    my_w.statusbar->clearMessage();
}

//////////////////////////////////////////////////////////

void Interferometry::loadSettings(QSettings &settings){

    for (std::map<QToolButton*,nLine*>::iterator it = my_shapes.begin(); it != my_shapes.end(); it++) {
        removeShape((*it).first);
    }

    QStringList names=settings.value("interpolateShape").toStringList();
    foreach (QString name, names) {
        bool found=false;
        foreach (QObject* widget, nparent->children()) {
            nLine *line=qobject_cast<nLine *>(widget);
            if (line && line->parent()==this) {
                if (line->toolTip()==name) found=true;
            }
        }
        if (!found) addShape(name);
    }

    settings.beginGroup("localPhys");
    foreach (const QString &childKey, settings.childKeys()) {
        QStringList qstr = settings.value(childKey).toStringList();
        if (qstr.size()==2) {
            std::string str0=qstr.at(0).toStdString();
            std::string str1=qstr.at(1).toStdString();
            for(auto & img : nparent->getBufferList()) {
                qDebug() << qstr << "Image" << img;
                if (img->getShortName() == str0 && img->getName() == str1) {
                    qDebug() << "Found localPhys" << childKey;
                    localPhys[childKey.toStdString()]=img;
                }
            }
        }
    }
    settings.endGroup();

    nGenericPan::loadSettings(settings);
}

void Interferometry::saveSettings(QSettings &settings){
    QStringList names;
    for (std::map<QToolButton*,nLine*>::iterator it = my_shapes.begin(); it != my_shapes.end(); it++) {
        names.append((*it).second->toolTip());
    }
    settings.setValue("interpolateShape", names);

    settings.beginGroup("localPhys");
    for(std::map<std::string, nPhysD *>::iterator itr = localPhys.begin(); itr != localPhys.end(); ++itr) {
        //        qDebug() << itr->first;
        if (itr->second && nparent->getBufferList().contains(itr->second)) {
            QStringList value;
            value << QString::fromStdString(itr->second->getShortName()) << QString::fromStdString(itr->second->getName());
            settings.setValue(QString::fromStdString(itr->first), value);
            DEBUG(itr->first << " " <<  nparent->indexOf(itr->second));
        }
    }
    settings.endGroup();

    nGenericPan::saveSettings(settings);
}



