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
#include "nPhysMaths.h"

// physWavelets

Interferometry::Interferometry(neutrino *nparent) : nGenericPan(nparent) {
    setupUi(this);

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
        images->setCurrentIndex(k);
        my_image[k].setupUi(images->widget(k));
        foreach (QWidget *my_obj,  images->widget(k)->findChildren<QWidget *>()) {
            if (!my_obj->objectName().isEmpty()) {
                my_obj->setObjectName(my_obj->objectName()+panName()+QLocale().toString(k));
                my_obj->setProperty("id",k);
            }
        }
        connect(my_image[k].doit, SIGNAL(released()), this, SLOT(doWavelet()));
    }

    show();

    connect(images, SIGNAL(currentChanged(int)), this, SLOT(imagesTabBarClicked(int)));
    connect(region, SIGNAL(key_pressed(int)), this, SLOT(line_key_pressed(int)));
    connect(maskRegion, SIGNAL(key_pressed(int)), this, SLOT(line_key_pressed(int)));
    connect(unwrapBarrier, SIGNAL(key_pressed(int)), this, SLOT(line_key_pressed(int)));

    connect(actionDoWavelet, SIGNAL(triggered()), this, SLOT(doWavelet()));

    connect(actionRect, SIGNAL(triggered()), region, SLOT(togglePadella()));
    connect(lineBarrier, SIGNAL(released()), unwrapBarrier, SLOT(togglePadella()));
    connect(maskRegionTB, SIGNAL(released()), maskRegion, SLOT(togglePadella()));

    connect(doCarrier, SIGNAL(released()), this, SLOT(guessCarrier()));
    connect(doUnwrapB, SIGNAL(released()), this, SLOT(doUnwrap()));
    connect(doSubtractB, SIGNAL(released()), this, SLOT(doSubtract()));
    connect(doMaskCutoffB, SIGNAL(released()), this, SLOT(doMaskCutoff()));
    connect(doInterpolate, SIGNAL(released()), this, SLOT(doShape()));
    connect(doCutoffB, SIGNAL(released()), this, SLOT(doCutoff()));

    connect(weightCarrier, SIGNAL(valueChanged(double)), this, SLOT(guessCarrier()));

    connect(useBarrier, SIGNAL(toggled(bool)), this, SLOT(useBarrierToggled(bool)));
    // the simple methods follow rows and columns: they cannot use a barrier
    connect(method, &QComboBox::currentTextChanged, this, [this](const QString &text) {
        useBarrier->setEnabled(!text.startsWith("Simple"));
        lineBarrier->setEnabled(!text.startsWith("Simple"));
    });
    useBarrier->setEnabled(!method->currentText().startsWith("Simple"));
    lineBarrier->setEnabled(!method->currentText().startsWith("Simple"));
    connect(useMask, SIGNAL(toggled(bool)), this, SLOT(maskRegionToggled(bool)));
    connect(useInterpolate, SIGNAL(toggled(bool)), this, SLOT(interpolateToggled(bool)));

    useBarrierToggled(useBarrier->isChecked());
    maskRegionToggled(useMask->isChecked());
    interpolateToggled(useInterpolate->isChecked());

    cropRegion =  new nRect(this,1);
    cropRegion->changeToolTip("CropRegion");
    cropRegion->setRect(QRectF(150,150,100,100));
    connect(useCropRegion, SIGNAL(toggled(bool)), this, SLOT(useCropRegionToggled(bool)));
    useCropRegionToggled(useCropRegion->isChecked());
    connect(cropRegionTB, SIGNAL(released()), cropRegion, SLOT(togglePadella()));


    connect(nparent, SIGNAL(physDel(nPhysD*)), this, SLOT(physDel(nPhysD*)));

    connect(rotAngle,SIGNAL(valueChanged(double)), this, SLOT(doSubtract()));

    connect(posZeroButton,SIGNAL(toggled(bool)), this, SLOT(getPosZero(bool)));
    connect(posZeroX,SIGNAL(editingFinished()), this, SLOT(doSubtract()));
    connect(posZeroY,SIGNAL(editingFinished()), this, SLOT(doSubtract()));

    connect(cutoffValue, SIGNAL(valueChanged(double)), this, SLOT(doMaskCutoff()));

    connect(addShapeB, SIGNAL(released()), this, SLOT(addShape()));
    connect(cutoffMin, SIGNAL(editingFinished()), this, SLOT(doCutoff()));
    connect(cutoffMax, SIGNAL(editingFinished()), this, SLOT(doCutoff()));

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
        connect(nparent->my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setPosZero(QPointF)));
    } else {
        disconnect(nparent->my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setPosZero(QPointF)));
    }
}

void Interferometry::setPosZero(QPointF point) {
    if(currentBuffer) {
        posZeroX->setValue(point.x());
        posZeroY->setValue(point.y());
        Zero->setChecked(true);
        doSubtract();
    }
}

void Interferometry::useBarrierToggled(bool val) {
    unwrapBarrier->setVisible(val);
}

void Interferometry::useCropRegionToggled(bool val) {
    // qDebug() << "HERE" << val;
    cropRegion->setVisible(val);
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
            maskRegion->setVisible(useMask->isChecked());
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
        datamatrix = image->sub(geom2);

        std::vector<vec2f> vecCarr=physWave::phys_guess_carrier(datamatrix, weightCarrier->value());

        if (vecCarr.size()==0) {
            statusbar->showMessage(tr("ERROR: Problem finding the carrier"), 5000);
        } else {
            statusbar->showMessage(tr("Carrier: ")+QLocale().toString(vecCarr[0].first())+"px "+QLocale().toString(vecCarr[0].second())+"deg", 5000);
            widthCarrier->setValue(vecCarr[0].first());
            angleCarrier->setValue(vecCarr[0].second());
        }
    }
}

void Interferometry::doWavelet () {
    QElapsedTimer timer;
    timer.start();
    nThread.n_iter=0; // runThread sets it to -1 only if a wavelet below is stopped
    if (sender() && sender()->property("id").isValid()) {
        doWavelet(sender()->property("id").toInt());
    } else {
        if (getPhysFromCombo(my_image[0].image) != getPhysFromCombo(my_image[1].image)) {
            doWavelet(0);
        }
        if (nThread.n_iter!=-1) doWavelet(1);
    }
    if (nThread.n_iter==-1) { // stopped: don't chain on the previous results
        statusbar->showMessage(tr("Wavelet stopped"), 5000);
        return;
    }
    statusbar->showMessage(QLocale().toString(timer.elapsed())+" msec", 5000);
    if (!chained->isChecked()) doUnwrap();
}

void Interferometry::doWavelet (int iimage) {
    std::string suffix=iimage==0?"_ref":"_shot";
    nPhysD *image=getPhysFromCombo(my_image[iimage].image);
    physWave::wavelet_params my_params;
    if (image) {
        saveDefaults();
        if (my_image[iimage].numAngle->value()==0) {
            my_params.init_angle=angleCarrier->value();
            my_params.end_angle=angleCarrier->value();
            my_params.n_angles=1;
        } else {
            my_params.init_angle=my_image[iimage].minAngle->value()+angleCarrier->value();
            my_params.end_angle=my_image[iimage].maxAngle->value()+angleCarrier->value();
            my_params.n_angles=my_image[iimage].numAngle->value();
        }
        if (my_image[iimage].numStretch->value()==0) {
            my_params.init_lambda=widthCarrier->value();
            my_params.end_lambda=widthCarrier->value();
            my_params.n_lambdas=1;
        } else {
            my_params.init_lambda=my_image[iimage].minStretch->value()*widthCarrier->value();
            my_params.end_lambda=my_image[iimage].maxStretch->value()*widthCarrier->value();
            my_params.n_lambdas=my_image[iimage].numStretch->value();
        }

        if (my_image[iimage].numThick->value()==0) {
            my_params.init_thick=widthCarrier->value()*thickness->value();
            my_params.end_thick=widthCarrier->value()*thickness->value();
            my_params.n_thicks=1;
        } else {
            my_params.init_thick=my_image[iimage].minThick->value()*widthCarrier->value()*thickness->value();
            my_params.end_thick=my_image[iimage].maxThick->value()*widthCarrier->value()*thickness->value();
            my_params.n_thicks=my_image[iimage].numThick->value();
        }

        // double thick = widthCarrier->value();
        my_params.damp=correlation->value();

        QRect geom2=region->getRect(image);

        // nPhysD datamatrix = image->sub(geom2,padding->isChecked()?thick:0);
        // Pad: the image around the region is used for the computation, 2 windows on each side as the padding of
        // the wavelet (it was 1 interfringe, less than the window, and filled with 0 beyond the image border:
        // a step). Clipped to the image: only real pixels, the rest is padded by the wavelet itself
        QRect geomPad=geom2;
        if (padding->isChecked()) {
            double window=std::max(my_params.damp*std::max(my_params.init_lambda,my_params.end_lambda),
                                   std::max(my_params.init_thick,my_params.end_thick));
            int margin=(int)std::ceil(2.0*window);
            geomPad=geom2.adjusted(-margin,-margin,margin,margin).intersected(QRect(0,0,image->getW(),image->getH()));
        }
        nPhysD datamatrix = image->sub(geomPad);

        std::ostringstream my_name;

        my_params.data=&datamatrix;

        int niter=my_params.n_angles*my_params.n_lambdas*my_params.n_thicks+1;

        QSettings settings("neutrino","");
        settings.beginGroup("nPreferences");
        if (physWave::openclEnabled()>0 && settings.value("openclUnit").toInt()>0) {
            DEBUG("Ready to run on OpenCL");
            my_params.opencl_unit=settings.value("openclUnit").toInt();
            runThread(&my_params, physWave::phys_wavelet_trasl_opencl, "Wavelet"+QString(suffix.c_str()), niter);
        } else {
            runThread(&my_params, physWave::phys_wavelet_trasl_cpu, "Wavelet"+QString(suffix.c_str()), niter);
        }
        if (nThread.n_iter==-1) {
            my_params.data=nullptr;
            return;
        }

        std::map<std::string,nPhysD *> retList;
        for (auto& it : my_params.olist) {
            nPhysD my_phys;
            if (padding->isChecked()) {
                // my_phys = it.second->sub(thick,thick,geom2.width(),geom2.height());
                // back to the region (the margin can differ on each side where it was clipped)
                my_phys = it.second->sub(geom2.x()-geomPad.x(),geom2.y()-geomPad.y(),geom2.width(),geom2.height());
            } else {
                my_phys = *it.second;
            }
            DEBUG(my_phys);
            retList[it.first] = new nPhysD(my_phys);
        }

        if (retList["phase_2pi"] && retList["contrast"]) {
            // retList["synthetic"] = new nPhysD(physWave::phys_synthetic_interferogram(retList["phase_2pi"], retList["contrast"]));
            retList["synthetic"] = new nPhysD(physWave::phys_synthetic_interferogram(retList["phase_2pi"], retList["contrast"], retList["intensity"]));
        }
        // qDebug() << retList.size();
        for(auto & itr : retList) {
            if (itr.second) {
                itr.second->setShortName(itr.second->getShortName()+suffix);
                localPhys[itr.first+suffix]=nparent->replacePhys(itr.second,localPhys[itr.first+suffix],false);
            }
        }
        my_params.data=nullptr;
    }
    // qDebug() << "exit here";
}

void Interferometry::doUnwrap () {
    statusbar->showMessage("Unwrapping");
    if (localPhys["phase_2pi_shot"]) {
        nPhysD diff=localPhys["phase_2pi_shot"]->copy();
        if ( !localPhys["phase_2pi_ref"] || getPhysFromCombo(my_image[0].image) == getPhysFromCombo(my_image[1].image)) {
            physWave::phys_subtract_carrier(diff, angleCarrier->value(), widthCarrier->value());
        } else if (localPhys["phase_2pi_ref"]) {
            nPhysD ref_phase=localPhys["phase_2pi_ref"]->copy();
            physMath::phys_point_subtract(diff,ref_phase);
        }
        physMath::phys_remainder(diff,1.0);
        diff.setShortName("phase_2pi_wrap"); // before replacePhys: pans and Image_list read it there
        localPhys["phase_2pi_wrap"]=nparent->replacePhys(new nPhysD(diff),localPhys["phase_2pi_wrap"]);
    }
    // qDebug() << "here";

    if (localPhys["contrast_shot"]) {
        nPhysD quality_loc= localPhys["contrast_shot"]->copy();
        if (localPhys["contrast_ref"] && (getPhysFromCombo(my_image[0].image) != getPhysFromCombo(my_image[1].image))) {
            physMath::phys_point_multiply(quality_loc,*localPhys["contrast_ref"]);
            physMath::phys_sqrt(quality_loc);
        }
        quality_loc.setShortName("phase_quality");
        localPhys["phase_quality"]=nparent->replacePhys(new nPhysD(quality_loc),localPhys["phase_quality"]);
    }

    if (nPhysExists(localPhys["phase_2pi_wrap"]) && nPhysExists(localPhys["phase_quality"])) {

        physD *phase = static_cast<physD*>(localPhys["phase_2pi_wrap"]);
        physD *qual = static_cast<physD*>(localPhys["phase_quality"]);

        // qDebug() << "here";
        if (qual && phase) {
            physD loc_qual(*qual);
            double mini=loc_qual.get_min();
            double maxi=loc_qual.get_max();
            double valDouble=mini+cutoffValue->value()*(maxi-mini)/100.0;
#pragma omp parallel for
            for (size_t k=0; k<loc_qual.getSurf(); k++)
                if (loc_qual.Timg_buffer[k] < valDouble)
                    loc_qual.Timg_buffer[k]=0.0;
            // qDebug() << "here";
            physD unwrap;

            physD barrierPhys = physD(phase->getW(),phase->getH(),1.0,"barrier");
            if (useBarrier->isChecked()) {
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

            QString methodName=method->currentText();

            // map given to the unwrap: barrier (0 on the line) times the quality for the quality guided methods
            // (computed once: it was multiplied twice when also unwrapping the angles)
            physD unwrapQual=barrierPhys;
            if (methodName=="Goldstein" || methodName=="Miguel+Quality" || methodName=="Quality") {
                physMath::phys_point_multiply(unwrapQual,loc_qual);
            }
            auto doPhaseUnwrap = [&](physD &wrapped, physD &unwrapped) {
                if (methodName=="Simple H+V") {
                    physWave::phys_phase_unwrap(wrapped, unwrapQual, physWave::SIMPLE_HV, unwrapped);
                } else if (methodName=="Simple V+H") {
                    physWave::phys_phase_unwrap(wrapped, unwrapQual, physWave::SIMPLE_VH, unwrapped);
                } else if (methodName=="Goldstein") {
                    physWave::phys_phase_unwrap(wrapped, unwrapQual, physWave::GOLDSTEIN, unwrapped);
                } else if (methodName=="Miguel" || methodName=="Miguel+Quality") {
                    physWave::phys_phase_unwrap(wrapped, unwrapQual, physWave::MIGUEL_QUALITY, unwrapped);
                } else if (methodName=="Quality") {
                    physWave::phys_phase_unwrap(wrapped, unwrapQual, physWave::QUALITY, unwrapped);
                }
            };

            physD phase2=phase->copy();
            bool haveAngles = nPhysExists(localPhys["angle_shot"]) && nPhysExists(localPhys["phase_2pi_shot"]);
            if (unwrapAngles->isChecked() && !haveAngles) {
                statusbar->showMessage(tr("Unwrap angles needs the wavelet of the shot"), 5000);
            }
            if (unwrapAngles->isChecked() && haveAngles) {
                // where the best filter points opposite to the (unwrapped) fringe direction, the phase of the SHOT changes sign:
                // flip the shot phase there, then subtract the reference (flipping the difference is wrong by 2*phase_ref)
                physD angle = (static_cast<physD*>(localPhys["angle_shot"]))->copy();
                physMath::phys_divide(angle,180); // the fringe direction has a period of 180 deg
                physD unwrapAngle = physD(angle.getW(),angle.getH(),0);
                doPhaseUnwrap(angle, unwrapAngle);

                physD shotFlipped = (static_cast<physD*>(localPhys["phase_2pi_shot"]))->copy();
                for (size_t k=0; k<unwrapAngle.getSurf(); k++) {
                    long swaps=std::lround(unwrapAngle.Timg_buffer[k]-angle.Timg_buffer[k]); // round: unwrapped values are not exact integers away
                    if (std::abs(swaps) % 2 == 1) {
                        shotFlipped.Timg_buffer[k] = -shotFlipped.Timg_buffer[k];
                        unwrapAngle.Timg_buffer[k]=1;
                    } else {
                        unwrapAngle.Timg_buffer[k]=0;
                    }
                }

                // reference phase: the reference wavelet or the carrier
                physD refPhase;
                if ( !localPhys["phase_2pi_ref"] || getPhysFromCombo(my_image[0].image) == getPhysFromCombo(my_image[1].image)) {
                    refPhase = physD(shotFlipped.getW(),shotFlipped.getH(),0.0);
                    physWave::phys_subtract_carrier(refPhase, angleCarrier->value(), widthCarrier->value());
                    physMath::phys_multiply(refPhase,-1.0);
                } else {
                    refPhase = static_cast<physD*>(localPhys["phase_2pi_ref"])->copy();
                }

                // the orientation of the whole unwrapped angle map is arbitrary: try shot-ref and shot+ref
                // and keep the one with the smaller range (max-min of the unwrapped phase)
                auto finiteRange = [](physD &img) {
                    double mini=std::numeric_limits<double>::max(), maxi=-std::numeric_limits<double>::max();
                    for (size_t k=0; k<img.getSurf(); k++) {
                        if (std::isfinite(img.Timg_buffer[k])) {
                            mini=std::min(mini,img.Timg_buffer[k]);
                            maxi=std::max(maxi,img.Timg_buffer[k]);
                        }
                    }
                    return maxi-mini;
                };
                physD phaseMinus=shotFlipped.copy(), phasePlus=shotFlipped.copy();
                physMath::phys_point_subtract(phaseMinus,refPhase);
                physMath::phys_point_add(phasePlus,refPhase);
                physMath::phys_remainder(phaseMinus,1.0);
                physMath::phys_remainder(phasePlus,1.0);
                physD unwrapMinus, unwrapPlus;
                doPhaseUnwrap(phaseMinus, unwrapMinus);
                doPhaseUnwrap(phasePlus, unwrapPlus);
                bool minus=finiteRange(unwrapMinus) <= finiteRange(unwrapPlus);
                phase2 = minus ? phaseMinus : phasePlus;
                unwrap = minus ? unwrapMinus : unwrapPlus;
                phase2.set_origin(phase->get_origin());
                phase2.set_scale(phase->get_scale());
                unwrap.set_origin(phase->get_origin());
                unwrap.set_scale(phase->get_scale());

                phase2.TscanBrightness();
                unwrapAngle.TscanBrightness();
                unwrapAngle.setShortName("unwrap angles");
                localPhys["unwrap_angles"]=nparent->replacePhys(new nPhysD(unwrapAngle),localPhys["unwrap_angles"]);
                phase2.setShortName(minus ? "phase2 (shot-ref)" : "phase2 (shot+ref)");
                localPhys["phase2"]=nparent->replacePhys(new nPhysD(phase2),localPhys["phase2"]);
            } else {
                doPhaseUnwrap(phase2, unwrap);
            }

            // old implementation: angle of shot-ref, flipped the difference (1-phase) with int() truncation
//            physD phase2=phase->copy();
//            bool haveAngles = nPhysExists(localPhys["angle_shot"]) && nPhysExists(localPhys["angle_ref"]);
//            if (unwrapAngles->isChecked() && !haveAngles) {
//                statusbar->showMessage(tr("Unwrap angles needs the wavelet of both reference and shot"), 5000);
//            }
//            if (unwrapAngles->isChecked() && haveAngles) {
//                physD angle = (static_cast<physD*>(localPhys["angle_shot"]))->copy();
//                physMath::phys_point_subtract(angle,*static_cast<physD*>(localPhys["angle_ref"]));
//                physMath::phys_divide(angle,180);
//                physD orig_angle=angle.copy();
//                physD unwrapAngle = physD(angle.getW(),angle.getH(),0);
//                doPhaseUnwrap(angle, unwrapAngle);
//                physMath::phys_point_subtract(unwrapAngle,orig_angle);
//                physMath::phys_add(unwrapAngle,100);
//                for (size_t k=0; k<unwrapAngle.getSurf(); k++)
//                    if (int(unwrapAngle.Timg_buffer[k]) % 2 == 1) {
//                        phase2.Timg_buffer[k] = 1.0-phase2.Timg_buffer[k];
//                        unwrapAngle.Timg_buffer[k]=1;
//                    } else {
//                        unwrapAngle.Timg_buffer[k]=0;
//                    }
//                phase2.TscanBrightness();
//                unwrapAngle.TscanBrightness();
//                unwrapAngle.setShortName("unwrap angles");
//                localPhys["unwrap_angles"]=nparent->replacePhys(new nPhysD(unwrapAngle),localPhys["unwrap_angles"]);
//                phase2.setShortName("phase2");
//                localPhys["phase2"]=nparent->replacePhys(new nPhysD(phase2),localPhys["phase2"]);
//            }

//            doPhaseUnwrap(phase2, unwrap);
#ifdef __phys_debug
            if (useBarrier->isChecked()) {
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
    statusbar->clearMessage();
    if (!chained->isChecked()) doSubtract();
}

void Interferometry::doSubtract () {
    statusbar->showMessage("Subtract and rotate");

    if (localPhys["intensity_ref"] && localPhys["intensity_shot"]) {
        nPhysD contrast_loc= *localPhys["intensity_shot"] / *localPhys["intensity_ref"];
        contrast_loc.TscanBrightness();
        nPhysD *contrast_rot=new nPhysD(contrast_loc.fast_rotated(rotAngle->value()));
        contrast_rot->setShortName("contrast");
        localPhys["contrast"]=nparent->replacePhys(contrast_rot,localPhys["contrast"]);
    }

    if (localPhys["angle_ref"] && localPhys["angle_shot"]) {
        nPhysD angle_loc= *localPhys["angle_shot"] - *localPhys["angle_ref"];
        angle_loc.TscanBrightness();
        nPhysD *angle_rot=new nPhysD(angle_loc.fast_rotated(rotAngle->value()));
        angle_rot->setShortName("angle");
        localPhys["angle"]=nparent->replacePhys(angle_rot,localPhys["angle"]);
    }

    if (localPhys["lambda_ref"] && localPhys["lambda_shot"]) {
        nPhysD lambda_loc= *localPhys["lambda_shot"] / *localPhys["lambda_ref"];
        lambda_loc.TscanBrightness();
        nPhysD *lambda_rot=new nPhysD(lambda_loc.fast_rotated(rotAngle->value()));
        lambda_rot->setShortName("lambda");
        localPhys["lambda"]=nparent->replacePhys(lambda_rot,localPhys["lambda"]);
    }

    if (localPhys["phase_quality"]) {
        nPhysD *quality_rot=new nPhysD(localPhys["phase_quality"]->fast_rotated(rotAngle->value()));
        quality_rot->setShortName("quality");
        localPhys["quality"]=nparent->replacePhys(quality_rot,localPhys["quality"]);
    }

    if (localPhys["phase_2pi_unwrap"]) {

        nPhysD phase=localPhys["phase_2pi_unwrap"]->copy();

        if (opposite->isChecked()) {
            physMath::phys_multiply(phase,-1.0);
        }

        // the zero is a pixel of the shot image (double click on it): convert to the phase image through the origins
        vec2f zero(posZeroX->value(),posZeroY->value());
        nPhysD *shot=getPhysFromCombo(my_image[1].image);
        if (shot) zero = zero - shot->get_origin() + phase.get_origin();
        double offset=phase.point(vec2i(std::lround(zero.x()),std::lround(zero.y())));
        // double offset=phase.point(posZeroX->value(),posZeroY->value());
        // qDebug() << "!@#!@$#!@##!@$!#@$@!##!@#!@$@@";
        // qDebug() << "!@#!@$#!@##!@$!#@$@!##!@#!@$@@";
        // qDebug() << "!@#!@$#!@##!@$!#@$@!##!@#!@$@@";
        // qDebug() << "!@#!@$#!@##!@$!#@$@!##!@#!@$@@";
        // qDebug() << "!@#!@$#!@##!@$!#@$@!##!@#!@$@@";
        // qDebug() << "!@#!@$#!@##!@$!#@$@!##!@#!@$@@";
        // qDebug() << "!@#!@$#!@##!@$!#@$@!##!@#!@$@@";
        // qDebug() << "";
        // qDebug() << "";
        // qDebug() << "";
        // qDebug() << posZeroX->value() << posZeroY->value() << offset;
        // qDebug() << "";
        // qDebug() << "";
        // qDebug() << "";
        // qDebug() << "!@#!@$#!@##!@$!#@$@!##!@#!@$@@";
        // qDebug() << "!@#!@$#!@##!@$!#@$@!##!@#!@$@@";
        // qDebug() << "!@#!@$#!@##!@$!#@$@!##!@#!@$@@";
        // qDebug() << "!@#!@$#!@##!@$!#@$@!##!@#!@$@@";
        // qDebug() << "!@#!@$#!@##!@$!#@$@!##!@#!@$@@";
        // qDebug() << "!@#!@$#!@##!@$!#@$@!##!@#!@$@@";
        // qDebug() << "!@#!@$#!@##!@$!#@$@!##!@#!@$@@";
        // qDebug() << "!@#!@$#!@##!@$!#@$@!##!@#!@$@@";
        if (std::isfinite(offset)) {
            physMath::phys_subtract(phase,offset);
        } else {
            statusbar->showMessage("Can't subtract point " + QLocale().toString(posZeroX->value()) + " , " + QLocale().toString(posZeroY->value()) + " is not finite", 5000);
        }

        nPhysD *phase_rot=new nPhysD(phase.fast_rotated(rotAngle->value()));
        phase_rot->setShortName("phase_2pi");
        localPhys["phase_2pi"]=nparent->replacePhys(phase_rot,localPhys["phase_2pi"]);

    }
    statusbar->clearMessage();

    if (!chained->isChecked()) doMaskCutoff();
}


void Interferometry::doMaskCutoff() {
    statusbar->showMessage("Mask");
    nPhysD *phase=localPhys["phase_2pi"];
    nPhysD *phaseMask=nullptr;
    if (nPhysExists(phase)) {
        if (useMask->isChecked()) {
            maskRegion->show();
            phaseMask =new nPhysD(phase->getW(),phase->getH(),0.0,"Mask");
            phaseMask->prop=phase->prop;
            // line points are relative to the image origin: pixels of the phase = points + its origin
            QPolygonF regionPoly=maskRegion->poly(1).translated(phase->get_origin().x(),phase->get_origin().y());
            QRect imageRect=QRect(0,0,phase->getW(),phase->getH());
            QRect rectRegion=regionPoly.boundingRect().toRect().intersected(imageRect);
            QProgressDialog progress("Extracting",QString(), 0, rectRegion.width(), this);
            progress.setWindowModality(Qt::WindowModal);
            progress.show();
            for (int i=rectRegion.left(); i<=rectRegion.right(); i++) {
                for (int j=rectRegion.top(); j<=rectRegion.bottom(); j++) {
                    if (regionPoly.containsPoint(QPointF(i,j),Qt::OddEvenFill)) {
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

        if (phaseMask==nullptr) {
            phaseMask=new nPhysD(*phase);
            phaseMask->setShortName("Mask");
        }
        nPhysD *quality=localPhys["quality"];
        if (nPhysExists(quality)) {
            nPhysD loc_qual(*quality);
            double mini=loc_qual.get_min();
            double maxi=loc_qual.get_max();
            double valDouble=mini+cutoffValue->value()*(maxi-mini)/100.0;
            for (size_t k=0; k<loc_qual.getSurf(); k++)
                if (loc_qual.Timg_buffer[k] < valDouble)
                    phaseMask->Timg_buffer[k]=0.0;
        }
        phaseMask->TscanBrightness();
        localPhys["phaseMask"]=nparent->replacePhys(phaseMask,localPhys["phaseMask"]);
    }
    statusbar->clearMessage();

    if (!chained->isChecked()) doShape();
}

void Interferometry::doShape(){
    statusbar->showMessage("Interpolating");

    nPhysD *image=localPhys["phaseMask"];

    if (nPhysExists(image)) {
        if (useInterpolate->isChecked()) {
            if (localPhys["interpPhase_2piMask"]==image) {
                localPhys["interpPhase_2piMask"]=nullptr; // old sessions aliased it to the mask: don't overwrite the mask
            }
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

                // harmonic (laplace) filling: the pixels around the shape are the boundary conditions
                QRect rectRegion=regionPoly.boundingRect().toRect().adjusted(-1,-1,+1,+1).intersected(QRect(0,0,regionPath->getW(),regionPath->getH()));
                std::vector<vec2i> pixels;
                for (int i=rectRegion.left(); i<=rectRegion.right(); i++) {
                    for (int j=rectRegion.top(); j<=rectRegion.bottom(); j++) {
                        vec2f pp(i,j);
                        if (point_inside_poly(pp,vecPoints)) {
                            pixels.push_back(vec2i(i,j));
                        }
                    }
                }
                physMath::laplace_fill(*regionPath, pixels);

                // old implementation: inverse distance (1/r^2) mean of 20 samples per segment of the shape
//                // these are the points on wich we will calculate the weighted mean
//                const unsigned int npoints=20;
//                std::vector<std::pair<vec2f, double> > vals;
//                for(unsigned int k=0;k+1<vecPoints.size();k++) {
//                    vec2f p1=vecPoints[k];
//                    vec2f p2=vecPoints[k+1];
//                    for(unsigned int kk=0;kk<npoints;kk++) {
//                        vec2f pp=p1+kk*(p2-p1)/npoints;
//                        double val=regionPath->point(pp);
//                        if (std::isfinite(val)) {
//                            vals.push_back(std::make_pair(pp, val));
//                        }
//                    }
//                }

//                if (vals.size()==0) continue;

//                QRect rectRegion=regionPoly.boundingRect().toRect().adjusted(-1,-1,+1,+1);

//                for (int i=rectRegion.left(); i<=rectRegion.right(); i++) {
//                    for (int j=rectRegion.top(); j<=rectRegion.bottom(); j++) {
//                        vec2f pp(i,j);
//                        if (point_inside_poly(pp,vecPoints)) {
//                            double mean=0;
//                            double weight=0;
//                            bool onSample=false;
//                            for(std::vector<std::pair<vec2f,double> >::iterator it=vals.begin();it!=vals.end();++it){
//                                vec2f p=it->first;
//                                double pval=it->second;
//                                double d2=(pp-p).mod2();
//                                if (d2==0) { // on a sample point: 1/0 weight would give inf/inf
//                                    mean=pval;
//                                    onSample=true;
//                                    break;
//                                }
//                                double wi=1.0/d2;
//                                mean+=wi*pval;
//                                weight+=wi;
//                            }
//                            double ppval=onSample ? mean : mean/weight;
//                            regionPath->set(pp,ppval);
//                        }
//                    }
//                }
            }
            localPhys["interpPhase_2piMask"]=nparent->replacePhys(regionPath,localPhys["interpPhase_2piMask"]);
        }
        // without interpolation doCutoff uses phaseMask directly (no alias: replacePhys would overwrite the mask)
    }
    statusbar->clearMessage();

    if (!chained->isChecked()) doCutoff();
}

void Interferometry::addShape(){
    int num=0;
    while (1) {
        num++;
        QString tooltipStr="interpolateShape"+QLocale().toString(num);
        bool found =false;
        // the shapes are children of the pan, not of nparent: the old search never found them and every new shape was "interpolateShape1"
//        foreach (QObject* widget, nparent->children()) {
//            nLine *line=qobject_cast<nLine *>(widget);
//            if (line && line->parent()==this) {
//                qDebug() << line->toolTip();
//                if (line->toolTip()==tooltipStr) {
//                    found=true;
//                }
//            }
//        }
        for (auto &it : my_shapes) {
            if (it.second->toolTip()==tooltipStr) found=true;
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
    shapes->layout()->addWidget(my_b);

    connect(my_l, SIGNAL(key_pressed(int)), this, SLOT(line_key_pressed(int)));
    connect(my_l, SIGNAL(destroyed(QObject*)), this, SLOT(removeShape(QObject*)));
    connect(my_b, SIGNAL(released()), my_l, SLOT(togglePadella()));
}

void Interferometry::removeShape(QObject *obj){
    QToolButton *found=nullptr;
    for (std::map<QToolButton*,nLine*>::iterator it = my_shapes.begin(); it != my_shapes.end(); it++) {
        if ((*it).second==obj) {
            (*it).first->deleteLater();
            found=(*it).first;
        }
    }
    if (found) my_shapes.erase(my_shapes.find(found));
}

void Interferometry::doCutoff(){
    statusbar->showMessage("Cutoff");
    nPhysD *image=useInterpolate->isChecked() ? localPhys["interpPhase_2piMask"] : localPhys["phaseMask"];
    if (nPhysExists(image)) {
        nPhysD *intNe= nullptr;
        if (useCropRegion->isChecked()) {
                QRect geom2=cropRegion->getRect(image);
                nPhysD datamatrix = image->sub(geom2);
                intNe=new nPhysD(datamatrix);
        } else {
            intNe = new nPhysD(*image);
        }
        intNe->reset_display();
        intNe->setShortName("interpPhase_2piMaskCutoff");
        if (display99->isChecked()) {
            physMath::cutoff(*intNe,physMath::getColorPrecentPixels(*intNe,99));
        }
        bool ok1,ok2;
        double mini=locale().toDouble(cutoffMin->text(),&ok1);
        double maxi=locale().toDouble(cutoffMax->text(),&ok2);
        if (!ok1) mini=intNe->get_min();
        if (!ok2) maxi=intNe->get_max();
        if (ok1||ok2) {
            physMath::cutoff(*intNe,mini,maxi);
            intNe->reset_display();

            if (nPhysExists(localPhys["interpPhase_2piMaskCutoff"])) {
                localPhys["interpPhase_2piMaskCutoff"]->prop["display_range"]=intNe->get_min_max();
                localPhys["interpPhase_2piMaskCutoff"]->reset_display();
            }
        }
        intNe->TscanBrightness();
        localPhys["interpPhase_2piMaskCutoff"]=nparent->replacePhys(intNe,localPhys["interpPhase_2piMaskCutoff"],true);
    }
    statusbar->clearMessage();
}

//////////////////////////////////////////////////////////

void Interferometry::loadSettings(QSettings &settings){

    // delete the current shapes: their destroyed() signal calls removeShape, which erases them from my_shapes
    std::vector<nLine*> oldShapes;
    for (auto &it : my_shapes) {
        oldShapes.push_back(it.second);
    }
    for (auto &line : oldShapes) {
        delete line;
    }

    QStringList names=settings.value("interpolateShape").toStringList();
    foreach (QString name, names) {
        bool found=false;
//        foreach (QObject* widget, nparent->children()) {
//            nLine *line=qobject_cast<nLine *>(widget);
//            if (line && line->parent()==this) {
//                if (line->toolTip()==name) found=true;
//            }
//        }
        for (auto &it : my_shapes) {
            if (it.second->toolTip()==name) found=true; // sessions saved with duplicated names: one shape per name
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

    for (const auto& one : property("NeuSave-localPhysNames").toStringList()) {
        std::string onestring=one.toStdString();
        for (const auto& img: nparent->getBufferList()) {
            if (img->getShortName() == onestring) {
                localPhys[onestring]=img;
            }
        }
    }
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
        if (itr->second && nPhysExists(itr->second)) {
            QStringList value;
            value << QString::fromStdString(itr->second->getShortName()) << QString::fromStdString(itr->second->getName());
            settings.setValue(QString::fromStdString(itr->first), value);
            DEBUG(itr->first << " " <<  nparent->indexOf(itr->second));
        }
    }
    settings.endGroup();

    QStringList localPhysNames;
    for (const auto& one : localPhys) {
        localPhysNames.append(QString::fromStdString(one.first));
    }
    setProperty("NeuSave-localPhysNames",localPhysNames);
    qDebug() << localPhysNames;

    nGenericPan::saveSettings(settings);
}



