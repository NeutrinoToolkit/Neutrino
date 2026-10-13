/*
 *
 *    Copyright (C) 2014 Alessandro Flacco, Tommaso Vinci All Rights Reserved
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


#include "FocalSpot.h"
#include "nPhysImageF.h"
#include "neutrino.h"

FocalSpot::FocalSpot(neutrino *nparent) : nGenericPan(nparent)
{
    my_w.setupUi(this);
    nContour = new nLine(this,3);
    nContour->toggleClosedLine(true);
    // (bufferChanged is already handled by the bufferChanged override)
    connect(my_w.zero_dsb, SIGNAL(editingFinished()), this, SLOT(calculate_stats()));
    connect(my_w.check_dsb, SIGNAL(editingFinished()), this, SLOT(calculate_stats()));
    connect(my_w.blur_radius_sb, SIGNAL(editingFinished()), this, SLOT(calculate_stats()));
    if (my_w.centroid->isChecked()) {
        connect(nparent->my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setPosZero(QPointF)));
    }
    my_w.plot->addGraph(my_w.plot->xAxis, my_w.plot->yAxis);
    my_w.plot->graph(0)->setPen(QPen(Qt::black));
    my_w.plot->show();

    connect(my_w.plot,SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(mouseAtPlot(QMouseEvent*)));
    show();
    QApplication::processEvents();
    calculate_stats();

}

void FocalSpot::loadSettings(QString my_settings) {
    nGenericPan::loadSettings(my_settings);
    calculate_stats();
}

void FocalSpot::on_centroid_toggled(bool tog) {
    if (tog) {
        connect(nparent->my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setPosZero(QPointF)));
    } else {
        disconnect(nparent->my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setPosZero(QPointF)));
    }
}

void FocalSpot::setPosZero(QPointF pos) {
    if (currentBuffer) {
        currentBuffer->set_origin(vec2f(pos.x(),pos.y()));
        nparent->emitPhysChanged(currentBuffer); // Image_list, pans, view; recomputes the stats via bufferChanged
    }
}

void
FocalSpot::calculate_stats()
{
    if (!currentBuffer || computing)
        return;

    if (currentBuffer->prop.count("FocalSpotDynamic") > 0) {
        // DEBUG("nFocal dynamic image --- skip");
        return;
    }
    computing=true;

    my_w.stats->clear();
    // 0. build decimated
    decimated = nPhysD(*currentBuffer);
    physMath::phys_fast_gaussian_blur(decimated, my_w.blur_radius_sb->value());
    decimated.TscanBrightness();

    // 1. find centroid: the origin if set, otherwise the maximum of the blurred image
    if (currentBuffer->get_origin() == vec2i(0,0)) {
        centre = decimated.max_Tv;
        if (my_w.moveOrigin->isChecked()) {
            currentBuffer->set_origin(centre);
            nparent->emitPhysChanged(currentBuffer);
        }
    } else {
        centre = currentBuffer->get_origin();
    }
    decimated.set_origin(centre);

    my_w.centroid_lbl->setText(QString("%1:%2").arg(centre.x()).arg(centre.y()));


    // 2. calculate integrals
    // reference value at the centre from the blurred image (a single raw pixel is too noisy)
    double c_value = decimated.point(centre.x(),centre.y());
    double total_energy = 0;
    double above_th_energy = 0;
    double below_zero_energy = 0;
    int point_count = 0, zero_point_count = 0;
    double th = my_w.check_dsb->value()/100.*(c_value-my_w.zero_dsb->value()) +my_w.zero_dsb->value() ;
    double zl = my_w.zero_dsb->value();
    for (size_t ii=0; ii<currentBuffer->getSurf(); ii++) {
        double zz=currentBuffer->point(ii);
        if (! isnan(zz)) {
            if (zz > th) {
                above_th_energy += zz;
                point_count++;
            } else if (zz < zl) {
                below_zero_energy += zz; // to minimize stupid results when zero level is far from reality
                zero_point_count++;
            } else {
                below_zero_energy += zl;
                zero_point_count++;
            }
        }
    }
    // background per pixel: no pixel below threshold -> use the zero level
    double background = zero_point_count ? below_zero_energy/zero_point_count : zl;
    double zero_energy_in_peak = background*point_count;
    total_energy = currentBuffer->sum()-below_zero_energy-zero_energy_in_peak;
    above_th_energy -= zero_energy_in_peak;

    double energy_ratio;
    if (total_energy != 0)
        energy_ratio = 100*(above_th_energy/total_energy);
    else
        energy_ratio = 0;

    my_w.stats->append(QString("Threshold integral %: %1").arg(energy_ratio));

    //std::cerr<<"min/max: "<<cur->get_min()<<"/"<<cur->get_max()<<", surf: "<<cur->getSurf()<<", point_count: "<<point_count<<std::endl;

    QList<double> c_integral = find_contour(th);
    double ath_integral=0, ath_points=0;

    //double contour_ratio = contour_integral();
    if (c_integral.size() > 0) {
        ath_integral = c_integral.front();
        c_integral.pop_front();
        ath_points = c_integral.front();
        c_integral.pop_front();
        my_w.stats->append(QString("Contour integral %: %1 (total: %2)").arg(100*(ath_integral-zero_energy_in_peak)/total_energy).arg(ath_integral));
    }

    // populate numerics
    my_w.stats->append(QString("total: %1").arg(currentBuffer->sum()));

    my_w.stats->append(QString("below zero average: %1").arg(background));

    my_w.stats->append(QString("points stats (bz / az) %1 / %2").arg(zero_point_count).arg(point_count));

    my_w.stats->append(QString("contour integral (contour points) %1 (%2)").arg(ath_integral).arg(ath_points));

    computing=false;
}

QList<double>
FocalSpot::find_contour(double th)
{
    QList<double> ql;
    if (currentBuffer) {
        std::list<vec2f> contour;
        physMath::contour_trace(decimated, contour, th);

        // DEBUG(5, "got contour of "<<contour.size()<<" points");

        QPolygonF myp;

        if (contour.size() > 1) {

            // set polygon
            for (auto& itr : contour) {
                myp<<QPointF(itr.x(), itr.y());
            }
            // centroid of the enclosed area (the vertices of a marching-squares line are unevenly spaced)
            vec2f centroid(0,0);
            {
                double a2=0, gx=0, gy=0;
                for (int k=0;k<myp.size();k++) {
                    const QPointF &p=myp[k], &q=myp[(k+1)%myp.size()];
                    double cr=p.x()*q.y()-q.x()*p.y();
                    a2+=cr; gx+=(p.x()+q.x())*cr; gy+=(p.y()+q.y())*cr;
                }
                if (a2!=0) centroid=vec2f(gx/(3*a2), gy/(3*a2));
                else { for (auto& itr : contour) centroid+=itr; centroid/=contour.size(); }
            }
            my_w.stats->append(QString("Contour centre: ( %1  , %2 )").arg(centroid.x()).arg(centroid.y()));



            // get stats
            vec2f c_scale = currentBuffer->get_scale();
            // DEBUG(c_scale);
            double min_r = std::numeric_limits<double>::max();
            double max_r = 0;
            double meanr=0;
            for (auto &itr : contour) {
                double dd = vmath::td<double>(itr-centroid, c_scale).mod();
                meanr+=abs(dd);
                if (dd > max_r) max_r = dd;
                if (dd < min_r) min_r = dd;
//                DEBUG(itr <<  " - "  << centroid)
            }
            meanr/=myp.size();

            my_w.stats->append(QString("Mean radius: %1").arg(meanr));
            my_w.stats->append(QString("Min Max radius:  %1  %2").arg(min_r).arg(max_r));

            my_w.statusBar->showMessage("Contour ok", 2000);

            // {integral, number of pixels} inside the contour (boundary included)
            // the contour is sub-pixel: the pixels whose centre is inside are the right estimate (no boundary pixels)
            physMath::contour_integral ci = physMath::contour_integrate(*currentBuffer, contour, false);
            ql << ci.integral << double(ci.count);

// PLOT radius: mean value in 100 rings around the centre (single pass, no shared accumulation between threads)
            const int len=100;
            vec2f c_origin(centre.x(),centre.y());
            const double W=currentBuffer->getW(), H=currentBuffer->getH();
            double rmax=0; // farthest corner
            for (double cx : {0.0, W-1}) for (double cy : {0.0, H-1}) {
                rmax=std::max(rmax, vec2f((cx-c_origin.x())*c_scale.x(),(cy-c_origin.y())*c_scale.y()).mod());
            }
            QVector <double> rplot(len,0.0);
            QVector <double> rnum(len,0.0);
            if (rmax>0) {
                for(size_t j = 0 ; j < currentBuffer->getH(); j++) {
                    for(size_t i = 0 ; i < currentBuffer->getW(); i++) {
                        double val=currentBuffer->point(i,j);
                        if (std::isfinite(val)) {
                            double rr=vec2f((i-c_origin.x())*c_scale.x(),(j-c_origin.y())*c_scale.y()).mod();
                            int r = std::min(len-1, static_cast<int>(len*rr/rmax));
                            rplot[r]+=val;
                            rnum[r]++;
                        }
                    }
                }
            }

            QVector <double> rdata(len);
            for (int i=0;i<len;i++) {
                rdata[i]=i*rmax/len;
                rplot[i] = rnum[i] ? rplot[i]/rnum[i] : std::numeric_limits<double>::quiet_NaN();
            }
            my_w.plot->graph(0)->setData(rdata,rplot,true);
            my_w.plot->graph(0)->rescaleAxes();

            my_w.plot->replot();
//  END PLOT radius

            nContour->setPoints(myp);
            nContour->show();
        } else {
            my_w.statusBar->showMessage("Contour not ok", 5000);
            nContour->hide(); // don't show the contour of another image/threshold
        }

    }
    return ql;
}

void FocalSpot::mouseAtWorld(QPointF p){
    double trueLength = std::sqrt(std::pow(p.x(), 2) + std::pow(p.y(), 2));
    if (currentBuffer) { // world -> pixel -> distance from the spot centre
        vec2f sc=currentBuffer->get_scale(), og=currentBuffer->get_origin();
        trueLength = vec2f((p.x()/sc.x()+og.x()-centre.x())*sc.x(), (p.y()/sc.y()+og.y()-centre.y())*sc.y()).mod();
    }
    my_w.plot->setMousePosition(trueLength);
    QString msg;
    QTextStream(&msg) << "Distance to center: " << trueLength;
    my_w.statusBar->showMessage(msg);
    my_w.plot->setMousePosition(trueLength);
}

void
FocalSpot::bufferChanged(nPhysD *buf)
{
    nGenericPan::bufferChanged(buf);
    calculate_stats();
}


void FocalSpot::mouseAtPlot(QMouseEvent* e) {
    QString msg;
    QTextStream(&msg) << my_w.plot->xAxis->pixelToCoord(e->pos().x()) << ","
                      << my_w.plot->yAxis->pixelToCoord(e->pos().y());

    my_w.statusBar->showMessage(msg);
}

