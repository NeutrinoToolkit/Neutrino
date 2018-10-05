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
#include "neutrino.h"

FocalSpot::FocalSpot(neutrino *nparent) : nGenericPan(nparent)
{
    my_w.setupUi(this);
    nContour = new nLine(this,3);

    show();
    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(calculate_stats()));
    connect(my_w.zero_dsb, SIGNAL(editingFinished()), this, SLOT(calculate_stats()));
    connect(my_w.check_dsb, SIGNAL(editingFinished()), this, SLOT(calculate_stats()));
    connect(my_w.blur_radius_sb, SIGNAL(editingFinished()), this, SLOT(calculate_stats()));
    if (my_w.centroid->isChecked()) {
        connect(nparent->my_w->my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setPosZero(QPointF)));
    }

    calculate_stats();
}

void FocalSpot::on_centroid_toggled(bool tog) {
    if (tog) {
        connect(nparent->my_w->my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setPosZero(QPointF)));
    } else {
        disconnect(nparent->my_w->my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(setPosZero(QPointF)));
    }
}

void FocalSpot::setPosZero(QPointF pos) {
    if (currentBuffer) {
        currentBuffer->set_origin(vec2f(pos.x(),pos.y()));
        nparent->my_w->my_view->update();
        QApplication::processEvents();
        calculate_stats();
    }
}

void
FocalSpot::calculate_stats()
{
    if (!currentBuffer)
        return;

    if (currentBuffer->prop.count("FocalSpotDynamic") > 0) {
        DEBUG("nFocal dynamic image --- skip");
        return;
    }

    my_w.stats->clear();
    // 0. build decimated
    decimated = nPhysD(*currentBuffer);
    physMath::phys_fast_gaussian_blur(decimated, my_w.blur_radius_sb->value());
    decimated.TscanBrightness();

    // 1. find centroid
    vec2i centr;
    if (currentBuffer->get_origin() == vec2i(0,0)) {
        centr = decimated.max_Tv;
        currentBuffer->set_origin(centr);
    } else {
        centr = currentBuffer->get_origin();
    }
    decimated.set_origin(centr);

    my_w.centroid_lbl->setText(QString("%1:%2").arg(centr.x()).arg(centr.y()));


    // 2. calculate integrals
    double c_value = currentBuffer->point(centr.x(),centr.y());
    double total_energy = 0;
    double above_th_energy = 0;
    double below_zero_energy = 0;
    int point_count = 0, zero_point_count = 0;
    double th = my_w.check_dsb->value()/100.*(c_value-my_w.zero_dsb->value()) +my_w.zero_dsb->value() ;
    double zl = my_w.zero_dsb->value();
    for (size_t ii=0; ii<currentBuffer->getSurf(); ii++) {
        if (currentBuffer->point(ii) > th) {
            above_th_energy += currentBuffer->point(ii);
            point_count++;
        } else if (currentBuffer->point(ii) < zl) {
            below_zero_energy += currentBuffer->point(ii); // to minimize stupid results when zero level is far from reality
            zero_point_count++;
        } else {
            below_zero_energy += zl;
            zero_point_count++;
        }
    }
    double zero_energy_in_peak = (below_zero_energy/zero_point_count)*point_count;
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

    my_w.stats->append(QString("below zero average: %1").arg(below_zero_energy/zero_point_count));

    my_w.stats->append(QString("points stats (bz/az) %1/%2").arg(zero_point_count).arg(point_count));

    my_w.stats->append(QString("contour integral (contour points) %1 (%2)").arg(ath_integral).arg(ath_points));

}

QList<double>
FocalSpot::find_contour(double th)
{
    QList<double> ql;
    if (currentBuffer) {
        std::list<vec2i> contour;
        physMath::contour_trace(decimated, contour, th);

        DEBUG(5, "got contour of "<<contour.size()<<" points");


        nContour->setPoints(QPolygonF());
        if (contour.size() > 0) {

            // set polygon
            nContour->setPoints(QPolygonF());
            QPolygonF myp;
            vec2f centroid(0,0);
            for (auto& itr : contour) {
                myp<<QPointF(itr.x(), itr.y());
                centroid+= itr;
                //std::cerr<<*itr<<std::endl;
            }
            centroid /= contour.size();

            // get stats
            vec2f c_scale = currentBuffer->get_scale();
            double min_r = std::numeric_limits<double>::max();
            double max_r = std::numeric_limits<double>::min();
            for (auto &itr : contour) {
                double dd = vmath::td<double>(itr-centroid, c_scale).mod();
                if (dd > max_r) max_r = dd;
                if (dd < min_r) min_r = dd;
            }

            my_w.stats->append(QString("min:max radius: %1 : %2").arg(min_r).arg(max_r));

            nContour->setPoints(myp);
            my_w.statusBar->showMessage("Contour ok", 2000);

            std::list<double> ci = physMath::contour_integrate(*currentBuffer, contour, true);
            while (ci.size() > 0) {
                ql.append(ci.front()); ci.pop_front();
            }
        } else {
            my_w.statusBar->showMessage("Contour not ok", 5000);
        }
    }
    return ql;
}


void
FocalSpot::bufferChanged(nPhysD *buf)
{
    nGenericPan::bufferChanged(buf);
    calculate_stats();
}
