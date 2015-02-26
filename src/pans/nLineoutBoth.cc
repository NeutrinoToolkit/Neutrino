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
#include "nLineoutBoth.h"
#include "neutrino.h"
#include <qwt_plot_zoomer.h>
#include <qwt_plot_panner.h>
#include <qwt_plot_renderer.h>

nLineoutBoth::nLineoutBoth(neutrino *parent, QString win_name)
: nGenericPan(parent, win_name)
{
	my_w.setupUi(this);

    my_w.statusBar->addPermanentWidget(my_w.autoscale, 0);
    my_w.statusBar->addPermanentWidget(my_w.lockClick, 0);
    
	connect(parent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updateLastPoint(void)));

    connect(my_w.autoscale, SIGNAL(released()), this, SLOT(updateLastPoint(void)));

    connect(my_w.lockClick,SIGNAL(released()), this, SLOT(setBehaviour()));
    setBehaviour();

	my_w.plot->enableAxis(QwtPlot::xTop);
	my_w.plot->enableAxis(QwtPlot::yRight);
	my_w.plot->enableAxis(QwtPlot::xBottom);
	my_w.plot->enableAxis(QwtPlot::yLeft);
	
	QPen marker_pen;
	marker_pen.setColor(QColor(255,0,0));
	marker.setLineStyle(QwtPlotMarker::Cross);
	marker.setLinePen(marker_pen);
	marker.attach(my_w.plot);
	marker_pen.setColor(QColor(0,0,255));
	markerRuler.setLinePen(marker_pen);
	markerRuler.attach(my_w.plot);

	markerRuler.setLineStyle(QwtPlotMarker::Cross);
	markerRuler.setValue(0,0);

	curve[0].setXAxis(QwtPlot::xBottom);
	curve[0].setYAxis(QwtPlot::yRight);
	curve[1].setXAxis(QwtPlot::xTop);
	curve[1].setYAxis(QwtPlot::yLeft);

	curve[0].setPen(QPen(Qt::red,1));
	curve[1].setPen(QPen(Qt::blue,1));
	
	for (int k=0;k<2;k++) {
		curve[k].attach(my_w.plot);
		curve[k].show();
		
	}
	

	my_w.plot->setAxisTitle(QwtPlot::xBottom, tr("X (red)"));
	my_w.plot->setAxisTitle(QwtPlot::yRight, tr("X value (red)"));
	my_w.plot->setAxisTitle(QwtPlot::yLeft, tr("Y (blue)"));
	my_w.plot->setAxisTitle(QwtPlot::xTop, tr("Y value (blue)"));
	(qobject_cast<QFrame*> (my_w.plot->canvas()))->setLineWidth(0);

	decorate();
	updateLastPoint();
    
}

void nLineoutBoth::setBehaviour() {
    if (my_w.lockClick->isChecked()) {
        disconnect(nparent->my_w.my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)));
        connect(nparent->my_w.my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)));
    } else {
        disconnect(nparent->my_w.my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(updatePlot(QPointF)));
        connect(nparent->my_w.my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(updatePlot(QPointF)));
    }
}


void nLineoutBoth::rescale(QPointF p) {
    DEBUG("HERE " << p.x() << " " << p.y());
    double minx = curve[0].minYValue();
    double maxx = curve[0].maxYValue();
    
    double miny = curve[1].minXValue();
    double maxy = curve[1].maxXValue();
    
    my_w.plot->setAxisScale(curve[0].xAxis(),curve[0].minXValue(), curve[0].maxXValue(),0);
    my_w.plot->setAxisScale(curve[0].yAxis(), minx, maxx, 0);
    my_w.plot->setAxisScale(curve[1].xAxis(), miny, maxy, 0);
	mouseAtMatrix(p);    
}


// mouse movement
void nLineoutBoth::updatePlot(QPointF p) {

	QPen marker_pen;
	marker_pen.setColor(nparent->my_mouse.color);
	marker.setLinePen(marker_pen);
	marker_pen.setColor(nparent->my_tics.rulerColor);
	markerRuler.setLinePen(marker_pen);
	
	marker.setValue(p);
	markerRuler.setVisible(nparent->my_tics.rulerVisible);
	
	if (currentBuffer != NULL) {
		for (int k=0;k<2;k++) {

			//get bounds from schermo
			QPointF orig, corner;
			orig = nparent->my_w.my_view->mapToScene(QPoint(0,0));
			QPoint lowerRight=QPoint(nparent->my_w.my_view->width(), nparent->my_w.my_view->height());
			corner = nparent->my_w.my_view->mapToScene(lowerRight);
			
			int b_o[2], b_c[2], b_p[2];
			double b_r[2], b_s[2];
			b_o[0] = int(orig.x()); b_o[1] = int(orig.y());
			b_c[0] = int(corner.x()); b_c[1] = int(corner.y());
			b_p[0] = int(p.x()); b_p[1] = int(p.y());
			b_r[0] = currentBuffer->get_origin().x(); b_r[1] = currentBuffer->get_origin().y();
			b_s[0] = currentBuffer->get_scale().x(); b_s[1] = currentBuffer->get_scale().y();
			
			statusBar()->showMessage(tr("Point (")+QString::number(p.x())+","+QString::number(p.y())+")="+QString::number(currentBuffer->point(p.x(),p.y())));
			
			double vmin=0, vmax=0;
			const double *dvec = currentBuffer->to_dvector((phys_direction)k, b_p[(k+1)%2]);
			size_t lat_skip = max(b_o[k], 0);
			size_t z_size = min(b_c[k]-lat_skip, currentBuffer->getSizeByIndex((phys_direction)k)-lat_skip);
			phys_get_vec_brightness(dvec+lat_skip, z_size, vmin, vmax);
			if (k==0) {
				curve[k].setRawSamples(currentBuffer->to_axis((phys_direction)k)+lat_skip, dvec+lat_skip, z_size);
			} else {
				curve[k].setRawSamples(dvec+lat_skip, currentBuffer->to_axis((phys_direction)k)+lat_skip, z_size);
			}
		}

        if (my_w.autoscale->isChecked()) {
            double minx = curve[0].minYValue();
            double maxx = curve[0].maxYValue();
            
            double miny = curve[1].minXValue();
            double maxy = curve[1].maxXValue();
            
            my_w.plot->setAxisScale(curve[0].xAxis(),curve[0].minXValue(), curve[0].maxXValue(),0);
            my_w.plot->setAxisScale(curve[0].yAxis(), minx, maxx, 0);
            my_w.plot->setAxisScale(curve[1].xAxis(), miny, maxy, 0);
//        } else {
//            double mini=nparent->colorMin;
//            double maxi=nparent->colorMax;
//            if (nparent->colorRelative) {
//                mini=currentBuffer->get_min()+nparent->colorMin*(currentBuffer->get_max() - currentBuffer->get_min());
//                maxi=currentBuffer->get_max()-(1.0-nparent->colorMax)*(currentBuffer->get_max() - currentBuffer->get_min());
//                my_w.plot->setAxisScale(curve[0].yAxis(), mini, maxi, 0);
//                my_w.plot->setAxisScale(curve[1].xAxis(), mini, maxi, 0);
//            }
        }
        
		my_w.plot->setAxisScale(curve[0].xAxis(),curve[0].minXValue(), curve[0].maxXValue(),0);
		my_w.plot->setAxisScale(curve[1].yAxis(),curve[1].maxYValue(), curve[1].minYValue(), 0);
		my_w.plot->replot();
	}		
}


void nLineoutBoth::nZoom(double) {
	updateLastPoint();
}

void nLineoutBoth::updateLastPoint() {
    if (!my_w.lockClick->isChecked()) {
        updatePlot(nparent->my_mouse.pos());
    }
    if (my_w.autoscale->isChecked()) {
        disconnect(nparent->my_w.my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(rescale(QPointF)));
    } else {
        connect(nparent->my_w.my_view, SIGNAL(mouseDoubleClickEvent_sig(QPointF)), this, SLOT(rescale(QPointF)));
    }
}




