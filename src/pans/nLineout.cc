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
#include "nLineout.h"

nLineout::nLineout(neutrino *parent, QString win_name, enum phys_direction plot_dir)
: nGenericPan(parent, win_name), lineout_zoom(false), cut_dir(plot_dir)
{
	my_w.setupUi(this);

	// indexes
	paxis_index = (int)plot_dir;
	naxis_index = (paxis_index+1)%2;

	connect(my_w.actionToggleZoom,SIGNAL(triggered()), this, SLOT(toggle_zoom()));
	connect(my_w.actionAutoscale,SIGNAL(triggered()), this, SLOT(toggle_scale()));
	connect(my_w.actionGetMinMax,SIGNAL(triggered()), this, SLOT(getMinMax()));
	autoScale=true;
	toggle_scale(autoScale);
	my_w.toolBar->insertWidget(my_w.actionMax,my_w.minVal);
	my_w.toolBar->insertWidget(my_w.actionToggleZoom,my_w.maxVal);
	my_w.toolBar->insertSeparator(my_w.actionToggleZoom);

	connect(parent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(updateLastPoint(void)));

	curve = new QwtPlotCurve(win_name);
	curve->show();
	curve->attach(my_w.the_plot);


	QPen marker_pen;
	marker_pen.setColor(QColor(255,0,0));
	marker.setLineStyle(QwtPlotMarker::VLine);
	marker.setLinePen(marker_pen);
	marker.attach(my_w.the_plot);
	marker_pen.setColor(QColor(0,0,255));
	markerRuler.setLineStyle(QwtPlotMarker::VLine);
	markerRuler.setLinePen(marker_pen);
	markerRuler.attach(my_w.the_plot);
	markerRuler.setXValue(0);

	decorate();
	updateLastPoint();
}

// mouse movemen
void
nLineout::mouseAtMatrix(QPointF p) {

	if (currentBuffer != NULL) {

		// get bounds from schermo
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
		const double *dvec = currentBuffer->to_dvector(cut_dir, b_p[naxis_index]);
		if (!lineout_zoom) {
			phys_get_vec_brightness(dvec, currentBuffer->getSizeByIndex(cut_dir), vmin, vmax);
			curve->setRawSamples(currentBuffer->to_axis(cut_dir), dvec, currentBuffer->getSizeByIndex(cut_dir));
			curve->attach(my_w.the_plot);
		} else {
			size_t lat_skip = max(b_o[paxis_index], 0);
			size_t z_size = min(b_c[paxis_index]-lat_skip, currentBuffer->getSizeByIndex(cut_dir)-lat_skip);
			phys_get_vec_brightness(dvec+lat_skip, z_size, vmin, vmax);
			curve->setRawSamples(currentBuffer->to_axis(cut_dir)+lat_skip, dvec+lat_skip, z_size);
		}

//  TODO: this might be ugly....
		QPen marker_pen;
		marker_pen.setColor(nparent->my_mouse.color);
		marker.setLinePen(marker_pen);
		marker_pen.setColor(nparent->my_tics.rulerColor);
		markerRuler.setLinePen(marker_pen);

		marker.setXValue((b_p[paxis_index]-b_r[paxis_index])*b_s[paxis_index]);

		markerRuler.setVisible(nparent->my_tics.rulerVisible);

		my_w.the_plot->setAxisScale(curve->xAxis(),curve->minXValue(),curve->maxXValue(),0);
		if (autoScale) {
			my_w.minVal->setText(QString::number(vmin));
			my_w.maxVal->setText(QString::number(vmax));
			my_w.the_plot->setAxisScale(curve->yAxis(),vmin,vmax,0);
		} else {
			my_w.the_plot->setAxisScale(curve->yAxis(),my_w.minVal->text().toDouble(),my_w.maxVal->text().toDouble(),0);
		}
		my_w.the_plot->replot();

	}

}

void
nLineout::toggle_zoom() {
	lineout_zoom = !lineout_zoom;
	if (lineout_zoom) {
		statusBar()->showMessage(tr("Lineout linked to visible part"),2000);
	} else {
		statusBar()->showMessage(tr("Lineout on the whole image"),2000);
	}
	updateLastPoint();
}

void
nLineout::toggle_scale() {
	updateLastPoint();
	toggle_scale(!autoScale);
}

void
nLineout::toggle_scale(bool val) {
	autoScale=val;
	my_w.minVal->setReadOnly(autoScale);
	my_w.maxVal->setReadOnly(autoScale);
	my_w.actionGetMinMax->setEnabled(!autoScale);
	if (autoScale) {
		statusBar()->showMessage(tr("Auto scale"),2000);
	} else {
		statusBar()->showMessage(tr("Fixed scale"),2000);
	}
}

void nLineout::getMinMax() {
	if (my_w.actionGetMinMax->isChecked()) {
		autoScale=true;
		my_w.actionAutoscale->setEnabled(false);
		statusBar()->showMessage(tr("Press mouse to get min and max"),5000);
		connect(nparent->my_w.my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(setMinMax(QPointF)));
	} else {
		autoScale=false;
		statusBar()->showMessage(tr("Canceled"),5000);
		disconnect(nparent->my_w.my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(setMinMax(QPointF)));
	}
}

void nLineout::setMinMax(QPointF point) {
	autoScale=false;
	my_w.actionAutoscale->setEnabled(true);
	statusBar()->showMessage(tr("Got min and max")+")"+QString::number(point.x())+","+QString::number(point.x())+")",5000);
	my_w.actionGetMinMax->setChecked(false);
	disconnect(nparent->my_w.my_view, SIGNAL(mousePressEvent_sig(QPointF)), this, SLOT(setMinMax(QPointF)));
}

void nLineout::nZoom(double) {
	if (lineout_zoom) updateLastPoint();
}

void nLineout::updateLastPoint() {
	mouseAtMatrix(nparent->my_mouse.pos());
}



