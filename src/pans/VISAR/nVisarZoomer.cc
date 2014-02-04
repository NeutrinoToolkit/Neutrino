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
#include "nVisarZoomer.h"
#include "neutrino.h"

#include <qwt_plot.h>
#include <qwt_plot_zoomer.h>


#if QWT_VERSION < 0x060100
nVisarZoomer::nVisarZoomer(QwtPlotCanvas *canvas): QwtPlotZoomer(canvas) {
#else
nVisarZoomer::nVisarZoomer(QWidget *canvas): QwtPlotZoomer(canvas) {
#endif
	setTrackerMode(AlwaysOn);
}

QwtText nVisarZoomer::trackerText(const QPoint &pos) const {
	QColor bg(Qt::white);
	bg.setAlpha(200);
	QwtText text;
	text.setBackgroundBrush( QBrush( bg ));

	double x=plot()->invTransform(QwtPlot::xBottom, pos.x());
	double y1=plot()->invTransform(QwtPlot::yLeft, pos.y());
	double y2=plot()->invTransform(QwtPlot::yRight, pos.y());

	text.setText(QString::number(x)+", "+QString::number(y1)+" ("+QString::number(y2)+")");
	return text;
}




