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
#include "nVisarPicker.h"
#include "neutrino.h"

#include <qwt_plot.h>
#include <qwt_plot_zoomer.h>


nVisarPicker::nVisarPicker(int xAxis, int yAxis, int selectionFlags, RubberBand rubberBand, DisplayMode trackerMode, QwtPlotCanvas* canvas)
: QwtPlotPicker(xAxis, yAxis, selectionFlags, rubberBand, trackerMode, canvas)
{}

QwtText nVisarPicker::trackerText (const QwtDoublePoint & pos) cons
{
	const QPoint point = pos.toPoint();
	emit mouseMoved(point);
	return QwtText(QString::number(point.x()) + ", " + QString::number(point.y()));
}



