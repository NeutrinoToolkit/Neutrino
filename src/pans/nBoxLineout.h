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
#include <QtGui>
#include <QWidget>

#include "nGenericPan.h"
#include "ui_nBoxLineout.h"

#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_plot_marker.h>
#include <qwt_plot_picker.h>
#include <qwt_picker_machine.h>
#include <qwt_plot_zoomer.h>

#ifndef __nBoxLineout
#define __nBoxLineout

#include "neutrino.h"


class nBoxLineoutZoomer: public QwtPlotPicker{
    
public:
#if QWT_VERSION < 0x060100
	nBoxLineoutZoomer(QwtPlotCanvas *canvas);
#else
	nBoxLineoutZoomer(QWidget *);
#endif
	
	virtual QwtText trackerText(const QPoint &pos) const;
	
};


class nRect;

class nBoxLineout : public nGenericPan {
	Q_OBJECT
	
public:
	
	nBoxLineout(neutrino *, QString);
	Ui::nBoxLineout my_w;
	
	public slots:

	void mouseAtWorld(QPointF);
	void updatePlot();
	
	void copy_clip();
	void export_txt();
	
	void export_pdf();
	
	// fun stuff

	void sceneChanged();

private:
    QPointer<nBoxLineoutZoomer> picker;

	QwtPlotCurve xCut,yCut;	
	QwtPlotMarker xMarker,yMarker, rxMarker, ryMarker;
	QPointer<nRect> box;
};

#endif
