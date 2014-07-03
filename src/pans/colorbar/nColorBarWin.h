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
#ifndef __ncolorbarwin_h
#define __ncolorbarwin_h

#include <iostream>

#include <QtGui>
#include <QWidget>

#include "nGenericPan.h"
#include "ui_nColorBarWin.h"
#include "nHistogram.h"

class neutrino;
using namespace std;

class nColorBarWin : public nGenericPan {
	Q_OBJECT
	
	
private:
	QDoubleValidator *dVal;
	
public:
	
	nColorBarWin (neutrino *, QString);
	neutrino *parent(){
		return (neutrino *) QWidget::parent();
	};
	nPhysD *cutOffPhys;
	Ui::Colorbar my_w;
	
	QColor colorBase;
	
public slots:
	void minChanged(QString);
	void maxChanged(QString);
	
	void getMinMax();

	void toggleAutoscale();
	
	void invertColors();
	void setToMin();
	void setToMax();
	void slider_min_changed(int);
	void slider_max_changed(int);

	void bufferChanged(nPhysD*);
	void updatecolorbar();

	void cutOff();

	void addColor();
	void addPalette();
	void removePalette();
	void savePalettes();
	void loadPalettes();
	void itemDoubleClicked(QTreeWidgetItem*,int);
	
	void addPaletteFile();
	void removePaletteFile();
signals:
	void change_contrast(double,double);

};

#endif
