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
#ifndef __nColorBar_h
#define __nColorBar_h

#include <iostream>

#include <QtGui>
#include <QWidget>

#include "nGenericPan.h"
#include "ui_nColorBar.h"
#include "nHistogram.h"

class neutrino;

class nColorBar : public nGenericPan {
	Q_OBJECT
	
	
private:
	QDoubleValidator *dVal;
	
public:
	
    nColorBar (neutrino *);
	neutrino *parent(){
		return (neutrino *) QWidget::parent();
	};
	nPhysD *cutOffPhys;
	Ui::Colorbar my_w;

    QComboBox palettes;

	QColor colorBase;
	
public slots:
    void minChanged();
    void maxChanged();
	
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
	void loadPalettes();
	void itemDoubleClicked(QTreeWidgetItem*,int);
	
	void addPaletteFile();
	void removePaletteFile();
    void on_gamma_valueChanged(int);

    vec2f sliderValues();

    void percentChange();


signals:
	void change_contrast(double,double);

};

#endif
