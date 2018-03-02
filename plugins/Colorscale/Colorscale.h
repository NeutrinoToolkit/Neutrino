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
#ifndef __Colorscale_h
#define __Colorscale_h

#include <iostream>

#include <QtGui>
#include <QWidget>

#include "nGenericPan.h"
#include "ui_Colorscale.h"
#include "nHistogram.h"

class neutrino;

class Colorscale : public nGenericPan {
	Q_OBJECT
	
	
private:
	QDoubleValidator *dVal;
	
public:
	
    Q_INVOKABLE Colorscale (neutrino *);
	neutrino *parent(){
		return (neutrino *) QWidget::parent();
	};
	nPhysD *cutOffPhys;
    Ui::Colorscale my_w;
	
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

    void loadPalettes();

    void resetPalettes();
	void addPaletteFile();
	void removePaletteFile();
    void on_gamma_valueChanged(int);

    void on_fileList_itemClicked(QListWidgetItem*);

    vec2f sliderValues();
    const QIcon getPaletteIconFile(QString my_file);
    const QIcon getPaletteIcon(QString paletteName);

signals:
	void change_contrast(double,double);

};

NEUTRINO_PLUGIN(Colorscale,Image;Colortable,":icons/colors.png", Qt::Key_C);


#endif
