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

#include <QtGui>
#include <QWidget>

#include "nGenericPan.h"
#include "ui_Contours.h"

#ifndef Contours_h_
#define Contours_h_

class neutrino;
class nLine;

class Contours : public nGenericPan, private Ui::Contours {
	Q_OBJECT
public:
    Q_INVOKABLE Contours(neutrino*);
	
	QPointer<nLine> my_c;

public slots:
	void on_draw_pb_released();
    void on_percent_released();
    void on_actionCenter_toggled(bool);

    void setOrigin(QPointF);

private:
	// blurred image for easier calculations
	nPhysD decimated;
};

NEUTRINO_PLUGIN(Contours,Analysis);

#endif
