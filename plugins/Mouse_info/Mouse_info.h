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
#include <iostream>

#include <QtGui>
#include <QWidget>

#include "nGenericPan.h"

#include "ui_Mouse_info.h"

#ifndef __Mouse_info_h
#define __Mouse_info_h

class neutrino;

class Mouse_info : public nGenericPan {
    Q_OBJECT

public:

    Q_INVOKABLE Mouse_info (neutrino *parent);

    Ui::Mouse_info my_w;
    QPointF mouse;

public slots:
    void setMouse(QPointF);
    void updateLabels();
    void updateOrigin();
    void updateScale();
    void updateUnits();
    void addPoint(QPointF);

    void remove_point();
    void copyPoints();
    void export_txt();
    QString getPointText();
    //	void bufferChanged(nPhysD*);
};

NEUTRINO_PLUGIN(Mouse_info,Analysis,":icons/ruler.png",Qt::CTRL | Qt::ALT | Qt::Key_M, 10);

#endif



