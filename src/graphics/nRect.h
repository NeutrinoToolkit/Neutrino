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
#include <QMainWindow>
#include <QGraphicsObject>
#include <QTableWidget>
#include <QTableWidgetItem>
#include "nPhysD.h"

#ifndef __nRect
#define __nRect

#include "nObject.h"

class neutrino;
class nGenericPan;

namespace Ui {
class nObject;
}

class nRect : public nObject {
    Q_OBJECT
public:

    nRect(neutrino *neu) : nObject(neu, QString("rect")) {
        changeColorHolder(QColor(0,0,255,200));
    };

    nRect(nGenericPan *pan, int level) : nObject(pan,level, QString("rect")) {};

    neutrino *nparent;

    enum { Type = QGraphicsItem::UserType + 2 };
    int type() const { return Type;}

    QPainterPath path() const {
        QPainterPath my_path;
        if (ref.size()>1) {
            my_path.addRect(QRectF(ref[0]->pos(),ref[1]->pos()));
        } else {
            my_path.addRect(QRectF(0,0,0,0));
        }
        return my_path;
    }

    void paint(QPainter* p, const QStyleOptionGraphicsItem* , QWidget* ) {
        //	p->setCompositionMode((QPainter::CompositionMode)22);
        QPen pen;
        pen.setWidthF(nWidth/zoom);
        pen.setColor(nColor);
        p->setPen(pen);
        p->drawPath(path());
    }

};

Q_DECLARE_METATYPE(nRect*);

#endif
