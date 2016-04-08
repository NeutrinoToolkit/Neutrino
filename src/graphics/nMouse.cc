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
#include "nMouse.h"
#include "neutrino.h"

nMouse::nMouse(neutrino *neu) : QGraphicsItem()
{
    nparent=neu;
    size=QSize(100,100);
    color=QColor(Qt::black);
    color.setAlpha(255);
}

// reimplementation
QRectF nMouse::boundingRect() const {
//	QRectF bbox(-pos().x(),-1,size.width(),3);
//	return bbox;
    return shape().boundingRect();
}

void nMouse::setSize(QSize my_size) {
    size=my_size;
}

void nMouse::changeColor() {
    QColorDialog colordial(color);
    colordial.setOption(QColorDialog::ShowAlphaChannel);
    colordial.exec();
    if (colordial.result() && colordial.currentColor().isValid()) {
        color=colordial.currentColor();
        update();
    }
}


QPainterPath nMouse::shape() const {
    QPainterPath my_path;

//    double r=4.0/nparent->my_w.my_view->transform().m11();
//    my_path.moveTo(0,-5*r);
//    my_path.lineTo(0,-r);
//    my_path.moveTo(0,r);
//    my_path.lineTo(0,5*r);

//    my_path.moveTo(-5*r,0);
//    my_path.lineTo(-r,0);
//    my_path.moveTo(r,0);
//    my_path.lineTo(5*r,0);
//    my_path.addEllipse(-r, -r, 2*r, 2*r);
    my_path.moveTo(0,-pos().y());
    my_path.lineTo(0,size.height()-pos().y());
    my_path.moveTo(-pos().x(),0);
    my_path.lineTo(size.width()-pos().x(),0);
    return my_path;
}

void
nMouse::paint(QPainter* p, const QStyleOptionGraphicsItem* option, QWidget* ) {
    p->setClipRect( option->exposedRect );
    p->setPen(QPen(color));
    p->drawPath(shape());
}








