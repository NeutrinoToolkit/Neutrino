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
#include <cmath>
#include <QGraphicsView>
#include <QPainter>
#include <QPainterPath>
#include <QSettings>

// arms of the "+" and "x" shapes and radius of the central circle, in screen pixels
static const double mouseArm=60.0;
static const double mouseRadius=4.0;

nMouse::nMouse(QGraphicsView *view) :
    my_shape(0),
    size(100,100),
    my_view(view),
    visible(true)
{
    pen.setCosmetic(true);

    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    my_shape=my_set.value("mouseShape",my_shape).toInt();
    pen.setColor(my_set.value("mouseColor",pen.color()).value<QColor>());
    pen.setWidthF(my_set.value("mouseThickness",pen.widthF()).toDouble());
    my_set.endGroup();
}

// saved when changed (it was in the destructor)
void nMouse::saveSettings() {
    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    my_set.setValue("mouseShape", my_shape);
    my_set.setValue("mouseColor", pen.color());
    my_set.setValue("mouseThickness", pen.widthF());
    my_set.endGroup();
}

QPointF nMouse::viewPos() const {
    return my_view->viewportTransform().map(position);
}

QRegion nMouse::region() const {
    if (!visible || my_shape%num_shape==0) return QRegion();
    int pad=static_cast<int>(std::ceil(std::max(1.0,pen.widthF())))+2; // cosmetic width 0 is 1 pixel, +2 for antialiasing
    QPoint c=viewPos().toPoint();
    if (my_shape%num_shape==1) {
        QRect vr=my_view->viewport()->rect();
        return QRegion(c.x()-pad,vr.top(),2*pad+1,vr.height()) + QRegion(vr.left(),c.y()-pad,vr.width(),2*pad+1);
    }
    int l=static_cast<int>(mouseArm)+pad;
    return QRegion(c.x()-l,c.y()-l,2*l+1,2*l+1);
}

void nMouse::update() {
    my_view->viewport()->update(region());
}

void nMouse::setPos(QPointF p) {
    if (p==position) return;
    QRegion old=region();
    position=p;
    my_view->viewport()->update(old+region());
}

void nMouse::setVisible(bool val) {
    QRegion old=region();
    visible=val;
    my_view->viewport()->update(old+region());
}

void nMouse::setShape(int num) {
    QRegion old=region();
    my_shape=((num%num_shape)+num_shape)%num_shape;
    my_view->viewport()->update(old+region());
}

void nMouse::setSize(QSize my_size) {
    QRegion old=region();
    size=my_size;
    my_view->viewport()->update(old+region());
}

void nMouse::setPen(QPen my_pen) {
    QRegion old=region();
    pen=my_pen;
    pen.setCosmetic(true);
    my_view->viewport()->update(old+region());
}

// in viewport pixels: the shape doesn't change with the zoom
void nMouse::paint(QPainter* p) {
    if (!visible) return;
    int shape=my_shape%num_shape;
    if (shape==0) return;

    p->save();
    p->resetTransform();
    p->setPen(pen);
    p->setBrush(Qt::NoBrush);

    QPointF c=viewPos();
    const double r=mouseRadius;
    const double len=mouseArm;
    QPainterPath my_path;
    switch (shape) {
        case 1: {
            QRectF img=my_view->viewportTransform().mapRect(QRectF(QPointF(0,0),QSizeF(size)));
            my_path.moveTo(c.x(),img.top());
            my_path.lineTo(c.x(),img.bottom());
            my_path.moveTo(img.left(),c.y());
            my_path.lineTo(img.right(),c.y());
            break;
        }
        case 2:
            my_path.moveTo(0,-len);
            my_path.lineTo(0,-r);
            my_path.moveTo(0,r);
            my_path.lineTo(0,len);

            my_path.moveTo(-len,0);
            my_path.lineTo(-r,0);
            my_path.moveTo(r,0);
            my_path.lineTo(len,0);
            my_path.addEllipse(-r, -r, 2*r, 2*r);
            my_path.translate(c);
            break;
        case 3:
            my_path.moveTo(-len,-len);
            my_path.lineTo(-r,-r);
            my_path.moveTo(r,r);
            my_path.lineTo(len,len);

            my_path.moveTo(-len,len);
            my_path.lineTo(-r,r);
            my_path.moveTo(r,-r);
            my_path.lineTo(len,-len);
            my_path.addEllipse(-r, -r, 2*r, 2*r);
            my_path.translate(c);
            break;
    }
    p->drawPath(my_path);
    p->restore();
}

#if 0
// old version (QGraphicsItem)
nMouse::~   nMouse() {
    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    my_set.setValue("mouseShape", my_shape);
    my_set.setValue("mouseColor", pen.color());
    my_set.setValue("mouseThickness", pen.widthF());
    my_set.endGroup();
}

nMouse::nMouse() : QGraphicsItem(),
    my_shape(0),
    size(100,100)
{
    pen.setCosmetic(true);

    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    my_shape=my_set.value("mouseShape",my_shape).toInt();
    pen.setColor(my_set.value("mouseColor",pen.color()).value<QColor>());
    pen.setWidthF(my_set.value("mouseThickness",pen.widthF()).toDouble());
    my_set.endGroup();


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

QPainterPath nMouse::shape() const {
    QPainterPath my_path;
    const double r=4.0;
    const int len=15*r;

    switch (my_shape%4) {
        case 0: break;
        case 1:
            my_path.moveTo(0,-pos().y());
            my_path.lineTo(0,size.height()-pos().y());
            my_path.moveTo(-pos().x(),0);
            my_path.lineTo(size.width()-pos().x(),0);

            break;
        case 2:
            my_path.moveTo(0,-len);
            my_path.lineTo(0,-r);
            my_path.moveTo(0,r);
            my_path.lineTo(0,len);

            my_path.moveTo(-len,0);
            my_path.lineTo(-r,0);
            my_path.moveTo(r,0);
            my_path.lineTo(len,0);
            my_path.addEllipse(-r, -r, 2*r, 2*r);
            break;

        case 3:
            my_path.moveTo(-len,-len);
            my_path.lineTo(-r,-r);
            my_path.moveTo(r,r);
            my_path.lineTo(len,len);

            my_path.moveTo(-len,len);
            my_path.lineTo(-r,r);
            my_path.moveTo(r,-r);
            my_path.lineTo(len,-len);
            my_path.addEllipse(-r, -r, 2*r, 2*r);
            break;
    }
    return my_path;
}

void
nMouse::paint(QPainter* p, const QStyleOptionGraphicsItem* option, QWidget* ) {
    p->setClipRect( option->exposedRect );
    p->setPen(pen);
    p->drawPath(shape());
}
#endif
