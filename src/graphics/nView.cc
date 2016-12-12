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
#include "neutrino.h"
#include "ui_neutrino.h"
#include "nView.h"

nView::~nView ()
{
}

nView::nView (QWidget *parent) : QGraphicsView (parent)
{
    trasformazione.reset();
    fillimage=true;
    setMouseTracking(true);
    setInteractive(true);
    grabGesture(Qt::SwipeGesture);
    QSettings settings("neutrino","");
    settings.beginGroup("Preferences");
    QVariant fontString=settings.value("defaultFont");
    if (fontString.isValid()) {
        QFont fontTmp;
        if (fontTmp.fromString(fontString.toString())) {
            setFont(fontTmp);
        }
    }
    showDimPixel=settings.value("showDimPixel",true).toBool();

    currentStepScaleFactor=settings.value("currentStepScaleFactor",15).toInt();

    setTransformationAnchor(QGraphicsView::AnchorViewCenter);
    
}

void nView::setZoomFactor(int val) {
    currentStepScaleFactor=val;
}

bool nView::event(QEvent *event)
{
    if (event->type() == QEvent::Gesture)
        return gestureEvent(static_cast<QGestureEvent*>(event));
    return QGraphicsView::event(event);
}

bool nView::gestureEvent(QGestureEvent *event)
{
    if (QGesture *swipe = event->gesture(Qt::SwipeGesture))
        swipeTriggered(static_cast<QSwipeGesture *>(swipe));
    return true;
}

void nView::swipeTriggered(QSwipeGesture *gesture)
{
    if (gesture->state() == Qt::GestureFinished) {
        //		qDebug() << "angle" << gesture->swipeAngle() << gesture->horizontalDirection() + gesture->verticalDirection();
        switch (gesture->horizontalDirection() + gesture->verticalDirection()) {
        case QSwipeGesture::Left:
            parent()->previousColorTable();
            break;
        case QSwipeGesture::Right:
            parent()->nextColorTable();
            break;
        case QSwipeGesture::Up:
            parent()->actionPrevBuffer();
            break;
        case QSwipeGesture::Down:
            parent()->actionNextBuffer();
            break;
        }
        update();
    }
}

void nView::focusInEvent (QFocusEvent *) {
    //	((neutrino *) parent())->emitBufferChanged();
}

void nView::zoomEq() {
    fillimage=!fillimage;
    if (!fillimage) resetMatrix();
    setSize();
}

void nView::zoomIn() {
    incrzoom(1.0+currentStepScaleFactor/100.);
}

void nView::zoomOut() {
    incrzoom(1.0-currentStepScaleFactor/100.);
}

void nView::incrzoom(double incr)
{
    scale(incr,incr);
    fillimage=false;
    setSize();
}

void
nView::setSize() {
    QSize my_size=parent()->my_pixitem.pixmap().size();
    QRectF bBox;
    if (parent()->currentBuffer) {
        scaledFont=font();
        if (fillimage) {
            double ratioFont=std::min(((double)width())/my_size.width(),((double)height())/my_size.height());
            scaledFont.setPointSizeF(std::max(6.0,font().pointSizeF()/ratioFont));
        }
        double fSize=QFontMetrics(scaledFont).size(Qt::TextSingleLine,"M").height();
        bBox=QRectF(-2.1*fSize,-2.2*fSize,my_size.width()+4.2*fSize, my_size.height()+5.1*fSize);
    } else {
        bBox=QRectF(0,0,my_size.width(),my_size.height());
    }
    setSceneRect(bBox);
    if (fillimage) {
        fitInView(bBox, Qt::KeepAspectRatio);
    }
    //	qDebug() << "nView::setSize" << bBox << font() << scaledFont << transform().m11();
    parent()->my_mouse.setSize(my_size);
    repaint();
    emit zoomChanged(transform().m11());
}

void
nView::resizeEvent(QResizeEvent *e) {
    QGraphicsView::resizeEvent(e);
    //qDebug() << "nView::resizeEvent" << e->size() << parent()->my_pixitem.pixmap().size() << transform().m11();
    setSize();
}

void nView::keyPressEvent (QKeyEvent *e)
{
    QGraphicsView::keyPressEvent(e);
    bool insideItem = false;
    foreach (QGraphicsItem *item, scene()->selectedItems()){
        insideItem = true;
        //		item->keyPressEvent(e);
        QGraphicsObject *itemObj=item->toGraphicsObject();
        switch (e->key()) {
        case Qt::Key_Backspace: {
            if (itemObj && itemObj->property("parentPanControlLevel").toInt()==0){
                parent()->statusBar()->showMessage(tr("Removed ")+item->toolTip(),2000);
                itemObj->deleteLater();
            } else {
                parent()->statusBar()->showMessage(tr("Can't remove ")+item->toolTip(),2000);
            }
            break;
        }
        }
    }
    if (!insideItem) {
        QPointF delta(0,0);
        switch (e->key()) {
        case Qt::Key_Up:
            delta=QPointF(0,-1);
            break;
        case Qt::Key_Down:
            delta=QPointF(0,+1);
            break;
        case Qt::Key_Left:
            delta=QPointF(-1,0);
            break;
        case Qt::Key_Right:
            delta=QPointF(+1,0);
            break;
        default:
            break;
        }
        if (delta!=QPointF(0,0)) {
            if (e->modifiers() & Qt::ShiftModifier) delta*=5;
            QPointF pos_mouse=parent()->my_mouse.pos()+delta;
            parent()->my_mouse.setPos(pos_mouse);
            emitMouseposition(pos_mouse);
        }
        //		QPoint posCursor=mapFromScene(pos_mouse)+mapToGlobal(QPoint(0,0));
        //		QCursor::setPos(posCursor);
        //		qDebug() << delta << pos_mouse << posCursor << mapToGlobal(QPoint(0,0));
    } else {
        parent()->keyPressEvent(e);
    }
    update();
    if (parent()->follower) parent()->follower->my_w->my_view->keyPressEvent(e);
}


void nView::keyReleaseEvent (QKeyEvent *e) {
    QGraphicsView::keyReleaseEvent(e);
}

void nView::wheelEvent(QWheelEvent *e) {
    switch (e->modifiers()) {
    case Qt::ControlModifier:
        if (e->orientation()==Qt::Vertical) {
            if (e->delta()>0) {
                incrzoom(1.05);
            } else {
                incrzoom(1.0/1.05);
            }
        }
        break;
    default:
        QGraphicsView::wheelEvent(e);
        break;
    }
    if (parent()->follower) {
        QPoint posFollow= parent()->follower->my_w->my_view->mapFromScene(mapToScene(e->pos()));
        QWheelEvent eFollow(posFollow,e->delta(),e->buttons(),e->modifiers(),e->orientation());
        parent()->follower->my_w->my_view->wheelEvent(&eFollow);
    }
}

void nView::mouseDoubleClickEvent (QMouseEvent *e) {
    QGraphicsView::mousePressEvent(e);
    if (parent()->follower) {
        QPoint posFollow= parent()->follower->my_w->my_view->mapFromScene(mapToScene(e->pos()));
        QMouseEvent eFollow(e->type(),posFollow,e->globalPos(),e->button(),e->buttons(),e->modifiers());
        parent()->follower->my_w->my_view->mousePressEvent(&eFollow);
    }
    emit mouseDoubleClickEvent_sig(mapToScene(e->pos()));
}

void nView::mousePressEvent (QMouseEvent *e)
{
    QGraphicsView::mousePressEvent(e);
    if (parent()->follower) {
        QPoint posFollow= parent()->follower->my_w->my_view->mapFromScene(mapToScene(e->pos()));
        QMouseEvent eFollow(e->type(),posFollow,e->globalPos(),e->button(),e->buttons(),e->modifiers());
        parent()->follower->my_w->my_view->mousePressEvent(&eFollow);
    }
    if (e->modifiers()&Qt::ControlModifier && parent()->currentBuffer) {
        minMax=parent()->currentBuffer->get_min_max().swap();
    }
    emit mousePressEvent_sig(mapToScene(e->pos()));
}

void nView::mouseReleaseEvent (QMouseEvent *e)
{
    QGraphicsView::mouseReleaseEvent(e);
    emit mouseReleaseEvent_sig(mapToScene(e->pos()));
    if (parent()->follower) {
        QPoint posFollow= parent()->follower->my_w->my_view->mapFromScene(mapToScene(e->pos()));
        QMouseEvent eFollow(e->type(),posFollow,e->globalPos(),e->button(),e->buttons(),e->modifiers());
        parent()->follower->my_w->my_view->mouseReleaseEvent(&eFollow);
    }
    if (e->modifiers()==Qt::ControlModifier && minMax.x()!=minMax.y()) {
        parent()->changeColorMinMax(minMax);
    }
}

void nView::mouseMoveEvent (QMouseEvent *e)
{
    QGraphicsView::mouseMoveEvent(e);
    if (QGraphicsItem *item = itemAt(e->pos())) {
        if (item->flags()&&QGraphicsItem::ItemIsFocusable) {
            parent()->statusBar()->showMessage(item->toolTip(),2000);
        }
    }
    if (parent()->follower) {
        QPoint posFollow= parent()->follower->my_w->my_view->mapFromScene(mapToScene(e->pos()));
        QMouseEvent eFollow(e->type(),posFollow,e->globalPos(),e->button(),e->buttons(),e->modifiers());
        parent()->follower->my_w->my_view->mouseMoveEvent(&eFollow);
    }

    QPointF pos_mouse=mapToScene(e->pos());
    parent()->my_mouse.setPos(pos_mouse);
    if (e->modifiers()==Qt::ControlModifier && parent()->currentBuffer) {
        double val=parent()->currentBuffer->point(mapToScene(e->pos()).x(),mapToScene(e->pos()).y());
        minMax=vec2f(std::min(minMax.x(),val),std::max(minMax.y(),val));
    }
    emitMouseposition(pos_mouse);
}

void nView::emitMouseposition (QPointF p) {
    emit mouseposition(p);
}


