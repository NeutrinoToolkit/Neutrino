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
    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    my_set.setValue("mouseShape", my_mouse.my_shape);
    my_set.setValue("mouseColor", my_mouse.pen.color());
    my_set.setValue("rulerVisible", my_tics.rulerVisible);
    my_set.setValue("gridVisible", my_tics.gridVisible);
    my_set.setValue("rulerColor", my_tics.rulerColor);
    my_set.endGroup();
}

nView::nView (QWidget *parent) : QGraphicsView (parent),
    my_scene(this),
    my_tics(this),
    currentBuffer(nullptr)
{
    DEBUG("HERE I AM");

    build_colormap();

    setScene(&my_scene);

    my_pixitem.setPixmap(QPixmap(":icons/icon.png"));
    //	my_pixitem.setFlag(QGraphicsItem::ItemIsMovable);

    my_pixitem.setEnabled(true);
    my_pixitem.setZValue(-1);

    setSize();

    my_scene.views().at(0)->viewport()->setCursor(QCursor(Qt::CrossCursor));
    setCursor(QCursor(Qt::CrossCursor));



    if (!parent) ERROREXIT("nView problem");

    DEBUG(qobject_cast<neutrino *>(parent));

    my_scene.addItem(&my_pixitem);
    my_scene.addItem(&my_mouse);
    my_scene.addItem(&my_tics);

    trasformazione.reset();
    fillimage=true;
    setMouseTracking(true);
    setInteractive(true);
    grabGesture(Qt::SwipeGesture);
    QSettings settings("neutrino","");
    settings.beginGroup("nPreferences");
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
    
    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    setMouseShape(my_set.value("mouseShape",my_mouse.my_shape).toInt());
    my_mouse.pen.setColor(my_set.value("mouseColor",my_mouse.pen.color()).value<QColor>());
    my_tics.rulerVisible=my_set.value("rulerVisible",my_tics.rulerVisible).toBool();
    my_tics.gridVisible=my_set.value("gridVisible",my_tics.gridVisible).toBool();
    my_tics.rulerColor=my_set.value("rulerColor",my_tics.rulerColor).value<QColor>();
    my_set.endGroup();
}

void nView::setLockColors(bool val) {
    lockColors=val;
}

void nView::showPhys(nPhysD *my_phys) {
    if (my_phys) {
        if (!physList.contains(my_phys)) physList << my_phys;

        if (currentBuffer) {
            if (lockColors) {
                my_phys->property["display_range"]=currentBuffer->property["display_range"];
                my_phys->property["gamma"]=currentBuffer->property["gamma"];
            } else {
                if (!my_phys->property.have("gamma")) {
                    my_phys->property["gamma"]=property("neuSave-gamma").toInt();
                }
            }
        }

        if (!my_phys->property.have("display_range")) {
            my_phys->property["display_range"]= my_phys->get_min_max();
        }
        currentBuffer=my_phys;

        if (!physList.contains(my_phys)) {
            // TODO: add memory copy...
            physList << my_phys;
        }

        createQimage();

        emit bufferChanged(my_phys);
    }
}


void
nView::createQimage() {
    QApplication::processEvents();
    if (currentBuffer && currentBuffer->getSurf()>0) {
        const unsigned char *nPhys_pointer=currentBuffer->to_uchar_palette(nPalettes[colorTable], colorTable.toStdString());
        const QImage tempImage(nPhys_pointer,
                               currentBuffer->getW(),
                               currentBuffer->getH(),
                               currentBuffer->getW()*3,
                               QImage::Format_RGB888);

        my_pixitem.setPixmap(QPixmap::fromImage(tempImage));
    }

    QApplication::processEvents();

    setSize();
}

void
nView::previousColorTable () {
    int idx=nPalettes.keys().indexOf(colorTable);
    if (idx>0) {
        colorTable=nPalettes.keys().at(idx-1);
    } else {
        colorTable=nPalettes.keys().last();
    }
    changeColorTable ();
};

void
nView::nextColorTable () {
    int indice=nPalettes.keys().indexOf(colorTable);
    if (indice<nPalettes.keys().size()-1) {
        colorTable=nPalettes.keys().at(indice+1);
    } else {
        colorTable=nPalettes.keys().first();
    }
    changeColorTable ();
};


void nView::setZoomFactor(int val) {
    currentStepScaleFactor=val;
}

bool nView::event(QEvent *event)
{
    qDebug() << event;
    if (event->type() == QEvent::Gesture)
        return gestureEvent(static_cast<QGestureEvent*>(event));
    return QGraphicsView::event(event);
}

bool nView::gestureEvent(QGestureEvent *event)
{
    qDebug() << event;
    if (QGesture *swipe = event->gesture(Qt::SwipeGesture)) {
        DEBUG("swipe");
        swipeTriggered(static_cast<QSwipeGesture *>(swipe));
    }
    return true;
}

void nView::swipeTriggered(QSwipeGesture *gesture)
{
    if (gesture->state() == Qt::GestureFinished) {
        //		qDebug() << "angle" << gesture->swipeAngle() << gesture->horizontalDirection() + gesture->verticalDirection();
        switch (gesture->horizontalDirection() + gesture->verticalDirection()) {
        case QSwipeGesture::Left:
            previousColorTable();
            break;
        case QSwipeGesture::Right:
            nextColorTable();
            break;
        case QSwipeGesture::Up:
            prevBuffer();
            break;
        case QSwipeGesture::Down:
            nextBuffer();
            break;
        }
        update();
    }
}

void nView::focusInEvent (QFocusEvent *) {
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
    QRectF bBox=my_tics.boundingRect();
    setSceneRect(bBox);
    if (fillimage) {
        fitInView(bBox, Qt::KeepAspectRatio);
    }
    my_mouse.setSize(my_pixitem.pixmap().size());
    my_mouse.pen.setWidthF(1.0/transform().m11());
    repaint();
    emit zoomChanged(transform().m11());
}

void
nView::resizeEvent(QResizeEvent *e) {
    QGraphicsView::resizeEvent(e);
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
                emit logging(tr("Removed ")+item->toolTip());
                itemObj->deleteLater();
            } else {
                emit logging(tr("Can't remove ")+item->toolTip());
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
        case Qt::Key_Return:
            emit mousePressEvent_sig(my_mouse.pos());
            break;
        default:
            break;
        }
        if (delta!=QPointF(0,0)) {
            if (e->modifiers() & Qt::ShiftModifier) delta*=5;
            QPointF pos_mouse=my_mouse.pos()+delta;
            my_mouse.setPos(pos_mouse);
            emitMouseposition(pos_mouse);
        }
    }

    // cycle over items
    switch (e->key()) {
    case Qt::Key_Tab: {
        QList<QGraphicsItem *> lista;
        foreach (QGraphicsItem *oggetto, scene()->items() ) {
            if (oggetto->type() > QGraphicsItem::UserType) {
                if (oggetto->isVisible()) lista << oggetto;
            }
        }
        scene()->clearSelection();
        if (lista.size()>0) {
            int found=0;
            for (int i=lista.size()-1; i >=0; i-- ) {
                if (lista.at(i)->hasFocus()) found=(i+lista.size()-1)%lista.size();
            }

            lista.at(found)->setFocus(Qt::TabFocusReason);
        }
        break;
    }
    case Qt::Key_Plus:
        zoomIn();
        break;
    case Qt::Key_Minus:
        zoomOut();
        break;
    case Qt::Key_Equal:
        zoomEq();
        break;
    case Qt::Key_M: {
        if (!(e->modifiers() & Qt::ShiftModifier)) {
            setMouseShape(my_mouse.my_shape+1);
        }
        break;
    }
    case Qt::Key_C:
        if ((e->modifiers() & Qt::ControlModifier))
            QApplication::clipboard()->setPixmap(QPixmap::grabWidget(this), QClipboard::Clipboard)   ;
        break;
    case Qt::Key_A: {
        if (e->modifiers() & Qt::ShiftModifier) {
            foreach (nPhysD* phys, physList) {
                currentBuffer->property["display_range"]=currentBuffer->get_min_max();
                setGamma(1);
                emit bufferChanged(phys);
            }
        } else {
            if (currentBuffer) {
                currentBuffer->property["display_range"]=currentBuffer->get_min_max();
                setGamma(1);
                emit updatecolorbar();
            }
        }
        createQimage();
        break;
    }
    case Qt::Key_Less:
        if (currentBuffer) {
            setGamma(int(currentBuffer->property["gamma"])-1);
        }
        break;
    case Qt::Key_Greater:
        if (currentBuffer) {
            setGamma(int(currentBuffer->property["gamma"])+1);
        }
        break;
    case Qt::Key_Period:
        if (currentBuffer) {
            setGamma(1);
        }
        break;
    }

    update();
    emit keypressed(e);
}

void nView::setGamma(int value) {
    if (currentBuffer) {
        currentBuffer->property["gamma"]=value;
        createQimage();
        emit bufferChanged(currentBuffer);
    }
}

void
nView::changeColorTable (QString ctname) {
    if (nPalettes.contains(ctname)) {
        colorTable=ctname;
    } else {
        colorTable=nPalettes.keys().first();
    }
    changeColorTable();
}


void
nView::changeColorTable () {
    createQimage();
    emit logging(colorTable);
    my_tics.update();
    emit updatecolorbar();
}

void nView::setMouseShape(int num) {
    num%=my_mouse.num_shape;
    my_mouse.hide();
    update();
    setSize();
    QCursor cur;
    if (num==0) {
        my_pixitem.setCursor(Qt::CrossCursor);
    } else {
        my_pixitem.setCursor(Qt::BlankCursor);
        my_mouse.show();
    }
    my_mouse.my_shape=num;
    my_mouse.update();
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
}

void nView::mouseDoubleClickEvent (QMouseEvent *e) {
    QGraphicsView::mousePressEvent(e);
    emit mouseDoubleClickEvent_sig(mapToScene(e->pos()));
}

void nView::mousePressEvent (QMouseEvent *e)
{
    QGraphicsView::mousePressEvent(e);
    if (e->modifiers()&Qt::ControlModifier && currentBuffer) {
        minMax=currentBuffer->get_min_max().swap();
    }
    emit mousePressEvent_sig(mapToScene(e->pos()));
}

void nView::mouseReleaseEvent (QMouseEvent *e)
{
    QGraphicsView::mouseReleaseEvent(e);
    emit mouseReleaseEvent_sig(mapToScene(e->pos()));
    if (e->modifiers()==Qt::ControlModifier && minMax.x()!=minMax.y()) {
        currentBuffer->property["display_range"]=minMax;
        createQimage();
    }
}

void nView::mouseMoveEvent (QMouseEvent *e)
{
    QGraphicsView::mouseMoveEvent(e);
    if (QGraphicsItem *item = itemAt(e->pos())) {
        if (item->flags() && QGraphicsItem::ItemIsFocusable) {
            emit logging (item->toolTip());
        }
    }

    QPointF pos_mouse=mapToScene(e->pos());
    my_mouse.setPos(pos_mouse);
    if (e->modifiers()==Qt::ControlModifier && currentBuffer) {
        double val=currentBuffer->point(mapToScene(e->pos()).x(),mapToScene(e->pos()).y());
        minMax=vec2f(std::min(minMax.x(),val),std::max(minMax.y(),val));
    }
    emitMouseposition(pos_mouse);
}

void nView::emitMouseposition (QPointF p) {
    emit mouseposition(p);
}

// switch buffers
void nView::prevBuffer() {
    int position=physList.indexOf(currentBuffer);
    if (position>-1) showPhys(physList.at((position+physList.size()-1)%physList.size()));
}

void nView::nextBuffer() {
    int position=physList.indexOf(currentBuffer);
    if (position>-1) showPhys(physList.at((position+1)%physList.size()));
}

