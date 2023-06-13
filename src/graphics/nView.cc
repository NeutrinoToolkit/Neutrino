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
#include "nApp.h"
#include "nView.h"
#include <QShortcut>
#include <QFileDialog>

nView::~nView ()
{
    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    my_set.setValue("pixmapFile", property("pixmapFile").toString());
    my_set.setValue("colorTable", colorTable);
    my_set.endGroup();
}

nView::nView (QWidget *parent) : QGraphicsView (parent),
    nparent(qobject_cast<neutrino *>(parent->parent())),
    nPalettes ((qobject_cast<nApp*> (qApp))->nPalettes),
    my_scene(this),
    my_tics(this),
    colorTable(":cmaps/Neutrino"),
    lockColors(false)
{

    setScene(&my_scene);

    my_pixitem.setPixmap(QPixmap(":icons/icon.png"));
    //	my_pixitem.setFlag(QGraphicsItem::ItemIsMovable);

    my_pixitem.setEnabled(true);
    my_pixitem.setZValue(-1);

    setSize();

    my_scene.views().at(0)->viewport()->setCursor(QCursor(Qt::CrossCursor));
    setCursor(QCursor(Qt::CrossCursor));

    if (!nparent) ERROREXIT("nView problem")

    my_scene.addItem(&my_pixitem);
    my_scene.addItem(&my_mouse);
    my_scene.addItem(&my_tics);

    trasformazione.reset();
    fillimage=true;
    setMouseTracking(true);
    setInteractive(true);
    QTapAndHoldGesture::setTimeout(5000);
    grabGesture(Qt::TapAndHoldGesture);
    grabGesture(Qt::SwipeGesture);
    grabGesture(Qt::PanGesture);
    grabGesture(Qt::PinchGesture);


    setTransformationAnchor(QGraphicsView::AnchorViewCenter);


    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    QVariant fontString=my_set.value("defaultFont");
    if (fontString.isValid()) {
        QFont fontTmp;
        if (fontTmp.fromString(fontString.toString())) {
            setFont(fontTmp);
        }
    }

    currentStepScaleFactor=my_set.value("currentStepScaleFactor",15).toInt();

    setProperty("pixmapFile",my_set.value("pixmapFile","Pixmap.png"));
    changeColorTable(my_set.value("colorTable",colorTable).toString());
    my_set.endGroup();

    setMouseShape(my_mouse.my_shape);

}

void nView::setLockColors(bool val) {
    lockColors=val;
}

void nView::exportPixmap() {
    QList<QByteArray> my_list=QImageWriter::supportedImageFormats();
    QByteArray my_suffix=QFileInfo(property("pixmapFile").toString()).suffix().toUtf8();
    if (my_list.contains(my_suffix)) {
        my_list.removeAll(my_suffix);
        my_list.prepend(my_suffix);
    }
    my_suffix.clear();

    QString allformats;
    foreach (QByteArray format, my_list ) {
        allformats += QString(format).toUpper()+" image (*."+format+");; ";
    }
    allformats.chop(1);
    DEBUG(">"<< allformats.toStdString() << "<");
    QString suffix=QFileInfo(property("pixmapFile").toString()).suffix();
    QString fname = QFileDialog::getSaveFileName(this,tr("Save Pixmap"),property("pixmapFile").toString(),allformats,&suffix );
    if (!fname.isEmpty()) {
        setProperty("pixmapFile",fname);
        my_pixitem.pixmap().save(fname);
    }

}

void nView::showPhys(nPhysD *my_phys) {
    DEBUG("ENTER");
    if (my_phys && nparent->physList.contains(my_phys)) {
        DEBUG(lockColors << "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>"  << my_phys->copies());

        if (nparent->physList.contains(nparent->currentBuffer)) {
            if (lockColors) {
                my_phys->prop["display_range"]=nparent->currentBuffer->prop["display_range"];
                my_phys->prop["gamma"]=nparent->currentBuffer->prop["gamma"];
            }
        }


        QApplication::processEvents();
        if (my_phys->getSurf()>0) {
            const QImage tempImage(my_phys->to_uchar_palette(nPalettes[colorTable], colorTable.toStdString()),
                                   my_phys->getW(),
                                   my_phys->getH(),
                                   my_phys->getW()*3,
                                   QImage::Format_RGB888);

            my_pixitem.setPixmap(QPixmap::fromImage(tempImage));
            nparent->currentBuffer=my_phys;

            QApplication::processEvents();

            setSize();

            emit bufferChanged(my_phys);
            QApplication::processEvents();
        }
    }
    DEBUG("EXIT");
}

void
nView::previousColorTable () {
    int idx=nPalettes.keys().indexOf(colorTable);
    if (idx>0) {
        colorTable=nPalettes.keys().at(idx-1);
    } else {
        colorTable=nPalettes.keys().last();
    }
    changeColorTable();
};

void
nView::nextColorTable () {
    int indice=nPalettes.keys().indexOf(colorTable);
    if (indice<nPalettes.keys().size()-1) {
        colorTable=nPalettes.keys().at(indice+1);
    } else {
        colorTable=nPalettes.keys().first();
    }
    changeColorTable();
};


void nView::setZoomFactor(int val) {
    currentStepScaleFactor=val;
}

bool nView::event(QEvent *event)
{
    if (event->type() == QEvent::Gesture) {
        return gestureEvent(static_cast<QGestureEvent*>(event));
    }
    return QGraphicsView::event(event);
}

bool nView::gestureEvent(QGestureEvent *event)
{
    qDebug() << "-------------------------------------------------------------------------";
    foreach (QGesture *gesture, event->gestures()) {
        qDebug() << "type: " << gesture->gestureType();
    }
    QGesture *gesture=nullptr;
    if ((gesture = event->gesture(Qt::TapAndHoldGesture))) {
        tapandholdTriggered(static_cast<QTapAndHoldGesture *>(gesture));
    }
    if ((gesture = event->gesture(Qt::SwipeGesture))) {
        swipeTriggered(static_cast<QSwipeGesture *>(gesture));
    }
    if ((gesture = event->gesture(Qt::PinchGesture))) {
        pinchTriggered(static_cast<QPinchGesture *>(gesture));
    }
    if ((gesture = event->gesture(Qt::PanGesture))) {
        qDebug() << static_cast<QPanGesture *>(gesture);
    }
    return true;
}

void nView::tapandholdTriggered(QTapAndHoldGesture *gesture) {
    DEBUG("-------------");
    qDebug() << gesture;
//    fillimage=true;
//    setSize();
//    update();
}

void nView::pinchTriggered(QPinchGesture *gesture)
{
    incrzoom(gesture->lastScaleFactor());
}

void nView::swipeTriggered(QSwipeGesture *gesture)
{
    if (gesture->state() == Qt::GestureFinished) {
        //		qDebug() << "angle" << gesture->swipeAngle() << gesture->horizontalDirection() +	 gesture->verticalDirection();
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

void nView::update () {
    my_mouse.update();
    my_tics.update();
    QGraphicsView::update();
}

void nView::focusInEvent (QFocusEvent *) {
}

void nView::zoomEq() {
    fillimage=!fillimage;
//    if (!fillimage) resetMatrix();
    if (!fillimage) resetTransform();
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
    if (my_mouse.size != my_pixitem.pixmap().size()) {
        my_mouse.setSize(my_pixitem.pixmap().size());
    }
    emit zoomChanged(transform().m11());
    repaint();
}

void
nView::resizeEvent(QResizeEvent *e) {
    QGraphicsView::resizeEvent(e);
    setSize();
}

void nView::keyPressEvent (QKeyEvent *e)
{
    qDebug() << e;
    QGraphicsView::keyPressEvent(e);
    if (scene()->selectedItems().size()) {
        if (e->key() == Qt::Key_Backspace) {
            foreach (QGraphicsItem *item, scene()->selectedItems()){
                QGraphicsObject *itemObj=item->toGraphicsObject();
                if (itemObj && itemObj->property("parentPanControlLevel").toInt()==0){
                    qInfo() << tr("Removed ") << item->toolTip();
                    itemObj->deleteLater();
                    break;
                } else {
                    qInfo() << tr("Can't remove ") << item->toolTip();
                }
            }
        }
    } else {
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

    update();
    emit keypressed(e);
}

void nView::rescaleColor(double val) {
    val=std::max<double>(std::min<double>(val,100),0);
    if (val==100) resetGamma();
    setProperty("percentPixels",val);
    if (nparent->currentBuffer) {
        if (QGuiApplication::keyboardModifiers() & Qt::AltModifier) {
            foreach (nPhysD* phys, nparent->physList) {
                phys->prop["display_range"]=physMath::getColorPrecentPixels(*phys,val);
                emit bufferChanged(phys);
            }
            qInfo() << "Colorscale of all images rescaled to show " << val << "% of the pixels";
        } else {
            nparent->currentBuffer->prop["display_range"]=physMath::getColorPrecentPixels(*nparent->currentBuffer,val);
            emit bufferChanged(nparent->currentBuffer);
            qInfo() << "Images colorscale rescaled to show " << val << "% of the pixels";
        }
        nparent->showPhys();
    }
}

void nView::rescale99() {
    rescaleColor(99.9);
}

void nView::rescaleLess() {
    rescaleColor(property("percentPixels").isValid() ? property("percentPixels").toDouble()-0.1 : 99.9);
}

void nView::rescaleMore() {
    rescaleColor(property("percentPixels").isValid() ? property("percentPixels").toDouble()+0.1 : 100);
}

void nView::cycleOverItems() {
    QList<QGraphicsItem *> lista;
    foreach (QGraphicsItem *oggetto, scene()->items() ) {
        if (oggetto->type() > QGraphicsItem::UserType) {
            if (oggetto->isVisible()) lista << oggetto;
        } else {
            qDebug() << oggetto;
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
}

void nView::copyImage() {
    QApplication::clipboard()->setPixmap(grab(), QClipboard::Clipboard);
    qInfo() << "Image copied in clipboard";
}

void nView::toggleRuler() {
    my_tics.rulerVisible=!my_tics.rulerVisible;
    my_tics.update();
    qInfo() << "Ruler" << (my_tics.rulerVisible? " " : " not") << "visible";
}

void nView::toggleGrid() {
    my_tics.gridVisible=!my_tics.gridVisible;
    my_tics.update();
    qInfo() << "Grid" << (my_tics.gridVisible? " " : " not") << "visible";
}


void nView::incrGamma() {
    if (nparent->currentBuffer) setGamma(int(nparent->currentBuffer->prop["gamma"])+1);
}

void nView::decrGamma() {
    if (nparent->currentBuffer) setGamma(int(nparent->currentBuffer->prop["gamma"])-1);
}

void nView::resetGamma() {
    setGamma(1);
}


void nView::setGamma(int value) {
    if (nparent->currentBuffer) {
        nparent->currentBuffer->prop["gamma"]=value;
        nparent->showPhys();
        emit bufferChanged(nparent->currentBuffer);
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
    nparent->showPhys();
    qInfo() << "Colortable:" << colorTable;
    my_tics.update();
    emit updatecolorbar(colorTable);
}

void nView::setMouseOrigin() {
    if (QGuiApplication::keyboardModifiers() & Qt::AltModifier) {
        foreach (nPhysD* phys, nparent->physList) {
            phys->set_origin(my_mouse.pos().x(),my_mouse.pos().y());
            emit bufferChanged(phys);
            qInfo() << "Origin set for all images";
        }
    } else {
        if (nparent->currentBuffer) {
            nparent->currentBuffer->set_origin(my_mouse.pos().x(),my_mouse.pos().y());
            emit bufferChanged(nparent->currentBuffer);
            qInfo() << "Origin set";
        }
    }
    my_tics.update();

    // I need a signal to communicate explicit origin change not to
    // be taken for a buffer change. Used in nWinList.
    emit bufferOriginChanged();
    emitMouseposition(my_mouse.pos());

}

void nView::nextMouseShape() {
    setMouseShape(my_mouse.my_shape+1);
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
    qInfo() << "Mouse " << QLocale().toString(num);
}

void nView::keyReleaseEvent (QKeyEvent *e) {
    QGraphicsView::keyReleaseEvent(e);
}

void nView::wheelEvent(QWheelEvent *event) {
    switch (event->modifiers()) {
        case Qt::ControlModifier:
            if (event->angleDelta().y() != 0) {
                if (event->angleDelta().y()>0) {
                    incrzoom(1.05);
                } else {
                    incrzoom(1.0/1.05);
                }
                return;
            }
            break;
        default:
            //            {
            //                int mindelta=2;
            //                QPoint numPixels = event->pixelDelta();

            //                if (!numPixels.isNull()) {
            //                    if (abs(numPixels.x()) > abs(numPixels.y())) {
            //                        if (numPixels.x()>mindelta) {
            //                            nextColorTable();
            //                        } else if(numPixels.x()<-mindelta) {
            //                            previousColorTable();
            //                        }
            //                    } else if (abs(numPixels.x()) < abs(numPixels.y())) {
            //                        if (numPixels.y()>mindelta) {
            //                            nextBuffer();
            //                        } else if(numPixels.y()<-mindelta) {
            //                            prevBuffer();
            //                        }
            //                    }
            //                    qDebug() << event;
            //                }

            //                event->accept();
            //            }
            break;
    }
    QGraphicsView::wheelEvent(event);
}

void nView::mouseDoubleClickEvent (QMouseEvent *e) {
    QGraphicsView::mouseDoubleClickEvent(e);
    emit mouseDoubleClickEvent_sig(mapToScene(e->pos()));
}

void nView::mousePressEvent (QMouseEvent *e)
{
//    qDebug() << "<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>";
    QGraphicsView::mousePressEvent(e);
    if (e->modifiers()&Qt::ControlModifier && nparent->currentBuffer) {
        minMax=nparent->currentBuffer->get_min_max().swap();
    }
    emit mousePressEvent_sig(mapToScene(e->pos()));
}

void nView::mouseReleaseEvent (QMouseEvent *e)
{
//    qDebug() << "end " << e;
    QGraphicsView::mouseReleaseEvent(e);
    emit mouseReleaseEvent_sig(mapToScene(e->pos()));
    if (e->modifiers()==Qt::ControlModifier && minMax.x()!=minMax.y()) {
        nparent->currentBuffer->prop["display_range"]=minMax;
        setProperty("percentPixels",QVariant());
        nparent->showPhys();
    }
}

void nView::mouseMoveEvent (QMouseEvent *e)
{
    QGraphicsView::mouseMoveEvent(e);
    if (QGraphicsItem *item = itemAt(e->pos())) {
        if (item->flags() & QGraphicsItem::ItemIsFocusable) {
            qDebug() << item->toolTip();
        }
    }

    QPointF pos_mouse=mapToScene(e->pos());
    my_mouse.setPos(pos_mouse);
    if (e->modifiers()==Qt::ControlModifier && nparent->currentBuffer) {
        double val=nparent->currentBuffer->point(mapToScene(e->pos()).x(),mapToScene(e->pos()).y());
        minMax=vec2f(std::min(minMax.x(),val),std::max(minMax.y(),val));
    }
    emitMouseposition(pos_mouse);
}

void nView::emitMouseposition (QPointF p) {
    emit mouseposition(p);
}

// switch buffers
void nView::prevBuffer() {
    int position=nparent->physList.indexOf(nparent->currentBuffer);
    if (position>-1) showPhys(nparent->physList.at((position+nparent->physList.size()-1)%nparent->physList.size()));
}

void nView::nextBuffer() {
    int position=nparent->physList.indexOf(nparent->currentBuffer);
    if (position>-1) showPhys(nparent->physList.at((position+1)%nparent->physList.size()));
}

