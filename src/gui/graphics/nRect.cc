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
#include "nRect.h"
#include "ui_nObject.h"
#include "neutrino.h"
#include "ui_neutrino.h"
#include "nGenericPan.h"

#include <iostream>

nRect::~nRect() {
    foreach (QGraphicsRectItem* item, ref) {
        delete item;
    }
}


nRect::nRect(nGenericPan *parentPan, int level) : nRect(parentPan->nparent)
{
    my_w->name->setText(parentPan->panName()+"Rect");
    setProperty("parentPan", QVariant::fromValue(parentPan));
    setProperty("parentPanControlLevel",level);
    if (level>0) {
        my_w->name->setReadOnly(true);
        disconnect(my_w->name, SIGNAL(textChanged(QString)), this, SLOT(changeToolTip(QString)));
    }

}


nRect::nRect(neutrino *my_parent) :
    QGraphicsObject(),
    nparent(my_parent),
    my_w(new Ui::nObject)
{
    if (my_parent) {
        my_parent->getScene().addItem(this);
        setParent(my_parent);
        int num=my_parent->property("numRect").toInt()+1;
        my_parent->setProperty("numRect",num);
        setProperty("numRect",num);
        setToolTip(tr("rect")+QString(" ")+QString::number(num));
        connect(my_parent, SIGNAL(mouseAtMatrix(QPointF)), this, SLOT(movePoints(QPointF)));
        connect(my_parent->my_w->my_view, SIGNAL(zoomChanged(double)), this, SLOT(zoomChanged(double)));
        connect(my_parent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(bufferChanged(nPhysD*)));
        zoom=my_parent->getZoom();
        if (my_parent->getCurrentBuffer()) {
            setPos(my_parent->getCurrentBuffer()->get_origin().x(),my_parent->getCurrentBuffer()->get_origin().y());
        }
    }
    
    setAcceptHoverEvents(true);
    setFlag(QGraphicsItem::ItemIsSelectable);
    setFlag(QGraphicsItem::ItemIsFocusable);
    setProperty("parentPan",QString(""));
    setProperty("parentPanControlLevel",0);

    setProperty("NeuSave-fileIni",toolTip()+".ini");

    nWidth=1.0;
    nSizeHolder=5.0;

    nColor=QColor(Qt::black);
    holderColor=QColor(0,0,255,200);

    setOrder(0.0);

    // PADELLA
    my_pad.setWindowTitle(toolTip());
    my_pad.setWindowIcon(QIcon(":rect"));
    my_w->setupUi(&my_pad);

    my_w->spinWidth->setValue(nWidth);
    my_w->spinDepth->setValue(zValue());
    my_w->colorLabel->setPalette(QPalette(nColor));

    connect(my_w->name, SIGNAL(textChanged(QString)), this, SLOT(changeToolTip(QString)));

    my_w->name->setText(toolTip());
    my_w->spinSizeHolder->setValue(nSizeHolder);
    my_w->colorHolderLabel->setPalette(QPalette(holderColor));

    connect(my_w->spinWidth, SIGNAL(valueChanged(double)), this, SLOT(setWidthF(double)));
    connect(my_w->spinDepth, SIGNAL(valueChanged(double)), this, SLOT(setOrder(double)));
    connect(my_w->colorButton, SIGNAL(pressed()), this, SLOT(changeColor()));
    connect(my_w->colorHolderButton, SIGNAL(pressed()), this, SLOT(changeColorHolder()));
    connect(my_w->spinSizeHolder, SIGNAL(valueChanged(double)), this, SLOT(sizeHolder(double)));
    connect(my_w->tableWidget, SIGNAL(itemChanged(QTableWidgetItem * )), this, SLOT(tableUpdated(QTableWidgetItem * )));

    connect(my_w->expandX, SIGNAL(pressed()), this, SLOT(expandX()));
    connect(my_w->expandY, SIGNAL(pressed()), this, SLOT(expandY()));
    connect(my_w->intersection, SIGNAL(pressed()), this, SLOT(intersection()));

    connect(my_w->sizeWidth, SIGNAL(editingFinished()), this, SLOT(changeWidth()));
    connect(my_w->sizeHeight, SIGNAL(editingFinished()), this, SLOT(changeHeight()));

    updateSize();
}

void nRect::contextMenuEvent ( QGraphicsSceneContextMenuEvent * e ) {
    QMenu menu;
    QAction *showPan = menu.addAction("Show control (w)");
    connect(showPan, SIGNAL(triggered()), this, SLOT(togglePadella()));
    QAction *expandx = menu.addAction("Expand width (x)");
    connect(expandx, SIGNAL(triggered()), this, SLOT(expandX()));
    QAction *expandy = menu.addAction("Expand height (y)");
    connect(expandy, SIGNAL(triggered()), this, SLOT(expandY()));
    menu.exec(e->screenPos());
}

void nRect::setRect(QRectF rect) {
    while (ref.size()<2) appendPoint();
    moveRef.clear();
    changeP(0,rect.topLeft(),true);
    changeP(1,rect.bottomRight(),true);
    itemChanged();
}

QRect nRect::getRect(nPhysD* image) {
    QRect geom2=QRectF(mapToScene(ref[0]->pos()),mapToScene(ref[1]->pos())).toRect().normalized();
    if (image && nparent->getCurrentBuffer()) {
        vec2f dx(image->get_origin()-nparent->getCurrentBuffer()->get_origin());
        geom2.translate(dx.x(),dx.y());
    }
    return geom2;
}

QRectF nRect::getRectF() {
    if (ref.size()<2) {
        return QRectF(0,0,0,0);
    } else {
        return QRectF(mapToScene(ref[0]->pos()),mapToScene(ref[1]->pos())).normalized();
    }
}

void nRect::bufferChanged(nPhysD* my_phys) {    
    if (my_phys) {
        setPos(my_phys->get_origin().x(),my_phys->get_origin().y());
    } else {
        setPos(0,0);
    }
}

void nRect::interactive ( ) {
    showMessage(tr("Click for the first point of the rectangle"));
    connect(nparent->my_w->my_view, SIGNAL(mouseReleaseEvent_sig(QPointF)), this, SLOT(addPointAfterClick(QPointF)));
    appendPoint();
}

void nRect::addPointAfterClick ( QPointF ) {
    showMessage(tr("Point added, click for the second point"));
    moveRef.clear();
    appendPoint();
    disconnect(nparent->my_w->my_view, SIGNAL(mouseReleaseEvent_sig(QPointF)), this, SLOT(addPointAfterClick(QPointF)));
}

void nRect::mousePressEvent ( QGraphicsSceneMouseEvent * e ) {
    for (int i=0;i<ref.size();i++) {
        if (ref.at(i)->rect().contains(mapToItem(ref.at(i), e->pos()))) {
            moveRef.append(i);
        }
    }
    if (moveRef.size()>0) { // if more that one just pick the last
        int keeplast=moveRef.last();
        moveRef.clear();
        moveRef.append(keeplast);
    } else { // if none is selected, append ref.size() to move the whole object
        moveRef.append(ref.size());
        click_pos= e->pos();
    }

    QGraphicsItem::mousePressEvent(e);
}

void nRect::mouseReleaseEvent ( QGraphicsSceneMouseEvent * e ) {
    moveRef.clear();
    showMessage("");
    QGraphicsItem::mouseReleaseEvent(e);
    itemChanged();
}

void nRect::mouseMoveEvent ( QGraphicsSceneMouseEvent * e ) {
    if (moveRef.contains(ref.size())) {
        QPointF delta=e->pos()-click_pos;
        moveBy(delta);
        click_pos=e->pos();
    }
    QGraphicsItem::mouseMoveEvent(e);
}

void nRect::togglePadella() {
    if (my_pad.isHidden()) {
        my_pad.show();
    } else {
        my_pad.hide();
    }
}

void nRect::mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * e ) {
    togglePadella();
    QGraphicsItem::mouseDoubleClickEvent(e);
}

void nRect::sizeHolder ( double val ) {
    nSizeHolder=val;
    QPointF p=QPointF(val/zoom,val/zoom);
    foreach(QGraphicsRectItem *item, ref){
        item->setRect(QRectF(-p,p));
    }
}

void
nRect::movePoints (QPointF p) {
    for (int i=0;i<ref.size(); i++) {
        if (moveRef.contains(i)) {
            changeP(i,p,true);
            showMessage("P"+QString::number(i)+" "+getRectString());
        }
    }
}

void
nRect::changeToolTip (QString n) {
    setToolTip(n);
    my_pad.setWindowTitle(n);
}

void
nRect::setWidthF (double w) {
    nWidth=w;
    update();
}

void
nRect::setOrder (double w) {
    setZValue(w);
}

void
nRect::tableUpdated (QTableWidgetItem * item) {
    QPointF p;
    p.rx()=QLocale().toDouble(my_w->tableWidget->item(item->row(),0)->text());
    p.ry()=QLocale().toDouble(my_w->tableWidget->item(item->row(),1)->text());

    changeP(item->row(),p, false);
    itemChanged();
}

void
nRect::changeColor () {
    QColorDialog colordial(my_w->colorLabel->palette().color(QPalette::Background));
    colordial.setOption(QColorDialog::ShowAlphaChannel);
    colordial.exec();
    if (colordial.result() && colordial.currentColor().isValid()) {
        changeColor(colordial.currentColor());
    }
    update();
}

void
nRect::changeColor (QColor col) {
    nColor=col;
    my_w->colorLabel->setPalette(QPalette(nColor));
}

void
nRect::changeColorHolder () {
    QColor color;
    QColorDialog colordial(my_w->colorHolderLabel->palette().color(QPalette::Background));
    colordial.setOption(QColorDialog::ShowAlphaChannel);
    colordial.exec();
    if (colordial.result() && colordial.currentColor().isValid()) {
        changeColorHolder(colordial.currentColor());
    }
}

void
nRect::changeColorHolder (QColor color) {
    my_w->colorHolderLabel->setPalette(QPalette(color));
    QBrush brush=ref[0]->brush();
    brush.setColor(color);
    foreach (QGraphicsRectItem *item, ref){
        item->setBrush(brush);
    }
}

void
nRect::changeP (int np, QPointF p, bool updatepad) {
    prepareGeometryChange();
    ref[np]->setPos(mapFromScene(p));
    ref[np]->setVisible(true);
    if (updatepad) changePointPad(np);
    updateSize();
}

void nRect::changePointPad(int nrow) {
    disconnect(my_w->tableWidget, SIGNAL(itemChanged(QTableWidgetItem * )), this, SLOT(tableUpdated(QTableWidgetItem * )));
    QPointF p=ref[nrow]->pos();
    QTableWidgetItem *xitem= new QTableWidgetItem(QString::number(p.x()));
    QTableWidgetItem *yitem= new QTableWidgetItem(QString::number(p.y()));
    xitem->setTextAlignment(Qt::AlignHCenter + Qt::AlignVCenter);
    yitem->setTextAlignment(Qt::AlignHCenter + Qt::AlignVCenter);
    my_w->tableWidget->setItem(nrow, 0, xitem);
    my_w->tableWidget->setItem(nrow, 1, yitem);
    my_w->tableWidget->resizeRowToContents(nrow);
    connect(my_w->tableWidget, SIGNAL(itemChanged(QTableWidgetItem * )), this, SLOT(tableUpdated(QTableWidgetItem * )));
}

void nRect::addPoint (int pos) {

    moveRef.append(pos);
    QPointF position;
    QBrush refBrush;
    QPen refPen;
    if (ref.size()>0) {
        int copyfrom=std::max(pos, 1);
        position=ref[copyfrom-1]->pos();
        refBrush=ref[copyfrom-1]->brush();
        refPen=ref[copyfrom-1]->pen();
    } else {
        position=QPointF(0,0);
        refPen.setStyle(Qt::NoPen);
        refBrush.setStyle(Qt::SolidPattern);
        refBrush.setColor(holderColor);
    }

    ref.insert(pos,new QGraphicsRectItem());
    ref[pos]->setPos(position);
    ref[pos]->setBrush(refBrush);
    ref[pos]->setPen(refPen);
    ref[pos]->setVisible(false);
    ref[pos]->setParentItem(this);
    sizeHolder(nSizeHolder);
    disconnect(my_w->tableWidget, SIGNAL(itemChanged(QTableWidgetItem * )), this, SLOT(tableUpdated(QTableWidgetItem * )));
    my_w->tableWidget->insertRow(pos);
    QTableWidgetItem *xitem= new QTableWidgetItem(QString::number(position.x()));
    QTableWidgetItem *yitem= new QTableWidgetItem(QString::number(position.y()));
    xitem->setTextAlignment(Qt::AlignHCenter + Qt::AlignVCenter);
    yitem->setTextAlignment(Qt::AlignHCenter + Qt::AlignVCenter);
    my_w->tableWidget->setItem(pos, 0, xitem);
    my_w->tableWidget->setItem(pos, 1, yitem);
    my_w->tableWidget->resizeRowToContents(pos);
    connect(my_w->tableWidget, SIGNAL(itemChanged(QTableWidgetItem * )), this, SLOT(tableUpdated(QTableWidgetItem * )));
}

void nRect::appendPoint () {
    addPoint(ref.size());
}

void nRect::expandX() {
    if (nparent->getCurrentBuffer()) {
        QRectF rect=getRectF();
        changeP(0,QPointF(0,rect.top()),true);
        changeP(1,QPointF(nparent->getCurrentBuffer()->getW(),rect.bottom()),true);
        itemChanged();
    }
}

void nRect::expandY() {
    if (nparent->getCurrentBuffer()) {
        QRectF rect=getRectF();
        changeP(0,QPointF(rect.left(),0),true);
        changeP(1,QPointF(rect.right(),nparent->getCurrentBuffer()->getH()),true);
        itemChanged();
    }
}

void nRect::intersection() {
    if (nparent->getCurrentBuffer()) {
        //QRectF rect=QRectF(0,0,nparent->getCurrentBuffer()->getW(),nparent->getCurrentBuffer()->getH()).intersect(getRectF());
        //obsolete
        QRectF rect=QRectF(0,0,nparent->getCurrentBuffer()->getW(),nparent->getCurrentBuffer()->getH()).intersected(getRectF());
        setRect(rect);
    }
}

void nRect::submatrix() {
    if (nparent->getCurrentBuffer()) {
        nPhysD subPhys=nparent->getCurrentBuffer()->sub(getRect().x(),getRect().y(),getRect().width(),getRect().height());
        nparent->showPhys(subPhys);
    }
}

void nRect::changeWidth () {
    if (nparent->getCurrentBuffer()) {
        QRectF rect=getRectF();
        bool ok;
        rect.setWidth(QLocale().toDouble(my_w->sizeWidth->text(),&ok));
        if (ok) {
            changeP(1,rect.bottomRight(),true);
            itemChanged();
        }
    }
}

void nRect::changeHeight () {
    if (nparent->getCurrentBuffer()) {
        QRectF rect=getRectF();
        bool ok;
        rect.setHeight(QLocale().toDouble(my_w->sizeHeight->text(),&ok));
        if (ok) {
            changeP(1,rect.bottomRight(),true);
            itemChanged();
        }
    }
}

void nRect::updateSize() {
    disconnect(my_w->sizeWidth, SIGNAL(editingFinished()), this, SLOT(changeWidth()));
    disconnect(my_w->sizeHeight, SIGNAL(editingFinished()), this, SLOT(changeHeight()));
    my_w->sizeWidth->setText(QString::number(getRectF().width()));
    my_w->sizeHeight->setText(QString::number(getRectF().height()));
    connect(my_w->sizeWidth, SIGNAL(editingFinished()), this, SLOT(changeWidth()));
    connect(my_w->sizeHeight, SIGNAL(editingFinished()), this, SLOT(changeHeight()));
}

void
nRect::showMessage ( QString s ) {
    nparent->statusBar()->showMessage(s);
    my_pad.statusBar()->showMessage(s);
}

void
nRect::keyPressEvent ( QKeyEvent * e ) {
    int delta=1.0;
    if (e->modifiers() & Qt::ShiftModifier) {
        delta =10.0;
    }
    switch (e->key()) {
    case Qt::Key_Escape:
        if (ref.size()<2) deleteLater();
        break;
    case Qt::Key_Up:
        moveBy(QPointF(0.0,-delta));
        itemChanged();
        break;
    case Qt::Key_Down:
        moveBy(QPointF(0.0,+delta));
        itemChanged();
        break;
    case Qt::Key_Left:
        moveBy(QPointF(-delta,0.0));
        itemChanged();
        break;
    case Qt::Key_Right:
        moveBy(QPointF(+delta,0.0));
        itemChanged();
        break;
    case Qt::Key_W:
        togglePadella();
        break;
    case Qt::Key_E:
        expandX();
        expandY();
        break;
    case Qt::Key_X:
        expandX();
        break;
    case Qt::Key_Y:
        expandY();
        break;
    case Qt::Key_I:
        intersection();
        break;
    case Qt::Key_S:
        submatrix();
        break;
    default:
        emit key_pressed(e->key());
        break;
    }
}

void
nRect::keyReleaseEvent ( QKeyEvent *  ) {
}


void
nRect::moveBy(QPointF delta) {
    for (int i =0; i<ref.size(); i++) {
        changeP(i,mapToScene(ref[i]->pos()+delta),true);
    }
    showMessage(getRectString());
}

void
nRect::focusInEvent( QFocusEvent *) {
    selectThis(true);
}

void
nRect::focusOutEvent( QFocusEvent *) {
    selectThis(false);
}

void
nRect::selectThis(bool val) {
    setSelected(val);
    for (int i =0; i<ref.size(); i++) {
        ref[i]->setVisible(val);
    }
    update();
    if (val) {
        nparent->my_w->statusbar->showMessage(toolTip());
    } else {
        nparent->my_w->statusbar->showMessage("");
    }
}

QString nRect::getRectString() {
    QRectF myR=getRectF();
    return QString::number(myR.left())+","+
            QString::number(myR.top())+" "+
            QString::number(myR.width())+"x"+
            QString::number(myR.height());
}

// reimplementation
QRectF
nRect::boundingRect() const {
    return shape().boundingRect();
}

QPainterPath nRect::shape() const {
    QPainterPathStroker stroker;
    double thickness=std::max(nWidth,10.0)/zoom;
    stroker.setWidth(thickness);
    QPainterPath my_shape = stroker.createStroke( path() );
    for (int i =0; i<ref.size(); i++) {
        my_shape.addPolygon(ref[i]->mapToScene(ref[i]->rect()));
    }
    return my_shape;
}

void
nRect::paint(QPainter* p, const QStyleOptionGraphicsItem* , QWidget* ) {
    //	p->setCompositionMode((QPainter::CompositionMode)22);
    QPen pen;
    pen.setWidthF(nWidth/zoom);
    pen.setColor(nColor);
    p->setPen(pen);
    p->drawPath(path());
}


QPainterPath nRect::path() const {
    QPainterPath my_path;
    if (ref.size()>1) {
        my_path.addRect(QRectF(ref[0]->pos(),ref[1]->pos()));
    } else {
        my_path.addRect(QRectF(0,0,0,0));
    }
    return my_path;
}

void nRect::zoomChanged(double val){
    zoom=val;
    sizeHolder(nSizeHolder);
    update();
}

void nRect::itemChanged() {
    emit sceneChanged();
}

// SETTINGS

void
nRect::loadSettings() {
    QString fnametmp = QFileDialog::getOpenFileName(&my_pad, tr("Open INI File"),property("NeuSave-fileIni").toString(), tr("INI Files (*.ini *.conf)"));
    if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileIni",fnametmp);
        QSettings settings(fnametmp,QSettings::IniFormat);
        loadSettings(&settings);
    }
}

void
nRect::saveSettings() {
    QString fnametmp = QFileDialog::getSaveFileName(&my_pad, tr("Save INI File"),property("NeuSave-fileIni").toString(), tr("INI Files (*.ini *.conf)"));
    if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileIni",fnametmp);
        QSettings settings(fnametmp,QSettings::IniFormat);
        settings.clear();
        saveSettings(&settings);
    }
}

void
nRect::loadSettings(QSettings *settings) {
    settings->beginGroup(toolTip());
    setPos(settings->value("position").toPoint());
    
    int size = settings->beginReadArray("points");
    QPolygonF poly_tmp;
    for (int i = 0; i < size; ++i) {
        settings->setArrayIndex(i);
        poly_tmp << QPointF(settings->value("x").toDouble(),settings->value("y").toDouble());
    }
    settings->endArray();
    if (poly_tmp.size()==2) {
        setRect(QRectF(poly_tmp.at(0),poly_tmp.at(1)));
    } else {
        showMessage(tr("Error reading from file"));
    }
    setToolTip(settings->value("name",toolTip()).toString());
    setZValue(settings->value("depth",zValue()).toDouble());
    setWidthF(settings->value("width",nWidth).toDouble());
    changeColor(settings->value("colorLine",nColor).value<QColor>());
    sizeHolder(settings->value("sizeHolder",nSizeHolder).toDouble());
    changeColorHolder(settings->value("colorHolder",ref[0]->brush().color()).value<QColor>());

    if (settings->childGroups().contains("Properties")) {
        settings->beginGroup("Properties");
        foreach(QString my_key, settings->allKeys()) {
            qDebug() << "load" <<  my_key << " : " << settings->value(my_key);
            setProperty(my_key.toStdString().c_str(), settings->value(my_key));
        }
        settings->endGroup();
    }

    settings->endGroup();
}

void
nRect::saveSettings(QSettings *settings) {
    settings->beginGroup(toolTip());
    settings->remove("");
    settings->setValue("position",pos());
    settings->beginWriteArray("points");
    for (int i = 0; i < ref.size(); ++i) {
        settings->setArrayIndex(i);
        QPointF ppos=mapToScene(ref.at(i)->pos());
        settings->setValue("x", ppos.x());
        settings->setValue("y", ppos.y());
    }
    settings->endArray();
    settings->setValue("name",toolTip());
    settings->setValue("depth",zValue());
    settings->setValue("width",nWidth);
    settings->setValue("colorLine",nColor);
    settings->setValue("sizeHolder",nSizeHolder);
    settings->setValue("colorHolder",ref[0]->brush().color());

    settings->beginGroup("Properties");
    qDebug() << dynamicPropertyNames().size();
    foreach(QByteArray ba, dynamicPropertyNames()) {
        qDebug() << "save" << ba << " : " << property(ba);
        if(ba.startsWith("NeuSave")) {
            qDebug() << "write" << ba << " : " << property(ba);
            settings->setValue(ba, property(ba));
        }
    }
    settings->endGroup();

    settings->endGroup();
}

