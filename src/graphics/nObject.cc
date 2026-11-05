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
#include "nCore.h"
#include "neutrino.h"
#include "nObject.h"
#include "ui_nObject.h"
#include "nGenericPan.h"

#include <iostream>

nObject::~nObject() {
    nCore::disconnectChildren(&my_pad, this); // its widgets are destroyed after this (see nCore::disconnectChildren)
    foreach (QGraphicsRectItem* item, ref) {
        delete item;
    }
}

nObject::nObject(nGenericPan *parentPan, int level, QString cname) : nObject(parentPan->nparent, cname)
{
    setParent(parentPan);
    my_w.name->setText(parentPan->panName()+cname);
    setProperty("parentPanControlLevel",level);
    if (level>0) {
        my_w.name->setReadOnly(true);
        disconnect(my_w.name, &QLineEdit::textChanged, this, &nObject::changeToolTip);

        disconnect(my_w.actionRemove, &QAction::triggered, this, &nObject::deleteLater);
        my_w.actionRemove->setVisible(false);
    }
}


nObject::nObject(neutrino *my_parent, QString cname) :
    QGraphicsObject(),
    nparent(my_parent)
{
    if (my_parent) {
        my_parent->getScene().addItem(this);
        setParent(my_parent);
        std::string my_prop_name=(QString("num")+cname).toStdString();
        int num=my_parent->property(my_prop_name.c_str()).toInt()+1;
        my_parent->setProperty(my_prop_name.c_str(),num);
        setProperty(my_prop_name.c_str(),num);

        setToolTip(cname+QString(" ")+QLocale().toString(num));
        connect(my_parent, &neutrino::mouseAtMatrix, this, &nObject::movePoints);
        connect(my_parent->my_view, &nView::zoomChanged, this, &nObject::zoomChanged);
        connect(my_parent, &neutrino::bufferChanged, this, &nObject::bufferChanged);
        zoom=my_parent->getZoom();
        if (my_parent->getCurrentBuffer()) {
            bufferChanged(my_parent->getCurrentBuffer());
        }
    }

    setAcceptHoverEvents(true);
    setFlag(QGraphicsItem::ItemIsSelectable);
    setFlag(QGraphicsItem::ItemIsFocusable);

    setProperty("NeuSave-fileIni",toolTip()+".ini");

    nWidth=1.0;
    nSizeHolder=5.0;

    nColor=QColor(Qt::black);

    setOrder(0.0);

    // PADELLA
    my_pad.setWindowTitle(toolTip());
    QFile my_icon_f(":"+cname+".png");

    my_pad.setWindowIcon(QIcon(my_icon_f.fileName()));
    my_w.setupUi(&my_pad);

    my_w.spinWidth->setValue(nWidth);
    my_w.spinDepth->setValue(zValue());
    my_w.colorLabel->setPalette(QPalette(nColor));

    connect(my_w.name, &QLineEdit::textChanged, this, &nObject::changeToolTip);

    my_w.name->setText(toolTip());
    my_w.spinSizeHolder->setValue(nSizeHolder);

    connect(my_w.spinWidth, &QDoubleSpinBox::valueChanged, this, &nObject::setWidthF);
    connect(my_w.spinDepth, &QDoubleSpinBox::valueChanged, this, &nObject::setOrder);
    connect(my_w.colorButton, &QPushButton::pressed, this, qOverload<>(&nObject::changeColor));
    connect(my_w.colorHolderButton, &QPushButton::pressed, this, qOverload<>(&nObject::changeColorHolder));
    connect(my_w.spinSizeHolder, &QDoubleSpinBox::valueChanged, this, &nObject::sizeHolder);
    connect(my_w.tableWidget, &QTableWidget::itemChanged, this, &nObject::tableUpdated);

    connect(my_w.actionFillH, &QAction::triggered, this, &nObject::expandX);
    connect(my_w.actionFillV, &QAction::triggered, this, &nObject::expandY);
    connect(my_w.actionFillBoth, &QAction::triggered, this, &nObject::expandX);
    connect(my_w.actionFillBoth, &QAction::triggered, this, &nObject::expandY);
    connect(my_w.actionIntersect, &QAction::triggered, this, &nObject::intersection);
    connect(my_w.actionSubmatrix, &QAction::triggered, this, &nObject::submatrix);
    connect(my_w.actionRemove, &QAction::triggered, this, &nObject::deleteLater);
    connect(my_w.actionCenter, &QAction::triggered, this, &nObject::centerInBuffer);

    connect(my_w.copyPoints, &QToolButton::released, this, &nObject::copy_points);
    connect(my_w.pastePoints, &QToolButton::released, this, &nObject::paste_points);

	connect(my_w.sizeWidth, &QLineEdit::editingFinished, this, &nObject::changeWidth);
	connect(my_w.sizeHeight, &QLineEdit::editingFinished, this, &nObject::changeHeight);

    updateSize();
}

void nObject::contextMenuEvent ( QGraphicsSceneContextMenuEvent * e ) {
	QMenu menu;
	QAction *expandx = menu.addAction("Expand width (x)");
	connect(expandx, &QAction::triggered, this, &nObject::expandX);
	QAction *expandy = menu.addAction("Expand height (y)");
	connect(expandy, &QAction::triggered, this, &nObject::expandY);
	QAction *expandBoth = menu.addAction("Expand Both (e)");
	connect(expandBoth, &QAction::triggered, this, &nObject::expandX);
	connect(expandBoth, &QAction::triggered, this, &nObject::expandY);
    QAction *intersect = menu.addAction("Intersection with the image (i)");
    connect(intersect, &QAction::triggered, this, &nObject::intersection);
    QAction *centerIt = menu.addAction("Center (c)");
    connect(centerIt, &QAction::triggered, this, &nObject::centerInBuffer);
    QAction *submat = menu.addAction("Submatrix (s)");
    connect(submat, &QAction::triggered, this, &nObject::submatrix);
    menu.addAction(menu.addSeparator());
    QAction *copy = menu.addAction("Copy points (shift-c)");
    connect(copy, &QAction::triggered, this, &nObject::copy_points);
    QAction *paste = menu.addAction("Paste points (shift-v)");
    connect(paste, &QAction::triggered, this, &nObject::paste_points);
    menu.addAction(menu.addSeparator());
    QAction *showPan = menu.addAction("Show control (w)");
    connect(showPan, &QAction::triggered, this, &nObject::togglePadella);

    menu.exec(e->screenPos());
}

void nObject::setRect(QRectF rect) {
    while (ref.size()<2) appendPoint();
    moveRef.clear();
    changeP(0,rect.topLeft(),true);
    changeP(1,rect.bottomRight(),true);
    itemChanged();
}

QRect nObject::getRect(nPhysD* image) {
    if (ref.size()<2) return QRect();
    QRect geom2=QRectF(mapToScene(ref[0]->pos()),mapToScene(ref[1]->pos())).toRect().normalized();
    // qDebug() << geom2;
    if (image && nparent->getCurrentBuffer()) {
        vec2f dx(image->get_origin()-nparent->getCurrentBuffer()->get_origin());
        geom2.translate(dx.x(),dx.y());
        // qDebug() << geom2;
        geom2=geom2.intersected(QRect(0,0,image->getW(), image->getH()));
        // qDebug() << geom2;
    }
    return geom2;
}

QRectF nObject::getRectF() {
    if (ref.size()<2) {
        return QRectF(0,0,0,0);
    } else {
        return QRectF(mapToScene(ref[0]->pos()),mapToScene(ref[1]->pos())).normalized();
    }
}

void nObject::bufferChanged(nPhysD* my_phys) {
    if (nparent->getBufferList().contains(my_phys)) {
        setPos(my_phys->get_origin().x(),my_phys->get_origin().y());
    } else {
        setPos(0,0);
    }
}

void nObject::interactive ( ) {
    showMessage(tr("Click and drag"));
    switch (ref.size()) {
        case 0:
            connect(nparent->my_view, &nView::mousePressEvent_sig, this, &nObject::interactive);
            connect(nparent->my_view, &nView::mouseReleaseEvent_sig, this, &nObject::interactive);
            appendPoint();
            moveRef << 0 ;
            break;
        case 1:
            appendPoint();
            moveRef.clear();
            moveRef << 1 ;
        case 2:
            disconnect(nparent->my_view, &nView::mousePressEvent_sig, this, &nObject::interactive);
            disconnect(nparent->my_view, &nView::mouseReleaseEvent_sig, this, &nObject::interactive);
        default:
            break;
    }
}

void nObject::hoverEnterEvent( QGraphicsSceneHoverEvent *){
    selectThis(true);
}

void nObject::hoverLeaveEvent( QGraphicsSceneHoverEvent *){
    selectThis(false);
}

void nObject::hoverMoveEvent( QGraphicsSceneHoverEvent *){}


void nObject::mousePressEvent ( QGraphicsSceneMouseEvent * e ) {

    if (e->button()==Qt::LeftButton) {
        for (int i=0;i<ref.size();i++) {
            if (ref.at(i)->rect().contains(mapToItem(ref.at(i), e->pos()))) {
                moveRef.append(i);
                break;
            }
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

void nObject::mouseReleaseEvent ( QGraphicsSceneMouseEvent * e ) {
    moveRef.clear();
    showMessage("");
    QGraphicsItem::mouseReleaseEvent(e);
    itemChanged();
}

void nObject::mouseMoveEvent ( QGraphicsSceneMouseEvent * e ) {
    if (moveRef.contains(ref.size())) {
        QPointF delta=e->pos()-click_pos;
        moveBy(delta);
        click_pos=e->pos();
    }
    QGraphicsItem::mouseMoveEvent(e);
}

void nObject::togglePadella() {
    if (my_pad.isHidden()) {
        my_pad.show();
    } else {
        my_pad.hide();
    }
}

void nObject::mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * e ) {
    togglePadella();
    QGraphicsItem::mouseDoubleClickEvent(e);
}

void nObject::sizeHolder ( double val ) {
    nSizeHolder=val;
    QPointF p=QPointF(val/zoom,val/zoom);
    foreach(QGraphicsRectItem *item, ref){
        item->setRect(QRectF(-p,p));
    }
}

void
nObject::movePoints (QPointF p) {
    for (int i=0;i<ref.size(); i++) {
        if (moveRef.contains(i)) {
            changeP(i,p,true);
            showMessage("P"+QLocale().toString(i)+" "+getRectString());
        }
    }
}

void
nObject::changeToolTip (QString n) {
    setToolTip(n);
    my_pad.setWindowTitle(n);
}

void
nObject::setWidthF (double w) {
    nWidth=w;
    update();
}

void
nObject::setOrder (double w) {
    setZValue(w);
}

void
nObject::tableUpdated (QTableWidgetItem * item) {
    QPointF p;
    bool okx=false, oky=false;
    p.rx()=QLocale().toDouble(my_w.tableWidget->item(item->row(),0)->text(), &okx);
    p.ry()=QLocale().toDouble(my_w.tableWidget->item(item->row(),1)->text(), &oky);
    if (!okx || !oky) { // not a number: back to the current value (it was set to 0)
        showMessage(tr("Not a number: ")+item->text());
        changePointPad(item->row());
        return;
    }
    // the table shows the coordinates of the object (changePointPad), changeP takes scene coordinates
    changeP(item->row(),mapToScene(p), false);
    itemChanged();
}

void
nObject::changeColor () {
    QColorDialog colordial(my_w.colorLabel->palette().color(QPalette::Window));
    colordial.setOption(QColorDialog::ShowAlphaChannel);
    colordial.exec();
    if (colordial.result() && colordial.currentColor().isValid()) {
        changeColor(colordial.currentColor());
    }
    update();
}

void
nObject::changeColor (QColor col) {
    nColor=col;
    my_w.colorLabel->setPalette(QPalette(nColor));
}

void
nObject::changeColorHolder () {
    QColorDialog colordial(my_w.colorHolderLabel->palette().color(QPalette::Window));
    colordial.setOption(QColorDialog::ShowAlphaChannel);
    colordial.exec();
    if (colordial.result() && colordial.currentColor().isValid()) {
        changeColorHolder(colordial.currentColor());
    }
}

void
nObject::changeColorHolder (QColor color) {
    holderColor=color;
    my_w.colorHolderLabel->setPalette(QPalette(color));
    if (ref.size()>0) {
        QBrush brush=ref[0]->brush();
        brush.setColor(color);
        foreach (QGraphicsRectItem *item, ref){
            item->setBrush(brush);
        }
    }
}

void
nObject::changeP (int np, QPointF p, bool updatepad) {
    prepareGeometryChange();
    ref[np]->setPos(mapFromScene(p));
    ref[np]->setVisible(true);
    if (updatepad) changePointPad(np);
    updateSize();
}

void nObject::changePointPad(int nrow) {
    disconnect(my_w.tableWidget, &QTableWidget::itemChanged, this, &nObject::tableUpdated);
    QPointF p=ref[nrow]->pos();
    QTableWidgetItem *xitem= new QTableWidgetItem(QLocale().toString(p.x()));
    QTableWidgetItem *yitem= new QTableWidgetItem(QLocale().toString(p.y()));
    xitem->setTextAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
    yitem->setTextAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
    my_w.tableWidget->setItem(nrow, 0, xitem);
    my_w.tableWidget->setItem(nrow, 1, yitem);
    my_w.tableWidget->resizeRowToContents(nrow);
    connect(my_w.tableWidget, &QTableWidget::itemChanged, this, &nObject::tableUpdated);
}

void nObject::addPoint (int pos) {

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
    disconnect(my_w.tableWidget, &QTableWidget::itemChanged, this, &nObject::tableUpdated);
    my_w.tableWidget->insertRow(pos);
    QTableWidgetItem *xitem= new QTableWidgetItem(QLocale().toString(position.x()));
    QTableWidgetItem *yitem= new QTableWidgetItem(QLocale().toString(position.y()));
    xitem->setTextAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
    yitem->setTextAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
    my_w.tableWidget->setItem(pos, 0, xitem);
    my_w.tableWidget->setItem(pos, 1, yitem);
    my_w.tableWidget->resizeRowToContents(pos);
    connect(my_w.tableWidget, &QTableWidget::itemChanged, this, &nObject::tableUpdated);
}

void nObject::appendPoint () {
    addPoint(ref.size());
}

void nObject::expandX() {
    if (nparent->getCurrentBuffer()) {
        QRectF rect=getRectF();
        changeP(0,QPointF(0,rect.top()),true);
        changeP(1,QPointF(nparent->getCurrentBuffer()->getW(),rect.bottom()),true);
        itemChanged();
    }
}

void nObject::expandY() {
    if (nparent->getCurrentBuffer()) {
        QRectF rect=getRectF();
        changeP(0,QPointF(rect.left(),0),true);
        changeP(1,QPointF(rect.right(),nparent->getCurrentBuffer()->getH()),true);
        itemChanged();
    }
}

void nObject::intersection() {
    if (nparent->getCurrentBuffer()) {
        //QRectF rect=QRectF(0,0,nparent->getCurrentBuffer()->getW(),nparent->getCurrentBuffer()->getH()).intersect(getRectF());
        //obsolete
        QRectF rect=QRectF(0,0,nparent->getCurrentBuffer()->getW(),nparent->getCurrentBuffer()->getH()).intersected(getRectF());
        setRect(rect);
    }
}

void nObject::submatrix() {
    if (nparent->getCurrentBuffer()) {
        nPhysD *copy=new nPhysD(nparent->getCurrentBuffer()->sub(getRect()));
        nparent->addShowPhys(copy);
    }
}

void nObject::centerInBuffer() {
    if (nparent->getCurrentBuffer()) {
        QRectF imageRect(0,0,nparent->getCurrentBuffer()->getW(),nparent->getCurrentBuffer()->getH());
        QRectF actualRect(getRectF());

        QSizeF my_size = (imageRect.size() - actualRect.size())/2.0;
        QPointF newcenter(my_size.width(), my_size.height());

        actualRect.moveTo(newcenter);

        setRect(imageRect.intersected(actualRect));
    }
}


void nObject::changeWidth () {
    if (ref.size()>1) { // (it needed an image, for nothing)
        QRectF rect=getRectF();
        bool ok;
        rect.setWidth(QLocale().toDouble(my_w.sizeWidth->text(),&ok));
        if (ok) {
            changeP(1,rect.bottomRight(),true);
            itemChanged();
        }
    }
}

void nObject::changeHeight () {
    if (ref.size()>1) { // (it needed an image, for nothing)
        QRectF rect=getRectF();
        bool ok;
        rect.setHeight(QLocale().toDouble(my_w.sizeHeight->text(),&ok));
        if (ok) {
            changeP(1,rect.bottomRight(),true);
            itemChanged();
        }
    }
}

void nObject::updateSize() {
    disconnect(my_w.sizeWidth, &QLineEdit::editingFinished, this, &nObject::changeWidth);
    disconnect(my_w.sizeHeight, &QLineEdit::editingFinished, this, &nObject::changeHeight);
    my_w.sizeWidth->setText(QLocale().toString(getRectF().width()));
    my_w.sizeHeight->setText(QLocale().toString(getRectF().height()));
    connect(my_w.sizeWidth, &QLineEdit::editingFinished, this, &nObject::changeWidth);
    connect(my_w.sizeHeight, &QLineEdit::editingFinished, this, &nObject::changeHeight);
}

void
nObject::showMessage ( QString s, int msec ) {
    nparent->statusBar()->showMessage(s,msec);
    my_pad.statusBar()->showMessage(s,msec);
}

void
nObject::keyPressEvent ( QKeyEvent * e ) {
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
        case Qt::Key_C:
            if (e->modifiers() & Qt::ShiftModifier) {
                copy_points();
            } else {
                centerInBuffer();
            }
            break;
        case Qt::Key_V:
            if (e->modifiers() & Qt::ShiftModifier) {
                paste_points();
            }
            break;
        default:
			emit key_pressed(e->key());
			break;
	}
}

void nObject::copy_points() {
    QString str_points;
    foreach(QGraphicsRectItem *r, ref) {
        str_points += QString::number(r->pos().x()) + " " + QString::number(r->pos().y()) + "\n";
    }
    DEBUG(str_points.toStdString());
    QApplication::clipboard()->setText(str_points);
    my_pad.statusBar()->showMessage("Points copied to clipboard");
}

void nObject::paste_points() {
    QStringList my_l=QApplication::clipboard()->text().split("\n",Qt::SkipEmptyParts);

    QPolygonF my_poly;
    for (int i=0; i<my_l.size(); i++) {
        QStringList my_p=my_l[i].split(" ",Qt::SkipEmptyParts);
        if (my_p.size()==2) {
            bool ok0=false, ok1=false;
            double p0=my_p[0].toDouble(&ok0);
            double p1=my_p[1].toDouble(&ok1);
            if (ok0 && ok1) {
                my_poly << QPointF(p0,p1);
            } else {
                my_poly.resize(0);
                break;
            }
        }
    }
    if (my_poly.size()>1) {
        moveRef.clear();
        while (ref.size()<2) appendPoint();
        // the clipboard has the coordinates of the object (copy_points), changeP takes scene coordinates
        for (int i=0; i<2; i++) {
            changeP(i,mapToScene(my_poly[i]), true);
        }
        itemChanged();
        moveRef.clear();
        my_pad.statusBar()->showMessage("Points pasted from clipboard");
    } else {
        my_pad.statusBar()->showMessage("Error pasting from clipboard");
    }
}


void
nObject::keyReleaseEvent ( QKeyEvent *  ) {
}


void
nObject::moveBy(QPointF delta) {
    for (int i =0; i<ref.size(); i++) {
        changeP(i,mapToScene(ref[i]->pos()+delta),true);
    }
    showMessage(getRectString());
}

void
nObject::focusInEvent( QFocusEvent *) {
    selectThis(true);
}

void
nObject::focusOutEvent( QFocusEvent *) {
    selectThis(false);
}

void
nObject::selectThis(bool val) {
    setSelected(val);
    for (int i =0; i<ref.size(); i++) {
        ref[i]->setVisible(val);
    }
    update();
    if (val) {
        nparent->statusbar->showMessage(toolTip());
    } else {
        nparent->statusbar->showMessage("");
    }
}

QString nObject::getRectString() {
    QRectF myR=getRectF();
    return QLocale().toString(myR.left())+","+
            QLocale().toString(myR.top())+" "+
            QLocale().toString(myR.width())+"x"+
            QLocale().toString(myR.height());
}

// reimplementation
QRectF
nObject::boundingRect() const {
    return shape().boundingRect();
}

QPainterPath nObject::shape() const {
	QPainterPathStroker stroker;
    double thickness=std::max(nWidth,10.0)/zoom;
	stroker.setWidth(thickness);
	QPainterPath my_shape = path();
	for (int i =0; i<ref.size(); i++) {
        // in the coordinates of the object (not of the scene: they differ by the image origin)
        my_shape.addPolygon(ref[i]->mapToParent(ref[i]->rect()));
	}
	return stroker.createStroke(my_shape);
}

QPainterPath nObject::path() const {
	QPainterPath my_path;
	if (ref.size()>1) {
        my_path.addRect(QRectF(ref[0]->pos(),ref[1]->pos()));
	} else {
		my_path.addRect(QRectF(0,0,0,0));
	}
	return my_path;
}

void nObject::zoomChanged(double val){
    zoom=val;
    sizeHolder(nSizeHolder);
    update();
}

void nObject::itemChanged() {
    emit sceneChanged();
}

// SETTINGS

void
nObject::loadSettings() {
    QString fnametmp = QFileDialog::getOpenFileName(&my_pad, tr("Open INI File"),property("NeuSave-fileIni").toString(), tr("INI Files (*.ini *.conf)"));
    if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileIni",fnametmp);
        QSettings settings(fnametmp,QSettings::IniFormat);
        loadSettings(settings);
    }
}

void
nObject::saveSettings() {
    QString fnametmp = QFileDialog::getSaveFileName(&my_pad, tr("Save INI File"),property("NeuSave-fileIni").toString(), tr("INI Files (*.ini *.conf)"));
    if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileIni",fnametmp);
        QSettings settings(fnametmp,QSettings::IniFormat);
        settings.clear();
        saveSettings(settings);
    }
}

void
nObject::loadSettings(QSettings &settings) {
    settings.beginGroup(toolTip());
    setPos(settings.value("position").toPointF());

    int size = settings.beginReadArray("points");
    QPolygonF poly_tmp;
    for (int i = 0; i < size; ++i) {
        settings.setArrayIndex(i);
        QPointF pp(settings.value("x").toDouble(),settings.value("y").toDouble());
        poly_tmp << pp;
	}
    settings.endArray();
    if (poly_tmp.size()>1) {
        // saveSettings writes the points in scene coordinates (mapToScene) and setRect takes scene coordinates
        // (changeP maps them): no mapFromScene here, it moved the object by -origin at each save/load
        setRect(QRectF(poly_tmp.at(0),poly_tmp.at(1)));
    } else {
        showMessage(tr("Error reading from file"));
    }
    // the saved position is the origin of the image of then: the points keep their place relative to it,
    // and the object goes to the origin of the current image now (it did at the next buffer change, jumping)
    if (nparent && nparent->getCurrentBuffer()) bufferChanged(nparent->getCurrentBuffer());
    setToolTip(settings.value("name",toolTip()).toString());
    setZValue(settings.value("depth",zValue()).toDouble());
    setWidthF(settings.value("width",nWidth).toDouble());
    changeColor(settings.value("colorLine",nColor).value<QColor>());
    sizeHolder(settings.value("sizeHolder",nSizeHolder).toDouble());
    changeColorHolder(settings.value("colorHolder",holderColor).value<QColor>());

    if (settings.childGroups().contains("Properties")) {
        settings.beginGroup("Properties");
        foreach(QString my_key, settings.allKeys()) {
            setProperty(my_key.toStdString().c_str(), settings.value(my_key));
        }
        settings.endGroup();
    }

    settings.endGroup();
}

void
nObject::saveSettings(QSettings &settings) {
    settings.beginGroup(toolTip());
    settings.remove("");
    settings.setValue("position",pos());
    settings.beginWriteArray("points");
    for (int i = 0; i < ref.size(); ++i) {
        settings.setArrayIndex(i);
        QPointF ppos=mapToScene(ref.at(i)->pos());
        settings.setValue("x", ppos.x());
        settings.setValue("y", ppos.y());
    }
    settings.endArray();
    settings.setValue("name",toolTip());
    settings.setValue("depth",zValue());
    settings.setValue("width",nWidth);
    settings.setValue("colorLine",nColor);
    settings.setValue("sizeHolder",nSizeHolder);
    settings.setValue("colorHolder",ref.size() ? ref[0]->brush().color() : holderColor);

    settings.beginGroup("Properties");
    foreach(QByteArray ba, dynamicPropertyNames()) {
        if(ba.startsWith("NeuSave")) {
            settings.setValue(ba, property(ba));
        }
    }
    settings.endGroup();

    settings.endGroup();
}

