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
#include "nPoint.h"
#include "neutrino.h"
#include <iostream>
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_plot_marker.h>

#include <qwt_spline.h>
#include <qwt_curve_fitter.h>

nPoint::nPoint(neutrino *nparent) : QGraphicsObject()
{
	nparent->my_s.addItem(this);
	setParent(nparent);

	setAcceptHoverEvents(true);
	setFlag(QGraphicsItem::ItemIsSelectable);
	setFlag(QGraphicsItem::ItemIsFocusable);
	setProperty("parentPan",QString(""));
	setProperty("parentPanControlLevel",0);

	nWidth=1.0;
	nSizeHolder=5.0;

	nColor=QColor(Qt::black);
	holderColor=QColor(0,255,255,200);

	int num=nparent->property("nunPoint").toInt()+1;
	nparent->setProperty("nunPoint",num);
	setProperty("nunPoint",num);

	setOrder(0.0);
	setToolTip(tr("point")+QString(" ")+QString::number(num));

	connect(parent(), SIGNAL(mouseAtMatrix(QPointF)), this, SLOT(movePoint(QPointF)));

	connect(parent()->my_w.my_view, SIGNAL(zoomChanged(double)), this, SLOT(zoomChanged(double)));

	zoom=parent()->getZoom();


	// PADELLA
	my_pad.setWindowTitle(toolTip());
	my_pad.setWindowIcon(QIcon(":center"));
	my_w.setupUi(&my_pad);

	connect(my_w.name, SIGNAL(textChanged(QString)), this, SLOT(changeToolTip(QString)));

	my_w.name->setText(toolTip());
	my_w.spinSizeHolder->setValue(nSizeHolder);
	my_w.colorHolderLabel->setPalette(QPalette(holderColor));

	connect(my_w.spinDepth, SIGNAL(valueChanged(double)), this, SLOT(setOrder(double)));
	connect(my_w.colorHolderButton, SIGNAL(pressed()), this, SLOT(changeColorHolder()));
	connect(my_w.spinSizeHolder, SIGNAL(valueChanged(double)), this, SLOT(sizeHolder(double)));
	
	connect(my_w.spinSizeHolder, SIGNAL(valueChanged(double)), this, SLOT(sizeHolder(double)));

	connect(my_w.xPos, SIGNAL(textChanged(QString)), this, SLOT(changePos(QString)));
	connect(my_w.yPos, SIGNAL(textChanged(QString)), this, SLOT(changePos(QString)));
	
	QBrush refBrush;
	refBrush.setStyle(Qt::SolidPattern);
	refBrush.setColor(holderColor);
	
	ref.setPos(QPointF(0,0));
	ref.setBrush(refBrush);
	ref.setPen(Qt::NoPen);
	ref.setVisible(false);
	ref.setParentItem(this);
	sizeHolder(nSizeHolder);
	
    connect(nparent, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(bufferChanged(nPhysD*)));

	moveRef=false;
}

void nPoint::setParentPan(QString winname, int level) {
	my_w.name->setText(winname+"Point");
	setProperty("parentPan",winname);
	setProperty("parentPanControlLevel",level);
	if (level>0) {
		my_w.name->setReadOnly(true);
		disconnect(my_w.name, SIGNAL(textChanged(QString)), this, SLOT(changeToolTip(QString)));
	}
}

QPoint nPoint::getPoint() {
	return ref.pos().toPoint();
}

QPointF nPoint::getPointF() {
	return ref.pos();
}

void nPoint::bufferChanged(nPhysD* my_phys) {    
    if (my_phys) {
        setPos(my_phys->get_origin().x(),my_phys->get_origin().y());
    } else {
        setPos(0,0);
    }
}

void nPoint::interactive ( ) {
	showMessage(tr("Click for the first point of the rectangle"));
	connect(parent()->my_w.my_view, SIGNAL(mouseReleaseEvent_sig(QPointF)), this, SLOT(addPointAfterClick(QPointF)));
}

void nPoint::addPointAfterClick ( QPointF p) {
	setPoint(p);
	showMessage(tr("Point added"));
	disconnect(parent()->my_w.my_view, SIGNAL(mouseReleaseEvent_sig(QPointF)), this, SLOT(addPointAfterClick(QPointF)));
}

void nPoint::mousePressEvent ( QGraphicsSceneMouseEvent * e ) {
	if (ref.rect().contains(mapToItem(&ref, e->pos()))) {
		moveRef=true;
		click_pos= e->pos();
	}
	QGraphicsItem::mousePressEvent(e);
}

void nPoint::mouseReleaseEvent ( QGraphicsSceneMouseEvent * e ) {
	moveRef=false;
	showMessage("");
	QGraphicsItem::mouseReleaseEvent(e);
	itemChanged();
}

void nPoint::mouseMoveEvent ( QGraphicsSceneMouseEvent * e ) {
	if (moveRef) {
		QPointF delta=e->pos()-click_pos;
		moveBy(delta);
		click_pos=e->pos();
	}
	QGraphicsItem::mouseMoveEvent(e);
}

void nPoint::togglePadella() {
	if (my_pad.isHidden()) {
		my_pad.show();
	} else {
		my_pad.hide();
	}
}

void nPoint::mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * e ) {
	togglePadella();
	QGraphicsItem::mouseDoubleClickEvent(e);
}

void nPoint::sizeHolder ( double val ) {
	nSizeHolder=val;
	QPointF p=QPointF(val/zoom,val/zoom);
	DEBUG(p.x() << " " << p.y());
	ref.setRect(QRectF(-p,p));
}

void
nPoint::movePoint (QPointF p) {
	if (moveRef) {
		setPoint(p);
		showMessage("P "+getPointString());
	}
}

void
nPoint::changeToolTip (QString n) {
	setToolTip(n);
	my_pad.setWindowTitle(n);
}

void
nPoint::setWidthF (double w) {
	nWidth=w;
	update();
}

void
nPoint::setOrder (double w) {
	setZValue(w);
}

void
nPoint::changeColorHolder () {
	QColor color;
	QColorDialog colordial(my_w.colorHolderLabel->palette().color(QPalette::Background));
	colordial.setOption(QColorDialog::ShowAlphaChannel);
	colordial.exec();
	if (colordial.result() && colordial.currentColor().isValid()) {
		changeColorHolder(colordial.currentColor());
	}
}

void
nPoint::changeColorHolder (QColor color) {
	my_w.colorHolderLabel->setPalette(QPalette(color));
	QBrush brush=ref.brush();
	brush.setColor(color);
	ref.setBrush(brush);
}

void
nPoint::setPoint (QPointF p) {
	prepareGeometryChange();
	ref.setPos(p);
	my_w.xPos->setText(QString::number(p.x()));
	my_w.yPos->setText(QString::number(p.y()));
	ref.setVisible(true);
	itemChanged();
}

void nPoint::changePos(QString valStr) {
	bool ok;
	double val=valStr.toDouble(&ok);
	if (ok) {
		disconnect(sender(), SIGNAL(textChanged(QString)), this, SLOT(changePos(QString)));
		if (sender()==my_w.xPos) {
			setPoint(QPointF(val,ref.pos().y()));
		} else if (sender()==my_w.yPos) {
			setPoint(QPointF(ref.pos().x(),val));
		}
		connect(sender(), SIGNAL(textChanged(QString)), this, SLOT(changePos(QString)));
	}
}


void
nPoint::showMessage ( QString s ) {
	parent()->statusBar()->showMessage(s);
}

void
nPoint::keyPressEvent ( QKeyEvent * e ) {
	int delta=1.0;
	if (e->modifiers() & Qt::ShiftModifier) {
		delta =10.0;
	}
	switch (e->key()) {
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
		case Qt::Key_Return:
			togglePadella();
			break;
		default:
			break;
	}
}

void
nPoint::keyReleaseEvent ( QKeyEvent *  ) {
}


void
nPoint::moveBy(QPointF delta) {
    setPoint(mapToScene(ref.pos()+delta));
	showMessage(getPointString());
}

void
nPoint::focusInEvent( QFocusEvent *) {
	selectThis(true);
}

void
nPoint::focusOutEvent( QFocusEvent *) {
	selectThis(false);
}


void
nPoint::hoverEnterEvent ( QGraphicsSceneHoverEvent * ) {
	selectThis(true);
}

void
nPoint::hoverLeaveEvent ( QGraphicsSceneHoverEvent * ) {
	selectThis(false);
}

void
nPoint::selectThis(bool val) {
	setSelected(val);
	ref.setVisible(val);
	update();
	if (val) {
		parent()->my_w.statusbar->showMessage(toolTip());
	} else {
		parent()->my_w.statusbar->showMessage("");
	}
}

QString nPoint::getPointString() {
	QPointF myR=getPointF();
	return QString::number(getPointF().x())+","+QString::number(getPointF().y());
}

// reimplementation
QRectF
nPoint::boundingRect() const {
	return shape().boundingRect();
}

QPainterPath nPoint::shape() const {
	QPainterPathStroker stroker;
	stroker.setWidth(4+nWidth/zoom);
	QPainterPath my_shape = stroker.createStroke( path() );
	my_shape.addPolygon(ref.mapToScene(ref.rect()));
	return my_shape;
}

void
nPoint::paint(QPainter* p, const QStyleOptionGraphicsItem* , QWidget* ) {
	//	p->setCompositionMode((QPainter::CompositionMode)22);
	QPen pen;
	pen.setWidthF(nWidth/zoom);
	pen.setColor(nColor);
	p->setPen(pen);
	p->drawPath(path());
}


QPainterPath nPoint::path() const {
	QPainterPath my_path;
	return my_path;
}

void nPoint::zoomChanged(double val){
	zoom=val;
	sizeHolder(nSizeHolder);
	update();
}

void nPoint::itemChanged() {
	emit sceneChanged();
}

// SETTINGS

void
nPoint::loadSettings() {
	if (!property("fileIni").isValid()) {
		setProperty("fileIni",QString("line.ini"));
	}
	QString fnametmp = QFileDialog::getOpenFileName(&my_pad, tr("Open INI File"),property("fileIni").toString(), tr("INI Files (*.ini *.conf)"));
	if (!fnametmp.isEmpty()) {
		setProperty("fileIni",fnametmp);
		QSettings settings(fnametmp,QSettings::IniFormat);
		loadSettings(&settings);
	}
}

void
nPoint::saveSettings() {
	if (!property("fileIni").isValid()) {
		setProperty("fileIni",QString("line.ini"));
	}
	QString fnametmp = QFileDialog::getSaveFileName(&my_pad, tr("Save INI File"),property("fileIni").toString(), tr("INI Files (*.ini *.conf)"));
	if (!fnametmp.isEmpty()) {
		setProperty("fileIni",fnametmp);
		QSettings settings(fnametmp,QSettings::IniFormat);
		settings.clear();
		saveSettings(&settings);
	}
}

void
nPoint::loadSettings(QSettings *settings) {
	settings->beginGroup(toolTip());
	setPoint(QPointF(settings->value("x").toDouble(), settings->value("y").toDouble()));
	setToolTip(settings->value("name",toolTip()).toString());
	setZValue(settings->value("depth",zValue()).toDouble());
	setWidthF(settings->value("width",nWidth).toDouble());
	sizeHolder(settings->value("sizeHolder",nSizeHolder).toDouble());
	changeColorHolder(settings->value("colorHolder",ref.brush().color()).value<QColor>());
	settings->endGroup();
}

void
nPoint::saveSettings(QSettings *settings) {
	settings->beginGroup(toolTip());
	settings->remove("");
	settings->setValue("x", ref.pos().x());
	settings->setValue("y", ref.pos().y());
	settings->setValue("name",toolTip());
	settings->setValue("depth",zValue());
	settings->setValue("width",nWidth);
	settings->setValue("colorLine",nColor);
	settings->setValue("sizeHolder",nSizeHolder);
	settings->setValue("colorHolder",ref.brush().color());
	settings->endGroup();
}



