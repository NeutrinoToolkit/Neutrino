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
#include <QGraphicsObject>
#include "ui_nPoint.h"
#include "nPhysImageF.h"

#ifndef __nPoint
#define __nPoint

class neutrino;

class nPoint : public QGraphicsObject {
	Q_OBJECT
public:
	
	nPoint(neutrino *);
	~nPoint(){};
	
	neutrino *parent(){
		return (neutrino *) QGraphicsObject::parent();
	};
	
	enum { Type = QGraphicsItem::UserType + 4 };
	int type() const { return Type;}
	
	void mousePressEvent ( QGraphicsSceneMouseEvent * );
	void mouseReleaseEvent ( QGraphicsSceneMouseEvent * );
	void mouseMoveEvent ( QGraphicsSceneMouseEvent * );
	void keyPressEvent ( QKeyEvent *);
	void keyReleaseEvent ( QKeyEvent *);
	void mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * );
	void focusInEvent(QFocusEvent * );
	void focusOutEvent(QFocusEvent * );
	void hoverEnterEvent ( QGraphicsSceneHoverEvent * );
	void hoverLeaveEvent ( QGraphicsSceneHoverEvent * );
	
	void moveBy(QPointF);
	
	qreal nWidth, nSizeHolder;
	QColor nColor, holderColor;
	
	// pure virtuals in QGraphicsObjec
	QRectF boundingRect() const;
	void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*);
	
	QGraphicsRectItem ref;
	bool moveRef;

	QPointF click_pos;
	
	double zoom;
	// roba da padelle
	QMainWindow my_pad;
	Ui::nPoint my_w;
	
	QPainterPath path() const;
	QPainterPath shape() const;
	
	void selectThis(bool);
		
public slots:
	
	void togglePadella();
	
	void itemChanged();

	void interactive();
	
	void setPoint(QPointF);
	QPoint getPoint();
	QPointF getPointF();
	QString getPointString();

    void bufferChanged(nPhysD*);

	void zoomChanged(double);
	void showMessage(QString);
	void sizeHolder(double);
	void setWidthF(double);
	void setOrder(double);
	void changeToolTip(QString);
	void changeColorHolder();
	void changeColorHolder(QColor);

	void changePos(QString);
		
	void movePoint(QPointF);
	
	void addPointAfterClick(QPointF);

	void setParentPan(QString,int);
	
	//SETTINGS
	void loadSettings();
	void saveSettings();
	void loadSettings(QSettings *);
	void saveSettings(QSettings *);
	
	
signals:
	void sceneChanged();
};

#endif
