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
#include <QGraphicsSceneContextMenuEvent>
#include <QColorDialog>
#include <QFileDialog>

#include "nPhysD.h"
#include "ui_nObject.h"

#ifndef nObject_h
#define nObject_h

class neutrino;
class nGenericPan;

namespace Ui {
class nObject;
}

class nObject : public QGraphicsObject {
	Q_OBJECT
public:
	
	nObject(neutrino *, QString name);
	nObject(nGenericPan *, int level, QString name);
    ~nObject() override;
	
	neutrino *nparent;
	
    virtual int type() const override = 0;

    void mousePressEvent ( QGraphicsSceneMouseEvent * ) override;
    void mouseReleaseEvent ( QGraphicsSceneMouseEvent * ) override;
    void mouseMoveEvent ( QGraphicsSceneMouseEvent * ) override;
    void keyPressEvent ( QKeyEvent *) override;
    void keyReleaseEvent ( QKeyEvent *) override;
    void mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * ) override;
    void hoverEnterEvent( QGraphicsSceneHoverEvent *) override;
    void hoverLeaveEvent( QGraphicsSceneHoverEvent *) override;
    void hoverMoveEvent( QGraphicsSceneHoverEvent *) override;
    void focusInEvent(QFocusEvent * event) override;
    void focusOutEvent(QFocusEvent * event) override;
    void contextMenuEvent ( QGraphicsSceneContextMenuEvent * event ) override;

	void moveBy(QPointF);
	
	qreal nWidth, nSizeHolder;
	QColor nColor, holderColor;
	
	// pure virtuals in QGraphicsObjec
    QRectF boundingRect() const override;
    virtual void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*) override = 0;
	
	QList<QGraphicsRectItem*> ref;
	QList<int> moveRef;

	QPointF click_pos;
	
	void changeP(int,QPointF,bool);
	
	double zoom;
	// roba da padelle
	QMainWindow my_pad;
	Ui::nObject my_w;
	
	virtual QPainterPath path() const = 0;
    QPainterPath shape() const override;
	
	void selectThis(bool);

public slots:
	
	void togglePadella();
	
	void itemChanged();

	void interactive();

	void setRect(QRectF);
    QRect getRect(nPhysD* image=nullptr);
	QRectF getRectF();
	QString getRectString();

    void bufferChanged(nPhysD*);

	void zoomChanged(double);
	void showMessage(QString, int=0);
	void changePointPad(int);
	void sizeHolder(double);
	void setWidthF(double);
	void setOrder(double);
	void changeToolTip(QString);
	void changeColor();
	void changeColor(QColor);
	void changeColorHolder();
    void changeColorHolder(QColor);
	void tableUpdated(QTableWidgetItem *);
	
	void expandX();
	void expandY();
    void intersection();
	void submatrix();
	
	void centerInBuffer();

	void changeWidth();
	void changeHeight();
	
	void updateSize();
	
	void movePoints(QPointF);
	
	void appendPoint();
	void addPoint(int);
	
    void copy_points();
    void paste_points();


	//SETTINGS
	void loadSettings();
	void saveSettings();
    void loadSettings(QSettings&);
    void saveSettings(QSettings&);
	
	
signals:
	void sceneChanged();
	void key_pressed(int);
};

//Q_DECLARE_METATYPE(nObject*);

#endif
