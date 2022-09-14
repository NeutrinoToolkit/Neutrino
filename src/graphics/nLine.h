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
#ifndef __nLine
#define __nLine

#include <QtGui>
#include <QClipboard>
#include <QGraphicsObject>
#include "ui_nLine.h"

#include "nPhysD.h"

class neutrino;
class nGenericPan;

class nLine : public QGraphicsObject {
Q_OBJECT

public:
    nLine(neutrino * = nullptr);
    nLine(nGenericPan *, int level);
    ~nLine();

    neutrino *nparent;

	enum { Type = QGraphicsItem::UserType + 1 };
	int type() const { return Type;}

	void mousePressEvent ( QGraphicsSceneMouseEvent * );
	void mouseReleaseEvent ( QGraphicsSceneMouseEvent * );
	void mouseMoveEvent ( QGraphicsSceneMouseEvent * );
	void keyPressEvent ( QKeyEvent *);
	void keyReleaseEvent ( QKeyEvent *);
	void hoverEnterEvent( QGraphicsSceneHoverEvent *);
	void hoverLeaveEvent( QGraphicsSceneHoverEvent *);
	void hoverMoveEvent( QGraphicsSceneHoverEvent *);
	void focusInEvent(QFocusEvent * event);
	void focusOutEvent(QFocusEvent * event);
	void contextMenuEvent ( QGraphicsSceneContextMenuEvent * event );

	void moveBy(QPointF);

	qreal nWidth;
	qreal nSizeHolder;
	int numPoints;
	QColor colorLine, colorHolder;

	// pure virtuals in QGraphicsObjec
	QRectF boundingRect() const;
	void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*);

	QList<QGraphicsEllipseItem*> ref;
	QList<int> moveRef;
	int nodeSelected;
	QPointF click_pos;


	bool bezier;
	bool closedLine;
	bool antialias;
	bool forceMonotone, forceInverseOrdering;

	double zoom;

	// roba da padelle
	QMainWindow my_pad;
    Ui::nLine my_w;

	QPainterPath path() const;
	QPolygonF poly(int) const;
	QPainterPath shape() const;

public slots:

    void selectThis(bool);

	void interactive();
	void togglePadella();

	void setPoints(QPolygonF);
	QPolygonF getPoints();
	QPolygonF getLine(int np=1);
	void bufferChanged(nPhysD*);

	void zoomChanged(double);
	void showMessage(QString);
	void changePointPad(int);
	void sizeHolder(double);
	void setNumPoints(int);
	void setWidthF(double);
	void setOrder(double);
	void changeToolTip(QString);
	void changeColor();
	void changeColor(QColor);
	void changeColorHolder();
	void changeColorHolder(QColor);
	void toggleBezier();
	void toggleBezier(bool);
	void tableUpdated(QTableWidgetItem *);
	void toggleClosedLine();
	void toggleClosedLine(bool);
	void toggleAntialias();
	void toggleAntialias(bool);


	void centerOnImage();

	void changeP(int,QPointF, bool isLocal=false);

	// methods to force monotonicity
	void setMonotone(bool);
	bool getHMonotone();
	void switchOrdering();
	void rearrange_monotone();

	virtual void updatePlot();

	void movePoints(QPointF);

	void appendPoint();
	void addPoint(int);
	void addPoint();// this is for the tablewidge
	void removeLastPoint();
	void removePoint(int);
	void removePoint();// this is for the tablewidge

	void addPointAfterClick(QPointF);

	void contextAppendPoint();
	void contextPrependPoint();
	void contextRemovePoint();

	void makeHorizontal();
	void makeVertical();
	void makeRectangle();

	void itemChanged();

	QString getPointsStr();
    void copy_points();
    void paste_points();
    void save_points();

	//SETTINGS
	void loadSettings();
	void saveSettings();
    void loadSettings(QSettings&);
    void saveSettings(QSettings&);

    void extractPath();

    // math facilities
    nPhysD getContourSubImage(double fill_value=std::numeric_limits<double>::quiet_NaN());
    QList<double> getContainedIntegral();

signals:
	void sceneChanged();
	void key_pressed(int);
};

bool orderMonotone_x(const QPointF &, const QPointF &);
bool orderMonotone_y(const QPointF &, const QPointF &);

Q_DECLARE_METATYPE(nLine*);

#endif
