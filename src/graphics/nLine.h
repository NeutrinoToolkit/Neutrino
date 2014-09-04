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
#include <QClipboard>
#include "nPhysImageF.h"
#include "ui_nLine.h"

#ifndef __nLine
#define __nLine

class neutrino;
class QwtPlot;
class QwtPlotCurve;
class QwtPlotMarker;

class nLine : public QGraphicsObject {
 Q_OBJECT
public:
		
	nLine(neutrino * = NULL);
	~nLine();
		
	neutrino *parent(){
		return (neutrino *) QGraphicsObject::parent();
	};
	
	void setNeutrino(neutrino*);
	
	enum { Type = QGraphicsItem::UserType + 1 };
	int type() const { return Type;}
	
	void mousePressEvent ( QGraphicsSceneMouseEvent * );
	void mouseReleaseEvent ( QGraphicsSceneMouseEvent * );
	void mouseMoveEvent ( QGraphicsSceneMouseEvent * );
	void keyPressEvent ( QKeyEvent *);
	void keyReleaseEvent ( QKeyEvent *);
	void mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * );
	void hoverEnterEvent( QGraphicsSceneHoverEvent *);
	void hoverMoveEvent( QGraphicsSceneHoverEvent *);
	void hoverLeaveEvent( QGraphicsSceneHoverEvent *);

	qreal nWidth;
	qreal nSizeHolder;
	int numPoints;
	QColor colorLine, colorHolder;
	
	// pure virtuals in QGraphicsObjec
	QRectF boundingRect() const;
	void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*);
		
	QList<QGraphicsEllipseItem*> ref;
	int moveRef;
	int nodeSelected;

	
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
	
    QPointF physOffset;
    
	void selectThis(bool);
	
	QwtPlot *my_qwt;
	QwtPlotCurve *lineOut;
	QList<QwtPlotMarker*> marker;

public slots:
	
	void interactive();
	void togglePadella();

	QString getStringData(QPolygonF);
	
	void setPoints(QPolygonF);
	QPolygonF getPoints();
	QPolygonF getLine(int np=1);

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

	void bufferChanged(nPhysD*);

	void changeP(int,QPointF);

	// methods to force monotonicity
	void setMonotone(bool);
	bool getHMonotone();
	void switchOrdering();
	void rearrange_monotone();

	void updatePlot();

	void movePoints(QPointF);

	void appendPoint();
	void addPoint(int);
	void addPoint();// this is for the tablewidge
	void removeLastPoint();
	void removePoint(int);
	void removePoint();// this is for the tablewidge

	void addPointAfterClick(QPointF);
	
	void export_txt();
	void export_txt_points();
	void copy_clip();
	void copyPoints();
	
	void itemChanged();

	void setParentPan(QString,int);

	//SETTINGS
	void loadSettings();
	void saveSettings();
	void loadSettings(QSettings *);
	void saveSettings(QSettings *);
	
signals:
	void sceneChanged();
	void key_pressed(int);
	
};

bool orderMonotone_x(const QPointF &, const QPointF &);
bool orderMonotone_y(const QPointF &, const QPointF &);

#endif
