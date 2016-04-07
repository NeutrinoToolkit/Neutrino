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
#include <iostream>

#include <QtGui>
#include <QWidget>
#include <QGraphicsView>
#include <QGestureEvent>

#include "nGenericPan.h"

#ifndef __nView_h
#define __nView_h

class neutrino;

class nView : public QGraphicsView {
	Q_OBJECT

public:
	nView (QWidget *parent=0);
	~nView ();

	neutrino *parent(){
		return (neutrino *) (QWidget::parent())->parent();
	};

	QSizeF my_size;
	void resizeEvent (QResizeEvent *);

	// events
	void keyPressEvent (QKeyEvent *);
	void keyReleaseEvent (QKeyEvent *);


	// zoom
	QTransform trasformazione;
	bool fillimage;
	void incrzoom(double);
    void zoomOut();
    void zoomIn();
    void zoomEq();

	void setSize();

	vec2f minMax;

	QFont scaledFont;

	// painter
	bool show_mouse;	
	
	bool showDimPixel;
	
public slots:
    void mouseDoubleClickEvent (QMouseEvent *);
	void mousePressEvent (QMouseEvent *);
	void mouseReleaseEvent (QMouseEvent *);
	void mouseMoveEvent (QMouseEvent *);
	void wheelEvent(QWheelEvent *);

	void emitMouseposition (QPointF);

private:
    bool gestureEvent(QGestureEvent *event);
    void swipeTriggered(QSwipeGesture*);
    float currentStepScaleFactor;

protected:
	// what does is this for??
	void focusInEvent (QFocusEvent *);
    bool event(QEvent *event);


signals:
	void mouseposition(QPointF);
	void mousePressEvent_sig(QPointF);
	void mouseDoubleClickEvent_sig(QPointF);
	void mouseReleaseEvent_sig(QPointF);
	void zoomChanged(double);
};

#endif
