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

#include "nMouse.h"
#include "nTics.h"
#include "neutrino.h"

#ifndef nView_H_
#define nView_H_

class neutrino;

class nView : public QGraphicsView {
    Q_OBJECT

public:
    nView (QWidget *parent=0);
    ~nView ();

    neutrino *nparent;
    QMap<QString, std::vector<unsigned char>>& nPalettes;

    void resizeEvent (QResizeEvent *) override;

    // events
    void keyPressEvent (QKeyEvent *) override;
    void keyReleaseEvent (QKeyEvent *) override;


    // zoom
    QTransform trasformazione;
    bool fillimage;
    void incrzoom(double);

    void setSize();

    vec2f minMax;

    bool show_mouse;

    QGraphicsScene my_scene;

    nMouse my_mouse;
    nTics my_tics;

    QGraphicsPixmapItem my_pixitem;

    QString colorTable;

public slots:

    void zoomOut();
    void zoomIn();
    void zoomEq();

    void mouseDoubleClickEvent (QMouseEvent *) override;
    void mousePressEvent (QMouseEvent *) override;
    void mouseReleaseEvent (QMouseEvent *) override;
    void mouseMoveEvent (QMouseEvent *) override;
    void wheelEvent(QWheelEvent *) override;

    void showPhys(nPhysD*);

    void emitMouseposition (QPointF);

    void setZoomFactor(int val);

    void setMouseShape(int);
    void nextMouseShape();

    void changeColorTable (QString);
    void changeColorTable ();

    void previousColorTable ();
    void nextColorTable ();

    void setLockColors(bool);

    void decrGamma();
    void incrGamma();
    void resetGamma();
    void setGamma(int value);
    void prevBuffer();
    void nextBuffer();

    void cycleOverItems();
    void rescale99();
    void rescaleColor(double=100.0);
    void rescaleLess();
    void rescaleMore();

    void setMouseOrigin();

    void toggleRuler();
    void toggleGrid();

    void copyImage();
    void exportPixmap();

    void update();

private:
    bool gestureEvent(QGestureEvent *event);
	void swipeTriggered(QSwipeGesture *);
    void pinchTriggered(QPinchGesture *);
    void tapandholdTriggered(QTapAndHoldGesture *);

    unsigned int currentStepScaleFactor;

    bool lockColors;

protected:
    // what does is this for??
    void focusInEvent (QFocusEvent *) override;
    bool event(QEvent *event) override;


signals:
    void updatecolorbar(QString);
    void keypressed(QKeyEvent*);
    void mouseposition(QPointF);
    void mousePressEvent_sig(QPointF);
    void mouseDoubleClickEvent_sig(QPointF);
    void mouseReleaseEvent_sig(QPointF);
    void zoomChanged(double);
    void bufferChanged(nPhysD*);
    void bufferOriginChanged();
};

#endif
