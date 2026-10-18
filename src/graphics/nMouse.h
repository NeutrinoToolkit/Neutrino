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
#include <QPen>
#include <QPointF>
#include <QRegion>
#include <QSize>

class QGraphicsView;
class QPainter;

#ifndef nMouse_H_
#define nMouse_H_

// the pointer drawn over the image by nView::drawForeground (not a scene item: not exported, printed or hit by itemAt)
// the position is in scene coordinates and can be moved with the keyboard; the drawing is in screen pixels:
// 0 system cross cursor, 1 crosshair over the whole image, 2 "+" and 3 "x" with fixed size on screen
class nMouse {
public:

    nMouse(QGraphicsView *view);

    static const int num_shape=4;
    int my_shape;

    QSize size;     // image size (crosshair extent)
    QPen pen;

    QPointF pos() const { return position; }
    void setPos(QPointF);

    bool isVisible() const { return visible; }
    void setVisible(bool);
    void show() { setVisible(true); }
    void hide() { setVisible(false); }

    void setShape(int);
    void setSize(QSize);
    void setPen(QPen);

    void update();                      // repaint the area of the pointer
    void paint(QPainter*);              // called by nView::drawForeground
    void saveSettings();

private:
    QGraphicsView *my_view;
    QPointF position;
    bool visible;
    QRegion region() const;             // viewport area covered by the pointer
    QPointF viewPos() const;
};

#endif

/* old version: a QGraphicsItem in the scene, with sizes in scene pixels (scaled with the zoom),
   the bounding rect changed without prepareGeometryChange and covering the whole image for the crosshair
#include <QtGui>
#include <QGraphicsItem>

#ifndef nMouse_H_
#define nMouse_H_


class nMouse : public QGraphicsItem {
public:
	
    nMouse();
    ~nMouse();
    int my_shape;
    const int num_shape=4;

    void setThickness(double);

	QSize size;
    QPen pen;

    void setSize(QSize);
	
	QRectF boundingRect() const;
	void paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*);
	QPainterPath shape() const;

};

#endif
*/
