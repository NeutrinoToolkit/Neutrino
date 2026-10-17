#include "ZoomView.h"

void ZoomView::drawForeground(QPainter* painter, const QRectF& rect) {
    painter->save();
    QColor pencolor(Qt::black);
    pencolor.setAlphaF(0.5);
    // painter->setPen(QPen(pencolor, 0.5));
    // (0.5 in scene units: 5 pixels wide at zoom 10, covering what is zoomed): 1 pixel at any zoom
    QPen pen(pencolor, 1);
    pen.setCosmetic(true);
    painter->setPen(pen);
    QPointF p1=mapToScene(QPoint(0,0));
    QPointF p2=mapToScene(QPoint(width(),height()));
    painter->drawLine(QLineF(p1,p2));
    painter->drawLine(QLineF(p1.x(), p2.y(), p2.x(),p1.y()));
    painter->restore();
}


