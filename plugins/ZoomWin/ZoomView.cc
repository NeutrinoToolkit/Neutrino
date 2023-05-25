#include "ZoomView.h"

void ZoomView::drawForeground(QPainter* painter, const QRectF& rect) {
    painter->save();
    QColor pencolor(Qt::black);
    pencolor.setAlphaF(0.5);
    painter->setPen(QPen(pencolor, 0.5));
    QPointF p1=mapToScene(QPoint(0,0));
    QPointF p2=mapToScene(QPoint(width(),height()));
//    QPointF p0=(p1+p1)/2.0;
    painter->drawLine(QLineF(p1,p2));
    painter->drawLine(QLineF(p1.x(), p2.y(), p2.x(),p1.y()));
    painter->restore();
}


