#ifndef __ZoomView
#define __ZoomView

#include <QGraphicsView>

class ZoomView : public QGraphicsView
{
    Q_OBJECT
    public:
    ZoomView(QWidget *parent = 0) {};

    void drawForeground(QPainter* painter, const QRectF& rect);
};

#endif
