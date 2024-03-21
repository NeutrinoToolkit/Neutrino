#ifndef ZoomView_H_
#define ZoomView_H_

#include <QGraphicsView>

class ZoomView : public QGraphicsView
{
    Q_OBJECT
    public:
    ZoomView(QWidget *parent = 0) {

    }

    void drawForeground(QPainter* painter, const QRectF& rect);
};

#endif
