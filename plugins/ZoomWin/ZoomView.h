#ifndef ZoomView_H_
#define ZoomView_H_

#include <QGraphicsView>

class ZoomView : public QGraphicsView
{
    Q_OBJECT
    public:
    // ZoomView(QWidget *parent = 0) {
    //
    // }
    // (the parent was not given to QGraphicsView)
    ZoomView(QWidget *parent = nullptr) : QGraphicsView(parent) {}

    void drawForeground(QPainter* painter, const QRectF& rect);
};

#endif
