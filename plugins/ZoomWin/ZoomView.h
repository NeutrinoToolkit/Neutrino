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
    // ZoomView(QWidget *parent = nullptr) : QGraphicsView(parent) {}
    // the whole viewport is repainted: when the view moves, Qt scrolls the old pixels (with the old cross)
    ZoomView(QWidget *parent = nullptr) : QGraphicsView(parent) { setViewportUpdateMode(QGraphicsView::FullViewportUpdate); }

    // void drawForeground(QPainter* painter, const QRectF& rect);
    void drawForeground(QPainter* painter, const QRectF& rect) override;
    // point marked by the cross (the view can't be centred on it near the edges of the scene)
    void setMark(QPointF p) { mark=p; viewport()->update(); }

private:
    QPointF mark;
};

#endif
