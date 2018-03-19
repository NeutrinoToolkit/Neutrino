#include <iostream>

#include <QObject>
#include <QGLFormat>
#include <QColorDialog>
#include <QMouseEvent>
#include <QResizeEvent>

#include <QDebug>

#if defined(Q_OS_MAC)
#include <glu.h>
#else
#include <GL/glu.h>
#endif

#include "tribox.h"

#ifndef __tpGlDraw
#define __tpGlDraw

class tpGlDraw : public QGLWidget
{
    Q_OBJECT

public:
    tpGlDraw(QWidget *parent = 0);
    
    void addTribox(tribox *tb)
    { 
	    if (tb) {
		    boxes.push_back(tb);
	    }
    }


protected:
    void initializeGL();
    void resizeGL(int width, int height);
    void paintGL();

    void wheelEvent(QWheelEvent *ev);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent *event);

    void resizeEvent (QResizeEvent *ev) {
	    std::cerr<<"viewport resize: "<<ev->size().width()<<":"<<ev->size().height()<<std::endl;
	    resizeGL(ev->size().width(), ev->size().height());
    }

private:
    void draw();
    int faceAtPosition(const QPoint &pos);
    void generateCube(GLfloat *, GLfloat *, int);

    void generateTribox(tribox, QColor);
    std::vector<tribox *> boxes;


    GLfloat rotationX;
    GLfloat rotationY;
    GLfloat rotationZ;
    GLfloat translationX;
    GLfloat translationY;
    GLfloat translationZ;

    GLfloat magnification;

    QColor faceColors[4];
    QPoint lastPos;
};

#endif
