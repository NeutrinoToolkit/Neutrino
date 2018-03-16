
#include "tpGlDraw.h"

tpGlDraw::tpGlDraw(QWidget *parent)
    : QGLWidget(parent)
{
    setFormat(QGLFormat(QGL::DoubleBuffer | QGL::DepthBuffer));

    // report resize
    resize(parent->size());

    rotationX = -21.0;
    rotationY = -57.0;
    rotationZ = 0.0;

    translationX = 0.;
    translationY = 0.;
    translationZ = -10.;

    magnification = 1.;

    faceColors[0] = Qt::red;
    faceColors[1] = Qt::green;
    faceColors[2] = Qt::blue;
    faceColors[3] = Qt::yellow;
}

void tpGlDraw::initializeGL()
{
    qglClearColor(Qt::white);
    glShadeModel(GL_FLAT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}

void tpGlDraw::resizeGL(int width, int height)
{
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    GLfloat x = GLfloat(width) / height;
    glOrtho(-x, +x, -1.0, +1.0, 4.0, 15.0);
    //glFrustum(-x, +x, -1.0, +1.0, 4.0, 15.0);
    glMatrixMode(GL_MODELVIEW);
}

void tpGlDraw::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    draw();
}

void tpGlDraw::draw()
{

	// e tutta la spiegazione sta qui:
	// gluLookAt is equivalent to
            
	//glMultMatrixf(M);
	//glTranslated(-eyex, -eyey, -eyez);
	//
	// from: https://www.opengl.org/sdk/docs/man2/xhtml/gluLookAt.xml

	static GLfloat P1[3] = { 1.0, -1.0, 2. };
	static GLfloat P2[3] = { 0, -1.0, 2. };

    	static GLfloat P3[3] = { 0, -1.0, 3. };
    	static GLfloat P4[3] = { 1.0, -1., 3. };

	GLfloat M[16] = {    magnification,0,0,0,
				    0,magnification,0,0,
				    0,0,magnification,0,
				    0,0,0,magnification};
//
//    static const GLfloat * const coords[4][3] = {
//        { P1, P2, P3 }, { P1, P3, P4 }, { P1, P4, P2 }, { P2, P4, P3 }
//    };
//
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(translationX, translationY, translationZ);
    glRotatef(rotationX, 1.0, 0.0, 0.0);
    glRotatef(rotationY, 0.0, 1.0, 0.0);
    glRotatef(rotationZ, 0.0, 0.0, 1.0);
    glScalef(magnification, magnification, magnification);

//    // print transformation matrix
//    GLfloat tm[16];
//    glGetFloatv(GL_MODELVIEW_MATRIX, tm);
//
//    for (int ii=0; ii<16; ii++) {
//	    std::cerr<<"\t"<<tm[ii];
//	    if (ii%4 == 3)
//		    std::cerr<<std::endl;
//    }
//    std::cerr<<"----------------------------------------------------------------------------"<<std::endl;
//    glMultMatrixf(M);
//
//    glGetFloatv(GL_MODELVIEW_MATRIX, tm);
//    for (int ii=0; ii<16; ii++) {
//	    std::cerr<<"\t"<<tm[ii];
//	    if (ii%4 == 3)
//		    std::cerr<<std::endl;
//    }
//    std::cerr<<"============================================================================"<<std::endl;
	
    
    // draw axis
	glLoadName(1);
	glBegin(GL_LINES);
	qglColor(Qt::black);


	// get min/max
	fp minv, maxv;
	for (int ii=0; ii<boxes.size(); ii++) {
		minv = min(minv, boxes[ii]->myvertex1);
		minv = min(minv, boxes[ii]->myvertex2);
		
		maxv = max(maxv, boxes[ii]->myvertex1);
		maxv = max(maxv, boxes[ii]->myvertex2);
	}

	glVertex3f(1.5*minv.x(),0,0);
	glVertex3f(1.5*maxv.x(),0,0);
	glVertex3f(0,1.5*minv.y(),0);
	glVertex3f(0,1.5*maxv.y(),0);
	glVertex3f(0,0,1.5*minv.z());
	glVertex3f(0,0,1.5*maxv.z());
	glEnd();

	renderText(1.6*maxv.x(),0,0,"x");
	renderText(0,1.6*maxv.y(),0,"y");
	renderText(0,0,1.6*maxv.z(),"z");

/*
//
//    for (int i = 0; i < 4; ++i) {
        glLoadName(0);
        //glBegin(GL_TRIANGLES);
        glBegin(GL_QUADS);
        qglColor(faceColors[2]);
//        for (int j = 0; j < 3; ++j) {
//            glVertex3f(coords[i][j][0], coords[i][j][1],
//                       coords[i][j][2]);
//        }
	glVertex3f(P1[0], P1[1], P1[2]);
	glVertex3f(P2[0], P2[1], P2[2]);
	glVertex3f(P3[0], P3[1], P3[2]);
	glVertex3f(P4[0], P4[1], P4[2]);
        glEnd();
//    }
*/

	//GLfloat cp1[3] = {0,0,0};
	//GLfloat cp2[3] = {2,2,2};
	//generateCube(cp1, cp2, 2);

	for (int ii=0; ii<boxes.size(); ii++) {
		//std::cerr<<"---- Adding tribox "<<boxes[ii]->myvertex1<<" <--> "<<boxes[ii]->myvertex2<<std::endl;
		generateTribox(*boxes[ii], Qt::blue);
	}
	//generateTribox(tribox(fp(-2,-2,3), fp(2,2,5)), Qt::blue);
	//generateTribox(tribox(fp(-4,-1,6), fp(4,1,9)), Qt::yellow);

}

void tpGlDraw::wheelEvent(QWheelEvent *ev) {
	int nDeg = ev->delta()/8;
	int nSteps = nDeg/15;

	ev->accept();

	magnification += nSteps/10.;
	//qDebug()<<"magnification: "<<magnification;
	updateGL();
}

void tpGlDraw::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
    //qDebug()<<"mouse event: "<<event->buttons();
}

void tpGlDraw::mouseMoveEvent(QMouseEvent *event)
{
    GLfloat dx = GLfloat(event->x() - lastPos.x()) / width();
    GLfloat dy = GLfloat(event->y() - lastPos.y()) / height();
    if (event->buttons() & Qt::LeftButton) {
        rotationX -= 180 * dy;
        rotationY -= 180 * dx;
        updateGL();
    } else if (event->buttons() & Qt::RightButton) {
        rotationX -= 180 * dy;
        rotationZ -= 180 * dx;
        updateGL();
    } else if (event->buttons() & Qt::MidButton) {
	translationX += dx;
	translationY += dy;
	updateGL();
    }
    lastPos = event->pos();

}

void tpGlDraw::mouseDoubleClickEvent(QMouseEvent *event)
{
    int face = faceAtPosition(event->pos());
    if (face != -1) {
        QColor color = QColorDialog::getColor(faceColors[face], this);
        if (color.isValid()) {
            faceColors[face] = color;
            updateGL();
        }
    }
}

int tpGlDraw::faceAtPosition(const QPoint &pos)
{
    const int MaxSize = 512;
    GLuint buffer[MaxSize];
    GLint viewport[4];

    makeCurrent();

    glGetIntegerv(GL_VIEWPORT, viewport);
    glSelectBuffer(MaxSize, buffer);
    glRenderMode(GL_SELECT);

    glInitNames();
    glPushName(0);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluPickMatrix(GLdouble(pos.x()), GLdouble(viewport[3] - pos.y()),
                  5.0, 5.0, viewport);
    GLfloat x = GLfloat(width()) / height();
    glFrustum(-x, x, -1.0, 1.0, 4.0, 15.0);
    draw();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    if (!glRenderMode(GL_RENDER))
        return -1;
    return buffer[3];
}

// helpers
//

void tpGlDraw::generateCube(GLfloat *P1, GLfloat *P2, int num)
{
	glLoadName(num);
	glBegin(GL_QUAD_STRIP);
	//qglColor(Qt::red);

	glVertex3f(P1[0], P1[1], P1[2]);
	glVertex3f(P1[0], P1[1], P2[2]);

	glVertex3f(P1[0], P2[1], P1[2]);
	glVertex3f(P1[0], P2[1], P2[2]);

	glVertex3f(P2[0], P2[1], P1[2]);
	glVertex3f(P2[0], P2[1], P2[2]);

	glVertex3f(P2[0], P1[1], P1[2]);
	glVertex3f(P2[0], P1[1], P2[2]);

	glVertex3f(P1[0], P1[1], P1[2]);
	glVertex3f(P1[0], P1[1], P2[2]);
	glEnd();

	glLoadName(num);
	glBegin(GL_QUADS);
	//qglColor(Qt::red);
	
	glVertex3f(P1[0], P1[1], P1[2]);
	glVertex3f(P1[0], P2[1], P1[2]);
	glVertex3f(P2[0], P2[1], P1[2]);
	glVertex3f(P2[0], P1[1], P1[2]);
	
	// questo deve girare al contrario
	glVertex3f(P1[0], P1[1], P2[2]);
	glVertex3f(P2[0], P1[1], P2[2]);
	glVertex3f(P2[0], P2[1], P2[2]);
	glVertex3f(P1[0], P2[1], P2[2]);
	
	glEnd();
	
}

// draw tribox
void tpGlDraw::generateTribox(tribox my_box, QColor color)
{
	qglColor(color);
	GLfloat cp1[3] = {(GLfloat) my_box.myvertex1.x(),
		(GLfloat) my_box.myvertex1.y(),
		(GLfloat) my_box.myvertex1.z()};
	GLfloat cp2[3] = {(GLfloat) my_box.myvertex2.x(),
		(GLfloat) my_box.myvertex2.y(),
		(GLfloat) my_box.myvertex2.z()};
	generateCube(cp1, cp2, 0);
}
