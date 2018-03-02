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
#ifndef __neutrino_h
#define __neutrino_h

#include <string>
#include <iostream>
#include <vector>
#include <limits>

#include <QVector>
#include <QList>

#include <QClipboard>
#include <QMainWindow>
#include <QMouseEvent>
#include <QString>
#include <QPrinter>
#include <QMap>

#include "nGenericPan.h"

// physImage
#include "nPhysImageF.h"
#include "nPhysMaths.h"


// Items
#include "nLine.h"
#include "nRect.h"
#include "nEllipse.h"
#include "nPoint.h"

#include "ui_neutrino.h"

namespace Ui {
class nSbarra;
}

class neutrino : public QMainWindow {

    Q_OBJECT

public:

    neutrino();
    ~neutrino();

    QGraphicsScene& getScene();

    Ui::neutrino *my_w;
    Ui::nSbarra *my_sbarra;

    // logging win
    QMainWindow log_win;
    QPlainTextEdit logger;

    static const int MaxRecentFiles=20;

    QList <QAction *> recentFileActs;
    void updateRecentFileActions(QString=QString());


private:
    QList<nGenericPan*> panList;
    QTimer timerSaveDefaults;


public slots:
    inline int indexOf(nPhysD* my_phys){return (my_w->my_view ? my_w->my_view->physList.indexOf(my_phys) : -1);};

    void setGamma(int value);

    void processEvents();
    void contextMenuEvent(QContextMenuEvent *);
    void menuFlipRotate();
    void rotateLeft();
    void rotateRight();
    void flipUpDown();
    void flipLeftRight();
    void transpose();

    void menuPaths();

    void loadPlugin();
    void loadPlugin(QString pname, bool launch);

    void scanPlugins();
    void scanPlugins(QString);

    void openRecentFile();
    void clearRecentFile();
    void openRecentBuffer();

    // save and reload across restar
    void saveDefaults();
    void loadDefaults();

    void scanDir(QString dirpath, QString pattern);
    void on_actionOpen_Glob_triggered();

    void addPhys(nPhysD*);
    nPhysD* replacePhys(nPhysD*,nPhysD*,bool=true);
    void removePhys(nPhysD*);
    void showPhys(nPhysD*);
    void updatePhys();
    void addShowPhys(nPhysD*);

    nPhysD* getBuffer(int);

    inline nPhysD* getCurrentBuffer() {
        return (my_w->my_view ? my_w->my_view->currentBuffer : nullptr);
    };

    inline QList<nPhysD *> getBufferList() {return (my_w->my_view ? my_w->my_view->physList : QList<nPhysD *>() );};
    inline QList<nGenericPan*> getPanList() {return panList;};

    // menu actions
    void addMenuBuffers(nPhysD*);
    // File
    neutrino* fileNew();

    void fileOpen();
    QList<nPhysD*> fileOpen(QString);
    void fileOpen(QStringList);
    void fileReopen();

    QString getFileSave();
    void fileSave();
    void fileSave(nPhysD*);
    void fileSave(QString);
    void fileSave(nPhysD*,QString);
    void closeCurrentBuffer();
    bool fileClose();
    //	void file_quit_slot();

    QList <nPhysD *> openSession(QString);
    void saveSession(QString=QString());

    void exportGraphics();
    void exportAllGraphics();
    void exportGraphics(QString);

    nGenericPan* Hlineout();
    nGenericPan* Vlineout();

    void createDrawLine();

    QString newRect(QRectF);
    void newRect(QRectF,QString);
    void createDrawRect();

    void createDrawEllipse();

    void createDrawPoint();

    void mouseposition(QPointF=QPointF());

    void zoomChanged(double);

    void keyPressEvent (QKeyEvent *);
    void keyReleaseEvent (QKeyEvent *);

    void closeEvent(QCloseEvent *);

    void print();

    double getZoom() const;

    void emitBufferChanged(nPhysD* = NULL);

    nGenericPan* openRAW();

    void about();

    nGenericPan* Preferences();

    nGenericPan* openPan(QString,bool=true);

    void emitPanAdd(nGenericPan*);
    void emitPanDel(nGenericPan*);

    // to python

    nLine* line(QString);
    nRect* rect(QString);

    void dragEnterEvent(QDragEnterEvent *);
    void dragMoveEvent(QDragMoveEvent *);
    void dropEvent(QDropEvent *);

    nGenericPan* newPan(QString=QString());
    nGenericPan* getPan(QString);

    void changeEvent(QEvent *e);

    void on_actionKeyboard_shortcut_triggered();

signals:
    void colorValue(double);

    // signals for communications with pans
    void mouseAtMatrix(QPointF);					// mouse position on the matrix, no scale
    void mouseAtWorld(QPointF);					// mouse position with scale, relative to reference

    void nZoom(double);

    void bufferChanged(nPhysD*);						// visible image update
    void physAdd(nPhysD*);
    void physDel(nPhysD*);
    void physReplace(std::pair<nPhysD*,nPhysD*>);

    void keyPressEvent();
    void closeAll();
    void panAdd(nGenericPan*);
    void panDel(nGenericPan*);

};

QVariant toVariant(anydata &my_data);

anydata toAnydata(QVariant &my_variant);

Q_DECLARE_METATYPE(nPhysD);

#endif
