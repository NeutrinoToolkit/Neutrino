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

#ifndef __neutrino_h
#define __neutrino_h

// physImage
#include "nPhysD.h"
#include "nPhysMaths.h"

#include "nGenericPan.h"

// Items
#include "nLine.h"
#include "nRect.h"
#include "nEllipse.h"
#include "nPoint.h"

#include "ui_neutrino.h"

class neutrino : public QMainWindow, public Ui::neutrino {
    Q_OBJECT

public:

    neutrino();
    ~neutrino();

    QGraphicsScene& getScene();

    QList <QAction *> recentFileActs;
    void updateRecentFileActions(QString=QString());

    nPhysD* currentBuffer;

    QList<nPhysD*> physList;

private:
    QList<nGenericPan*> panList;


public slots:
    inline bool nPhysExists(nPhysD* my_phys) {return physList.contains(my_phys);}
    inline int indexOf(nPhysD* my_phys){return physList.indexOf(my_phys);}

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
    void on_actionClose_All_Buffers_triggered();
    void closeCurrentBuffer();
    void closeBuffer(nPhysD*);

    void addPhys(nPhysD*);
    nPhysD* replacePhys(nPhysD*,nPhysD*,bool=true);
    void removePhys(nPhysD*);
    void delayedDeletePhsy(nPhysD*);
    void showPhys(nPhysD* my_phys=nullptr);
    void addShowPhys(nPhysD*);

    nPhysD* getBuffer(int);

    nPhysD* getBuffer(QString);

    inline nPhysD* getCurrentBuffer() {
        return (physList.size() ? (nPhysExists(currentBuffer)? currentBuffer : nullptr) : nullptr);
    }

    inline QList<nPhysD *> getBufferList() {return physList;}
    inline QList<nGenericPan*> getPanList() {return panList;}

    // menu actions
    void addMenuBuffers(nPhysD*);
    // File
    neutrino* fileNew();

    QList<nPhysD*> fileOpen(QString=QString(""));
    void fileReopen();

    QString getFileSave();
    void fileSave();
    void fileSave(nPhysD*);
    void fileSave(QString);
    void fileSave(nPhysD*,QString);
    bool fileClose();
    //	void file_quit_slot();

    QList <nPhysD *> openSession(QString);
    void saveSession(QString=QString());
    std::string getPanData();
    void setPanData(std::string);

    void exportGraphics();
    void exportAllGraphics();
    void exportAllGraphicsOriginalName();
    void exportGraphics(QString);

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

    void emitBufferChanged(nPhysD* = nullptr);

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

#endif
