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

// plugins
#include <QPluginLoader>

#include "nGenericPan.h"
#include "nPlug.h"
#include "nView.h"

// base ui
#include "ui_neutrino.h"
#include "ui_nSbarra.h"
#include "ui_nAbout.h"

//#include "nColorBar.h"
#include "nColorBarWin.h"

// physImage
#include "nPhysImageF.h"
#include "nPhysMaths.h"


// Items
#include "nLine.h"
#include "nRect.h"
#include "nEllipse.h"
#include "nPoint.h"

#include "nMouse.h"

#include "nTics.h"

class neutrino : public QMainWindow {

Q_OBJECT

public:

	neutrino();
	~neutrino();
	
	QGraphicsScene *getScene();
	Ui::neutrino my_w;

    QGraphicsScene my_s;

    Ui::nSbarra my_sbarra;
    Ui::nAbout my_about;

	QPluginLoader *plug_loader;
	//nPlug *plug_iface;
	
	static const int MaxRecentFiles=20;

	QList <QAction *> recentFileActs;
	void updateRecentFileActions(QString=QString());


	QGraphicsPixmapItem my_pixitem;
	
	nMouse my_mouse;
	
	nTics my_tics;

	nPhysD* currentBuffer;
	QPointer<neutrino> follower;

	QString colorTable;
    QMap<QString, std::vector<unsigned char>> nPalettes;

private:
    QList<nGenericPan*> panList;
	QList<nPhysD*> physList;

public slots:
    inline int indexOf(nPhysD* my_phys){return physList.indexOf(my_phys);};

	nGenericPan* existsPan(QString);

	void build_colormap();
    void setGamma(int value);

	void processEvents();
	void contextMenuEvent(QContextMenuEvent *);
	void menuFlipRotate();
	void rotateLeft();
	void rotateRight();
	void flipUpDown();
	void flipLeftRight();
	
	void menuPaths();
	
    void loadPlugin();
    void scanPlugins();
    void scanPlugins(QString);
    void scanPlugins(QDir);

	void openRecentFile();
	void clearRecentFile();
	void openRecentBuffer();

	// save and reload across restar
	void saveDefaults();
	void loadDefaults();

	//colortable
	bool addPaletteFromString(QString,QString);
	QString addPaletteFromFile(QString);
	
	void changeColorTable (QString);
	void changeColorTable ();
	void previousColorTable ();
	void nextColorTable ();

	void changeColorMinMax (vec2f);

	void addPhys(nPhysD*);
	void addPhys(nPhysD&);
	nPhysD* replacePhys(nPhysD*,nPhysD*,bool=true);
	void removePhys(nPhysD*);
    void removePhys(nPhysD&);
	void showPhys(nPhysD*);
	void showPhys(nPhysD&);
	void addShowPhys(nPhysD*);
	void addShowPhys(nPhysD&);
    nPhysD* getBuffer(int=-1,bool=true);

	inline QList<nPhysD *> getBufferList() {return physList;};
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

	nGenericPan* Monitor();
	
	// Image
	void createQimage();
	
	void toggleMouse();
	void toggleMouse(bool);
	void toggleRuler();
	void toggleGrid();

	nGenericPan* Shortcuts();

	// Analysis
	nGenericPan* FocalSpot();
	nGenericPan* Contours();
	nGenericPan* MathOperations();
	
    nGenericPan* ZoomWin();
	
	// cutoff mask
	nGenericPan* CutoffImage();

	nGenericPan* WinList();
	
	nGenericPan* Properties();

	void cycleOverItems();
//	void setItemSelect(QGraphicsItem *, bool);

	nGenericPan* Hlineout();
	nGenericPan* Vlineout();
	nGenericPan* bothLineout();
#ifdef HAVE_HDF5
	nGenericPan* openHDF5();
#endif
	nGenericPan* BoxLineout();
	nGenericPan* FindPeaks();
	nGenericPan* CompareLines();

	void createDrawLine();

	QString newRect(QRectF);
	void newRect(QRectF,QString);
	void createDrawRect();

	void createDrawEllipse();

	void createDrawPoint();
	
	void mouseposition(QPointF);

	void zoomChanged(double);

	void keyPressEvent (QKeyEvent *);
	void keyReleaseEvent (QKeyEvent *);

	void closeEvent(QCloseEvent *);

	void actionPrevBuffer();
	void actionNextBuffer();

	void print();

	double getZoom() const;

	void emitBufferChanged(nPhysD* = NULL);

	// remote control of another neutrino
	void createFollower();

	nGenericPan* openRAW();
	
	// VISAR STUFF
	nGenericPan* Colorbar();

	nGenericPan* Visar();
	
	// WAVELET STUFF
	nGenericPan* Wavelet();
    
	// interferometry STUFF
	nGenericPan* Interferometry();
    
	// SPECTRAL ANALYSIS
	nGenericPan* SpectralAnalysis();

	// INTEGRAL INVERSION STUFF
	nGenericPan* Inversions();

	// Region path
	nGenericPan* RegionPath();

	// ROTATE STUFF
	nGenericPan* Rotate();
	
	// Affine STUFF
	nGenericPan* Affine();
	
	// remove ghost Fringes
	nGenericPan* Ghost();
    
	// grab picture from camera
    nGenericPan* Camera();

	// interpolate inside path
	nGenericPan* InterpolatePath();
    
	nGenericPan* MouseInfo();

	void about();
	nGenericPan* Preferences();

    nGenericPan* openPan(QString,bool=true);

	void emitPanAdd(nGenericPan*);
	void emitPanDel(nGenericPan*);
	
	// to python
	
	nLine* line(QString);
	
    void dragEnterEvent(QDragEnterEvent *);
	void dragMoveEvent(QDragMoveEvent *);
	void dropEvent(QDropEvent *);

#ifdef HAVE_PYTHONQT
    void loadPyScripts();
    void runPyScript();
    void runPyScript(QString);
    // pythonqt STUFF
    nGenericPan* Python();
    nGenericPan* newPan(QString=QString());
    nGenericPan* getPan(QString);
#endif

signals:
	void updatecolorbar();
    void colorValue(double);

	// signals for communications with pans
	void mouseAtMatrix(QPointF);					// mouse position on the matrix, no scale
	void mouseAtWorld(QPointF);					// mouse position with scale, relative to reference

	void nZoom(double);

	void bufferChanged(nPhysD*);						// visible image update
	void bufferOriginChanged();
	void physAdd(nPhysD*);
	void physDel(nPhysD*);
    void physMod(std::pair<nPhysD*,nPhysD*>);
    
    

	void keyPressEvent();
	void closeAll();
	void panAdd(nGenericPan*);
	void panDel(nGenericPan*);

};

QVariant toVariant(anydata &my_data);

anydata toAnydata(QVariant &my_variant);

#ifdef HAVE_PYTHONQT
class nPyWrapper : public QObject {
    Q_OBJECT

    public slots:
    neutrino* new_neutrino() {return new neutrino();};
    void delete_neutrino(neutrino* neu) {neu->deleteLater();};

};
#endif

Q_DECLARE_METATYPE(nPhysD);

#endif
