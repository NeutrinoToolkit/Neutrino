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

#include <QMainWindow>
#include <QMouseEvent>
#include <QMap>

// plugins
#include <QPluginLoader>

#include "nGenericPan.h"
#include "nPlug.h"
#include "nView.h"

// base ui
#include "ui_neutrino.h"
#include "ui_nSbarra.h"

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

template<class T>
inline T SIGN(T x) { return (x > 0) ? 1 : ((x < 0) ? -1 : 0); }

class neutrino : public QMainWindow {

Q_OBJECT

public:

	neutrino();
	~neutrino();
	
	QGraphicsScene *getScene();
	Ui::neutrino my_w;

//	QDockWidget *leftDock;

	QGraphicsScene my_s;

	Ui::nSbarra my_sbarra;

	QList<nPhysD*> physList;
									
	nPlug *plug_iface;
	
	QString colorTable;
	
	
	static const int MaxRecentFiles=20;

	QList <QAction *> recentFileActs;
	void updateRecentFileActions(QString=QString());

	QList<QAction *> listabuffer;

	//	QGraphicsScene my_s;
	QGraphicsPixmapItem my_pixitem;
	
	double colorMin, colorMax;
	bool colorRelative;

	nMouse my_mouse;
	
	nTics my_tics;

	nPhysD* currentBuffer;
	QPointer<neutrino> follower;

	QMap<QString, unsigned char *> nPalettes;

public slots:
	nGenericPan* existsPan(QString, bool=false);
	QList<nGenericPan*> getPans();


	void build_colormap();
	
	void processEvents();
	void contextMenuEvent(QContextMenuEvent *);
	void menuFlipRotate();
	void rotateLeft();
	void rotateRight();
	void flipUpDown();
	void flipLeftRight();
	
	void menuPaths();
	
	void loadPlugin();
	
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

	void changeColorMinMax (double,double);

	void addPhys(nPhysD*);
	void addPhys(nPhysD&);
	nPhysD* replacePhys(nPhysD*,nPhysD*,bool=true);
	void removePhys(nPhysD*);
	void removePhys(nPhysD&);
	void showPhys(nPhysD*);
	void showPhys(nPhysD&);
	void addShowPhys(nPhysD*);
	void addShowPhys(nPhysD&);
	nPhysD* getBuffer(int=-1);

	QList<nPhysD *> getBufferList();

							
	// menu actions
	void addMenuBuffers(nPhysD*);
	// File
	void fileNew();

	void openFile(QString);
	void fileOpen();
	std::vector<nPhysD*> fileOpen(QString, QString=QString());
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

	vector <nPhysD *> openSession(QString);
	void saveSession(QString=QString());

	void exportGraphics();
	void exportGraphics(QString);

	nGenericPan* Monitor();
	
	// Image
	void createQimage();
	
	void zoomIn();
	void zoomOut();
	void zoomEq();

	void toggleMouse();
	void toggleMouse(bool);
	void toggleRuler();
	void toggleRuler(bool);
	void toggleGrid();
	void toggleGrid(bool);
	
	nGenericPan* Shortcuts();

	// Analysis
	nGenericPan* FocalSpot();
	nGenericPan* MathOperations();
	nGenericPan* AutoAlign();

	// cutoff mask
	nGenericPan* CutoffImage();

	nGenericPan* WinList();
	
	nGenericPan* Properties();

	void cycleOverItems();
//	void setItemSelect(QGraphicsItem *, bool);

	nGenericPan* Hlineout();
	nGenericPan* Vlineout();
	nGenericPan* bothLineout();

#ifdef __phys_HDF
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

	void dragEnterEvent(QDragEnterEvent *);
	void dragMoveEvent(QDragMoveEvent *);
	void dropEvent(QDropEvent *);

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
	
	// BLUR STUFF
	nGenericPan* Blur();

	nGenericPan* MouseInfo();

	void about();
	nGenericPan* Preferences();

	void emitPanAdd(nGenericPan*);
	void emitPanDel(nGenericPan*);
	
	// to python
	QList<QList<qreal> > getData(int=-1);
	bool setData(QList<QList<qreal> >,int=-1);
	
	nLine* line(QString);
	
signals:
	void passdropEvent(QDropEvent *);
	void updatecolorbar();
	void colorValue(double);

	// signals for communications with pans
	void mouseAtMatrix(QPointF);					// mouse position on the matrix, no scale
	void mouseAtWorld(QPointF);					// mouse position with scale, relative to reference

	void nZoom(double);

	void bufferChanged(nPhysD*);						// visible image update
	void physAdd(nPhysD*);
	void physDel(nPhysD*);

	void keyPressEvent();
	void closeAll();
	void panAdd(nGenericPan*);
	void panDel(nGenericPan*);

};


Q_DECLARE_METATYPE(nPhysD);

#endif
