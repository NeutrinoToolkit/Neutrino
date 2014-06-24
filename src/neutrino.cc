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
#include <QtSvg>

#include "neutrino.h"
#include "nMouseInfo.h"

#include "nBoxLineout.h"
#include "nFindPeaks.h"
#include "nCompareLines.h"
#include "nVisar.h"
#include "nWavelet.h"
#include "nSpectralAnalysis.h"
#include "nIntegralInversion.h"
#include "nRotate.h"
#include "nBlur.h"
#include "nRegionPath.h"
#include "nShortcuts.h"
#include "nAffine.h"

#include "nFocalSpot.h"
#include "nLineout.h"
#include "nLineoutBoth.h"

#include "nOperator.h"
#include "nCutoffMask.h"

#include "nMonitor.h"

#ifdef __phys_HDF
#include "nHDF5.h"
#endif

#include "nPreferences.h"
#include "nWinList.h"
#include "nPhysProperties.h"

#include "nPhysFormats.h"

#include "nOpenRAW.h"

#include "nPhysFormats.h"

neutrino::~neutrino()
{
	DEBUG("destructor");
	currentBuffer=NULL;
	saveDefaults();
	QApplication::processEvents();
	foreach (nGenericPan *pan, getPans()) {
		pan->close();
	}
	QApplication::processEvents();
	foreach (nPhysD *phys, physList) {
		delete phys;
	}
}

/// Creator
neutrino::neutrino(): my_mouse(this), my_tics(this) {

	my_w.setupUi(this);

// this below works if there is just one neutrino win open
//#ifdef Q_OS_MAC
//	DEBUG("command to have the main menu always visible!!! http://qt-project.org/doc/qt-4.8/mac-differences.html#menu-bar");
//	my_w.menubar->setParent(NULL);
//#endif

	
	currentBuffer=NULL;

	follower=NULL;
	
	int numwin=qApp->property("numWin").toInt()+1;
	qApp->setProperty("numWin",numwin);
	setProperty("winId",numwin);

	setWindowTitle(QString::number(numwin)+QString(": Neutrino"));

	QString menuTransformationDefault=QSettings("neutrino","").value("menuTransformationDefault", "").toString();
	foreach (QAction * act, my_w.menuTransformation->actions()) {
		if (!menuTransformationDefault.isEmpty()) {
			if (act->text()==menuTransformationDefault) {
				my_w.menuTransformation->setDefaultAction(act);
				my_w.actionFlipRotate->setIcon(my_w.menuTransformation->defaultAction()->icon());
			}
		} else if (act->icon().cacheKey()==my_w.actionFlipRotate->icon().cacheKey()) {
			my_w.menuTransformation->setDefaultAction(act);
		}
	}	
	connect(my_w.actionFlipRotate, SIGNAL(triggered()), this, SLOT(menuFlipRotate()));


	QString defualtActionPath=QSettings("neutrino","").value("defualtActionPath", "Rectangle").toString();

	foreach (QAction * act, my_w.menuPaths->actions()) {
		if (!defualtActionPath.isEmpty()) {		
			if (act->text()==defualtActionPath) {
				my_w.menuPaths->setDefaultAction(act);
				my_w.actionPaths->setIcon(my_w.menuPaths->defaultAction()->icon());
			}
		} else if (act->icon().cacheKey()==my_w.actionPaths->icon().cacheKey()) {
			my_w.menuPaths->setDefaultAction(act);
		}
	}	
	connect(my_w.actionPaths, SIGNAL(triggered()), this, SLOT(menuPaths()));
	

	connect(my_w.actionWinlist, SIGNAL(triggered()), this, SLOT(WinList()));
	connect(my_w.actionColors, SIGNAL(triggered()), this, SLOT(Colorbar()));
	connect(my_w.actionMouseInfo, SIGNAL(triggered()), this, SLOT(MouseInfo()));


	connect(my_w.actionLine, SIGNAL(triggered()), this, SLOT(createDrawLine()));
	connect(my_w.actionRect, SIGNAL(triggered()), this, SLOT(createDrawRect()));
	connect(my_w.actionPoint, SIGNAL(triggered()), this, SLOT(createDrawPoint()));
	connect(my_w.actionEllipse, SIGNAL(triggered()), this, SLOT(createDrawEllipse()));
	connect(my_w.actionLineoutH, SIGNAL(triggered()), this, SLOT(Hlineout()));
	connect(my_w.actionLineoutV, SIGNAL(triggered()), this, SLOT(Vlineout()));

	connect(my_w.actionNew, SIGNAL(triggered()), this, SLOT(fileNew()));
	connect(my_w.actionOpen, SIGNAL(triggered()), this, SLOT(fileOpen()));
	connect(my_w.actionOpen_RAW, SIGNAL(triggered()), this, SLOT(openRAW()));
#ifdef __phys_HDF
	connect(my_w.actionOpen_HDF5, SIGNAL(triggered()), this, SLOT(openHDF5()));
#endif
	connect(my_w.actionSave, SIGNAL(triggered()), this, SLOT(fileSave()));

	connect(my_w.actionMonitor_Directory, SIGNAL(triggered()), this, SLOT(Monitor()));

	
	
	connect(my_w.actionReopen_to_saved, SIGNAL(triggered()), this, SLOT(fileReopen()));

	connect(my_w.actionSave_Session, SIGNAL(triggered()), this, SLOT(saveSession()));

	connect(my_w.actionExport, SIGNAL(triggered()), this, SLOT(exportGraphics()));

	connect(my_w.actionPrint, SIGNAL(triggered()), this, SLOT(print()));

	connect(my_w.actionQuit, SIGNAL(triggered()), qApp, SLOT(closeAllWindows())) ;
	
	connect(my_w.actionClose, SIGNAL(triggered()), this, SLOT(fileClose()));
	connect(my_w.actionAbout, SIGNAL(triggered()), this, SLOT(about()));
	connect(my_w.actionPreferences, SIGNAL(triggered()), this, SLOT(Preferences()));


	connect(my_w.actionPrev_Buffer, SIGNAL(triggered()), this, SLOT(actionPrevBuffer()));
	connect(my_w.actionNext_Buffer, SIGNAL(triggered()), this, SLOT(actionNextBuffer()));
	connect(my_w.actionClose_Buffer, SIGNAL(triggered()), this, SLOT(closeCurrentBuffer()));
	connect(my_w.actionCycle_over_paths, SIGNAL(triggered()), this, SLOT(cycleOverItems()));
	connect(my_w.actionRect, SIGNAL(triggered()), this, SLOT(createDrawRect()));
	connect(my_w.actionLine, SIGNAL(triggered()), this, SLOT(createDrawLine()));
	connect(my_w.actionEllipse, SIGNAL(triggered()), this, SLOT(createDrawEllipse()));

	connect(my_w.actionShow_mouse, SIGNAL(triggered()), this, SLOT(toggleMouse()));
	connect(my_w.actionShow_ruler, SIGNAL(triggered()), this, SLOT(toggleRuler()));
	connect(my_w.actionShow_grid, SIGNAL(triggered()), this, SLOT(toggleGrid()));

	connect(my_w.actionRotate_left, SIGNAL(triggered()), this, SLOT(rotateLeft()));
	connect(my_w.actionRotate_right, SIGNAL(triggered()), this, SLOT(rotateRight()));
	connect(my_w.actionFlip_up_down, SIGNAL(triggered()), this, SLOT(flipUpDown()));
	connect(my_w.actionFlip_left_right, SIGNAL(triggered()), this, SLOT(flipLeftRight()));
	
	connect(my_w.actionProperties, SIGNAL(triggered()), this, SLOT(Properties()));

	
	connect(my_w.actionZoom_in, SIGNAL(triggered()), this, SLOT(zoomIn()));
	connect(my_w.actionZoom_out, SIGNAL(triggered()), this, SLOT(zoomOut()));
	connect(my_w.actionZoom_eq, SIGNAL(triggered()), this, SLOT(zoomEq()));

	connect(my_w.actionMouse_Info, SIGNAL(triggered()), this, SLOT(MouseInfo()));

	connect(my_w.actionFocal_Spot, SIGNAL(triggered()), this, SLOT(FocalSpot()));

	connect(my_w.actionMath_operations, SIGNAL(triggered()), this, SLOT(MathOperations()));
	connect(my_w.actionCutoff_Mask, SIGNAL(triggered()), this, SLOT(CutoffImage()));

	connect(my_w.actionNext_LUT, SIGNAL(triggered()), this, SLOT(nextColorTable()));
	connect(my_w.actionPrevious_LUT, SIGNAL(triggered()), this, SLOT(previousColorTable()));
	connect(my_w.actionShow_colortable, SIGNAL(triggered()), this, SLOT(Colorbar()));

	connect(my_w.actionDrawLine, SIGNAL(triggered()), this, SLOT(createDrawLine()));

	connect(my_w.actionHorizontal, SIGNAL(triggered()), this, SLOT(Hlineout()));
	connect(my_w.actionVertical, SIGNAL(triggered()), this, SLOT(Vlineout()));
	connect(my_w.actionBoth, SIGNAL(triggered()), this, SLOT(bothLineout()));
	connect(my_w.actionBoxLineout, SIGNAL(triggered()), this, SLOT(BoxLineout()));
	connect(my_w.actionFind_Peaks, SIGNAL(triggered()), this, SLOT(FindPeaks()));
	connect(my_w.actionCompareLines, SIGNAL(triggered()), this, SLOT(CompareLines()));
	connect(my_w.actionPlugin, SIGNAL(triggered()), this, SLOT(loadPlugin()));

	connect(my_w.actionSpectral_Analysis, SIGNAL(triggered()), this, SLOT(SpectralAnalysis()));
	connect(my_w.actionVisar, SIGNAL(triggered()), this, SLOT(Visar()));
	connect(my_w.actionWavelet, SIGNAL(triggered()), this, SLOT(Wavelet()));
	connect(my_w.actionInversions, SIGNAL(triggered()), this, SLOT(Inversions()));
	connect(my_w.actionRegionPath, SIGNAL(triggered()), this, SLOT(RegionPath()));
	
	connect(my_w.actionRotate, SIGNAL(triggered()), this, SLOT(Rotate()));
	connect(my_w.actionAffine_Transform, SIGNAL(triggered()), this, SLOT(Affine()));
	connect(my_w.actionBlur, SIGNAL(triggered()), this, SLOT(Blur()));

	connect(my_w.actionFollower, SIGNAL(triggered()), this, SLOT(createFollower()));

	connect(my_w.actionKeyborard_shortcuts, SIGNAL(triggered()), this, SLOT(Shortcuts()));

	
	// ---------------------------------------------------------------------------------------------

	QWidget *sbarra=new QWidget(this);
	my_sbarra.setupUi(sbarra);
	my_w.statusbar->addPermanentWidget(sbarra, 0);

	setAttribute(Qt::WA_DeleteOnClose);

	setCentralWidget(my_w.centralwidget);

	my_w.my_view->setAcceptDrops(false);

	my_w.my_view->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);

	connect(my_w.my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(mouseposition(QPointF)));

	connect(my_w.my_view, SIGNAL(zoomChanged(double)), this, SLOT(zoomChanged(double)));

	build_colormap();

	
	QSettings my_set("neutrino","");
	my_set.beginGroup("Palettes");
	QStringList paletteNamesClean,paletteNames=my_set.value("paletteNames","").toStringList();
	QStringList paletteColorsClean,paletteColors=my_set.value("paletteColors","").toStringList();
	if (paletteNames.size()==paletteColors.size()) {
		for (int i=0;i<paletteNames.size();i++) {
			if (addPaletteFromString(paletteNames.at(i), paletteColors.at(i))) {
				paletteNamesClean << paletteNames.at(i);
				paletteColorsClean << paletteColors.at(i);
			}
		}
	}
	my_set.setValue("paletteNames",paletteNamesClean);
	my_set.setValue("paletteColors",paletteColorsClean);

	QStringList paletteFilesClean,paletteFiles=my_set.value("paletteFiles","").toStringList();
	QStringList paletteFilesNameClean;
	foreach (QString paletteFile, paletteFiles) {
		QString name=addPaletteFromFile(paletteFile);
		if (!name.isEmpty()) {
			paletteFilesClean << paletteFile;
			paletteFilesNameClean<< name;
		}
	}
	my_set.setValue("paletteFiles",paletteFilesClean);
	my_set.setValue("paletteFilesNames",paletteFilesNameClean);
	my_set.endGroup();

	changeColorTable(nPalettes.keys().first());

	colorMin=0.0;
	colorMax=1.0;
	colorRelative=true;

	//recent file stuff

	for (int i = 0; i < MaxRecentFiles; ++i) {
		QAction *act = new QAction(this);
		act->setVisible(false);
		connect(act, SIGNAL(triggered()),this, SLOT(openRecentFile()));
		my_w.menuOpen_Recent->addAction(act);
		recentFileActs << act;
	}
	my_w.menuOpen_Recent->addSeparator();
	QAction *act = new QAction("Clear Menu", this);
	act->setVisible(true);
	connect(act, SIGNAL(triggered()),this, SLOT(clearRecentFile()));
	my_w.menuOpen_Recent->addAction(act);

	// lasciamo questo per ultimo
	//	my_w.scrollArea->setWidget(my_view);
	my_s.setParent(this);

	my_w.my_view->setScene(&my_s);
	
	
	my_pixitem.setPixmap(QPixmap(":icons/icon.png"));
	//	my_pixitem.setFlag(QGraphicsItem::ItemIsMovable);

	my_s.addItem(&my_pixitem);
	my_pixitem.setEnabled(true);
	my_pixitem.setZValue(-1);

	toggleRuler(false);
	toggleGrid(false);

	my_s.addItem(&my_mouse);

	my_w.my_view->setSize();
	my_s.addItem(&my_tics);
	
	my_s.views().at(0)->viewport()->setCursor(QCursor(Qt::CrossCursor));
	my_w.my_view->setCursor(QCursor(Qt::CrossCursor));


//	my_s.setBackgroundBrush(QBrush(QColor(255,255,255,255)));
	
	updateRecentFileActions();

	loadDefaults();
	show();
	
	//!enable this for testing

//	Palettes();
//	Colorbar();
//	recentFileActs.first()->trigger();
//	newRect(QRectF(100,100,300,300), "pippo");

//	WinList();
	//	if (numwin <10) runPyScript("/Users/tommaso/local/src/neutrino/test.py");
//	MathOperations();
//	Visar();
//	RegionPath();
//	FindPeaks();
//	Rotate();
	//openSession("/Users/tommaso/local/src/neutrino/build/Debug/pippo.neus");
	//	openSession("/Users/tommaso/local/src/neutrino/build/Release/Untitled.neus");
	//	saveSession("/Users/tommaso/local/src/neutrino/build/Release/Untitled.neus");
//		fileOpen("/Volumes/Gigante/xpp44712-r0177.h5");
	//	
	//	Rotate();
	//	currentBuffer->set_scale(-1,-1);
	//	if (numwin==1)createFollower();
	//	exportGraphics("/Users/tommaso/Desktop/test.png");
	//	exportGraphics("/Users/tommaso/Desktop/test.svg");
	//	exportGraphics("/Users/tommaso/Desktop/test.pdf");
	//	currentBuffer->set_origin(555,555);
	//	currentBuffer->set_scale(1500.0,1500.0);
	//	Visar();
	
}

void neutrino::processEvents()
{ QApplication::processEvents(); }

void neutrino::contextMenuEvent (QContextMenuEvent *) {
	my_w.menuImage->exec(QCursor::pos());
}

void neutrino::menuPaths() {
	if ( QApplication::keyboardModifiers()!=Qt::NoModifier) {
		my_w.menuPaths->exec(QCursor::pos());
	} else {
		my_w.menuPaths->defaultAction()->trigger();
	}
}

void neutrino::menuFlipRotate() {
	if ( QApplication::keyboardModifiers()!=Qt::NoModifier) {
		my_w.menuTransformation->exec(QCursor::pos());
	} else {
		my_w.menuTransformation->defaultAction()->trigger();
	}
}

void neutrino::rotateLeft() {
	my_w.menuTransformation->setDefaultAction(my_w.actionRotate_left);
	if (currentBuffer) {
		phys_rotate_left(*currentBuffer);
		createQimage();				
	}
	my_w.actionFlipRotate->setIcon(my_w.menuTransformation->defaultAction()->icon());
	QSettings("neutrino","").setValue("menuTransformationDefault",my_w.menuTransformation->defaultAction()->text());

}

void neutrino::rotateRight() {
	my_w.menuTransformation->setDefaultAction(my_w.actionRotate_right);
	if (currentBuffer) {
		phys_rotate_right(*currentBuffer);
		createQimage();		
	}
	my_w.actionFlipRotate->setIcon(my_w.menuTransformation->defaultAction()->icon());
	QSettings("neutrino","").setValue("menuTransformationDefault",my_w.menuTransformation->defaultAction()->text());
}

void neutrino::flipUpDown() {
	my_w.menuTransformation->setDefaultAction(my_w.actionFlip_up_down);
	if (currentBuffer) {
		phys_flip_ud(*currentBuffer);
		createQimage();
	}
	QSettings("neutrino","").setValue("menuTransformationDefault",my_w.menuTransformation->defaultAction()->text());
	my_w.actionFlipRotate->setIcon(my_w.menuTransformation->defaultAction()->icon());
}

void neutrino::flipLeftRight() {
	my_w.menuTransformation->setDefaultAction(my_w.actionFlip_left_right);
	if (currentBuffer) {
		phys_flip_lr(*currentBuffer);
		createQimage();		
	}
	QSettings("neutrino","").setValue("menuTransformationDefault",my_w.menuTransformation->defaultAction()->text());
	my_w.actionFlipRotate->setIcon(my_w.menuTransformation->defaultAction()->icon());
}

nPhysD* neutrino::getBuffer(int i) {
	if (i>=0 && i<physList.size()) return physList.at(i);
	return currentBuffer;
}

//void
//neutrino::scanPlugins()
//{
//   	QDir pluginsDir(qApp->applicationDirPath());
//#if defined(Q_OS_WIN)
//   	if (pluginsDir.dirName().toLower() == "debug" || pluginsDir.dirName().toLower() == "release")
//       		pluginsDir.cdUp();
//#elif defined(Q_OS_MAC)
//   	if (pluginsDir.dirName() == "MacOS") {
//       		pluginsDir.cdUp();
//       		pluginsDir.cdUp();
//       		pluginsDir.cdUp();
//   	}
//#endif
//   	pluginsDir.cd("plugins");
//   	foreach (QString fileName, pluginsDir.entryList(QDir::Files)) {
//		QPluginLoader pluginLoader(pluginsDir.absoluteFilePath(fileName));
//
//		
//	
//	}
//}

void
neutrino::loadPlugin()
 {
//     QDir pluginsDir(qApp->applicationDirPath());
// #if defined(Q_OS_WIN)
//     if (pluginsDir.dirName().toLower() == "debug" || pluginsDir.dirName().toLower() == "release")
//         pluginsDir.cdUp();
// #elif defined(Q_OS_MAC)
//     if (pluginsDir.dirName() == "MacOS") {
//         pluginsDir.cdUp();
//         pluginsDir.cdUp();
//         pluginsDir.cdUp();
//     }
// #endif
//     pluginsDir.cd("plugins");
//     foreach (QString fileName, pluginsDir.entryList(QDir::Files)) {
//       QPluginLoader pluginLoader(pluginsDir.absoluteFilePath(fileName));


	QString pname = QFileDialog::getOpenFileName(this,tr("Load Plugin"), property("loadPlugin").toString(),tr("Neutrino Plugins")+QString(" (*.dylib *.so *.dll);;")+tr("Any files")+QString(" (*)"));

	if (!pname.isEmpty()) {

		QPluginLoader pluginLoader(pname);
		QObject *plugin = pluginLoader.instance();
		if (plugin) {
			plug_iface = qobject_cast<nPlug *>(plugin);
			if (plug_iface) {
				cerr<<"plugin"<<plug_iface->name().toAscii().constData()<<" cast success"<<endl;
				plug_iface->instantiate(this);
			} else
				cerr<<"plugin load fail"<<endl;
		}
	}
 }

QList<nPhysD *>
neutrino::getBufferList()
{ return physList; }

void neutrino::emitBufferChanged(nPhysD *phys) {
	if (!phys) phys=currentBuffer;
	my_w.my_view->update();
	emit bufferChanged(phys);
}

void neutrino::emitPanAdd(nGenericPan* pan) {
	emit panAdd(pan);
}

void neutrino::emitPanDel(nGenericPan* pan) {
	emit panDel(pan);
}


/// Returns the QT drawings area
QGraphicsScene *
neutrino::getScene () {
	return &my_s;
}

/// This is called form a Open recent file menu.
void neutrino::openRecentFile()
{
	QAction *action = qobject_cast<QAction *>(sender());
	if (action) {
		QString fname=action->data().toString();
		fileOpen(fname);
		setProperty("fileOpen", fname);
	}
}

// Remove Open recent files menu.
void neutrino::clearRecentFile()
{
	for (int i = 0; i < recentFileActs.size(); ++i) {
		recentFileActs[i]->setVisible(false);
	}
	QSettings my_set("neutrino","");
	my_set.remove("recentFiles");
}


/// This is called form a Open recent buffer menu.
void neutrino::openRecentBuffer()
{
	QAction *action = qobject_cast<QAction *>(sender());
	if (action) {
		nPhysD *phys=(nPhysD*) (action->data().value<void*>());
		showPhys(phys);
	}
}

/// This resets the recent Open file menu after opening a new file.
void neutrino::updateRecentFileActions(QString fname)
{
	QSettings my_set("neutrino","");
	QStringList listarecentfiles=my_set.value("recentFiles").toStringList();
	listarecentfiles.prepend(fname);
	foreach (QString str, listarecentfiles) {
		if (!QFile(str).exists()) {
			listarecentfiles.removeAll(str);
		}
	}
	listarecentfiles.removeDuplicates();
	while (listarecentfiles.size()>20) listarecentfiles.removeLast();

	int i=0;
	foreach (QString fname, listarecentfiles) {
		if (QFile(fname).exists()) {
			recentFileActs[i]->setText(QFileInfo(fname).fileName()+QString(": ")+fname);
			recentFileActs[i]->setData(fname);
			recentFileActs[i]->setVisible(true);
			i++;
		}
	}

	my_set.setValue("recentFiles",listarecentfiles);			
}

/// Using TAB you can cycle over the items in the canvas (lines, rectangles, ovals)
void neutrino::cycleOverItems() {
	QList<QGraphicsItem *> lista;
	foreach (QGraphicsItem *oggetto, my_s.items() ) {
		if (oggetto->type() > QGraphicsItem::UserType) {
			if (oggetto->isVisible()) lista << oggetto;
		}
	}
	my_s.clearSelection();
	if (lista.size()>0) {
		int found=0;
		for (int i=lista.size()-1; i >=0; i-- ) {
			if (lista.at(i)->hasFocus()) found=(i+lista.size()-1)%lista.size();
		}

		lista.at(found)->setFocus(Qt::TabFocusReason);
	}
}

nGenericPan* neutrino::existsPan(QString name, bool reShow) {
	foreach (nGenericPan *pan, getPans()) {
		if (pan->panName.startsWith(name)) {
			if (reShow) {
				pan->show();
				pan->raise();
			}
			return pan;
		}		
	}
	return NULL;
}

QList<nGenericPan*> neutrino::getPans() {
	QList<nGenericPan*> retList;
	foreach (QWidget *widget, QApplication::allWidgets()) {
		nGenericPan *pan=qobject_cast<nGenericPan *>(widget);
		if (pan && pan->nparent==this) {
			retList << pan;
		}		
	}
	return retList;
}

// public slots

// file menu actions


void neutrino::fileNew() {
	QThread *m_thread = new QThread();
	neutrino*my_neu= new neutrino();
	my_neu->moveToThread(m_thread);
	m_thread->start();
}

void
neutrino::fileReopen() {
	if(currentBuffer && currentBuffer->getType()==PHYS_FILE) {
		QString fname=QString::fromUtf8(currentBuffer->getFromName().c_str());
		fileOpen(fname);
	}
}

void neutrino::fileOpen()
{
	QStringList fnames = QFileDialog::getOpenFileNames(this,tr("Open Image(s)"),property("fileOpen").toString(),"neutrino (*.neu *.neus *.hdf *.h5 *.imd *.png *.pgm *.ppm *.jpg *.tiff *.tif *.gif *.sif *.spe *.img *.raw *.txt *.fits *.inf *.gz);; Any files (*)");
	fileOpen(fnames);
}

void neutrino::fileOpen(QStringList fnames) {
	setProperty("fileOpen", fnames);
	foreach (QString fname, fnames) {
		qDebug() << " opening " << fname;
		vector <nPhysD *> imagelist = fileOpen(fname);
		if (imagelist.size()==0 && QFileInfo(fname).suffix().toLower()!="h5"){
			QString vwinname="OpenRaw";
			nOpenRAW *openRAW=(nOpenRAW *)existsPan(vwinname,true);
			if (!openRAW) {
				openRAW = new nOpenRAW(this, vwinname);	}
			openRAW->add(fname);
		}
	}
}


void neutrino::openFile(QString fname) {
	fileOpen(fname);
}

vector <nPhysD *> neutrino::fileOpen(QString fname, QString optString) {
	vector <nPhysD *> imagelist;
	if (QFileInfo(fname).suffix().toLower()=="neus") {
		imagelist=openSession(fname);
	} else {
		imagelist=phys_open(fname.toUtf8().constData(),optString.toUtf8().constData());
	}
	if (imagelist.size()==0) {
		// resta quasi solo QImage
		if (QFileInfo(fname).suffix().toLower()=="h5") {
#ifdef __phys_HDF
			QString vwinname="HDF5";
			nHDF5 *openHDF5=(nHDF5 *)existsPan(vwinname,true);
			if (!openHDF5) {
				openHDF5 = new nHDF5(this, vwinname);
			}
			openHDF5->showFile(fname);
#endif
		} else {
			QImage image(fname);
			if (!image.isNull()) {
				if (image.isGrayscale()) {
					nPhysD *datamatrix = new nPhysD(fname.toStdString());
					datamatrix->resize(image.width(), image.height());
					for (int i=0;i<image.height();i++) {
						for (int j=0;j<image.width();j++) {
							datamatrix->Timg_matrix[i][j]= qRed(image.pixel(j,i));
						}
					}
					datamatrix->TscanBrightness();
					datamatrix->setType(PHYS_FILE);
					imagelist.push_back(datamatrix);
				} else {
					nPhysD *datamatrix[3];
					string name[3];
					name[0]="Red";
					name[1]="Green";
					name[2]="Blue";
					for (int k=0;k<3;k++) {
						datamatrix[k] = new nPhysD(QFileInfo(fname).fileName().toStdString());
						datamatrix[k]->setShortName(name[k]);
						datamatrix[k]->setName(name[k]+" "+QFileInfo(fname).fileName().toStdString());
						datamatrix[k]->setFromName(fname.toStdString());
						datamatrix[k]->resize(image.width(), image.height());
						datamatrix[k]->setType(PHYS_FILE);
						imagelist.push_back(datamatrix[k]);
					}
					for (int i=0;i<image.height();i++) {
						for (int j=0;j<image.width();j++) {
							QRgb px = image.pixel(j,i);
							datamatrix[0]->Timg_matrix[i][j]= (double) (qRed(px));
							datamatrix[1]->Timg_matrix[i][j]= (double) (qGreen(px));
							datamatrix[2]->Timg_matrix[i][j]= (double) (qBlue(px));
						}
					}
				}
			}
		}
	}
		
	if (imagelist.size()>0) {
		updateRecentFileActions(fname);
		for (vector<nPhysD *>::iterator it=imagelist.begin(); it!=imagelist.end();) {
			if ((*it)->getSurf()>0) {
				addShowPhys((*it));
				it++;
			} else {
				delete (*it);
				it=imagelist.erase(it);
			}
		}
	}
	QApplication::processEvents();
	return imagelist;
}

void neutrino::saveSession (QString fname) {
	if (fname.isEmpty()) {
		QString fnameSave = QFileDialog::getSaveFileName(this,tr("Open Session"),property("fileOpen").toString(),tr("Session files")+QString(" (*.neus);;")+tr("Any files")+QString(" (*)"));
		if (!fnameSave.isEmpty()) {
			saveSession(fnameSave);
		}		
	} else {
		setProperty("fileOpen", fname);
		QString suffix=QFileInfo(fname).suffix().toLower();
		QList<nGenericPan *> pans=getPans();
		for(int k = 0; k < (pans.size()/2); k++) pans.swap(k,pans.size()-(1+k));
		
		QProgressDialog progress("Save session", "Cancel", 0, pans.size()+physList.size(), this);
		progress.setWindowModality(Qt::WindowModal);
		progress.show();
		
		ofstream ofile(fname.toUtf8().constData(), ios::out | ios::binary);
		ofile << "Neutrino " << __VER << " " << physList.size() << " " << getPans().size() << endl;
		
		for (int i=0;i<physList.size(); i++) {
			if (progress.wasCanceled()) break;
			progress.setValue(i);
			progress.setLabelText(QString::fromUtf8(physList.at(i)->getShortName().c_str()));
			QApplication::processEvents();
			ofile << "NeutrinoImage" << endl;
			phys_dump_binary(physList.at(i),ofile);
		}
		for (int i=0;i<pans.size(); i++) {
			if (progress.wasCanceled()) break;
			QString namePan=pans.at(i)->property("panName").toString();
			progress.setValue(physList.size()+i);
			progress.setLabelText(namePan);
			QApplication::processEvents();
			ofile << "NeutrinoPan-begin " << namePan.toStdString() << endl;
			QTemporaryFile tmpFile(pans.at(i)->panName);
			tmpFile.setAutoRemove(false);
			tmpFile.open();
			QSettings my_set(tmpFile.fileName(),QSettings::IniFormat);
			my_set.clear();
			pans.at(i)->saveSettings(&my_set);
			my_set.sync();
			tmpFile.close();
			QFile file(tmpFile.fileName());
			if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
				return;
			
			while (!file.atEnd()) {
				QByteArray line = file.readLine();
				ofile << line.constData();
			}
			file.close();
			file.remove();
			ofile << "NeutrinoPan-end " << namePan.toStdString() << endl;
		}
		ofile.close();
	}
}

vector <nPhysD *> neutrino::openSession (QString fname) {
	vector <nPhysD *> imagelist;
	if (!fname.isEmpty()) {
		updateRecentFileActions(fname);
		setProperty("fileOpen", fname);
		if (physList.size()!=0) {
			QThread *m_thread = new QThread();
			neutrino*my_neu= new neutrino();
			my_neu->moveToThread(m_thread);
			m_thread->start();
			my_neu->fileOpen(fname);
		} else {
			QProgressDialog progress("Load session", "Cancel", 0, 0, this);
			ifstream ifile(fname.toUtf8().constData(), ios::in | ios::binary);
			string line;
			getline(ifile,line);
			QString qLine=QString::fromStdString(line);
			int counter=0;
			if (qLine.startsWith("Neutrino")) {
				progress.setWindowModality(Qt::WindowModal);
				progress.setLabelText(tr("Load Session ")+qLine.split(" ").at(1));
				progress.setMaximum(1+qLine.split(" ").at(2).toInt()+qLine.split(" ").at(3).toInt());
				progress.show();
				QApplication::processEvents();
				while(ifile.peek()!=-1) {
					getline(ifile,line);
					QString qLine=QString::fromStdString(line);
					if (progress.wasCanceled()) break;
					if (qLine.startsWith("NeutrinoImage")) {
						counter++;
						progress.setValue(counter);
						nPhysD *my_phys=new nPhysD();
						
//						int ret=phys_resurrect_old_binary(my_phys,ifile);
//						if (ret>=0 && my_phys->getSurf()>0) {
//							addPhys(my_phys);
//							imagelist.push_back(my_phys);
//						} else {
//							int 	ret=phys_resurrect_binary(my_phys,ifile);
//							if (ret>=0 && my_phys->getSurf()>0) {
//								addPhys(my_phys);
//								imagelist.push_back(my_phys);
//							} else {
//								delete my_phys;
//							}
//						}

						int ret=phys_resurrect_binary(my_phys,ifile);
						if (ret>=0 && my_phys->getSurf()>0) {
							addPhys(my_phys);
							imagelist.push_back(my_phys);
						} else {
							delete my_phys;
						}
						progress.setLabelText(QString::fromUtf8(my_phys->getShortName().c_str()));
						QApplication::processEvents();
					} else if (qLine.startsWith("NeutrinoPan-begin")) {
                        DEBUG("NeutrinoPan detected >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
						QStringList listLine=qLine.split(" ");
						QString panName=listLine.at(1);
						counter++;
						progress.setValue(counter);
						progress.setLabelText(panName);
						QApplication::processEvents();
						if (metaObject()->indexOfMethod((panName+"()").toLatin1().constData())>0) {
							nGenericPan *my_pan=NULL;
							QMetaObject::invokeMethod(this,panName.toLatin1().constData(),Q_RETURN_ARG(nGenericPan*, my_pan));
							QApplication::processEvents();
							if (my_pan) {
								QTemporaryFile tmpFile("."+my_pan->panName);
								// tmpFile.setAutoRemove(false);
								tmpFile.open();
								while(!qLine.startsWith("NeutrinoPan-end") && !ifile.eof()) {
									getline(ifile,line);
									qLine=QString::fromStdString(line);
									line+="\n";
									if (!qLine.startsWith("NeutrinoPan-end")) tmpFile.write(line.c_str());
								}
								tmpFile.flush();
								my_pan->loadSettings(tmpFile.fileName());
								tmpFile.close(); // this should also remove it...
							}
						}
					}
					QApplication::processEvents();
				}
			}
			ifile.close();
		}
	}
	return imagelist;
}

void neutrino::addShowPhys(nPhysD* datamatrix) {
	addPhys(datamatrix);
	showPhys(datamatrix);
}

void neutrino::addPhys(nPhysD* datamatrix) {
	if (datamatrix && !physList.contains(datamatrix))	{
		physList << datamatrix;
		addMenuBuffers(datamatrix);
		emit physAdd(datamatrix);
	}
}


void neutrino::addMenuBuffers (nPhysD* datamatrix) {
	QAction *action=new QAction(this);
	QString name=QString::fromUtf8(datamatrix->getName().c_str());
	if (datamatrix->getType()==PHYS_FILE || datamatrix->getType()==PHYS_RFILE) {
		action->setText(QFileInfo(name).fileName()+QString(": ")+name);
	} else {
		action->setText(name);
	}
	action->setData(qVariantFromValue((void*) datamatrix));
	connect(action, SIGNAL(triggered()),this, SLOT(openRecentBuffer()));

	listabuffer << action;
	my_w.menuBuffers->addAction(action);
}

nPhysD* neutrino::replacePhys(nPhysD* newPhys, nPhysD* oldPhys, bool show) { //TODO: this should be done in nPhysImage...
	if (newPhys) {
		bool redisplay = (currentBuffer==oldPhys); 
		if (physList.contains(oldPhys)) {
			if (oldPhys==NULL) oldPhys=new nPhysD();
			*oldPhys=*newPhys;
			delete newPhys;
			newPhys=oldPhys;
		} else {
			addPhys(newPhys);
		}
		if (show || redisplay) showPhys(newPhys);
	}
	return newPhys;
}

void neutrino::removePhys(nPhysD* datamatrix) {
	if (datamatrix) {
		emit physDel(datamatrix);
		int position=physList.indexOf(datamatrix);
		if (position != -1) {
			physList.removeAll(datamatrix);

			if (physList.size()>0) {
				showPhys(physList.at(min(position,physList.size()-1)));
			} else {
				currentBuffer=NULL;
				emitBufferChanged();
				setWindowTitle(property("winId").toString()+QString(": Neutrino"));
				setWindowFilePath("");
				zoomChanged(1);
				my_pixitem.setPixmap(QPixmap(":icons/icon.png"));
				my_w.my_view->setSize();
			}

			QList<QAction *> lista=my_w.menuBuffers->actions();
			foreach (QAction* action, my_w.menuBuffers->actions()) {
				if (action->data() == qVariantFromValue((void*) datamatrix)) {
					my_w.menuBuffers->removeAction(action);
				}
			}

			foreach (QAction* action, listabuffer) {
				if (action->data() == qVariantFromValue((void*) datamatrix)) {
					listabuffer.removeAll(action);
				}
			}
			QApplication::processEvents();
			delete datamatrix;
			datamatrix=NULL;
		}
	}
}


void neutrino::showPhys(nPhysD& datamatrixRef) {
	bool found=false;
	foreach (nPhysD* datamatrix, physList) {	
		if (*datamatrix == datamatrixRef) found=true;
	}
	if (!found) {
		nPhysD *datamatrix =  new nPhysD;
		*datamatrix=datamatrixRef;
		showPhys(datamatrix);
	}
}

void neutrino::addShowPhys(nPhysD& datamatrixRef) {
	bool found=false;
	foreach (nPhysD* datamatrix, physList) {	
		if (*datamatrix == datamatrixRef) found=true;
	}
	if (!found) {
		nPhysD *datamatrix =  new nPhysD;
		*datamatrix=datamatrixRef;
		addShowPhys(datamatrix);
	}
}

void neutrino::addPhys(nPhysD& datamatrixRef) {
	bool found=false;
	foreach (nPhysD* datamatrix, physList) {	
		if (*datamatrix == datamatrixRef) found=true;
	}
	if (!found) {
		nPhysD *datamatrix =  new nPhysD;
		*datamatrix=datamatrixRef;
		addPhys(datamatrix);
	}
}

void neutrino::removePhys(nPhysD& datamatrixRef) {
	foreach (nPhysD* datamatrix, physList) {	
		if (*datamatrix == datamatrixRef) removePhys(datamatrix);
	}
}

void
neutrino::showPhys(nPhysD* datamatrix) {
	if (datamatrix) {
		if (!physList.contains(datamatrix)) addPhys(datamatrix);
		currentBuffer=datamatrix;
		if (!physList.contains(datamatrix)) {
			// TODO: add memory copy...
			physList << datamatrix;
		}
		
		emitBufferChanged();
		
		QString winName=QString::fromUtf8(datamatrix->getShortName().c_str());
		winName.prepend(property("winId").toString()+QString(":")+QString::number(physList.indexOf(datamatrix))+QString(" "));
		
		QString mypath=QString::fromUtf8(datamatrix->getFromName().c_str());
		winName.append(QString(" ")+mypath);
		setWindowTitle(winName);

		DEBUG(datamatrix->getType() << " : " << datamatrix->getFromName());
		if (datamatrix->getType()==PHYS_FILE || datamatrix->getType()==PHYS_RFILE) {
			setWindowFilePath(mypath);
		} else {
			setWindowFilePath("");
		}
		
		createQimage();
	} else {
		statusBar()->showMessage("Image not valid",2000);
	}
}


void
neutrino::createQimage() {
	if (currentBuffer) {
		double mini=colorMin;
		double maxi=colorMax;
		if (colorRelative) {
			mini=currentBuffer->Tminimum_value+colorMin*(currentBuffer->Tmaximum_value - currentBuffer->Tminimum_value);
			maxi=currentBuffer->Tmaximum_value-(1.0-colorMax)*(currentBuffer->Tmaximum_value - currentBuffer->Tminimum_value);
		}
//		DEBUG(">>>>>>>>>>>>>>>>>> " << mini << " " << maxi << " " << colorRelative);
		const unsigned char *buff=currentBuffer->to_uchar_palette(mini,maxi,nPalettes[colorTable]);
		const QImage tempImage(buff, currentBuffer->getW(), currentBuffer->getH(), 
							   currentBuffer->getW()*4, QImage::Format_ARGB32_Premultiplied);
		my_pixitem.setPixmap(QPixmap::fromImage(tempImage));
	}
	my_w.my_view->setSize();
}

// Export

void neutrino::exportGraphics () {
	QString fout = QFileDialog::getSaveFileName(this,tr("Save Drawing"),property("fileExport").toString(),"Available formats (*.svg, *.pdf, *.png);; Any files (*)");
	if (!fout.isEmpty()) {
		exportGraphics(fout);
	}
}

void neutrino::exportGraphics (QString fout) {
	setProperty("fileExport",fout);
	bool resetmouse=my_mouse.isVisible();
	my_mouse.setVisible(false);
	QSize my_size=QSize(my_s.width(), my_s.height());
	if (QFileInfo(fout).suffix().toLower()==QString("pdf")) {
		QPrinter printer(QPrinter::HighResolution);
		printer.setOutputFormat(QPrinter::PdfFormat);
		printer.setOutputFileName(fout);
		printer.setColorMode(QPrinter::Color);
		printer.setPaperSize(my_tics.boundingRect().size().toSize(),QPrinter::DevicePixel);
		printer.setOrientation(QPrinter::Portrait);
		printer.setCreator("Neutrino "+QString(__VER));
		printer.setDocName(windowTitle());
		QPainter my_painter( &printer );
		my_s.render(&my_painter);
		my_painter.end();
	} else if	(QFileInfo(fout).suffix().toLower()==QString("svg")) {
		QSvgGenerator svgGen;
		svgGen.setFileName(fout);
		svgGen.setSize(my_size);
		svgGen.setViewBox(QRect(-my_tics.boundingRect().left(),-my_tics.boundingRect().top(),my_tics.boundingRect().width(),my_tics.boundingRect().height()));
		svgGen.setTitle("Neutrino");
		svgGen.setDescription(windowFilePath());
		QPainter painter( &svgGen );		
		my_s.render(&painter);
	} else {
		QImage image(my_size,QImage::Format_ARGB32);
		image.fill(0);
		QPainter painter(&image);
		my_s.render(&painter);
		image.save(fout);
	}
	my_mouse.setVisible(resetmouse);
}

void neutrino::toggleMouse() {
	toggleMouse(!my_mouse.isVisible());
}

void neutrino::toggleMouse(bool stat) {
	my_mouse.setVisible(stat);
	QCursor cur;
	if (stat) {
		cur=QCursor(Qt::BlankCursor);
		my_w.actionShow_mouse->setText("Hide mouse");
	} else {
		cur=QCursor(Qt::CrossCursor);
		my_w.actionShow_mouse->setText("Show mouse");
	}
	my_pixitem.setCursor(cur);
}

void neutrino::toggleRuler() {
	toggleRuler(!my_tics.rulerVisible);
}

void neutrino::toggleRuler(bool stat) {
	my_tics.rulerVisible=stat;
	if (stat) {
		my_w.actionShow_ruler->setText("Hide ruler");
	} else {
		my_w.actionShow_ruler->setText("Show ruler");
	}
	my_tics.update();
}

void neutrino::toggleGrid() {
	toggleGrid(!my_tics.gridVisible);
}

void neutrino::toggleGrid(bool stat) {
	my_tics.gridVisible=stat;
	if (stat) {
		my_w.actionShow_grid->setText("Hide grid");
	} else {
		my_w.actionShow_grid->setText("Show grid");
	}
	my_tics.update();
}

void neutrino::closeEvent (QCloseEvent *e) {
	disconnect(my_w.my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(mouseposition(QPointF)));
	if (fileClose()) {
		foreach (nGenericPan* pan, getPans()) {
			pan->close();
		}
		e->accept();
	} else {
		e->ignore();
	}
	connect(my_w.my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(mouseposition(QPointF)));
}

// keyevents: pass to my_view!
void neutrino::keyPressEvent (QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_Question:
			Shortcuts();
			break;
		case Qt::Key_Plus:
			zoomIn();
			break;
		case Qt::Key_Minus:
			zoomOut();
			break;
		case Qt::Key_Equal:
			zoomEq();
			break;
		case Qt::Key_A: {
			if (currentBuffer) {
				if (e->modifiers() & Qt::ShiftModifier) {
					colorRelative=!colorRelative;
				}
				if (colorRelative) {
					colorMin=0.0;
					colorMax=1.0;
				} else {
					colorMin=currentBuffer->Tminimum_value;
					colorMax=currentBuffer->Tmaximum_value;
				}
				createQimage();
				emit updatecolorbar();
			}
			break;
		}
		case Qt::Key_O:
			if (e->modifiers() & Qt::ShiftModifier) {
				foreach (nPhysD* phys, physList) {
					phys->set_origin(my_mouse.pos().x(),my_mouse.pos().y());
					emit bufferChanged(phys);
				}
			} else {
				if (currentBuffer) currentBuffer->set_origin(my_mouse.pos().x(),my_mouse.pos().y());
			}
			mouseposition(my_mouse.pos());
			my_tics.update();
			emitBufferChanged();
			break;
		case Qt::Key_C:
			if (e->modifiers() & Qt::ShiftModifier) {
				Colorbar();
			} else {
				std::swap(colorMin,colorMax);
				createQimage();
				emit updatecolorbar();			
			}
			break;
		case Qt::Key_I:
			if (e->modifiers() & Qt::ShiftModifier) {
				MouseInfo();
			} else {
				WinList();
			}
			break;
		case Qt::Key_S:
			if (e->modifiers() & Qt::ShiftModifier) {
				if (currentBuffer) {
					currentBuffer->TscanBrightness();
					createQimage();					
				}
			}
			break;
		case Qt::Key_P:
			if (e->modifiers() & Qt::ShiftModifier) {
				Properties();
			}
			break;
		case Qt::Key_R:
			if (!(e->modifiers() & Qt::ShiftModifier))
				toggleRuler();
			break;
		case Qt::Key_G:
			if (!(e->modifiers() & Qt::ShiftModifier))
				toggleGrid();
			break;
		case Qt::Key_M: {
			if (!(e->modifiers() & Qt::ShiftModifier))
				toggleMouse();
			break;
		}
		case Qt::Key_V: {
			if (!(e->modifiers() & Qt::ShiftModifier))
				Vlineout();
			break;
		}
		case Qt::Key_H: {
			if (!(e->modifiers() & Qt::ShiftModifier))
				Hlineout();
			break;
		}
		default:
			break;
	}
	if (follower) {
		follower->keyPressEvent(e);
	}
}

void neutrino::keyReleaseEvent (QKeyEvent *e)
{
	if (follower) follower->keyReleaseEvent(e);
}

// Drag and Drop
void neutrino::dragEnterEvent(QDragEnterEvent *e)
{
	statusBar()->showMessage(tr("Drop content"), 2000);
	e->acceptProposedAction();
}

void neutrino::dragMoveEvent(QDragMoveEvent *e)
{
	e->accept();
}

void neutrino::dropEvent(QDropEvent *e) {
	if (e->mimeData()->hasFormat("data/neutrino")) {
		nPhysD *my_phys=(nPhysD *) e->mimeData()->data("data/neutrino").toLong();
		DEBUG("DROP pointer: " << (void*)my_phys);
		if (my_phys) {
			if (physList.contains(my_phys)) {
				showPhys(my_phys);
			} else {
				nPhysD *copyhere;
				copyhere = new nPhysD(*my_phys);
				addShowPhys(copyhere);
			}
		}
	}	else if (e->mimeData()->hasUrls()) {
		QStringList fileList;
		foreach (QUrl qurl, e->mimeData()->urls()) {
			fileList << qurl.toLocalFile();
		}
		if (fileList.size()>0) {
			fileOpen(fileList);
		}
	}
	e->acceptProposedAction();
}

// switch buffers
void neutrino::actionPrevBuffer() {
	int position=physList.indexOf(currentBuffer);
	if (position>-1) showPhys(physList.at((position+physList.size()-1)%physList.size()));
}

void neutrino::actionNextBuffer() {
	int position=physList.indexOf(currentBuffer);
	if (position>-1) showPhys(physList.at((position+1)%physList.size()));
}

// zoom
void
neutrino::zoomChanged(double zoom) {
	QString tmp;
	tmp.sprintf(" %.1f%%",100.0*zoom);
	statusBar()->showMessage(tr("Zoom :")+tmp,2000);
	emit nZoom(zoom);
	update();
}

void
neutrino::zoomIn() {
	my_w.my_view->incrzoom(1.05);
}

void
neutrino::zoomOut() {
	my_w.my_view->incrzoom(1.0/1.05);
}

double
neutrino::getZoom() const {
	return my_w.my_view->transform().m11();
}

void
neutrino::zoomEq() {
	my_w.my_view->zoomEq();
}

void
neutrino::mouseposition(QPointF pos_mouse) {
	my_sbarra.pos_x->setNum((int)pos_mouse.x());
	my_sbarra.pos_y->setNum((int)pos_mouse.y());

	
	if (currentBuffer) {
		vec2f vec=currentBuffer->to_real(vec2f(pos_mouse.x(),pos_mouse.y()));
		QPointF pos=QPointF(vec.x(),vec.y());
		my_sbarra.dx->setNum(pos.x());
		my_sbarra.dy->setNum(pos.y());
		double val=currentBuffer->point(pos_mouse.x(),pos_mouse.y());
		my_sbarra.pos_z->setNum(val);
		emit colorValue(val);
		emit mouseAtWorld(pos);
	} else {
		my_sbarra.dx->setText("");
		my_sbarra.dy->setText("");
	}
	
	emit mouseAtMatrix(pos_mouse);
}

QString neutrino::getFileSave() {
	return QFileDialog::getSaveFileName(this, "Save to...",property("fileOpen").toString(),"neutrino (*.txt *.neu *.neus *.tif *.tiff *.hdf *.h5 *.fits);; Any files (*)");
}

void
neutrino::fileSave() {
	fileSave(getFileSave());
}

void neutrino::fileSave(nPhysD *phys) {
	fileSave(phys,getFileSave());
}

void neutrino::fileSave(QString fname) {
	if (!fname.isEmpty()) {
		setProperty("fileOpen", fname);
		QString suffix=QFileInfo(fname).suffix().toLower();
		if (suffix.startsWith("neus")) {
			saveSession(fname);
		} else {
			fileSave(currentBuffer,fname);
		}
	}
}

void neutrino::fileSave(nPhysD* phys, QString fname) {
	if (phys) {
		QString suffix=QFileInfo(fname).suffix().toLower();
		int ret=-1;
		if (suffix.startsWith("neu")) {
			ret = phys_dump_binary(phys,fname.toUtf8().constData());
		} else if (suffix.startsWith("tif")) {
			ret = phys_write_tiff(phys,fname.toUtf8().constData());
		} else if (suffix.startsWith("fit")) {
			ret = phys_write_fits(phys,("!"+fname).toUtf8().constData(),4);
#ifdef __phys_HDF
		} else if (suffix.startsWith("hdf")) {
			ret = phys_write_HDF4(phys,fname.toUtf8().constData());
		} else if (suffix.startsWith("h5")) {
			ret = phys_write_HDF5(phys,fname.toUtf8().constData());
#endif
		} else {
			ret = phys->writeASC(fname.toUtf8().constData());
		}
		if (ret==0) {
			phys->setType(PHYS_FILE);
			phys->setShortName(QFileInfo(fname).fileName().toStdString());
			phys->setName(phys->getShortName());
			phys->setFromName(fname.toStdString());
		}
	}
}

bool
neutrino::fileClose() {
	if (QApplication::activeWindow() == this) {
		bool askAll=true;
		foreach (nGenericPan* pan, getPans()) {
			pan->hide();
			pan->close();
			pan->deleteLater();
			QApplication::processEvents();
		}	
		foreach (nPhysD *phys, physList) {
			if (askAll && phys->getType()==	PHYS_DYN) {
				int res=QMessageBox::warning(this,tr("Attention"),
											 tr("The image")+QString("\n")+QString::fromUtf8(phys->getName().c_str())+QString("\n")+tr("has not been saved. Do you vant to save it now?"),
											 QMessageBox::Yes | QMessageBox::No  | QMessageBox::NoToAll | QMessageBox::Cancel);
				switch (res) {
					case QMessageBox::Yes:
						fileSave(phys); // TODO: add here a check for a cancel to avoid exiting
						break;
					case QMessageBox::NoToAll:
						askAll=false;
						break;
					case QMessageBox::Cancel:
						return false;
						break;
				}
			}
		}
		deleteLater();
		return true;
	} else {
		nGenericPan *pan=qobject_cast<nGenericPan *>(QApplication::activeWindow());
		if (pan) pan->close();
		return false;
	}
}

void
neutrino::closeCurrentBuffer() {
	if (currentBuffer)  {
		if (currentBuffer->getType()==PHYS_DYN) {
			int res=QMessageBox::warning(this,tr("Attention"),
										 tr("The image")+QString("\n")+QString::fromUtf8(currentBuffer->getName().c_str())+QString("\n")+tr("has not been saved. Do you vant to save it now?"),
										 QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
			switch (res) {
				case QMessageBox::Yes:
					fileSave(currentBuffer); // TODO: add here a check for a cancel to avoid exiting
					break;
				case QMessageBox::No:
					removePhys(currentBuffer);
					break;
			}
		} else {
			removePhys(currentBuffer);			
		}
	}
}

nGenericPan*
neutrino::Shortcuts() {
	QString vwinname=tr("Shortcuts");
	nGenericPan* win=existsPan(vwinname,true);
	if (!win) win=new nShortcuts(this,vwinname);
	return win;	
}

nGenericPan*
neutrino::FocalSpot() {
	QString vwinname=tr("FocalSpot");
	nGenericPan* win=existsPan(vwinname,true);
	if (!win) win=new nFocalSpot(this,vwinname);
	return win;
}

nGenericPan*
neutrino::MathOperations() {
	QString vwinname=tr("MathOperations");
	nGenericPan* win=existsPan(vwinname,true);
	if (!win) win=new nOperator(this,vwinname);
	return win;
}

nGenericPan*
neutrino::CutoffImage() {
	QString vwinname=tr("CutoffImage");
	nGenericPan* win=existsPan(vwinname,true);
	if (!win) win=new nCutoffMask(this,vwinname);
	return win;
}

// Window List pan
nGenericPan*
neutrino::WinList() {
	QString namepad=tr("WinList");
	nGenericPan *win = existsPan(namepad,true);
	if (!win) win = new nWinList (this, namepad);
	return win;
}

nGenericPan*
neutrino::Properties() {
	QString namepad=tr("Properties");
	nGenericPan *win = existsPan(namepad,true);
	if (!win) win = new nPhysProperties (this, namepad);
	return win;
}

nGenericPan*
neutrino::MouseInfo() {
	QString namepad=tr("MouseInfo");
	nGenericPan *win = existsPan(namepad,true);
	if (!win) win = new nMouseInfo (this, namepad);
	return win;
}

// colortables

nGenericPan*
neutrino::Colorbar() {
	QString namepad=tr("Colorbar");
	nGenericPan *win = existsPan(namepad,true);
	if (!win) win = new nColorBarWin (this, namepad);
	return win;
}

struct QPairFirstComparer {
    template<typename T1, typename T2>
    bool operator()(const QPair<T1,T2> & a, const QPair<T1,T2> & b) const {
        return a.first <= b.first;
    }
};

bool neutrino::addPaletteFromString(QString paletteName, QString paletteStr) {
	qDebug() << "addPaletteFromString" << paletteStr;
	if (paletteStr.contains(",")) {
		QList<QPair<double,QColor> > listDoubleColor;
		QStringList paletteList=paletteStr.split(",",QString::SkipEmptyParts);
		for (int i=0;i<paletteList.size();i++) {
			QStringList colorValueName=paletteList.at(i).split(" ",QString::SkipEmptyParts);
			if (colorValueName.size()==2) {
				bool ok;
				double my_val=colorValueName.first().toDouble(&ok);
				QColor my_color(colorValueName.last());
				if (ok && my_color.isValid()) {
					listDoubleColor.append(qMakePair(my_val,my_color));
				}
			}
		}
		qSort(listDoubleColor.begin(), listDoubleColor.end(), QPairFirstComparer());	
		if (listDoubleColor.size()>=2) {
			double minVal=listDoubleColor.first().first;
			double maxVal=listDoubleColor.last().first;
			if (minVal!=maxVal) {
				for(int i=0;i<listDoubleColor.size();i++) {
					listDoubleColor[i]=qMakePair(256.0*(listDoubleColor.at(i).first-minVal)/(maxVal-minVal),listDoubleColor.at(i).second);
				}
				
				unsigned char *palC = new unsigned char [768];
				
				int counter=1;
				for (int i=0;i<256;i++) {
					
					QColor col1=listDoubleColor.at(counter-1).second;
					QColor col2=listDoubleColor.at(counter).second;
					
					double delta=(listDoubleColor.at(counter).first-i)/(listDoubleColor.at(counter).first-listDoubleColor.at(counter-1).first); 
					
					palC[i*3+0]=(unsigned char) (delta*col1.red()+(1.0-delta)*col2.red());
					palC[i*3+1]=(unsigned char) (delta*col1.green()+(1.0-delta)*col2.green());
					palC[i*3+2]=(unsigned char) (delta*col1.blue()+(1.0-delta)*col2.blue());

					while (i+1>listDoubleColor.at(counter).first) counter++;
				}
				if (nPalettes.contains(paletteName)) {
					delete nPalettes[paletteName];
				}				
				nPalettes[paletteName] = palC;
				changeColorTable(paletteName);
				return true;
			}
		}
	}
	return false;
}

QString neutrino::addPaletteFromFile(QString paletteFile) {
	QFile file(paletteFile);
	QString paletteName=QFileInfo(paletteFile).baseName();
	if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		unsigned char *palette = new unsigned char[256*3]();
		bool allOk=true;
		int i=0;
		while (!file.atEnd() && allOk) {
			QString line = QString(file.readLine()).trimmed();
			if (line.startsWith("#")) {
				paletteName=line.remove(0,1).trimmed();
			} else {
				QStringList colorsToSplit=line.split(QRegExp("\\s+"),QString::SkipEmptyParts);
				if (colorsToSplit.size()==3) {
					bool ok0,ok1,ok2;
					unsigned int redInt=colorsToSplit.at(0).toUInt(&ok0);
					unsigned int greenInt=colorsToSplit.at(1).toUInt(&ok1);
					unsigned int blueInt=colorsToSplit.at(2).toUInt(&ok2);
					if (ok0 && ok1 && ok2 && redInt<256 && greenInt<256 && blueInt<256) {
						if (i<256) {
							palette[3*i+0]=redInt;
							palette[3*i+1]=greenInt;
							palette[3*i+2]=blueInt;
							i++;
						} else {
							allOk=false;
						}
					}
				}
			}
		}
		if (allOk) {
			QSettings my_set("neutrino","");
			my_set.beginGroup("Palettes");
			QStringList paletteFiles=my_set.value("paletteFiles","").toStringList();
			paletteFiles << paletteFile;
			my_set.setValue("paletteFiles",paletteFiles);
			QStringList paletteFilesNames=my_set.value("paletteFilesNames","").toStringList();
			paletteFilesNames << paletteName;
			my_set.setValue("paletteFilesNames",paletteFilesNames);
			my_set.endGroup();

			nPalettes[paletteName] = palette;
			changeColorTable(paletteName);
		} else {
			delete palette;
			paletteName.clear();
		}
	}
	return paletteName;
}

void
neutrino::changeColorTable (QString ctname) {
	if (nPalettes.contains(ctname)) {
		colorTable=ctname;
	} else {
		colorTable=nPalettes.keys().first();
	}
	changeColorTable();
}


void
neutrino::changeColorTable () {
	createQimage();
	statusBar()->showMessage(colorTable, 1500);
	my_tics.update();
	emit updatecolorbar();
}

void
neutrino::changeColorMinMax (double mini, double maxi) {
	if (colorRelative) {
		colorMin = (mini-currentBuffer->Tminimum_value)/(currentBuffer->Tmaximum_value - currentBuffer->Tminimum_value);
		colorMax = 1.0-(currentBuffer->Tmaximum_value-maxi)/(currentBuffer->Tmaximum_value - currentBuffer->Tminimum_value);
	} else {
		colorMin=mini;
		colorMax=maxi;
	}
	createQimage();
	emit updatecolorbar();	
}

void
neutrino::previousColorTable () {
	int indice=nPalettes.keys().indexOf(colorTable);
	if (indice>0) {
		colorTable=nPalettes.keys().at(indice-1);
	} else {
		colorTable=nPalettes.keys().last();
	}
	changeColorTable ();
};

void
neutrino::nextColorTable () {
	int indice=nPalettes.keys().indexOf(colorTable);
	if (indice<nPalettes.keys().size()-1) {
		colorTable=nPalettes.keys().at(indice+1);
	} else {
		colorTable=nPalettes.keys().first();
	}
	changeColorTable ();
};

// testing
void
neutrino::createDrawLine() {
	my_w.menuPaths->setDefaultAction(my_w.actionLine);
	statusBar()->showMessage(tr("Click for points, press Esc to finish"),5000);
	nLine *item = new nLine(this);
	item->interactive();
	if (follower) follower->createDrawLine();
	my_w.actionPaths->setIcon(my_w.menuPaths->defaultAction()->icon());
	QSettings("neutrino","").setValue("defualtActionPath",my_w.menuPaths->defaultAction()->text());
}

QString
neutrino::newRect(QRectF rectangle) {
	nRect *item = new nRect(this);
	item->setRect(rectangle);
	return item->toolTip();
}

void
neutrino::newRect(QRectF rectangle, QString name) {
	nRect *item = new nRect(this);
	item->setRect(rectangle);
	item->setToolTip(name);
}

void
neutrino::createDrawRect() {
	my_w.menuPaths->setDefaultAction(my_w.actionRect);
	statusBar()->showMessage(tr("Click for the first point of the rectangle"),5000);
	nRect *item = new nRect(this);
	item->interactive();
	if (follower) follower->createDrawRect();
	my_w.actionPaths->setIcon(my_w.menuPaths->defaultAction()->icon());
	QSettings("neutrino","").setValue("defualtActionPath",my_w.menuPaths->defaultAction()->text());
}

void
neutrino::createDrawPoint() {
	my_w.menuPaths->setDefaultAction(my_w.actionPoint);
	statusBar()->showMessage(tr("Click for the point"),5000);
	nPoint *item = new nPoint(this);
	item->interactive();
	if (follower) follower->createDrawPoint();
	my_w.actionPaths->setIcon(my_w.menuPaths->defaultAction()->icon());
	QSettings("neutrino","").setValue("defualtActionPath",my_w.menuPaths->defaultAction()->text());
}

void
neutrino::createDrawEllipse() {
	my_w.menuPaths->setDefaultAction(my_w.actionEllipse);
	statusBar()->showMessage(tr("Click and release to create the ellipse"),5000);
	nEllipse *item = new nEllipse(this);
	item->interactive();
	if (follower) follower->createDrawEllipse();
	my_w.actionPaths->setIcon(my_w.menuPaths->defaultAction()->icon());
	QSettings("neutrino","").setValue("defualtActionPath",my_w.menuPaths->defaultAction()->text());
}

nGenericPan*
neutrino::Hlineout() {
	QString namepad=tr("Hlineout");
	nGenericPan *win = existsPan(namepad,true);
	if (!win) win = new nLineout(this, namepad, PHYS_X);
	return win;
}

nGenericPan*
neutrino::Vlineout() {
	QString namepad=tr("Vlineout");
	nGenericPan *win = existsPan(namepad,true);
	if (!win) win = new nLineout(this, namepad, PHYS_Y);
	return win;
}

nGenericPan*
neutrino::bothLineout() {
	QString namepad=tr("Bothlineout");
	nGenericPan *win = existsPan(namepad,true);
	if (!win) win = new nLineoutBoth(this, namepad);
	return win;
}

void neutrino::print()
{
	QPrinter printer(QPrinter::HighResolution);
	QPrintDialog *printDialog = new QPrintDialog(&printer, this);
	if (printDialog->exec() == QDialog::Accepted) {
		my_mouse.setVisible(false);
		QPainter painter(&printer);
		foreach (QGraphicsItem *oggetto, my_s.items() ) {
			if (qgraphicsitem_cast<nLine *>(oggetto)) {
				nLine *my_nline = (nLine *)oggetto;
				my_nline->selectThis(false);
			}
		}
		my_s.render(&painter);
		my_mouse.setVisible(true);
	}
}

#ifdef __phys_HDF
/// HDF5 treeview
nGenericPan*
neutrino::openHDF5() {
	QString namepad=tr("HDF5");
	nGenericPan *win = existsPan(namepad,true);
	if (!win) win = new nHDF5(this, namepad);
	return win;
}
#endif


/// rectangle lineout
nGenericPan*
neutrino::BoxLineout() {
	return new nBoxLineout(this, "BoxLineout");
}

/// Find peaks
nGenericPan*
neutrino::FindPeaks() {
	return new nFindPeaks(this, "FindPeaks");
}

/// compare lines between images
nGenericPan*
neutrino::CompareLines() {
	return new nCompareLines(this, "CompareLines");
}

/// Open raw window
nGenericPan*
neutrino::openRAW() {
	QStringList fnames;
	nGenericPan *win = NULL;
	fnames = QFileDialog::getOpenFileNames(this,tr("Open RAW"),NULL,tr("Any files")+QString(" (*)"));
	if (fnames.size()) {
		QString vwinname="OpenRaw";
		win=existsPan(vwinname,true);
		if (!win) win= new nOpenRAW(this, vwinname);
		nOpenRAW *winRAW=qobject_cast<nOpenRAW*>(win);
		if (winRAW) winRAW->add(fnames);
	}
	return win;
}

/// Spectral Analysis (FT, filtering and stuff)
nGenericPan*
neutrino::SpectralAnalysis() {
	QString namepad=tr("SpectralAnalysis");
	nGenericPan *win = existsPan(namepad,true);
	if (!win) win = new nSpectralAnalysis(this, namepad);
	return win;
}

nGenericPan*
neutrino::Visar() {
	QString vwinname=tr("Visar");
	return new nVisar(this, vwinname);
}

/// Wavelet analysis window
nGenericPan*
neutrino::Wavelet() {
	QString vwinname=tr("Wavelet");
	nGenericPan *ret=existsPan(vwinname,true);
	if (!ret) ret = new nWavelet(this, vwinname);
	return ret;
}

/// Integral inversion (Abel etc...)
nGenericPan*
neutrino::Inversions() {
	QString vwinname=tr("Inversions");
	nGenericPan *ret=existsPan(vwinname,true);
	if (!ret) ret = new nIntegralInversion(this, vwinname);
	return ret;
}

/// Region Path
nGenericPan*
neutrino::RegionPath() {
	QString vwinname=tr("RegionPath");
	return new nRegionPath(this, "RegionPath");
}


/// ROTATE STUFF
nGenericPan*
neutrino::Rotate() {
	QString vwinname=tr("Rotate");
	nGenericPan *ret=existsPan(vwinname,true);
	if (!ret) ret = new nRotate(this, vwinname);
	return ret;
}

/// Affine STUFF
nGenericPan*
neutrino::Affine() {
	QString vwinname=tr("Affine");
	nGenericPan *ret=existsPan(vwinname,true);
	if (!ret) ret = new nAffine(this, vwinname);
	return ret;
}

/// Blur STUFF
nGenericPan*
neutrino::Blur() {
	QString vwinname=tr("Blur");
	nGenericPan *ret=existsPan(vwinname,true);
	if (!ret) ret = new nBlur(this, vwinname);
	return ret;
}

// FOLLOWER
void
neutrino::createFollower() {
	follower = new neutrino ();
	follower->toggleMouse(true);
}

// MONIOR DIRECTORY
nGenericPan*
neutrino::Monitor() {
	QString vwinname=tr("Monitor");
	nGenericPan *ret=existsPan(vwinname,true);
	if (!ret) ret = new nMonitor(this, vwinname);
	return ret;
}

//save and load across restar
void neutrino::saveDefaults(){
	QSettings my_set("neutrino","");
	my_set.beginGroup("Preferences");
	my_set.setValue("geometry", pos());
	my_set.setValue("mouseVisible", my_mouse.isVisible());
	my_set.setValue("mouseColor", my_mouse.color);
	my_set.setValue("rulerVisible", my_tics.rulerVisible);
	my_set.setValue("gridVisible", my_tics.gridVisible);
	my_set.setValue("rulerColor", my_tics.rulerColor);
	my_set.setValue("colorTable", colorTable);
	my_set.setValue("fileExport", property("fileExport"));
	my_set.setValue("fileOpen", property("fileOpen"));
	my_set.endGroup();
}

void neutrino::loadDefaults(){
	QSettings my_set("neutrino","");
	my_set.beginGroup("Preferences");
	move(my_set.value("geometry",pos()).toPoint());
	toggleMouse(my_set.value("mouseVisible",my_mouse.isVisible()).toBool());
	my_mouse.color=my_set.value("mouseColor",my_mouse.color).value<QColor>();
	my_tics.rulerVisible=my_set.value("rulerVisible",my_tics.rulerVisible).toBool();
	my_tics.gridVisible=my_set.value("gridVisible",my_tics.gridVisible).toBool();
	my_tics.rulerColor=my_set.value("rulerColor",my_tics.rulerColor).value<QColor>();
	changeColorTable(my_set.value("colorTable",colorTable).toString());
	QVariant variant=my_set.value("comboIconSizeDefault");
	if (variant.isValid()) {
		int val=my_set.value("comboIconSizeDefault", my_w.toolBar->iconSize()).toInt();
		my_set.setValue("comboIconSizeDefault",val);
		if (val>=0) {
			QSize mysize=QSize(10*(val+1),10*(val+1));
			foreach (QToolBar *obj, findChildren<QToolBar *>()) {
				if (obj->iconSize()!=mysize) {
					obj->hide();
					obj->setIconSize(mysize);
					obj->show();
				}
			}
		}
	}
	setProperty("fileExport", my_set.value("fileExport", "Untitled.pdf"));
	setProperty("fileOpen", my_set.value("fileOpen",""));
	my_set.endGroup();
}

nGenericPan*
neutrino::Preferences() {
	QString vwinname=tr("Preferences");
	nGenericPan *ret=existsPan(vwinname,true);
	if (!ret) ret = new nPreferences(this, vwinname);
	return ret;
}

void neutrino::about() {
	QMessageBox credits(QString("About"),QString("neutrino"),
						QMessageBox::NoIcon,
						QMessageBox::Close,
						QMessageBox::NoButton,
						QMessageBox::NoButton,
						this);
	
	credits.setText(QString("<h1>Neutrino</h1><br><i>the only neutrino faster than light</i><br>")+tr("version: ")+__VER);
	credits.setInformativeText("Alessandro Flacco: alessandro.flacco@polytechnique.edu <br>Tommaso Vinci: tommaso.vinci@polytechnique.edu<hr>http://web.luli.polytechnique.fr/Neutrino<hr>");
	QPixmap icona(":icons/icon.png");
	credits.setIconPixmap(icona.scaledToHeight(100,Qt::SmoothTransformation));
	credits.exec();
}

QList<QList<qreal> > neutrino::getData(int num) {
	QList<QList<qreal> > myListList;
	nPhysD *phys=getBuffer(num);
	if (phys) {
		for (size_t i=0; i<phys->getH(); i++) {
			QList<qreal> myList;
			for (size_t j=0; j<phys->getW(); j++) {
				myList.append(phys->point(j,i));
			}
			myListList.append(myList);
		}
	}
	return myListList;
}

bool neutrino::setData(QList<QList<qreal> >myListList,int num) {
	size_t sizeH = myListList.size();
	if (sizeH > 0) {
		size_t sizeW = myListList.first().size();
		nPhysD *newPhys=new nPhysD(sizeW,sizeH,0.0,"Python");
		for (size_t i=0; i<sizeH; i++) {
			if (myListList.at(i).size()!=int(sizeW)) {
				delete newPhys;
				return false;
			}
			for (size_t j=0; j<sizeW; j++) {
				newPhys->set(j,i,myListList.at(i).at(j));
			}
		}
        newPhys->TscanBrightness();
		if (num>=0 && num<physList.size()) {			
			replacePhys(newPhys, physList.at(num), true);
		} else {
			addShowPhys(newPhys);
		}
		return true;
	} else {
		return false;
	}
}

nGenericPan* neutrino::openPan(QString panName) {
    nGenericPan *my_pan=NULL;
    QMetaObject::invokeMethod(this,panName.toLatin1().constData(),Q_RETURN_ARG(nGenericPan*, my_pan));
    return my_pan;
}

nLine* neutrino::line(QString name) {
	foreach (QObject* widget, children()) {
		nLine *linea=qobject_cast<nLine *>(widget);
		if (linea && linea->my_w.name->text() == name) {
			return linea;
		}
	}
	return NULL;
}

