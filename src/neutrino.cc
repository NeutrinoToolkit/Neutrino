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

#include <QVector>
#include <QList>

#include "neutrino.h"
#include "nApp.h"

#include <QMetaObject>
#include <QtSvg>

#include <QtUiTools>

#include <QPrintDialog>

#include "nView.h"

#include "nColorBar.h"

#include "nMouseInfo.h"
#include "nZoomWin.h"
#include "nPluginLoader.h"
#include "nPanPlug.h"
#include "nBoxLineout.h"
#include "nCompareLines.h"
#include "nWavelet.h"
#include "nSpectralAnalysis.h"
#include "nIntegralInversion.h"
#include "nRotate.h"
#include "nRegionPath.h"
#include "nInterpolatePath.h"
#include "nShortcuts.h"
#include "nAffine.h"

#include "nCamera.h"

#include "nFocalSpot.h"
#include "nContours.h"
#include "nLineout.h"
#include "nLineoutBoth.h"

#include "nOperator.h"
#include "nCutoffMask.h"

#include "nMonitor.h"


#include "nPreferences.h"
#include "nWinList.h"
#include "nPhysProperties.h"

#include "nPhysFormats.h"

#include "nOpenRAW.h"

#include "nPhysFormats.h"

#include "ui_nSbarra.h"
#include "ui_nAbout.h"


void neutrino::changeEvent(QEvent *e)
{
	QWidget::changeEvent(e);
	switch (e->type()) {
		case QEvent::LanguageChange: {
				for(auto& pan: getPanList())
					for(int i =  0; i < pan->metaObject()->methodCount(); ++i) {
						if (pan->metaObject()->method(i).methodSignature() == "retranslateUi(QMainWindow*)") {
							qDebug() << "found retranslateUi";
							QMetaObject::invokeMethod(pan,"retranslateUi",Q_ARG(QMainWindow *,pan));
						}
					}
				my_w->retranslateUi(this);
				break;
			}
		default:
			break;
	}
}


neutrino::~neutrino()
{
}

/// Creator
neutrino::neutrino():
	my_w(new Ui::neutrino),
	my_sbarra(new Ui::nSbarra),
	my_about(new Ui::nAbout)
{
	my_w->setupUi(this);
	setAcceptDrops(true);

	connect(qApp,SIGNAL(aboutToQuit()),this,SLOT(saveDefaults()));

	int numwin=qApp->property("numWin").toInt()+1;
	qApp->setProperty("numWin",numwin);
	setProperty("winId",numwin);

	setProperty("neuSave-gamma",1);
    setProperty("neuSave-physNameLength",40);

	setWindowTitle(QString::number(numwin)+QString(": Neutrino"));

	QString menuTransformationDefault=QSettings("neutrino","").value("menuTransformationDefault", "").toString();
	foreach (QAction * act, my_w->menuTransformation->actions()) {
		if (!menuTransformationDefault.isEmpty()) {
			if (act->text()==menuTransformationDefault) {
				my_w->menuTransformation->setDefaultAction(act);
				my_w->actionFlipRotate->setIcon(my_w->menuTransformation->defaultAction()->icon());
			}
		} else if (act->icon().cacheKey()==my_w->actionFlipRotate->icon().cacheKey()) {
			my_w->menuTransformation->setDefaultAction(act);
		}
	}
	connect(my_w->actionFlipRotate, SIGNAL(triggered()), this, SLOT(menuFlipRotate()));


	QString defualtActionPath=QSettings("neutrino","").value("defualtActionPath", "Rectangle").toString();

	foreach (QAction * act, my_w->menuPaths->actions()) {
		if (!defualtActionPath.isEmpty()) {
			if (act->text()==defualtActionPath) {
				my_w->menuPaths->setDefaultAction(act);
				my_w->actionPaths->setIcon(my_w->menuPaths->defaultAction()->icon());
			}
		} else if (act->icon().cacheKey()==my_w->actionPaths->icon().cacheKey()) {
			my_w->menuPaths->setDefaultAction(act);
		}
	}
	connect(my_w->actionPaths, SIGNAL(triggered()), this, SLOT(menuPaths()));


	connect(my_w->actionWinlist, SIGNAL(triggered()), this, SLOT(WinList()));
	connect(my_w->actionColors, SIGNAL(triggered()), this, SLOT(ColorBar()));

	connect(my_w->actionMouseInfo, SIGNAL(triggered()), this, SLOT(MouseInfo()));
	connect(my_w->actionOperator, SIGNAL(triggered()), this, SLOT(MathOperations()));

	connect(my_w->actionCamera, SIGNAL(triggered()), this, SLOT(Camera()));

	connect(my_w->actionLine, SIGNAL(triggered()), this, SLOT(createDrawLine()));
	connect(my_w->actionRect, SIGNAL(triggered()), this, SLOT(createDrawRect()));
	connect(my_w->actionPoint, SIGNAL(triggered()), this, SLOT(createDrawPoint()));
	connect(my_w->actionEllipse, SIGNAL(triggered()), this, SLOT(createDrawEllipse()));
	connect(my_w->actionLineoutH, SIGNAL(triggered()), this, SLOT(Hlineout()));
	connect(my_w->actionLineoutV, SIGNAL(triggered()), this, SLOT(Vlineout()));

	connect(my_w->actionNew, SIGNAL(triggered()), this, SLOT(fileNew()));
	connect(my_w->actionOpen, SIGNAL(triggered()), this, SLOT(fileOpen()));
	connect(my_w->actionOpen_RAW, SIGNAL(triggered()), this, SLOT(openRAW()));
	connect(my_w->actionSave, SIGNAL(triggered()), this, SLOT(fileSave()));

	connect(my_w->actionMonitor_Directory, SIGNAL(triggered()), this, SLOT(Monitor()));



	connect(my_w->actionReopen_to_saved, SIGNAL(triggered()), this, SLOT(fileReopen()));

	connect(my_w->actionSave_Session, SIGNAL(triggered()), this, SLOT(saveSession()));

	connect(my_w->actionExport, SIGNAL(triggered()), this, SLOT(exportGraphics()));
	connect(my_w->actionExport_all, SIGNAL(triggered()), this, SLOT(exportAllGraphics()));

	connect(my_w->actionPrint, SIGNAL(triggered()), this, SLOT(print()));

	connect(my_w->actionQuit, SIGNAL(triggered()), qApp, SLOT(closeAllWindows())) ;

	connect(my_w->actionClose, SIGNAL(triggered()), this, SLOT(fileClose()));
	connect(my_w->actionAbout, SIGNAL(triggered()), this, SLOT(about()));
	connect(my_w->actionPreferences, SIGNAL(triggered()), this, SLOT(Preferences()));


	connect(my_w->actionPrev_Buffer, SIGNAL(triggered()), my_w->my_view, SLOT(prevBuffer()));
	connect(my_w->actionNext_Buffer, SIGNAL(triggered()), my_w->my_view, SLOT(nextBuffer()));
	connect(my_w->actionClose_Buffer, SIGNAL(triggered()), this, SLOT(closeCurrentBuffer()));

	connect(my_w->actionShow_ruler, SIGNAL(triggered()), this, SLOT(toggleRuler()));
	connect(my_w->actionShow_grid, SIGNAL(triggered()), this, SLOT(toggleGrid()));

	connect(my_w->actionMouse_Zoom, SIGNAL(triggered()), this, SLOT(ZoomWin()));

	connect(my_w->actionRotate_left, SIGNAL(triggered()), this, SLOT(rotateLeft()));
	connect(my_w->actionRotate_right, SIGNAL(triggered()), this, SLOT(rotateRight()));
	connect(my_w->actionFlip_up_down, SIGNAL(triggered()), this, SLOT(flipUpDown()));
    connect(my_w->actionFlip_left_right, SIGNAL(triggered()), this, SLOT(flipLeftRight()));
    connect(my_w->actionTranspose, SIGNAL(triggered()), this, SLOT(transpose()));

	connect(my_w->actionProperties, SIGNAL(triggered()), this, SLOT(Properties()));

	connect(my_w->actionZoom_in, SIGNAL(triggered()), my_w->my_view, SLOT(zoomIn()));
	connect(my_w->actionZoom_out, SIGNAL(triggered()), my_w->my_view, SLOT(zoomOut()));
	connect(my_w->actionZoom_eq, SIGNAL(triggered()), my_w->my_view, SLOT(zoomEq()));

	connect(my_w->actionMouse_Info, SIGNAL(triggered()), this, SLOT(MouseInfo()));

	connect(my_w->actionFocal_Spot, SIGNAL(triggered()), this, SLOT(FocalSpot()));
	connect(my_w->actionContours, SIGNAL(triggered()), this, SLOT(Contours()));

	connect(my_w->actionMath_operations, SIGNAL(triggered()), this, SLOT(MathOperations()));
	connect(my_w->actionCutoff_Mask, SIGNAL(triggered()), this, SLOT(CutoffImage()));

	connect(my_w->actionNext_LUT, SIGNAL(triggered()), my_w->my_view, SLOT(nextColorTable()));
	connect(my_w->actionPrevious_LUT, SIGNAL(triggered()), my_w->my_view, SLOT(previousColorTable()));
	connect(my_w->actionShow_colortable, SIGNAL(triggered()), this, SLOT(ColorBar()));

	connect(my_w->actionHorizontal, SIGNAL(triggered()), this, SLOT(Hlineout()));
	connect(my_w->actionVertical, SIGNAL(triggered()), this, SLOT(Vlineout()));
	connect(my_w->actionBoth, SIGNAL(triggered()), this, SLOT(bothLineout()));
	connect(my_w->actionBoxLineout, SIGNAL(triggered()), this, SLOT(BoxLineout()));
	connect(my_w->actionCompareLines, SIGNAL(triggered()), this, SLOT(CompareLines()));
	connect(my_w->actionPlugin, SIGNAL(triggered()), this, SLOT(loadPlugin()));

	connect(my_w->actionSpectral_Analysis, SIGNAL(triggered()), this, SLOT(SpectralAnalysis()));
	connect(my_w->actionWavelet, SIGNAL(triggered()), this, SLOT(Wavelet()));
	connect(my_w->actionInversions, SIGNAL(triggered()), this, SLOT(IntegralInversion()));
	connect(my_w->actionRegionPath, SIGNAL(triggered()), this, SLOT(RegionPath()));
	connect(my_w->actionInterpolate_Path, SIGNAL(triggered()), this, SLOT(InterpolatePath()));

	connect(my_w->actionRotate, SIGNAL(triggered()), this, SLOT(Rotate()));
	connect(my_w->actionAffine_Transform, SIGNAL(triggered()), this, SLOT(Affine()));
	connect(my_w->actionKeyborard_shortcuts, SIGNAL(triggered()), this, SLOT(Shortcuts()));

	connect(my_w->actionLockColors, SIGNAL(toggled(bool)), my_w->my_view, SLOT(setLockColors(bool)));

	connect(my_w->my_view, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(emitBufferChanged(nPhysD*)));
	connect(my_w->my_view, SIGNAL(logging(QString)), statusBar(), SLOT(showMessage(QString)));


    connect(my_w->actionExport_pixmap, SIGNAL(triggered()), my_w->my_view, SLOT(exportPixmap()));



    // ---------------------------------------------------------------------------------------------

    QWidget* spacer = new QWidget();
    spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    my_w->toolBar->addWidget(spacer);


	// ---------------------------------------------------------------------------------------------

	QWidget *sbarra=new QWidget(this);
	my_sbarra->setupUi(sbarra);
	my_w->statusbar->addPermanentWidget(sbarra, 0);

	setAttribute(Qt::WA_DeleteOnClose);
	setCentralWidget(my_w->centralwidget);

	connect(my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(mouseposition(QPointF)));
	connect(my_w->my_view, SIGNAL(zoomChanged(double)), this, SLOT(zoomChanged(double)));


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

	if (my_w->my_view->nPalettes.keys().contains("Neutrino"))
		my_w->my_view->changeColorTable("Neutrino");
	else
		my_w->my_view->changeColorTable(my_w->my_view->nPalettes.keys().first());

	//recent file stuff

	for (int i = 0; i < MaxRecentFiles; ++i) {
		QAction *act = new QAction(this);
		act->setVisible(false);
		connect(act, SIGNAL(triggered()),this, SLOT(openRecentFile()));
		my_w->menuOpen_Recent->addAction(act);
		recentFileActs << act;
	}
	my_w->menuOpen_Recent->addSeparator();
	QAction *act = new QAction("Clear Menu", this);
	act->setVisible(true);
	connect(act, SIGNAL(triggered()),this, SLOT(clearRecentFile()));
	my_w->menuOpen_Recent->addAction(act);

	// lasciamo questo per ultimo
	//	my_w->scrollArea->setWidget(my_view);



	updateRecentFileActions();

	loadDefaults();

	// plugins
	scanPlugins();

	show();

	QApplication::processEvents();

	//#ifdef  __phys_debug
	//    if (numwin==1 && recentFileActs.size()>0)	recentFileActs.first()->trigger();
	//#endif

    connect(&timerSaveDefaults, SIGNAL(timeout()), this, SLOT(saveDefaults()));
    timerSaveDefaults.start(60000); // 1 min


}

void neutrino::on_action99_triggered() {
    if (my_w->my_view->currentBuffer) {
        my_w->my_view->currentBuffer->property["display_range"]=getColorPrecentPixels(*(my_w->my_view->currentBuffer),99);
        updatePhys();
        emitBufferChanged();
    }
}

void neutrino::scanDir(QString dirpath, QString pattern)
{
    qDebug() << "Scanning: " << pattern << dirpath;
    QDir dir(dirpath);
    dir.setNameFilters(QStringList(pattern));
    foreach (QFileInfo my_info, dir.entryInfoList(QDir::Files | QDir::Readable | QDir::NoDotAndDotDot)) {
        qDebug() << my_info.absoluteFilePath();
        fileOpen(my_info.absoluteFilePath());
    }

    foreach (QFileInfo my_info, dir.entryInfoList(QDir::AllDirs | QDir::NoDotAndDotDot | QDir::NoSymLinks)) {
        scanDir(my_info.absoluteFilePath(), pattern);
    }
}

void neutrino::on_actionOpen_Glob_triggered() {
    QString dirName = QFileDialog::getExistingDirectory(this,tr("Change monitor directory"),property("NeuSave-globdir").toString());
    if (!dirName.isEmpty()) {
        bool ok;
        QString globstring = QInputDialog::getText(this, dirName,tr("Pattern"), QLineEdit::Normal, property("NeuSave-globstring").toString(), &ok);
        if (ok) {

            setProperty("NeuSave-globdir",dirName);
            setProperty("NeuSave-globstring",globstring);

            foreach (QString my_filter, globstring.split(";")) {
                scanDir(dirName, my_filter.trimmed());
            }
        }

    }
}



void neutrino::processEvents()
{ QApplication::processEvents(); }

void neutrino::contextMenuEvent (QContextMenuEvent *ev) {
	QMenu *my_menu=new QMenu(this);
	my_menu->setAttribute(Qt::WA_DeleteOnClose);
	foreach(QAction *act, my_w->menubar->actions()) {
		my_menu->addAction(act);
	}
	my_menu->exec(ev->globalPos());

}

void neutrino::menuPaths() {
	if ( QApplication::keyboardModifiers()!=Qt::NoModifier) {
		my_w->menuPaths->exec(QCursor::pos());
	} else {
		my_w->menuPaths->defaultAction()->trigger();
	}
}

void neutrino::menuFlipRotate() {
	if ( QApplication::keyboardModifiers()!=Qt::NoModifier) {
		my_w->menuTransformation->exec(QCursor::pos());
	} else {
		my_w->menuTransformation->defaultAction()->trigger();
	}
}

void neutrino::rotateLeft() {
	my_w->menuTransformation->setDefaultAction(my_w->actionRotate_left);
	if (my_w->my_view->currentBuffer) {
		phys_rotate_left(*my_w->my_view->currentBuffer);
        updatePhys();
	}
	my_w->actionFlipRotate->setIcon(my_w->menuTransformation->defaultAction()->icon());
	QSettings("neutrino","").setValue("menuTransformationDefault",my_w->menuTransformation->defaultAction()->text());

}

void neutrino::rotateRight() {
	my_w->menuTransformation->setDefaultAction(my_w->actionRotate_right);
	if (my_w->my_view->currentBuffer) {
		phys_rotate_right(*my_w->my_view->currentBuffer);
        updatePhys();
	}
	my_w->actionFlipRotate->setIcon(my_w->menuTransformation->defaultAction()->icon());
	QSettings("neutrino","").setValue("menuTransformationDefault",my_w->menuTransformation->defaultAction()->text());
}

void neutrino::flipUpDown() {
	my_w->menuTransformation->setDefaultAction(my_w->actionFlip_up_down);
	if (my_w->my_view->currentBuffer) {
		phys_flip_ud(*my_w->my_view->currentBuffer);
        updatePhys();
	}
	QSettings("neutrino","").setValue("menuTransformationDefault",my_w->menuTransformation->defaultAction()->text());
	my_w->actionFlipRotate->setIcon(my_w->menuTransformation->defaultAction()->icon());
}

void neutrino::flipLeftRight() {
    my_w->menuTransformation->setDefaultAction(my_w->actionFlip_left_right);
    if (my_w->my_view->currentBuffer) {
        phys_flip_lr(*my_w->my_view->currentBuffer);
        updatePhys();
    }
    QSettings("neutrino","").setValue("menuTransformationDefault",my_w->menuTransformation->defaultAction()->text());
    my_w->actionFlipRotate->setIcon(my_w->menuTransformation->defaultAction()->icon());
}

void neutrino::transpose() {
    my_w->menuTransformation->setDefaultAction(my_w->actionTranspose);
    if (my_w->my_view->currentBuffer) {
        phys_transpose(*my_w->my_view->currentBuffer);
        updatePhys();
    }
    QSettings("neutrino","").setValue("menuTransformationDefault",my_w->menuTransformation->defaultAction()->text());
    my_w->actionFlipRotate->setIcon(my_w->menuTransformation->defaultAction()->icon());
}

nPhysD* neutrino::getBuffer(int i) {
	return my_w->my_view->physList.value(i);
}

// ------------------ PLUGINS -----------------------

void
neutrino::scanPlugins(QString pluginsDirStr)
{
	scanPlugins(QDir(pluginsDirStr));
}

void
neutrino::scanPlugins(QDir pluginsDir) {
	if (pluginsDir.exists()) {
		foreach (QString fileName, pluginsDir.entryList(QDir::Files)) {
			if (QFileInfo(fileName).suffix() == nPlug::extension()) {
				loadPlugin(pluginsDir.absoluteFilePath(fileName), false);
			}
		}
		QStringList listdirPlugins=property("NeuSave-plugindirs").toStringList();
		qDebug() << pluginsDir.absolutePath() << property("defaultPluginDir").toString();
		if (!listdirPlugins.contains(pluginsDir.absolutePath()) && pluginsDir.absolutePath() != property("defaultPluginDir").toString())
			listdirPlugins.append(pluginsDir.absolutePath());
		setProperty("NeuSave-plugindirs",listdirPlugins);
	}
}

void
neutrino::scanPlugins() {
	QDir pluginsDir;
#if defined(Q_OS_WIN)
	pluginsDir.setPath(qApp->applicationDirPath());
	if (pluginsDir.dirName().toLower() == "debug" || pluginsDir.dirName().toLower() == "release")
		pluginsDir.cdUp();
#elif defined(Q_OS_MAC)
	pluginsDir.setPath(qApp->applicationDirPath());
	pluginsDir.cdUp();
	pluginsDir.cd("Resources");
#elif defined(Q_OS_LINUX)
	pluginsDir.setPath("/usr/share/neutrino");
#endif
	pluginsDir.cd("plugins");
	qDebug() << "defaultPluginDir:" << pluginsDir.absolutePath();
	setProperty("defaultPluginDir",pluginsDir.absolutePath());
	scanPlugins(pluginsDir);

	if (property("NeuSave-plugindirs").isValid()) {
		for (auto& d : property("NeuSave-plugindirs").toStringList()) {
			if (d!=pluginsDir.absolutePath())
				scanPlugins(d);
		}
	}
}

void
neutrino::loadPlugin()
{
	QStringList pnames=QFileDialog::getOpenFileNames(this,tr("Load Plugin"), property("NeuSave-loadPlugin").toString(),tr("Neutrino Plugins")+QString(" (*.dylib *.so *.dll);;")+tr("Any files")+QString(" (*)"));
	loadPlugins(pnames);

}

void
neutrino::loadPlugins(QStringList pnames)
{
	bool launch(pnames.size()==1);
	for(auto& pname: pnames) {
		loadPlugin(pname, launch);
	}
}

void
neutrino::loadPlugin(QString pname, bool launch)
{
	if (!property("NeuSave-loadPlugin").isValid()) {
		setProperty("NeuSave-loadPlugin",QString("plugin.so"));
	}

	if (QFileInfo(pname).exists()) {
		setProperty("NeuSave-loadPlugin",pname);
		DEBUG(10, "loading plugin "<<pname.toStdString());

		nPluginLoader *my_npl = new nPluginLoader(pname, this);
		qDebug() << "here" << my_npl->ok();
		if (launch) my_npl->launch();
	}
}

void neutrino::emitBufferChanged(nPhysD *my_phys) {
    qDebug() << sender();
	if (!my_phys) my_phys=my_w->my_view->currentBuffer;

	if (my_phys) {
        double gamma_val=my_phys->gamma();
		my_sbarra->gamma->setText(QString(QChar(0x03B3))+" "+QString(gamma_val<1? "1/"+ QString::number(int(1.0/gamma_val)) : QString::number(int(gamma_val))));

		QString winName=QString::fromUtf8(my_phys->getShortName().c_str());
		winName.prepend(property("winId").toString()+QString(":")+QString::number(indexOf(my_phys))+QString(" "));

		QString mypath=QString::fromUtf8(my_phys->getFromName().c_str());
		winName.append(QString(" ")+mypath);
		setWindowTitle(winName);

		if (my_phys->getType()==PHYS_FILE || my_phys->getType()==PHYS_RFILE) {
			setWindowFilePath(mypath);
		} else {
			setWindowFilePath("");
		}
	}
	my_w->my_view->update();
	emit bufferChanged(my_phys);
}

void neutrino::emitPanAdd(nGenericPan* pan) {
	panList.removeAll(pan);
	panList.append(pan);
	emit panAdd(pan);
}

void neutrino::emitPanDel(nGenericPan* pan) {
	panList.removeAll(pan);
	emit panDel(pan);
}


/// Returns the QT drawings area
QGraphicsScene&
neutrino::getScene() {
	return my_w->my_view->my_scene;
}

/// This is called form a Open recent file menu.
void neutrino::openRecentFile()
{
	QAction *action = qobject_cast<QAction *>(sender());
	if (action) {
		QString fname=action->data().toString();
		fileOpen(fname);
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

void neutrino::setGamma(int value) {
	my_w->my_view->setGamma(value);
	setProperty("neuSave-gamma",value);
}

nGenericPan* neutrino::existsPan(QString name) {
	foreach (nGenericPan *pan, panList) {
		if (pan->panName()==name) {
			pan->show();
			pan->raise();
			return pan;
		}
	}
	return NULL;
}


// public slots

// file menu actions


neutrino* neutrino::fileNew() {
	//	QThread *m_thread = new QThread();
	return new neutrino();
	//	my_neu->moveToThread(m_thread);
	//	m_thread->start();
}

void
neutrino::fileReopen() {
	if(my_w->my_view->currentBuffer && my_w->my_view->currentBuffer->getType()==PHYS_FILE) {
		QString fname=QString::fromUtf8(my_w->my_view->currentBuffer->getFromName().c_str());
		fileOpen(fname);
	}
}

void neutrino::fileOpen()
{
    QString formats("Neutrino Images (");
    for (auto &format : phys_image_formats()) {
        formats+="*."+ QString::fromStdString(format)+" ";
    }
    formats+=" *.neus);; Images (";
    foreach (QByteArray format, QImageReader::supportedImageFormats() ) {
        formats+="*."+format+" ";
    }
    formats.chop(1);
    formats+=");;";
    formats+=("Any files (*)");

    QStringList fnames = QFileDialog::getOpenFileNames(this,tr("Open Image(s)"),property("NeuSave-fileOpen").toString(),formats);
    fileOpen(fnames);
}

void neutrino::fileOpen(QStringList fnames) {
	foreach (QString fname, fnames) {
		QList<nPhysD *> imagelist = fileOpen(fname);
		if (imagelist.size()==0){
			QString vwinname="OpenRaw";
			nOpenRAW *openRAW=(nOpenRAW *)existsPan(vwinname);
			if (!openRAW) {
				openRAW = new nOpenRAW(this);
			}
			openRAW->add(fname);
		}
	}
}


QList <nPhysD *> neutrino::fileOpen(QString fname) {
    setProperty("NeuSave-fileOpen", fname);
    QSettings my_set("neutrino","");
	my_set.beginGroup("nPreferences");
	bool separate_rgb= my_set.value("separateRGB",false).toBool();
	my_set.endGroup();

	QList <nPhysD *> imagelist;
	QString suffix=QFileInfo(fname).suffix().toLower();
	if (suffix=="neus") {
		imagelist=openSession(fname);
	} else {
		std::vector<nPhysD*> my_vec;
		try {
			my_vec=phys_open(fname.toUtf8().constData(),separate_rgb);
		} catch (std::exception &e) {
			QMessageBox dlg(QMessageBox::Critical, tr("Exception"), e.what());
			dlg.setWindowFlags(dlg.windowFlags() | Qt::WindowStaysOnTopHint);
			dlg.exec();
		}
		for(std::vector<nPhysD*>::iterator it=my_vec.begin();it!=my_vec.end();it++) {
			imagelist.push_back(*it);
		}
		if (imagelist.size()==0) {
			QImage image(fname);
			if (!image.isNull()) {
				if (image.isGrayscale() || !separate_rgb) {
					nPhysD *datamatrix = new nPhysD(fname.toStdString());
					datamatrix->resize(image.width(), image.height());
					for (int i=0;i<image.height();i++) {
						for (int j=0;j<image.width();j++) {
							datamatrix->Timg_matrix[i][j]= qGray(image.pixel(j,i));
						}
					}
					imagelist.push_back(datamatrix);
				} else {
					std::array<nPhysD*,3> datamatrix;
					std::array<std::string,3> name;
					name[0]="Red";
					name[1]="Green";
					name[2]="Blue";
					for (int k=0;k<3;k++) {
						datamatrix[k] = new nPhysD(QFileInfo(fname).fileName().toStdString());
						datamatrix[k]->setShortName(name[k]);
						datamatrix[k]->setName(name[k]+" "+QFileInfo(fname).fileName().toStdString());
						datamatrix[k]->setFromName(fname.toStdString());
						datamatrix[k]->resize(image.width(), image.height());
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
				for (int k=0;k<imagelist.size();k++) {
					imagelist[k]->TscanBrightness();
					imagelist[k]->setType(PHYS_FILE);
				}

			}
		}
	}

	if (imagelist.size()>0) {
		updateRecentFileActions(fname);
		QMutableListIterator<nPhysD*> i(imagelist);
		while (i.hasNext()) {
			if (i.next()->getSurf()>0) {
				addShowPhys(i.value());
			} else {
				i.remove();
			}
		}
	}
	QApplication::processEvents();
	return imagelist;
}

void neutrino::saveSession (QString fname) {
	if (fname.isEmpty()) {
		QString extensions=tr("Neutrino session")+QString(" (*.neus);;");
#ifdef HAVE_LIBTIFF
		extensions+=tr("Tiff session")+" (*.tiff *.tif);;";
#endif
        QString fnameSave = QFileDialog::getSaveFileName(this,tr("Save Session"),property("NeuSave-fileOpen").toString(),extensions+tr("Any files")+QString(" (*)"));
		if (!fnameSave.isEmpty()) {
			saveSession(fnameSave);
		}
	} else {
		QFileInfo file_info(fname);
		if (file_info.suffix()=="neus") {
			setProperty("NeuSave-fileOpen", fname);
			//            for(int k = 0; k < (panList.size()/2); k++) panList.swap(k,panList.size()-(1+k));

			QProgressDialog progress("Save session", "Cancel", 0, my_w->my_view->physList.size()+1, this);
			progress.setWindowModality(Qt::WindowModal);
			progress.show();

			std::ofstream ofile(fname.toUtf8().constData(), std::ios::out | std::ios::binary);
            ofile << "Neutrino " << __VER_LATEST << " " << my_w->my_view->physList.size() << " " << panList.size() << std::endl;

			for (int i=0;i<my_w->my_view->physList.size(); i++) {
				if (progress.wasCanceled()) break;
				progress.setValue(i);
				progress.setLabelText(QString::fromUtf8(my_w->my_view->physList.at(i)->getShortName().c_str()));
				QApplication::processEvents();
				ofile << "NeutrinoImage" << std::endl;
				phys_dump_binary(my_w->my_view->physList.at(i),ofile);
				my_w->my_view->physList.at(i)->setType(PHYS_FILE);
			}
			for (int i=0;i<panList.size(); i++) {
				QString namePan=panList.at(i)->metaObject()->className();
				QTemporaryFile tmpFile(this);
				if (tmpFile.open()) {
					QString tmp_filename=tmpFile.fileName();
					QApplication::processEvents();
					QSettings my_set(tmp_filename,QSettings::IniFormat);
					my_set.clear();
					panList.at(i)->saveSettings(&my_set);
					my_set.sync();
					tmpFile.close();
					QFile file(tmp_filename);
					if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
						ofile << "NeutrinoPan-begin " << namePan.toStdString() << std::endl;
						ofile.flush();
						while (!file.atEnd()) {
							QByteArray line = file.readLine();
							ofile << line.constData();
						}
						file.close();
						file.remove();
						ofile << "NeutrinoPan-end " << namePan.toStdString() << std::endl;
						ofile.flush();
					} else {
						QMessageBox::warning(this,tr("Attention"),tr("Cannot write values for ")+panList.at(i)->panName()+QString("\n")+tmp_filename+QString("\n")+tr("Contact dev team."), QMessageBox::Ok);
					}
				} else {
					QMessageBox::warning(this,tr("Attention"),tr("Cannot write values for ")+panList.at(i)->panName(), QMessageBox::Ok);
				}
			}
			progress.setValue(my_w->my_view->physList.size()+1);
			ofile.close();
		} else if (file_info.suffix().startsWith("tif")) {
			std::vector <nPhysD *> vecPhys;
			foreach (nPhysD * my_phys, my_w->my_view->physList) {
				vecPhys.push_back(my_phys);
			}
			phys_write_tiff(vecPhys,fname.toUtf8().constData());
		} else {
			QMessageBox::warning(this,tr("Attention"),tr("Unknown extension: ")+file_info.suffix(), QMessageBox::Ok);
		}
		updateRecentFileActions(fname);
	}
}

QList <nPhysD *> neutrino::openSession (QString fname) {
	QList <nPhysD *> imagelist;
	if (!fname.isEmpty()) {
		updateRecentFileActions(fname);
		setProperty("NeuSave-fileOpen", fname);
		if (my_w->my_view->physList.size()!=0) {
			QThread *m_thread = new QThread();
			neutrino*my_neu= new neutrino();
			my_neu->moveToThread(m_thread);
			m_thread->start();
			my_neu->fileOpen(fname);
		} else {
			QProgressDialog progress("Load session", "Cancel", 0, 0, this);
			std::ifstream ifile(fname.toUtf8().constData(), std::ios::in | std::ios::binary);
			std::string line;
			getline(ifile,line);
			QString qLine=QString::fromStdString(line);
			int counter=0;
			if (qLine.startsWith("Neutrino")) {
				progress.setWindowModality(Qt::WindowModal);
				progress.setLabelText(tr("Load Session ")+qLine.split(" ").at(1));
				progress.setMaximum(1+qLine.split(" ").at(2).toInt()+1);
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
						int ret=phys_resurrect_binary(my_phys,ifile);
						if (ret>=0 && my_phys->getSurf()>0) {
							imagelist.push_back(my_phys);
						} else {
							delete my_phys;
						}
						progress.setLabelText(QString::fromUtf8(my_phys->getShortName().c_str()));
						QApplication::processEvents();
					} else if (qLine.startsWith("NeutrinoPan-begin")) {
                        for (auto& my_phys: imagelist) {
                            addPhys(my_phys);
                        }
						QStringList listLine=qLine.split(" ");
						QString panName=listLine.at(1);
						QApplication::processEvents();

						for(int i =  metaObject()->methodOffset(); i < metaObject()->methodCount(); ++i) {
							qDebug() << "method:" << metaObject()->method(i).methodSignature();
						}

						nGenericPan *my_pan=openPan(panName, false);

						if (my_pan) {
							QApplication::processEvents();
							QTemporaryFile tmpFile(this);
							tmpFile.open();
							while(!ifile.eof()) {
								getline(ifile,line);
								// WARNING(line);
								qLine=QString::fromStdString(line);
								if (qLine.startsWith("NeutrinoPan-end"))
									break;
								tmpFile.write(line.c_str());
								tmpFile.write("\n");
							}
							tmpFile.flush();
							QApplication::processEvents();
							QMetaObject::invokeMethod(my_pan,"loadSettings",Q_ARG(QString,tmpFile.fileName()));
							QApplication::processEvents();
							tmpFile.close(); // this should also remove it...
						} else {
							QMessageBox::critical(this,tr("Session error"),tr("Cannot find method or plugin for ")+panName,  QMessageBox::Ok);
						}

					}
					progress.setValue(counter++);
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
    if (datamatrix && datamatrix->getSurf()>0 && !my_w->my_view->physList.contains(datamatrix))	{
		my_w->my_view->physList << datamatrix;
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
	my_w->menuBuffers->addAction(action);
}

nPhysD* neutrino:: replacePhys(nPhysD* newPhys, nPhysD* oldPhys, bool show) { //TODO: this should be done in nPhysImage...
    if (newPhys && newPhys->getSurf()) {
		bool redisplay = (my_w->my_view->currentBuffer==oldPhys);
		if (my_w->my_view->physList.contains(oldPhys)) {
//			newPhys->property["display_range"]=oldPhys->property["display_range"];
			if (oldPhys==NULL) oldPhys=new nPhysD();
			*oldPhys=*newPhys;
			delete newPhys;
			newPhys=oldPhys;
		} else {
			newPhys->property.erase("display_range");
			addPhys(newPhys);
		}
		if (show || redisplay) {
			showPhys(newPhys);
		}
        emit physMod(std::make_pair(oldPhys, newPhys));
    }
    QApplication::processEvents();
	return newPhys;
}

void neutrino::removePhys(nPhysD* datamatrix) {
	if (datamatrix) {
		emit physDel(datamatrix);
		int position=indexOf(datamatrix);
		if (position != -1) {
			my_w->my_view->physList.removeAll(datamatrix);
            if (my_w->my_view->physList.size()>0) {
				showPhys(my_w->my_view->physList.at(std::min<int>(position,my_w->my_view->physList.size()-1)));
			} else {
				my_w->my_view->currentBuffer=nullptr;
				emitBufferChanged();
				setWindowTitle(property("winId").toString()+QString(": Neutrino"));
				setWindowFilePath("");
				zoomChanged(1);
				my_w->my_view->my_pixitem.setPixmap(QPixmap(":icons/icon.png"));
				my_w->my_view->setSize();
			}
			QList<QAction *> lista=my_w->menuBuffers->actions();
			foreach (QAction* action, my_w->menuBuffers->actions()) {
				if (action->data() == qVariantFromValue((void*) datamatrix)) {
					my_w->menuBuffers->removeAction(action);
				}
			}
            if (datamatrix->property["keep_phys_alive"].get_i()!=42){
                DEBUG("PLEASE NOTE that this is a failsafe to avoid deleting stuff owned by python")
                delete datamatrix;
            }
            datamatrix=NULL;
        }
	}
}

void
neutrino::showPhys(nPhysD* my_phys) {
    if (my_phys && !my_phys->property.have("gamma")) {
        my_phys->property["gamma"]=property("neuSave-gamma").toInt();
    }
    my_w->my_view->showPhys(my_phys);
}

void
neutrino::updatePhys() {
    my_w->my_view->updatePhys();
}

void neutrino::exportGraphics () {
	QString ftypes="SVG (*.svg);; PDF (*.PDF);; PNG (*.png);; Any files (*)";
	QString fout = QFileDialog::getSaveFileName(this,tr("Save Drawing"),property("NeuSave-fileExport").toString(),ftypes);
	if (!fout.isEmpty())
		exportGraphics(fout);
}

void neutrino::exportAllGraphics () {
	QString ftypes="SVG (*.svg);; PDF (*.PDF);; PNG (*.png);; Any files (*)";
	QString fout = QFileDialog::getSaveFileName(this,tr("Save All Drawings"),property("NeuSave-fileExport").toString(),ftypes);
	if (!fout.isEmpty()) {
		for (int i=0;i<my_w->my_view->physList.size() ; i++) {
			my_w->my_view->nextBuffer();
			QFileInfo fi(fout);
			exportGraphics(fi.path()+"/"+fi.baseName()+QString("_")+QString("%1").arg(i, 3, 10, QChar('0'))+QString("_")+QString::fromStdString(my_w->my_view->currentBuffer->getShortName())+"."+fi.completeSuffix());
		}
		setProperty("NeuSave-fileExport",fout);
	}
}

void neutrino::exportGraphics (QString fout) {
	setProperty("NeuSave-fileExport",fout);
	bool resetmouse=my_w->my_view->my_mouse.isVisible();
	my_w->my_view->my_mouse.setVisible(false);
	QSize my_size=QSize(getScene().width(), getScene().height());
	if (QFileInfo(fout).suffix().toLower()==QString("pdf")) {
		QPrinter myPrinter(QPrinter::ScreenResolution);
		myPrinter.setOutputFileName(fout);
		myPrinter.setOrientation(QPrinter::Landscape);
		myPrinter.setPaperSize(QPrinter::A4);
		int newWidth=myPrinter.paperSize(QPrinter::DevicePixel).height() * ((double) my_size.width())/((double)my_size.height());
		QSize newSize=QSize(newWidth,myPrinter.paperSize(QPrinter::DevicePixel).height());
		myPrinter.setPaperSize(newSize,QPrinter::DevicePixel);

		myPrinter.setOutputFormat(QPrinter::PdfFormat);
		myPrinter.setPageMargins(0.0, 0.0, 0.0, 0.0, QPrinter::DevicePixel);

		QPainter myPainter(&myPrinter);
		myPainter.setViewport(0, 0, myPrinter.width(), myPrinter.height());
		getScene().render(&myPainter);
	} else if (QFileInfo(fout).suffix().toLower()==QString("svg")) {
		QSvgGenerator svgGen;
		svgGen.setFileName(fout);
		svgGen.setSize(my_size);
		QRect my_rect(0,0,my_w->my_view->my_tics.boundingRect().width(),my_w->my_view->my_tics.boundingRect().height());
		svgGen.setViewBox(my_rect);
		svgGen.setTitle("Neutrino");
		svgGen.setDescription(windowFilePath());
		QPainter painter( &svgGen );
		getScene().render(&painter);
	} else {
		QPixmap::grabWidget(my_w->my_view).save(fout);
	}
	my_w->my_view->my_mouse.setVisible(resetmouse);
}

void neutrino::toggleRuler() {
	my_w->my_view->my_tics.rulerVisible=!my_w->my_view->my_tics.rulerVisible;
	if (my_w->my_view->my_tics.rulerVisible) {
		my_w->actionShow_ruler->setText("Hide ruler");
	} else {
		my_w->actionShow_ruler->setText("Show ruler");
	}
	my_w->my_view->my_tics.update();
}

void neutrino::toggleGrid() {
	my_w->my_view->my_tics.gridVisible=!my_w->my_view->my_tics.gridVisible;
	if (my_w->my_view->my_tics.gridVisible) {
		my_w->actionShow_grid->setText("Hide grid");
	} else {
		my_w->actionShow_grid->setText("Show grid");
	}
	my_w->my_view->my_tics.update();
}

void neutrino::closeEvent (QCloseEvent *e) {
	disconnect(my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(mouseposition(QPointF)));
	if (fileClose()) {
		saveDefaults();
		e->accept();
	} else {
		e->ignore();
		connect(my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(mouseposition(QPointF)));
	}
}

// keyevents: pass to my_view!
void neutrino::keyPressEvent (QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_Question:
			Shortcuts();
			break;
		case Qt::Key_O:
			if (e->modifiers() & Qt::ShiftModifier) {
				foreach (nPhysD* phys, my_w->my_view->physList) {
					phys->set_origin(my_w->my_view->my_mouse.pos().x(),my_w->my_view->my_mouse.pos().y());
					emit bufferChanged(phys);
				}
			} else {
				if (my_w->my_view->currentBuffer) my_w->my_view->currentBuffer->set_origin(my_w->my_view->my_mouse.pos().x(),my_w->my_view->my_mouse.pos().y());
			}
			mouseposition(my_w->my_view->my_mouse.pos());
			my_w->my_view->my_tics.update();
			emitBufferChanged();

			// I need a signal to communicate explicit origin change not to
			// be taken for a buffer change. Used in nWinList.
			emit bufferOriginChanged();
			break;
		case Qt::Key_C:
			if (e->modifiers() & Qt::ShiftModifier) {
				ColorBar();
			}
			break;
		case Qt::Key_I:
			if (e->modifiers() & Qt::ShiftModifier) {
				MouseInfo();
			} else {
				WinList();
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
		case Qt::Key_X: {
				if ((e->modifiers() & Qt::ControlModifier)) {
					bool ok;
					QString text = QInputDialog::getText(this,"","Open", QLineEdit::Normal,QString(""), &ok, Qt::Sheet);
					if (ok && !text.isEmpty()) {
						nGenericPan *my_pan= openPan(text,false);
						if(!my_pan) {
							statusBar()->showMessage(tr("Can't find ")+text, 2000);
						}
					}
				}
				break;
			}
		default:
			break;
	}
}

void neutrino::keyReleaseEvent (QKeyEvent *e)
{
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
		e->acceptProposedAction();
		QList<QByteArray> pippo=e->mimeData()->data("data/neutrino").split(' ');
		foreach(QByteArray bytephys, pippo) {
			bool ok=false;
			nPhysD *my_phys=(nPhysD *) bytephys.toLongLong(&ok);
			if (ok && my_phys) {
				if (my_w->my_view->physList.contains(my_phys)) {
					showPhys(my_phys);
				} else {
					nPhysD *copyhere;
					copyhere = new nPhysD(*my_phys);
					addShowPhys(copyhere);
				}
			}
		}

	} else if (e->mimeData()->hasUrls()) {
		e->acceptProposedAction();
		QStringList fileList;
		foreach (QUrl qurl, e->mimeData()->urls()) {
			fileList << qurl.toLocalFile();
		}
		if (fileList.size()>0) {
			fileOpen(fileList);
		}
	}
}

// zoom

nGenericPan*
neutrino::ZoomWin() {
	return new nZoomWin(this);
}


void
neutrino::zoomChanged(double zoom) {
	QString tmp;
	tmp.sprintf(" %.1f%%",100.0*zoom);
	statusBar()->showMessage(tr("Zoom :")+tmp,2000);
	emit nZoom(zoom);
	update();
}

double
neutrino::getZoom() const {
	return my_w->my_view->transform().m11();
}

void
neutrino::mouseposition(QPointF pos_mouse) {
	my_sbarra->pos_x->setNum((int)pos_mouse.x());
	my_sbarra->pos_y->setNum((int)pos_mouse.y());


	if (my_w->my_view->currentBuffer) {
		vec2f vec=my_w->my_view->currentBuffer->to_real(vec2f(pos_mouse.x(),pos_mouse.y()));
		QPointF pos=QPointF(vec.x(),vec.y());
		my_sbarra->dx->setNum(pos.x());
		my_sbarra->dy->setNum(pos.y());
		double val=my_w->my_view->currentBuffer->point(pos_mouse.x(),pos_mouse.y());
		my_sbarra->pos_z->setNum(val);
		emit colorValue(val);
		emit mouseAtWorld(pos);
	} else {
		my_sbarra->dx->setText("");
		my_sbarra->dy->setText("");
	}

	emit mouseAtMatrix(pos_mouse);
}

QString neutrino::getFileSave() {
    QString allformats;
    QStringList formats;
    formats << "txt" << "neu" << "neus";
#ifdef HAVE_LIBTIFF
    formats << "tif" << "tiff";
#endif
#ifdef HAVE_LIBCFITSIO
    formats << "fits";
#endif
#if defined(HAVE_LIBMFHDF) || defined(HAVE_LIBMFHDFDLL)
    formats << "hdf";
#endif
    foreach(QString format, formats ) {
        allformats += format + " (*."+format+");; ";
    }
	foreach (QByteArray format, QImageWriter::supportedImageFormats() ) {
        allformats += format + " (*."+format+");; ";
    }
    allformats.chop(1);
    allformats+=("Any files (*)");
    return QFileDialog::getSaveFileName(this, "Save to...",property("NeuSave-fileOpen").toString(),allformats);
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
		setProperty("NeuSave-fileOpen", fname);
		QString suffix=QFileInfo(fname).suffix().toLower();
		if (suffix.isEmpty()) {
			fname+=".neus";
			suffix=QFileInfo(fname).suffix().toLower();

			if (QFile(fname).exists()) {
				int res=QMessageBox::warning(this,tr("Attention"), fname+QString("\n")+tr("exists. Overwrite?"),
											 QMessageBox::Yes | QMessageBox::No  | QMessageBox::Cancel);
				switch (res) {
					case QMessageBox::No:
						fileSave();
						return;
						break;
					case QMessageBox::Cancel:
						return;
						break;
				}
			}

		}

		if (suffix.startsWith("neus")) {
			saveSession(fname);
		} else {
			fileSave(my_w->my_view->currentBuffer,fname);
		}
	}
}

void neutrino::fileSave(nPhysD* phys, QString fname) {
	if (phys) {
		QString suffix=QFileInfo(fname).suffix().toLower();
		if (suffix.startsWith("neu")) {
			phys_dump_binary(phys,fname.toUtf8().constData());
		} else if (fname.endsWith("tif") || fname.endsWith("tiff")) {
			phys_write_tiff(phys,fname.toUtf8().constData());
		} else if (suffix.startsWith("fit")) {
			phys_write_fits(phys,("!"+fname).toUtf8().constData(),4);
#ifdef __phys_HDF
		} else if (suffix.startsWith("hdf")) {
			phys_write_HDF4(phys,fname.toUtf8().constData());
#endif
		} else if (suffix.startsWith("txt") || suffix.startsWith("dat")) {
			phys->writeASC(fname.toUtf8().constData());
		} else {
			my_w->my_view->my_pixitem.pixmap().save(fname);
		}
		phys->setType(PHYS_FILE);
		phys->setShortName(QFileInfo(fname).fileName().toStdString());
		phys->setName(phys->getShortName());
		phys->setFromName(fname.toStdString());
	}
}

bool
neutrino::fileClose() {
	saveDefaults();
	if (QApplication::activeWindow() == this) {
		bool askAll=true;
		QApplication::processEvents();
		foreach (nGenericPan* pan, panList) {
			pan->hide();
			pan->close();
			QApplication::processEvents();
		}
		foreach (nPhysD *phys, my_w->my_view->physList) {
			if (askAll && phys->getType()==	PHYS_DYN && property("askCloseUnsaved").toBool()==true) {
				int res=QMessageBox::warning(this,tr("Attention"),
											 tr("The image")+QString("\n")+QString::fromUtf8(phys->getName().c_str())+QString("\n")+tr("has not been saved. Do you want to save it now?"),
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

		foreach (nGenericPan *pan, panList) {
			pan->close();
		}
		QApplication::processEvents();
		my_w->my_view->currentBuffer=NULL;
		foreach (nPhysD *phys, my_w->my_view->physList) {
			delete phys;
		}
		//#ifdef HAVE_PYTHONQT
		//        PythonQt::self()->getMainModule().removeVariable(objectName());
		//#endif

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
	if (my_w->my_view->currentBuffer)  {
		if (my_w->my_view->currentBuffer->getType()==PHYS_DYN && property("askCloseUnsaved").toBool()==true) {
			int res=QMessageBox::warning(this,tr("Attention"),
										 tr("The image")+QString("\n")+QString::fromUtf8(my_w->my_view->currentBuffer->getName().c_str())+QString("\n")+tr("has not been saved. Do you vant to save it now?"),
										 QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
			switch (res) {
				case QMessageBox::Yes:
					fileSave(my_w->my_view->currentBuffer); // TODO: add here a check for a cancel to avoid exiting
					break;
				case QMessageBox::No:
					removePhys(my_w->my_view->currentBuffer);
					break;
			}
		} else {
			removePhys(my_w->my_view->currentBuffer);
		}
	}
	QApplication::processEvents();
}

nGenericPan*
neutrino::Shortcuts() {
	return new nShortcuts(this);
}

nGenericPan*
neutrino::FocalSpot() {
	return new nFocalSpot(this);
}

nGenericPan*
neutrino::Contours() {
	return new nContours(this);
}

nGenericPan*
neutrino::MathOperations() {
	return new nOperator(this);
}

nGenericPan*
neutrino::CutoffImage() {
	return new nCutoffMask(this);
}

// Window List pan
nGenericPan*
neutrino::WinList() {
	return new nWinList(this);
}

nGenericPan*
neutrino::Properties() {
	return new nPhysProperties(this);
}

nGenericPan*
neutrino::MouseInfo() {
	return new nMouseInfo(this);
}

// colortables

nGenericPan*
neutrino::ColorBar() {
	return new nColorBar(this);
}

struct QPairFirstComparer {
	template<typename T1, typename T2>
	bool operator()(const QPair<T1,T2> & a, const QPair<T1,T2> & b) const {
		return a.first <= b.first;
	}
};

bool neutrino::addPaletteFromString(QString paletteName, QString paletteStr) {
	if (paletteStr.contains(",")) {
		QList<QPair<double,QColor> > listDoubleColor;
		QStringList paletteList=paletteStr.split(",",QString::SkipEmptyParts);
		for (int i=0;i<paletteList.size();i++) {
			QStringList colorValueName=paletteList.at(i).split(" ",QString::SkipEmptyParts);
			if (colorValueName.size()==2) {
				bool ok;
				double my_val=QLocale().toDouble(colorValueName.first(),&ok);
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

				std::vector<unsigned char> palC(768);

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
				my_w->my_view->nPalettes[paletteName] = palC;
				my_w->my_view->changeColorTable(paletteName);
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
		std::vector<unsigned char> palette(768);
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

			my_w->my_view->nPalettes[paletteName] = palette;
			my_w->my_view->changeColorTable(paletteName);
		} else {
			paletteName.clear();
		}
	}
	return paletteName;
}

// testing
void
neutrino::createDrawLine() {
	my_w->menuPaths->setDefaultAction(my_w->actionLine);
	statusBar()->showMessage(tr("Click for points, press Esc to finish"),5000);
	nLine *item = new nLine(this);
	item->interactive();
	my_w->actionPaths->setIcon(my_w->menuPaths->defaultAction()->icon());
	QSettings("neutrino","").setValue("defualtActionPath",my_w->menuPaths->defaultAction()->text());
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
	my_w->menuPaths->setDefaultAction(my_w->actionRect);
	statusBar()->showMessage(tr("Click for the first point of the rectangle"),5000);
	nRect *item = new nRect(this);
	item->interactive();
	my_w->actionPaths->setIcon(my_w->menuPaths->defaultAction()->icon());
	QSettings("neutrino","").setValue("defualtActionPath",my_w->menuPaths->defaultAction()->text());
}

void
neutrino::createDrawPoint() {
	my_w->menuPaths->setDefaultAction(my_w->actionPoint);
	statusBar()->showMessage(tr("Click for the point"),5000);
	nPoint *item = new nPoint(this);
	item->interactive();
	my_w->actionPaths->setIcon(my_w->menuPaths->defaultAction()->icon());
	QSettings("neutrino","").setValue("defualtActionPath",my_w->menuPaths->defaultAction()->text());
}

void
neutrino::createDrawEllipse() {
	my_w->menuPaths->setDefaultAction(my_w->actionEllipse);
	statusBar()->showMessage(tr("Click and release to create the ellipse"),5000);
	nEllipse *item = new nEllipse(this);
	item->interactive();
	my_w->actionPaths->setIcon(my_w->menuPaths->defaultAction()->icon());
	QSettings("neutrino","").setValue("defualtActionPath",my_w->menuPaths->defaultAction()->text());
}

nGenericPan*
neutrino::Hlineout() {
	return new nHlineout(this);
}

nGenericPan*
neutrino::Vlineout() {
	return new nVlineout(this);
}

nGenericPan*
neutrino::bothLineout() {
	return new nLineoutBoth(this);
}

void neutrino::print()
{
	QPrinter printer(QPrinter::HighResolution);
	QPrintDialog *printDialog = new QPrintDialog(&printer, this);
	if (printDialog->exec() == QDialog::Accepted) {
		my_w->my_view->my_mouse.hide();
		QPainter painter(&printer);
		foreach (QGraphicsItem *oggetto, getScene().items() ) {
			if (qgraphicsitem_cast<nLine *>(oggetto)) {
				nLine *my_nline = (nLine *)oggetto;
				my_nline->selectThis(false);
			}
		}
		getScene().render(&painter);
		my_w->my_view->my_mouse.show();
	}
}

/// rectangle lineout
nGenericPan*
neutrino::BoxLineout() {
	return new nBoxLineout(this);
}

/// compare lines between images
nGenericPan*
neutrino::CompareLines() {
	return new nCompareLines(this);
}

/// Open raw window
nGenericPan*
neutrino::openRAW() {
	QStringList fnames;
	nGenericPan *win = NULL;
	fnames = QFileDialog::getOpenFileNames(this,tr("Open RAW"),NULL,tr("Any files")+QString(" (*)"));
	if (fnames.size()) {
		win=existsPan("nOpenRAW");
		if (!win) win= new nOpenRAW(this);
		nOpenRAW *winRAW=qobject_cast<nOpenRAW*>(win);
		if (winRAW) winRAW->add(fnames);
	}
	return win;
}

/// Spectral Analysis (FT, filtering and stuff)
nGenericPan*
neutrino::SpectralAnalysis() {
	return new nSpectralAnalysis(this);
}


/// Wavelet analysis window
nGenericPan*
neutrino::Wavelet() {
	return new nWavelet(this);
}

/// Integral inversion (Abel etc...)
nGenericPan*
neutrino::IntegralInversion() {
	return new nIntegralInversion(this);
}

/// Region Path
nGenericPan*
neutrino::RegionPath() {
	return new nRegionPath(this);
}

/// Region Path
nGenericPan*
neutrino::InterpolatePath() {
	return new nInterpolatePath(this);
}


/// ROTATE STUFF
nGenericPan*
neutrino::Rotate() {
	return new nRotate(this);
}

/// Affine STUFF
nGenericPan*
neutrino::Affine() {
	return new nAffine(this);
}

/// camera
nGenericPan*
neutrino::Camera() {
	return new nCamera(this);
}


// MONIOR DIRECTORY
nGenericPan*
neutrino::Monitor() {
	return new nMonitor(this);
}

//save and load across restart
void neutrino::saveDefaults(){
	QSettings my_set("neutrino","");
	my_set.beginGroup("nPreferences");
	my_set.setValue("geometry", pos());
	my_set.setValue("colorTable", my_w->my_view->colorTable);
	my_set.setValue("comboIconSizeDefault", my_w->toolBar->iconSize().width()/10-1);

	my_set.beginGroup("Properties");
	foreach(QByteArray ba, dynamicPropertyNames()) {
		if(ba.startsWith("NeuSave")) {
			my_set.setValue(ba, property(ba));
			qDebug() << ba;
		}
	}
	my_set.endGroup();
	my_set.endGroup();
}

void neutrino::loadDefaults(){
	QSettings my_set("neutrino","");
	my_set.beginGroup("nPreferences");
	move(my_set.value("geometry",pos()).toPoint());

	my_w->my_view->changeColorTable(my_set.value("colorTable",my_w->my_view->colorTable).toString());
	qDebug() << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << my_w->toolBar->iconSize();
	int comboIconSizeDefault=my_set.value("comboIconSizeDefault", my_w->toolBar->iconSize().width()/10-1).toInt();

	QSize mysize=QSize(10*(comboIconSizeDefault+1),10*(comboIconSizeDefault+1));
	foreach (QToolBar *obj, findChildren<QToolBar *>()) {
		if (obj->iconSize()!=mysize) {
			obj->hide();
			obj->setIconSize(mysize);
			obj->show();
		}
	}

	if (my_set.childGroups().contains("Properties")) {
		my_set.beginGroup("Properties");
		foreach(QString my_key, my_set.allKeys()) {
			setProperty(my_key.toUtf8().constData(), my_set.value(my_key));
			qDebug() << my_key;
		}
		my_set.endGroup();
	}

	my_set.endGroup();
}

nGenericPan*
neutrino::Preferences() {
	return new nPreferences(this);
}

void neutrino::about() {

	QDialog myabout(this);
	Ui::nAbout my_about;
	my_about.setupUi(&myabout);
	connect(my_about.buttonBox, SIGNAL(accepted()), &myabout, SLOT(close()));
	connect(my_about.buttonBox, SIGNAL(rejected()), &myabout, SLOT(close()));

    my_about.version->setText(QString(__VER_LATEST));
#ifdef __neutrino_key
	QString serial(qApp->property("nHash").toString());
	// copy serial to clipboard
	myabout.label->setText(myabout.label->text()+"\nSerial number:"+serial);
	QApplication::clipboard()->setText(serial);
#endif

	my_about.creditsText->setLineWrapMode(QTextEdit::FixedColumnWidth);
	my_about.creditsText->setLineWrapColumnOrWidth(80);
	QScrollBar *vScrollBar = my_about.creditsText->verticalScrollBar();
	vScrollBar->triggerAction(QScrollBar::SliderToMinimum);
	QApplication::processEvents();

	QDirIterator it(":licenses/", QDirIterator::Subdirectories);
	while (it.hasNext()) {
		QString fname=it.next();
		QFile lic(fname);
		if (lic.open(QFile::ReadOnly | QFile::Text)) {
			QString licenseText=QTextStream(&lic).readAll();
			if (!licenseText.isEmpty()) {
				my_about.creditsText->insertHtml("<h2>"+QFileInfo(fname).completeBaseName()+" license :</h2><PRE>");
				my_about.creditsText->insertPlainText(licenseText);
				my_about.creditsText->insertHtml("</PRE><br><hr><br>");
			}
		}
	}
	myabout.exec();
}

nLine* neutrino::line(QString name) {
	foreach (QObject* widget, children()) {
		nLine *obj=qobject_cast<nLine *>(widget);
		if (obj && obj->my_w.name->text() == name) {
			return obj;
		}
	}
	return NULL;
}

nRect* neutrino::rect(QString name) {
	foreach (QObject* widget, children()) {
		nRect *obj=qobject_cast<nRect *>(widget);
		if (obj && obj->my_w.name->text() == name) {
			return obj;
		}
	}
	return NULL;
}


nGenericPan* neutrino::openPan(QString panName, bool force) {

	nGenericPan *my_pan=nullptr;

	qDebug() << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << panName;
	int methodIdx=metaObject()->indexOfMethod((panName+"()").toLatin1().constData());
	qDebug() << "methodIdx" << methodIdx;
	if (methodIdx<0 && panName.size()>1) {
		QString other_panName=panName;
		other_panName.remove(0,1);
		qDebug() << "methodIdx" << methodIdx << panName << other_panName;
		methodIdx=metaObject()->indexOfMethod((other_panName+"()").toLatin1().constData());
		qDebug() << "methodIdx" << methodIdx << panName << other_panName;
		if (methodIdx>=0) {
			panName=other_panName;
		}
	}
	qDebug() << "methodIdx" << methodIdx;

	if (methodIdx>=0) {
		if (!strcmp(metaObject()->method(methodIdx).typeName(),"nGenericPan*") &&
				metaObject()->method(methodIdx).parameterTypes().empty()) {
			QMetaObject::invokeMethod(this,panName.toLatin1().constData(),Q_RETURN_ARG(nGenericPan*, my_pan));
		}
	}
	if (my_pan==nullptr) {
		foreach (QAction *my_action, findChildren<QAction *>()) {
			if (!my_action->data().isNull()) {
				nPluginLoader *my_qplugin=my_action->data().value<nPluginLoader*>();
				qDebug() << my_action->data() << my_qplugin;
				if (my_qplugin!=nullptr) {
                    qDebug() << panName << "plugin action" << my_qplugin->name();
                    if (panName==my_qplugin->name() || (panName.left(1)=="n" && panName.right(panName.size()-1) == my_qplugin->name())) {
						my_qplugin->launch();
						QApplication::processEvents();
						QObject *p_obj = my_qplugin->instance();
						if (p_obj) {
							nPanPlug *iface = qobject_cast<nPanPlug *>(p_obj);
							if (iface) {
								qDebug() << "reloaded";
								my_pan=iface->pan();
							}
						}
					}
				}
			}
		}
	}
	QApplication::processEvents();

	if (force && my_pan==nullptr) {
		my_pan=new nGenericPan(this);
	}
	return my_pan;
}


// ----------------------------------- scripting --------------------------------------

nGenericPan* neutrino::getPan(QString name) {
	foreach(nGenericPan* pan, getPanList()) {
		if(pan->panName()==name) return pan;
	}
	return nullptr;
}

nGenericPan* neutrino::newPan(QString my_string) {

    nGenericPan *my_pan=new nGenericPan(this);

    if (!my_string.isEmpty() && QFileInfo(my_string).exists()) {
        QFile file(my_string);
        file.open(QFile::ReadOnly);
        QUiLoader loader;
        QWidget *uiwidget = loader.load(&file);
        file.close();

        qDebug() << uiwidget->objectName();

        if (uiwidget) {
            my_pan->setProperty("panName",uiwidget->objectName());
            uiwidget->setParent(my_pan);

            my_pan->setUnifiedTitleAndToolBarOnMac(uiwidget->property("unifiedTitleAndToolBarOnMac").toBool());
            foreach (QWidget *my_widget, uiwidget->findChildren<QWidget *>()) {
                if(my_widget->objectName()=="centralwidget") {
                    my_pan->setCentralWidget(my_widget);
                }
            }
            foreach (QStatusBar *my_widget, uiwidget->findChildren<QStatusBar *>()) {
                my_pan->setStatusBar(my_widget);
            }
            foreach (QToolBar *my_widget, uiwidget->findChildren<QToolBar *>()) {
                my_pan->addToolBar(my_widget);
            }

            const QMetaObject *metaobject=uiwidget->metaObject();
            for (int i=0; i<metaobject->propertyCount(); ++i) {
                QMetaProperty metaproperty = metaobject->property(i);
                const char *name = metaproperty.name();
                QVariant value = uiwidget->property(name);
            }

            //            my_pan->setCentralWidget(uiwidget);
            my_pan->show();
        }
    }


	return my_pan;
}


// cool functions outside neutrino....
QVariant toVariant(anydata &my_data) {
	if (my_data.is_i()) {
		return QVariant::fromValue((int)my_data);
	} else if (my_data.is_d()) {
		return QVariant::fromValue((double)my_data);
	} else if (my_data.is_vec()) {
		vec2f my_val(my_data.get_str());
		return QVariant::fromValue(QPointF(my_val.x(),my_val.y()));
	} else if (my_data.is_str()) {
		return QVariant::fromValue(QString::fromStdString((std::string)my_data));
	}
	return QVariant();
}

anydata toAnydata(QVariant &my_variant) {
	bool ok;
	anydata my_data;
	int valInt=my_variant.toInt(&ok);
	if (ok) {
		my_data=valInt;
	} else {
		double valDouble=my_variant.toDouble(&ok);
		if (ok) {
			my_data=valDouble;
		} else {
			QPointF valPoint=my_variant.toPointF();
			if (!valPoint.isNull()) {
				vec2f my_vec2f(valPoint.x(),valPoint.y());
				my_data=my_vec2f;
			} else {
				std::string valStr=my_variant.toString().toStdString();
				if (!valStr.empty()) {
					my_data=valStr;
				}
			}
		}
	}
	return my_data;
}

