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


#ifdef Q_CC_GNU
#define QT_INIT_METAOBJECT
#endif

#include <QVector>
#include <QList>

#include "neutrino.h"

#include "nApp.h"

#include <QMetaObject>
#include <QtSvg>
#include <QDirIterator>

#include <QtUiTools>

#include <QPrintDialog>

#include "nView.h"

#include "nPluginLoader.h"
#include "nPanPlug.h"


#include "nPreferences.h"

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
    currentBuffer=nullptr;
    foreach (nPhysD *phys, physList) {
        delete phys;
    }
}

/// Creator
neutrino::neutrino():
    my_w(new Ui::neutrino),
    my_sbarra(new Ui::nSbarra),
    currentBuffer(nullptr)
{
    my_w->setupUi(this);
    setAcceptDrops(true);

    setProperty("winId",qApp->property("numWin").toInt()+1);
    qApp->setProperty("numWin",property("winId"));

    setProperty("NeuSave-gamma",1);
    setProperty("NeuSave-physNameLength",40);

    setProperty("NeuSave-askCloseUnsaved",true);

    setWindowTitle(property("winId").toString()+QString(": Neutrino"));

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


    connect(my_w->actionLine, SIGNAL(triggered()), this, SLOT(createDrawLine()));
    connect(my_w->actionRect, SIGNAL(triggered()), this, SLOT(createDrawRect()));
    connect(my_w->actionPoint, SIGNAL(triggered()), this, SLOT(createDrawPoint()));
    connect(my_w->actionEllipse, SIGNAL(triggered()), this, SLOT(createDrawEllipse()));

    connect(my_w->actionNew, SIGNAL(triggered()), this, SLOT(fileNew()));
    connect(my_w->actionOpen, SIGNAL(triggered()), this, SLOT(fileOpen()));
    connect(my_w->actionOpen_RAW, SIGNAL(triggered()), this, SLOT(openRAW()));
    connect(my_w->actionSave, SIGNAL(triggered()), this, SLOT(fileSave()));

    connect(my_w->actionReopen_to_saved, SIGNAL(triggered()), this, SLOT(fileReopen()));

    connect(my_w->actionSave_Session, SIGNAL(triggered()), this, SLOT(saveSession()));

    connect(my_w->actionExport, SIGNAL(triggered()), this, SLOT(exportGraphics()));
    connect(my_w->actionExport_all, SIGNAL(triggered()), this, SLOT(exportAllGraphics()));
    connect(my_w->actionExport_all_original_name, SIGNAL(triggered()), this, SLOT(exportAllGraphicsOriginalName()));
    connect(my_w->actionPrint, SIGNAL(triggered()), this, SLOT(print()));

    connect(my_w->actionQuit, SIGNAL(triggered()), qApp, SLOT(closeAllWindows())) ;

    connect(my_w->actionClose, SIGNAL(triggered()), this, SLOT(close()));
    connect(my_w->actionAbout, SIGNAL(triggered()), this, SLOT(about()));
    connect(my_w->actionPreferences, SIGNAL(triggered()), this, SLOT(Preferences()));


    connect(my_w->actionPrev_Buffer, SIGNAL(triggered()), my_w->my_view, SLOT(prevBuffer()));
    connect(my_w->actionNext_Buffer, SIGNAL(triggered()), my_w->my_view, SLOT(nextBuffer()));

    connect(my_w->actionClose_Buffer, SIGNAL(triggered()), this, SLOT(closeCurrentBuffer()), Qt::UniqueConnection);

    connect(my_w->actionShow_mouse, SIGNAL(triggered()), my_w->my_view, SLOT(nextMouseShape()));

    connect(my_w->actionRescale99, SIGNAL(triggered()), my_w->my_view, SLOT(rescale99()));
    connect(my_w->actionRescale_max, SIGNAL(triggered()), my_w->my_view, SLOT(rescaleColor()));

    connect(my_w->actionShow_less_pixels, SIGNAL(triggered()), my_w->my_view, SLOT(rescaleLess()));
    connect(my_w->actionShow_more_pixels, SIGNAL(triggered()), my_w->my_view, SLOT(rescaleMore()));

    connect(my_w->actionShow_ruler, SIGNAL(triggered()), my_w->my_view, SLOT(toggleRuler()));
    connect(my_w->actionShow_grid, SIGNAL(triggered()), my_w->my_view, SLOT(toggleGrid()));

    connect(my_w->actionRotate_left, SIGNAL(triggered()), this, SLOT(rotateLeft()));
    connect(my_w->actionRotate_right, SIGNAL(triggered()), this, SLOT(rotateRight()));
    connect(my_w->actionFlip_up_down, SIGNAL(triggered()), this, SLOT(flipUpDown()));
    connect(my_w->actionFlip_left_right, SIGNAL(triggered()), this, SLOT(flipLeftRight()));
    connect(my_w->actionTranspose, SIGNAL(triggered()), this, SLOT(transpose()));

    connect(my_w->actionSet_origin, SIGNAL(triggered()), my_w->my_view, SLOT(setMouseOrigin()));

    connect(my_w->actionZoom_in, SIGNAL(triggered()), my_w->my_view, SLOT(zoomIn()));
    connect(my_w->actionZoom_out, SIGNAL(triggered()), my_w->my_view, SLOT(zoomOut()));
    connect(my_w->actionZoom_eq, SIGNAL(triggered()), my_w->my_view, SLOT(zoomEq()));

    connect(my_w->actionIncrement, SIGNAL(triggered()), my_w->my_view, SLOT(incrGamma()));
    connect(my_w->actionDecrement, SIGNAL(triggered()), my_w->my_view, SLOT(decrGamma()));
    connect(my_w->actionReset, SIGNAL(triggered()), my_w->my_view, SLOT(resetGamma()));
    connect(my_w->actionCycle_over_items, SIGNAL(triggered()), my_w->my_view, SLOT(cycleOverItems()));

    connect(my_w->actionNext_LUT, SIGNAL(triggered()), my_w->my_view, SLOT(nextColorTable()));
    connect(my_w->actionPrevious_LUT, SIGNAL(triggered()), my_w->my_view, SLOT(previousColorTable()));

    connect(my_w->actionPlugin, SIGNAL(triggered()), this, SLOT(loadPlugin()));


    connect(my_w->actionLockColors, SIGNAL(toggled(bool)), my_w->my_view, SLOT(setLockColors(bool)));

    connect(my_w->actionCopy, SIGNAL(triggered()), my_w->my_view, SLOT(copyImage()));
    connect(my_w->actionExport_pixmap, SIGNAL(triggered()), my_w->my_view, SLOT(exportPixmap()));


    // ---------------------------------------------------------------------------------------------

    QWidget *sbarra=new QWidget(this);
    my_sbarra->setupUi(sbarra);
    my_w->statusbar->addPermanentWidget(sbarra, 0);

    setAttribute(Qt::WA_DeleteOnClose);
    setCentralWidget(my_w->centralwidget);

    connect(my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(mouseposition(QPointF)));
    connect(my_w->my_view, SIGNAL(zoomChanged(double)), this, SLOT(zoomChanged(double)));
    connect(my_w->my_view, SIGNAL(bufferChanged(nPhysD*)), this, SLOT(emitBufferChanged(nPhysD*)));
    connect(my_w->my_view, SIGNAL(updatecolorbar(QString)), my_w->statusbar, SLOT(showMessage(QString)));

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
    show();

    // plugins
    scanPlugins();


    QApplication::processEvents();

    // autosave config
    QTimer *timerSaveDefaults =  new QTimer(this);
    connect(timerSaveDefaults, SIGNAL(timeout()), this, SLOT(saveDefaults()));
    timerSaveDefaults->start(60000); // 1 min

    for (int i=0; i<metaObject()->methodCount(); i++){
        if (strcmp(metaObject()->method(i).typeName(),"nGenericPan*")==0 && metaObject()->method(i).parameterCount() == 0 )
            qDebug() << metaObject()->method(i).name() << metaObject()->method(i).methodSignature();
    }

    nApp *napp(qobject_cast<nApp*> (qApp));

    my_w->actionLog_info->setChecked(napp->log_win.isVisible());
    connect (my_w->actionLog_info, SIGNAL(toggled(bool)), &(napp->log_win), SLOT(setVisible(bool)));

    connect(my_w->actionCheck_for_updates, SIGNAL(triggered()),napp,SLOT(checkUpdates()));

//#define xxstring(s) xstring(s)
//#define xstring(s) #s

//    QString lista(xxstring(NEU_PLUGIN_LIST));
//    QStringList lista2=lista.split("_nplug_",Qt::SkipEmptyParts);
//    qCritical() <<lista2;
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

void neutrino::on_actionOpen_Glob_triggered () {
    QString dirName = QFileDialog::getExistingDirectory(this,tr("Change glob directory"),property("NeuSave-globdir").toString());
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
    if (currentBuffer) {
        physMath::phys_rotate_left(*dynamic_cast<physD*>(currentBuffer));
        currentBuffer->reset_display();
        showPhys();
    }
    my_w->actionFlipRotate->setIcon(my_w->menuTransformation->defaultAction()->icon());
    QSettings("neutrino","").setValue("menuTransformationDefault",my_w->menuTransformation->defaultAction()->text());

}

void neutrino::rotateRight() {
    my_w->menuTransformation->setDefaultAction(my_w->actionRotate_right);
    if (currentBuffer) {
        physMath::phys_rotate_right(*dynamic_cast<physD*>(currentBuffer));
        currentBuffer->reset_display();
        showPhys();
    }
    my_w->actionFlipRotate->setIcon(my_w->menuTransformation->defaultAction()->icon());
    QSettings("neutrino","").setValue("menuTransformationDefault",my_w->menuTransformation->defaultAction()->text());
}

void neutrino::flipUpDown() {
    my_w->menuTransformation->setDefaultAction(my_w->actionFlip_up_down);
    if (currentBuffer) {
        physMath::phys_flip_ud(*dynamic_cast<physD*>(currentBuffer));
        currentBuffer->reset_display();
        showPhys();
    }
    QSettings("neutrino","").setValue("menuTransformationDefault",my_w->menuTransformation->defaultAction()->text());
    my_w->actionFlipRotate->setIcon(my_w->menuTransformation->defaultAction()->icon());
}

void neutrino::flipLeftRight() {
    my_w->menuTransformation->setDefaultAction(my_w->actionFlip_left_right);
    if (currentBuffer) {
        physMath::phys_flip_lr(*dynamic_cast<physD*>(currentBuffer));
        currentBuffer->reset_display();
        showPhys();
    }
    QSettings("neutrino","").setValue("menuTransformationDefault",my_w->menuTransformation->defaultAction()->text());
    my_w->actionFlipRotate->setIcon(my_w->menuTransformation->defaultAction()->icon());
}

void neutrino::transpose() {
    my_w->menuTransformation->setDefaultAction(my_w->actionTranspose);
    if (currentBuffer) {
        physMath::phys_transpose(*dynamic_cast<physD*>(currentBuffer));
        currentBuffer->reset_display();
        showPhys();
    }
    QSettings("neutrino","").setValue("menuTransformationDefault",my_w->menuTransformation->defaultAction()->text());
    my_w->actionFlipRotate->setIcon(my_w->menuTransformation->defaultAction()->icon());
}

nPhysD* neutrino::getBuffer(int i) {
    return physList.value(i);
}

nPhysD* neutrino::getBuffer(QString name) {
    for(auto &phys : getBufferList()) {
        if (phys->getShortName() == name.toStdString() || phys->getName() == name.toStdString() )
            return phys;
    }
    return nullptr;
}

// ------------------ PLUGINS -----------------------

void
neutrino::scanPlugins(QString pluginsDirStr) {
    qInfo() << "Looking for plugins in" << pluginsDirStr;
    QDir pluginsDir(pluginsDirStr);
    if (pluginsDir.exists()) {

#if defined(Q_OS_WIN)
        QString extension("dll");
#elif defined(Q_OS_MAC)
        QString extension("dylib");
#elif defined(Q_OS_LINUX)
        QString extension("so");
#endif

        QStringList pluginlist;
        QDirIterator it(pluginsDir.absolutePath(), QStringList() << "*."+extension, QDir::Files, QDirIterator::Subdirectories);
        while (it.hasNext()) {
            pluginlist.append(it.next());
        }
        pluginlist.sort();
        QProgressDialog progress("Loading plugin", "Cancel", 0, pluginlist.size(), this);
        progress.setWindowModality(Qt::WindowModal);
        progress.show();
        progress.setCancelButton(nullptr);
        for (auto &pluginfile : pluginlist) {
            QString name_plugin=QFileInfo(pluginfile).baseName().replace("_"," ");
#if defined(Q_OS_MAC) || defined(Q_OS_LINUX)
                if (name_plugin.startsWith("lib")) {
                    name_plugin.remove(0,3);
                }
#endif
            progress.setLabelText("Plugin: "+name_plugin);
            progress.setValue(progress.value()+1);
            loadPlugin(pluginfile, false);
        }
        progress.close();
//        QStringList listdirPlugins=property("NeuSave-plugindirs").toStringList();
//        qDebug() << pluginsDir.absolutePath() << property("defaultPluginDir").toString();
//        if (!listdirPlugins.contains(pluginsDir.absolutePath()) && pluginsDir.absolutePath() != property("defaultPluginDir").toString())
//            listdirPlugins.append(pluginsDir.absolutePath());
//        setProperty("NeuSave-plugindirs",listdirPlugins);
    }
}

void
neutrino::scanPlugins() {
    QDir pluginsDir;
    pluginsDir.setPath(qApp->applicationDirPath());
#if defined(Q_OS_WIN)
    if (pluginsDir.dirName().toLower() == "debug" || pluginsDir.dirName().toLower() == "release")
        pluginsDir.cdUp();
#elif defined(Q_OS_MAC)
    pluginsDir.cdUp();
    pluginsDir.cd("Resources");
#elif defined(Q_OS_LINUX)
    pluginsDir.cdUp();
    pluginsDir.cd("lib/neutrino");
#endif
    pluginsDir.cd("plugins");
    qDebug() << "defaultPluginDir:" << pluginsDir.absolutePath();
    setProperty("defaultPluginDir",pluginsDir.absolutePath());
    scanPlugins(pluginsDir.absolutePath());

    QMap<QString, QVariant> pluginList(property("NeuSave-plugindirs").toMap());
    qDebug() << pluginList;
    for (auto& k : pluginList.keys()) {
        qDebug() << k << pluginList[k];
        if (pluginList[k].toInt() == 2) {
            scanPlugins(k);
        }
    }


//    if (property("NeuSave-plugindirs").isValid()) {
//        for (auto& d : property("NeuSave-plugindirs").toStringList()) {
//            QStringList dir_flag=d.split("!#!");
//            qDebug() << dir_flag;
//            if (dir_flag.size()>1) {
//                qDebug() << dir_flag[1].toInt();
//            }
//            if (dir_flag.size()>1 && dir_flag[1].toInt() == 1 && d!=pluginsDir.absolutePath())
//                scanPlugins(dir_flag[0]);
//        }
//    }
}

void
neutrino::loadPlugin()
{
    QStringList pnames=QFileDialog::getOpenFileNames(this,tr("Load Plugin"), property("NeuSave-loadPlugin").toString(),tr("Neutrino Plugins")+QString(" (*.dylib *.so *.dll);;")+tr("Any files")+QString(" (*)"));
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
        nPluginLoader *my_npl = new nPluginLoader(pname, this);
        qDebug() << "Loading plugin" <<  QFileInfo(pname).baseName() << " : " << my_npl->ok();
        if (launch) my_npl->run();
    }
}

void neutrino::emitBufferChanged(nPhysD *my_phys) {
    qDebug() << "here";
    QApplication::processEvents();
    if (my_phys) {
        double gamma_val=1.0;
        if (nPhysExists(my_phys)) {
            gamma_val=my_phys->gamma();
        }

        my_sbarra->gamma->setText(QString(QChar(0x03B3))+" "+QString(gamma_val<1? "1/"+ QLocale().toString(int(1.0/gamma_val)) : QLocale().toString(int(gamma_val))));

        QString winName=QString::fromUtf8(my_phys->getShortName().c_str());
        winName.prepend(property("winId").toString()+QString(":")+QLocale().toString(indexOf(my_phys))+QString(" "));

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
    QApplication::processEvents();
    emit bufferChanged(my_phys);
    QApplication::processEvents();
}

void neutrino::emitPanAdd(nGenericPan* pan) {
    QAction *act = new QAction(pan->windowTitle(),this);
    //    QAction *act = new QAction(pan->panName().replace("_"," "),this);
    QVariant v;
    v.setValue(pan);
    act->setData(v);
    connect(act, SIGNAL(triggered()),pan, SLOT(raiseIt()));
    my_w->menuWindow->addAction(act);

    panList.removeAll(pan);
    panList.append(pan);
    emit panAdd(pan);
}

void neutrino::emitPanDel(nGenericPan* pan) {
    foreach (QAction *action,  my_w->menuWindow->actions()) {
        if (action->data().value<nGenericPan*>() == pan) {
            action->deleteLater();
        }
    }
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
        nPhysD *phys=(nPhysD*) (action->data().value<nPhysD*>());
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
    setProperty("NeuSave-gamma",value);
}

// public slots

// file menu actions
neutrino* neutrino::fileNew() {
    return new neutrino();
}

void
neutrino::fileReopen() {
    if(currentBuffer && currentBuffer->getType()==PHYS_FILE) {
        QString fname=QString::fromUtf8(currentBuffer->getFromName().c_str());
        fileOpen(fname);
    }
}

QList <nPhysD *> neutrino::fileOpen(QString fname) {
    neutrino *my_neu=this;

    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    if (getBufferList().size()!=0 && my_set.value("openNewWindow",false).toBool()) {
        my_neu=fileNew();
    }
    my_set.endGroup();

    QList <nPhysD *> imagelist;
    if (QFile(fname).exists()) {

        setProperty("NeuSave-fileOpen", fname);
        if (!property("NeuSave-fileSave").isValid()) {
            setProperty("NeuSave-fileSave",property("NeuSave-fileOpen"));
        }
        my_set.beginGroup("nPreferences");
        bool separate_rgb= my_set.value("separateRGB",false).toBool();
        my_set.endGroup();

        QString suffix=QFileInfo(fname).suffix().toLower();
        if (suffix=="neus") {
            imagelist=openSession(fname);
        } else {
            std::vector<physD> my_vec;
            try {
                my_vec=physFormat::phys_open(QFile::encodeName(fname).toStdString(),separate_rgb);
            } catch (std::exception &e) {
                QMessageBox dlg(QMessageBox::Critical, tr("Exception"), e.what());
                dlg.setWindowFlags(dlg.windowFlags() | Qt::WindowStaysOnTopHint);
                dlg.exec();
            }
            for(std::vector<physD>::iterator it=my_vec.begin();it!=my_vec.end();it++) {
                nPhysD *ceppa = new nPhysD(*it);
                imagelist.push_back(ceppa);
            }
            if (imagelist.size()==0) {
                QImage image(fname);
                if (!image.isNull()) {
                    if (image.isGrayscale() || !separate_rgb) {
                        qDebug() << image.size();
                        nPhysD *datamatrix = new nPhysD(fname.toStdString());
                        datamatrix->resize(image.width(), image.height());
                        datamatrix->setType(PHYS_FILE);
                        for (int i=0;i<image.height();i++) {
                            for (int j=0;j<image.width();j++) {
                                datamatrix->Timg_matrix[i][j]= (double) (qGray(image.pixel(j,i)));
                            }
                        }
                        datamatrix->TscanBrightness();
                        imagelist.push_back(datamatrix);
                    } else {
                        std::array<nPhysD*,3> datamatrix;
                        std::array<std::string,3> name;
                        name[0]="Red";
                        name[1]="Green";
                        name[2]="Blue";
                        for (unsigned int k=0;k<3;k++) {
                            datamatrix[k] = new nPhysD(QFileInfo(fname).fileName().toStdString());
                            datamatrix[k]->setShortName(name[k]);
                            datamatrix[k]->setName(name[k]+" "+QFileInfo(fname).fileName().toStdString());
                            datamatrix[k]->setFromName(fname.toStdString());
                            datamatrix[k]->resize(image.width(), image.height());
                            datamatrix[k]->setType(PHYS_FILE);
                        }
                        for (int i=0;i<image.height();i++) {
                            for (int j=0;j<image.width();j++) {
                                QRgb px = image.pixel(j,i);
                                datamatrix[0]->Timg_matrix[i][j]= static_cast<double>(qRed(px));
                                datamatrix[1]->Timg_matrix[i][j]= static_cast<double>(qGreen(px));
                                datamatrix[2]->Timg_matrix[i][j]= static_cast<double>(qBlue(px));
                            }
                        }
                        for (unsigned int k=0;k<3;k++) {
                            datamatrix[k]->TscanBrightness();
                            imagelist.push_back(datamatrix[k]);
                        }
                    }

                }
            }
            if (imagelist.size()>0) {
                QList <nPhysD *> imagelistold;
                for (auto &my_phys : imagelist) {
                    if (my_phys) {
                        if (my_phys->getSurf()>0) {
                            my_set.beginGroup("nPreferences");
                            if (my_set.value("enableNewImageSettings",false).toBool()) {

                                if (my_set.value("lockOrigin",false).toBool()) {
                                    my_phys->set_origin(my_set.value("originX",0.0).toDouble(),my_set.value("originY",0.0).toDouble());
                                }
                                if (my_set.value("lockScale",false).toBool()) {
                                    my_phys->set_scale(my_set.value("scaleX",1.0).toDouble(),my_set.value("scaleY",1.0).toDouble());
                                }

                                if (my_set.value("lockRotate",false).toBool()) {
                                    nPhysD rotated = my_phys->rotated(my_set.value("rotate",0.0).toDouble());
                                    delete my_phys;
                                    my_phys=new nPhysD(rotated);
                                }

                                if (my_set.value("lockFlip",false).toBool()) {
                                    if (my_set.value("flipX").toBool()) {
                                        physMath::phys_flip_ud(*dynamic_cast<physD*>(my_phys));
                                        my_phys->reset_display();
                                    }
                                    if (my_set.value("flipY").toBool()) {
                                        physMath::phys_flip_ud(*dynamic_cast<physD*>(my_phys));
                                        my_phys->reset_display();
                                    }
                                    if (my_set.value("transpose").toBool()) {
                                        physMath::phys_transpose(*dynamic_cast<physD*>(my_phys));
                                        my_phys->reset_display();
                                    }
                                }

                                if (my_set.value("lockMath",false).toBool()) {
                                    physMath::phys_subtract(*dynamic_cast<physD*>(my_phys),my_set.value("subtract",0).toDouble());
                                    physMath::phys_multiply(*dynamic_cast<physD*>(my_phys),my_set.value("multiply",1).toDouble());
                                }

                                if (my_set.value("lockBlur",false).toBool()) {
                                    physMath::phys_fast_gaussian_blur(*my_phys,my_set.value("blurX",1).toDouble(),my_set.value("blurY",1).toDouble());
                                }

                                if (my_set.value("lockCrop",false).toBool()) {
                                    physMath::phys_crop(*my_phys,my_set.value("cropW",my_phys->getW()).toInt(),my_set.value("cropH",my_phys->getH()).toInt(),my_set.value("cropDx",0).toInt(),my_set.value("cropDy",0).toInt());
                                }

                                if (my_set.value("lockColors",false).toBool()) {
                                    bool ok1, ok2;
                                    double mymin=my_set.value("colorMin").toDouble(&ok1);
                                    double mymax=my_set.value("colorMax").toDouble(&ok2);
                                    if (ok1 && ok2) {
                                        my_phys->prop["display_range"]=vec2f(mymin,mymax);
                                    }
                                    if (my_set.value("colorSpin").toInt() != 100) {
                                        DEBUG(my_phys->prop["display_range"]);
                                        qDebug() << my_set.value("colorSpin").toInt();
                                        my_phys->prop["display_range"]=physMath::getColorPrecentPixels(*my_phys,my_set.value("colorSpin").toInt());
                                        DEBUG(my_phys->prop["display_range"]);
                                    }
                                }
                            }
                            my_set.endGroup();
                            my_neu->addShowPhys(my_phys);
                            imagelistold << my_phys;
                        } else {
                            delete my_phys;
                        }
                    }
                }
                imagelist=imagelistold;
            } else if (suffix!="neus") {
                nOpenRAW *openRAW=(nOpenRAW *)getPan("OpenRaw");
                if (!openRAW) {
                    openRAW = new nOpenRAW(my_neu);
                }
                openRAW->add(fname);
            }
        }

        my_neu->updateRecentFileActions(fname);

        QApplication::processEvents();
    } else {
        QString formats("Neutrino Images (");
        for (auto &format : physFormat::phys_image_formats()) {
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
        foreach (QString fname, fnames) {
            imagelist.append(fileOpen(fname));
        }
    }
    for (auto& img: imagelist) {
        if (img->prop.have("neutrinoPanData")) {
            std::string panString=img->prop["neutrinoPanData"];
            img->prop.erase("neutrinoPanData");
            setPanData(panString);
        }
    }
    return imagelist;
}

void neutrino::setPanData(std::string panstring){
    DEBUG(">\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n>\n" << panstring);
    std::istringstream ifile(panstring);
    std::string line;
    while(ifile.peek()!=-1) {
        getline(ifile,line);
        QString qLine=QString::fromStdString(line);
        if (line.find("NeutrinoPan-begin")==0) {
            QStringList listLine=QString::fromStdString(line).split(" ");
            QString pName=listLine.at(1);

            nGenericPan *my_pan=openPan(pName, false);

            if (my_pan) {
                QApplication::processEvents();
                QTemporaryFile tmpFile(this);
                tmpFile.open();

                while(!ifile.eof()) {
                    getline(ifile,line);
                    qLine=QString::fromStdString(line);
                    qDebug() << qLine;
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
                QMessageBox::critical(this,tr("Session error"),tr("Cannot find method or plugin for ")+pName,  QMessageBox::Ok);
            }
        }
    }
}

std::string neutrino::getPanData(){
    std::stringstream ofile;
    for (int i=0;i<panList.size(); i++) {
        QString pName=panList.at(i)->metaObject()->className();
        QTemporaryFile tmpFile(this);
        if (tmpFile.open()) {
            QString tmp_filename=tmpFile.fileName();
            tmpFile.close();
            tmpFile.remove();
            panList.at(i)->saveSettings(tmp_filename);

            QFile file(tmp_filename);
            if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
                ofile << "NeutrinoPan-begin " << pName.toStdString() << std::endl;
                ofile.flush();
                file.seek(0);
                while (!file.atEnd()) {
                    QByteArray line = file.readLine();
                    qDebug() << line;
                    ofile << line.toStdString();
                }
                file.close();
                file.remove();
                ofile << "NeutrinoPan-end " << pName.toStdString() << std::endl;
                ofile.flush();
            } else {
                qWarning() << tr("Cannot write values for ")+panList.at(i)->panName()+QString("\n")+tmp_filename+QString("\n")+tr("Contact dev team.");
            }
        } else {
            qWarning() << tr("Cannot write values for ")+panList.at(i)->panName();
        }
    }
    DEBUG("><><><><" << ofile.str());
    return ofile.str();
}

void neutrino::saveSession (QString fname) {
    qDebug() << property("NeuSave-fileSave");
    if (fname.isEmpty()) {
        QString extensions=tr("Neutrino session")+QString(" (*.neus);;");
#ifdef HAVE_LIBTIFF
        if (QFileInfo(property("NeuSave-fileSave").toString()).suffix() == "neus") {
            extensions+=tr("Tiff session")+" (*.tiff *.tif);;";
           } else {
            extensions=tr("Tiff session")+" (*.tiff *.tif);;"+extensions;
        }
#endif
        QString fnameSave = QFileDialog::getSaveFileName(this,tr("Save Session"),property("NeuSave-fileSave").toString(),extensions+tr("Any files")+QString(" (*)"));
        if (!fnameSave.isEmpty()) {
            saveSession(fnameSave);
        }
    } else {
        QFileInfo file_info(fname);
        if (file_info.suffix()=="neus") {
            setProperty("NeuSave-fileSave", fname);
            //            for(int k = 0; k < (panList.size()/2); k++) panList.swap(k,panList.size()-(1+k));

            QProgressDialog progress("Save session", "Cancel", 0, physList.size()+1, this);
            progress.setWindowModality(Qt::WindowModal);
            progress.show();

            std::ofstream ofile(QFile::encodeName(fname).toStdString().c_str(), std::ios::out | std::ios::binary);
            ofile << "Neutrino " << __VER << " " << physList.size() << " " << panList.size() << std::endl;

            for (int i=0;i<physList.size(); i++) {
                if (progress.wasCanceled()) break;
                progress.setValue(i);
                if (! (physList.at(i)->getType()==PHYS_DYN)) {
                    progress.setLabelText(QString::fromUtf8(physList.at(i)->getShortName().c_str()));
                    QApplication::processEvents();
                    ofile << "NeutrinoImage" << std::endl;
                    physFormat::phys_dump_binary(physList.at(i),ofile);
                    physList.at(i)->setType(PHYS_FILE);
                } else {
                    qWarning() << "not saving " << QString::fromUtf8(physList.at(i)->getShortName().c_str());
                }
            }
            progress.setValue(physList.size());
            ofile << getPanData();
            progress.setValue(physList.size()+1);
            ofile.close();
        } else if (file_info.suffix().startsWith("tif")) {
            setProperty("NeuSave-fileSave", fname);
            std::vector <physD *> vecPhys;
            foreach (nPhysD * my_phys, physList) {
                if (! (my_phys->getType()==PHYS_DYN)) {
                    vecPhys.push_back(dynamic_cast<physD*>(my_phys));
                } else {
                    qWarning() << "not saving " << QString::fromUtf8(my_phys->getShortName().c_str());
                }
            }
            if (vecPhys.size()) {
                std::string pandata=getPanData();
                DEBUG(pandata);
                vecPhys[0]->prop["neutrinoPanData"] = pandata;
            }
            physFormat::phys_write_tiff(vecPhys,QFile::encodeName(fname).toStdString().c_str());
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
        if (physList.size()!=0) {
            neutrino*my_neu= new neutrino();
            my_neu->fileOpen(fname);
        } else {
            QProgressDialog progress("Load session", "Cancel", 0, 0, this);
            std::ifstream ifile(QFile::encodeName(fname).toStdString(), std::ios::in | std::ios::binary);
            std::string line;
            getline(ifile,line);
            QString qLine=QString::fromStdString(line);
            qInfo() << "Session version" << qLine;
            int counter=0;
            if (qLine.startsWith("Neutrino")) {
                progress.setWindowModality(Qt::WindowModal);
                progress.setLabelText(tr("Load Session ")+qLine.split(" ").at(1));
                progress.setMaximum(1+qLine.split(" ").at(2).toInt());
                progress.show();
                QApplication::processEvents();
                std::string panString;
                while(ifile.peek()!=-1) {
                    getline(ifile,line);
                    QString qLine=QString::fromStdString(line);
                    if (progress.wasCanceled()) break;
                    if (qLine.startsWith("NeutrinoImage")) {
                        progress.setValue(++counter);
                        nPhysD *my_phys=new nPhysD();
                        int ret=physFormat::phys_resurrect_binary(*my_phys,ifile);
                        if (ret>=0 && my_phys->getSurf()>0) {
                            progress.setLabelText(QString::fromUtf8(my_phys->getShortName().c_str()));
                            QApplication::processEvents();
                            imagelist.push_back(my_phys);
                            addShowPhys(my_phys);
                        } else {
                            delete my_phys;
                        }
                        QApplication::processEvents();
                    } else if (qLine.startsWith("NeutrinoPan-begin")) {
                        panString+=line+'\n';
                        while (line.find("NeutrinoPan-end")!=0) {
                            getline(ifile,line);
                            panString+=line+'\n';
                        }
                    }
                    QApplication::processEvents();
                }
                setPanData(panString);
            }
            ifile.close();
        }
    }
    qInfo() << fname << "contains" << imagelist.size() << " images";
    return imagelist;
}

void neutrino::addShowPhys(nPhysD* datamatrix) {
    addPhys(datamatrix);
    showPhys(datamatrix);
}

void neutrino::addPhys(nPhysD* datamatrix) {
    if ((!nPhysExists(datamatrix)) && datamatrix->getSurf()>0)	{
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
    action->setData(QVariant::fromValue( datamatrix));
    connect(action, SIGNAL(triggered()),this, SLOT(openRecentBuffer()));
    my_w->menuBuffers->addAction(action);
}

nPhysD* neutrino:: replacePhys(nPhysD* newPhys, nPhysD* oldPhys, bool show) { //TODO: this should be done in nPhysImage...
    qDebug() << newPhys << oldPhys;
    if (newPhys && newPhys->getSurf()) {
        bool redisplay = (currentBuffer==oldPhys);
        if (oldPhys && nPhysExists(oldPhys)) {
            *oldPhys=*newPhys;
            delete newPhys;
            newPhys=oldPhys;
        } else {
//            newPhys->prop.erase("display_range");
            addPhys(newPhys);
        }
        if (show || redisplay) {
            showPhys(newPhys);
        }
        emit physReplace(std::make_pair(oldPhys, newPhys));
    }
    QApplication::processEvents();
    return newPhys;
}

void neutrino::removePhys(nPhysD* datamatrix) {
    DEBUG(">>>>>>>>>>>>>>>>> ENTER ");
    if (nPhysExists(datamatrix)) {
        std::string physremovename = datamatrix->getShortName();
        DEBUG(">>>>>>>>>>>>>>>>> ENTER " << physremovename<< "  :  " << physList.size());
        int position=indexOf(datamatrix);
        if (position != -1) {
            physList.removeAll(datamatrix);
            QList<QAction *> lista=my_w->menuBuffers->actions();
            foreach (QAction* action, my_w->menuBuffers->actions()) {
                if (action->data() == QVariant::fromValue( datamatrix)) {
                    my_w->menuBuffers->removeAction(action);
                }
            }
        }
//        nApp::processEvents();
        if (datamatrix && !datamatrix->prop.have("keep_phys_alive")){
            DEBUG("removing from neutrino.cc");
            delete datamatrix;
//            datamatrix=nullptr;
        } else {
            DEBUG("not removing. PLEASE NOTE that this is a failsafe to avoid deleting stuff owned by python");
        }
        if (physList.size()>0) {
            int pos = position%physList.size();
            qDebug() << physList.size() << pos;
            qDebug() << physList;
            showPhys(physList.at(pos));
        } else {
            currentBuffer=nullptr;
            emitBufferChanged();
            setWindowTitle(property("winId").toString()+QString(": Neutrino"));
            setWindowFilePath("");
            zoomChanged(1);
            my_w->my_view->my_pixitem.setPixmap(QPixmap(":icons/icon.png"));
            my_w->my_view->setSize();
        }
        //    QApplication::processEvents(QEventLoop::WaitForMoreEvents);
        DEBUG(">>>>>>>>>>>>>>>>> EXIT " << physremovename << "  :  " << physList.size());
    }
    emit physDel(datamatrix);
    DEBUG(">>>>>>>>>>>>>>>>> EXIT ");
}

void
neutrino::showPhys(nPhysD* my_phys) {
    if (!my_phys) my_phys=currentBuffer;
    if (my_phys && !my_phys->prop.have("gamma")) {
        my_phys->prop["gamma"]=property("NeuSave-gamma").toInt();
    }
    my_w->my_view->showPhys(my_phys);
}

QString graphicsTypes(QString fname) {
    QStringList exts={"pdf","svg","png"};
    QFileInfo finfo(fname);
    QString ext=finfo.suffix().toLower();
    if (exts.contains(ext)) {
        exts.removeAll(ext);
        exts.push_back(ext);
    }
    QString ftypes="Any files (*)";
    for (auto &str: exts) {
        ftypes.prepend(str.toUpper()+" (*."+str+");; ");
    }
    return ftypes;
}

void neutrino::exportGraphics () {
    QString ftypes=graphicsTypes(property("NeuSave-fileExport").toString());
    QString fout = QFileDialog::getSaveFileName(this,tr("Save Drawing"),property("NeuSave-fileExport").toString(),ftypes);
    if (!fout.isEmpty())
        exportGraphics(fout);
}


void neutrino::exportAllGraphics () {
    QString ftypes=graphicsTypes(property("NeuSave-fileExport").toString());
    QString fout = QFileDialog::getSaveFileName(this,tr("Save All Drawings"),property("NeuSave-fileExport").toString(),ftypes);
    if (!fout.isEmpty()) {
        for (int i=0;i<physList.size() ; i++) {
            my_w->my_view->nextBuffer();
            if (currentBuffer) {
                QFileInfo fi(fout);
                exportGraphics(fi.path()+"/"+fi.baseName()+QString("_")+QString("%1").arg(i, 3, 10, QChar('0'))+QString("_")+QString::fromStdString(currentBuffer->getShortName())+"."+fi.completeSuffix());
            }
        }
        setProperty("NeuSave-fileExport",fout);
    }
}

void neutrino::exportAllGraphicsOriginalName () {
    QString ftypes=graphicsTypes(property("NeuSave-fileExport").toString());
    QString fout = QFileDialog::getSaveFileName(this,tr("Save All Drawings"),property("NeuSave-fileExport").toString(),ftypes);
    if (!fout.isEmpty()) {
        for (int i=0;i<physList.size() ; i++) {
            my_w->my_view->nextBuffer();
            if (currentBuffer) {
                QFileInfo fi(fout);
                QString origfname=QFileInfo(QString::fromStdString(currentBuffer->getFromName())).baseName();
                exportGraphics(fi.path()+"/"+origfname+"."+fi.completeSuffix());
            }
        }
        setProperty("NeuSave-fileExport",fout);
    }
}

void neutrino::exportGraphics (QString fout) {
    setProperty("NeuSave-fileExport",fout);
    bool resetmouse=my_w->my_view->my_mouse.isVisible();
    my_w->my_view->my_mouse.setVisible(false);
    QSize my_size=QSize(getScene().width(), getScene().height());
    QRect my_rect=my_w->my_view->my_tics.boundingRect().toRect();
    if (QFileInfo(fout).suffix().toLower()==QString("pdf")) {
        QPrinter myPrinter(QPrinter::ScreenResolution);
        myPrinter.setOutputFileName(fout);
        myPrinter.setOutputFormat(QPrinter::PdfFormat);
        myPrinter.setPaperSize(my_rect.size(),QPrinter::DevicePixel);
        //        myPrinter.setPageMargins(my_size.width()/10.0, my_size.height()/10.0, my_size.width()/10.0, my_size.height()/10.0, QPrinter::DevicePixel);
        QPainter myPainter(&myPrinter);
        myPainter.setViewport(my_rect);
        getScene().render(&myPainter);
    } else if (QFileInfo(fout).suffix().toLower()==QString("svg")) {
        QSvgGenerator svgGen;
        svgGen.setFileName(fout);
        svgGen.setSize(my_size);
        svgGen.setViewBox(my_w->my_view->my_tics.boundingRect().toRect());
        svgGen.setTitle("Neutrino");
        svgGen.setDescription(windowFilePath());
        QPainter painter(&svgGen);
        getScene().render(&painter);
    } else {
        QPixmap(my_w->my_view->grab()).save(fout);
    }
    my_w->my_view->my_mouse.setVisible(resetmouse);
}

void neutrino::closeEvent (QCloseEvent *e) {
    qDebug() << "here" << sender();
    qDebug() << QApplication::activeWindow();
    if (!QApplication::activeWindow() || QApplication::activeWindow() == this) {
        disconnect(my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(mouseposition(QPointF)));
        if (fileClose()) {
            saveDefaults();
            e->accept();
        } else {
            e->ignore();
            connect(my_w->my_view, SIGNAL(mouseposition(QPointF)), this, SLOT(mouseposition(QPointF)));
        }
    } else {
        QApplication::activeWindow()->close();
        e->ignore();
    }
}

void neutrino::on_actionKeyboard_shortcut_triggered() {
    bool ok;
    QString text = QInputDialog::getText(this,"Open","", QLineEdit::Normal,property("NeuSave-shortcut").toString(), &ok, Qt::Sheet);
    if (ok && !text.isEmpty()) {
        setProperty("NeuSave-shortcut",text.replace(" ","_"));
        nGenericPan *my_pan= openPan(text,false);
        if(!my_pan) {
            statusBar()->showMessage(tr("Can't find ")+text, 2000);
        }
    }
}

void neutrino::keyPressEvent (QKeyEvent *)
{
}

void neutrino::keyReleaseEvent (QKeyEvent *)
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
        QList<QByteArray> my_data=e->mimeData()->data("data/neutrino").split(' ');
        foreach(QByteArray bytephys, my_data) {
            bool ok=false;
            DEBUG("here\ndrop\nhere\ndrop\nhere\ndrop\nhere\ndrop\nhere\ndrop\nhere\ndrop\nhere\ndrop\nhere\ndrop\nhere\ndrop\nhere\ndrop\nhere\ndrop\nhere\ndrop\nhere\ndrop\n");
            nPhysD *my_phys=reinterpret_cast<nPhysD *> (bytephys.toLongLong(&ok));
            if (ok && my_phys) {
                if (nPhysExists(my_phys)) {
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
            fileOpen(qurl.toLocalFile());
        }
    }
}

void
neutrino::zoomChanged(double zoom) {
    statusBar()->showMessage(QString(tr("Zoom")+" : %1").arg(100.0*zoom,0,' ',1),2000);
    emit nZoom(zoom);
    update();
}

double
neutrino::getZoom() const {
    return my_w->my_view->transform().m11();
}

void
neutrino::mouseposition(QPointF pos_mouse) {
    my_sbarra->pos_x->setNum(static_cast<int>(pos_mouse.x()));
    my_sbarra->pos_y->setNum(static_cast<int>(pos_mouse.y()));

    if (nPhysExists(currentBuffer)) {
        vec2f vec=currentBuffer->to_real(vec2f(pos_mouse.x(),pos_mouse.y()));
        QPointF pos=QPointF(vec.x(),vec.y());
        my_sbarra->dx->setNum(pos.x());
        my_sbarra->dy->setNum(pos.y());
        double val=currentBuffer->point(pos_mouse.x(),pos_mouse.y());
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

    QString suffix=QFileInfo(property("NeuSave-fileSave").toString()).suffix().toLower();

    for (auto &format : physFormat::phys_image_formats()) {
        formats << QString::fromStdString(format);
    }
    formats << "neus";
    foreach (QByteArray format, QImageWriter::supportedImageFormats() ) {
        if (!formats.contains(format))
            formats << format ;
    }

    if (formats.contains(suffix)) {
        formats.removeAll(suffix);
        formats.prepend(suffix);
    }

    foreach(QString format, formats ) {
        allformats += format + " files (*."+format+");; ";
    }

    allformats+=("Any files (*)");

    return QFileDialog::getSaveFileName(this, "Save to...",property("NeuSave-fileSave").toString(),allformats);
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
        setProperty("NeuSave-fileSave", fname);
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
                    case QMessageBox::Cancel:
                        return;
                }
            }

        }

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
        if (suffix.startsWith("neu")) {
            physFormat::phys_dump_binary(phys,QFile::encodeName(fname).toStdString().c_str());
        } else if (fname.endsWith("tif") || fname.endsWith("tiff")) {
            physFormat::phys_write_tiff(phys,QFile::encodeName(fname).toStdString().c_str());
        } else if (suffix.startsWith("fit")) {
            physFormat::phys_write_fits(phys,QFile::encodeName("!"+fname).toStdString().c_str(),4);
        } else if (suffix.startsWith("hdf")) {
            physFormat::phys_write_HDF4(phys,QFile::encodeName(fname).toStdString().c_str());
        } else if (suffix.startsWith("txt") || suffix.startsWith("dat")) {
            phys->writeASC(QFile::encodeName(fname).toStdString().c_str());
        } else {
            my_w->my_view->my_pixitem.pixmap().save(QFile::encodeName(fname));
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
    qDebug() << "here" << sender();
    bool askAll=true;
    foreach (nPhysD *phys, physList) {
        DEBUG( phys->getName() << " " << phys->getType());
        if (askAll && phys->getType()==	PHYS_DYN && property("NeuSave-askCloseUnsaved").toBool()==true) {
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
            }
        }
    }

    QApplication::processEvents();
    for (auto &pan : getPanList()) {
        qDebug() << pan->windowTitle();
    }
    for (auto &pan : getPanList()) {
        pan->hide();
        pan->close();
        QApplication::processEvents();
    }

    QApplication::processEvents();

    deleteLater();
    return true;
}

void neutrino::on_actionClose_All_Buffers_triggered() {
    while (physList.size()) closeCurrentBuffer();
//    for (auto &my_phys : physList) {
//        qDebug() << "->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->->" << my_phys;
//        removePhys(my_phys);
//    }
}
void neutrino::closeBuffer(nPhysD* my_phys) {
    QApplication::processEvents();
    if (nPhysExists(my_phys))  {
        if (my_phys->getType()==PHYS_DYN && property("NeuSave-askCloseUnsaved").toBool()==true) {
            int res=QMessageBox::warning(this,tr("Attention"),
                                         tr("The image")+QString("\n")+QString::fromUtf8(my_phys->getName().c_str())+QString("\n")+tr("has not been saved. Do you vant to save it now?"),
                                         QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
            switch (res) {
                case QMessageBox::Yes:
                    fileSave(my_phys); // TODO: add here a check for a cancel to avoid exiting
                    break;
                case QMessageBox::No:
                    removePhys(my_phys);
                    break;
            }
        } else {
            removePhys(my_phys);
        }
    }
    QApplication::processEvents();
}

void neutrino::closeCurrentBuffer() {
    nPhysD *my_phys=getCurrentBuffer();
    closeBuffer(my_phys);
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

void neutrino::print()
{
    QPrinter printer(QPrinter::HighResolution);
    QPrintDialog *printDialog = new QPrintDialog(&printer, this);
    if (printDialog->exec() == QDialog::Accepted) {
        my_w->my_view->my_mouse.hide();
        QPainter painter(&printer);
        foreach (QGraphicsItem *oggetto, getScene().items() ) {
            if (qgraphicsitem_cast<nLine *>(oggetto)) {
                nLine *my_nline = static_cast<nLine*>(oggetto);
                my_nline->selectThis(false);
            }
        }
        getScene().render(&painter);
        my_w->my_view->my_mouse.show();
    }
}

/// Open raw window
nGenericPan*
neutrino::openRAW() {
    QStringList fnames;
    nGenericPan *win = nullptr;
    fnames = QFileDialog::getOpenFileNames(this,tr("Open RAW"),"",tr("Any files")+QString(" (*)"));
    if (fnames.size()) {
        win=getPan("nOpenRAW");
        if (!win) win= new nOpenRAW(this);
        nOpenRAW *winRAW=qobject_cast<nOpenRAW*>(win);
        if (winRAW) winRAW->add(fnames);
    }
    return win;
}

//save and load across restart
void neutrino::saveDefaults() {
    QSettings my_set("neutrino","");
    qDebug() << my_set.fileName();
    my_set.beginGroup("nPreferences");
    my_set.setValue("geometry", pos());
    my_set.setValue("comboIconSizeDefault", my_w->toolBar->iconSize().width()/10-1);

    my_set.beginGroup("Properties");
    foreach(QByteArray ba, dynamicPropertyNames()) {
        if(ba.startsWith("NeuSave")) {
            my_set.setValue(ba, property(ba));
        }
    }
    my_set.endGroup();
    my_set.endGroup();
}

void neutrino::loadDefaults(){
    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    move(my_set.value("geometry",pos()).toPoint());
    qDebug() << "Reading defaults from" <<   my_set.fileName();
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

    my_about.version->setText(QApplication::applicationVersion());
    my_about.sha->setText(QString(__VER_SHA));
    QDirIterator it(":licenses/", QDirIterator::Subdirectories);
    QStringList licenses;
    while (it.hasNext()) {
        QString lic=it.next();
        if (lic.contains("Neutrino")) {
            licenses.prepend(lic);
        } else {
            licenses.append(lic);
        }
    }

    for(auto& fname: licenses) {
        QString basename=QFileInfo(fname).completeBaseName();
        QFile lic(fname);
        if (lic.open(QFile::ReadOnly | QFile::Text)) {
            QString licenseText=QTextStream(&lic).readAll().replace("\n","<br>");
            if (!licenseText.isEmpty()) {
                my_about.creditsText->insertHtml("<h1>"+basename+" license :</h1><PRE>");
                my_about.creditsText->insertHtml(licenseText);
                my_about.creditsText->insertHtml("</PRE><br><hr><br>");
            }
        }
    }

    my_about.creditsText->insertHtml("<h2>QT version :"+QLibraryInfo::version().toString()+"</h2>");

    my_about.creditsText->moveCursor(QTextCursor::Start);
    my_about.creditsText->ensureCursorVisible();
//    for (int id=QMetaType::User; id< 2000; id++){
//        qDebug() << id;
//        if (QMetaType(id).isRegistered()) {
//            qDebug() << "registered";
//            void *myClassPtr = QMetaType::create(id);
//            qDebug() << id << myClassPtr;
//            if(myClassPtr) {
//                QObject *my_qobject = static_cast<QObject*>(myClassPtr);
//                if (my_qobject) {
//                    qWarning() << my_qobject->metaObject()->className();
//                }
//                QMetaType::destroy(id, myClassPtr);
//            }
//        }
//    }

    myabout.exec();
}

nLine* neutrino::line(QString name) {
    foreach (QObject* widget, children()) {
        nLine *obj=qobject_cast<nLine *>(widget);
        if (obj && obj->my_w.name->text() == name) {
            return obj;
        }
    }
    return nullptr;
}

nRect* neutrino::rect(QString name) {
    foreach (QObject* widget, children()) {
        nRect *obj=qobject_cast<nRect *>(widget);
        if (obj && obj->my_w.name->text() == name) {
            return obj;
        }
    }
    return nullptr;
}


nGenericPan* neutrino::openPan(QString pName, bool force) {

    nGenericPan *my_pan=nullptr;

    qDebug() << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << pName;
    int methodIdx=metaObject()->indexOfMethod((pName+"()").toLatin1().constData());
    qDebug() << "methodIdx" << methodIdx;
    if (methodIdx<0 && pName.size()>1) {
        QString tmp_pName=pName;
        tmp_pName.remove(0,1);
        qDebug() << "methodIdx" << methodIdx << pName << tmp_pName;
        methodIdx=metaObject()->indexOfMethod((tmp_pName+"()").toLatin1().constData());
        qDebug() << "methodIdx" << methodIdx << pName << tmp_pName;
        if (methodIdx>=0) {
            pName=tmp_pName;
        }
    }
    qDebug() << "methodIdx" << methodIdx;

    if (methodIdx>=0) {
        if (!strcmp(metaObject()->method(methodIdx).typeName(),"nGenericPan*") &&
                metaObject()->method(methodIdx).parameterTypes().empty()) {
            QMetaObject::invokeMethod(this,pName.toLatin1().constData(),Q_RETURN_ARG(nGenericPan*, my_pan));
        }
    }
    if (my_pan==nullptr) {
        foreach (QAction *my_action, findChildren<QAction *>()) {
            if (!my_action->data().isNull()) {
                nPluginLoader *my_qplugin=my_action->data().value<nPluginLoader*>();
                qDebug() << my_action->data() << my_qplugin;
                if (my_qplugin!=nullptr) {
                    qDebug() << pName << "plugin action" << my_qplugin->name();
                    if (pName==my_qplugin->name()) {
                        my_qplugin->run();
                        QApplication::processEvents();
                        QObject *p_obj = my_qplugin->instance();
                        if (p_obj) {
                            nPanPlug *iface = qobject_cast<nPanPlug *>(p_obj);
                            if (iface) {
                                qDebug() << "reloaded";
                                my_pan=iface->pan();
                                break; // important otherwise pan might get closed in the meanwhile and give segfault
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
        if(pan->panName()==name) {
            pan->raiseIt();
            return pan;
        }
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
        return QVariant::fromValue(static_cast<int>(my_data));
    } else if (my_data.is_d()) {
        return QVariant::fromValue(static_cast<double>(my_data));
    } else if (my_data.is_vec()) {
        vec2f my_val(my_data.get_str());
        return QVariant::fromValue(QPointF(my_val.x(),my_val.y()));
    } else if (my_data.is_str()) {
        return QVariant::fromValue(QString::fromStdString(static_cast<std::string>(my_data)));
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

