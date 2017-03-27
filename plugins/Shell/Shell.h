#ifndef __Shell
#define __Shell

#include <cmath>
#include <PythonQt.h>

#include <QtGui>
#include <QWidget>
#include <QShortcut>

#include "PythonQt_QtBindings.h"
#include "PythonQt.h"
#include "nPhysPyWrapper.h"

#include "nGenericPan.h"
#include "ui_Shell.h"
#include "nPluginLoader.h"

class neutrino;
/**
 @short Python Subsystem

 This class provides a simple implementation of python bindings via PythonQt library
 */
class Shell : public nGenericPan {
    Q_OBJECT

public:

    Q_INVOKABLE Shell(neutrino *);
    Ui::Shell my_w;

public slots:
    void loadScript(void);
    void runScript(void);
    void runScript(QString);
    void changeScriptsFolder();
    void changeSiteFolder();
};

class nPyWrapper : public QObject {
    Q_OBJECT

    public slots:
    neutrino* new_neutrino() {return new neutrino();};
    void delete_neutrino(neutrino* neu) {neu->deleteLater();};

};

class nPanPyWrapper : public QObject {
    Q_OBJECT

    public slots:
    nGenericPan* new_nPan(neutrino* neu) {
        DEBUG("here");
        return new nGenericPan(neu);
    }; // opens new neutrino with that image

    void delete_nPan(nGenericPan* pan) {
        DEBUG("here "<< pan->panName().toStdString());
        pan->deleteLater();
    };

};

class ShellPlug : public QObject, nPanPlug {
    Q_OBJECT Q_INTERFACES(nPanPlug)
    Q_PLUGIN_METADATA(IID "org.neutrino.panPlug")

public:

    ShellPlug() :
    nparent(nullptr)
    {
        qRegisterMetaType<Shell *>(name()+"*");
    }

    QByteArray name() {
        return "Shell";
    }

    QString menuEntryPoint() {
        return QString("Python");
    }

    nGenericPan* Python()
    {
        DEBUG("anyway I'm here " << nparent);
        if (nparent)
            return new Shell(nparent);
    }

    bool instantiate(neutrino *neu) {
        nparent=neu;
        PythonQt::init(PythonQt::IgnoreSiteModule|PythonQt::RedirectStdOut);

        PythonQt_init_QtBindings();

        PythonQt::self()->addDecorators(new nPhysPyWrapper());
        PythonQt::self()->registerCPPClass("nPhysD",NULL,"neutrino");

        PythonQt::self()->addDecorators(new nPanPyWrapper());
        PythonQt::self()->registerClass(& nGenericPan::staticMetaObject, "nPan", PythonQtCreateObject<nPanPyWrapper>);

		PythonQt::self()->registerClass(& nCustomPlot::staticMetaObject, "nPlot");
		PythonQt::self()->registerClass(&nLine::staticMetaObject, "nLine");
		PythonQt::self()->registerClass(&nRect::staticMetaObject, "nRect");
		PythonQt::self()->registerClass(&nEllipse::staticMetaObject, "nEllipse");
		PythonQt::self()->registerClass(&nPoint::staticMetaObject, "nPoint");

        PythonQt::self()->addDecorators(new nPyWrapper());
        PythonQt::self()->registerClass(& neutrino::staticMetaObject, "neutrino", PythonQtCreateObject<nPyWrapper>);

        QSettings settings("neutrino","");
        settings.beginGroup("Shell");
        foreach (QString spath, settings.value("siteFolder").toString().split(QRegExp("\\s*:\\s*"))) {
            qDebug() << "Python site folder " << spath;
            if (QFileInfo(spath).isDir()) PythonQt::self()->addSysPath(spath);
        }
        settings.endGroup();

        PythonQt::self()->getMainModule().addObject("nApp", qApp);



        QPointer<QMenu> menuPython = nPluginLoader::getMenu(menuEntryPoint(),neu);

        neu->my_w->menubar->addMenu(menuPython);
        settings.beginGroup("Shell");
        QDir scriptdir(settings.value("scriptsFolder").toString());
        qDebug() << scriptdir.exists() << scriptdir;
        if (scriptdir.exists()) {
            QStringList scriptlist = scriptdir.entryList(QStringList("*.py"));
            qDebug() << scriptlist;

            if (scriptlist.size() > 0) {
                foreach (QAction* myaction, menuPython->actions()) {
                    if (QFileInfo(myaction->data().toString()).suffix()=="py")
                        menuPython->removeAction(myaction);
                }
            }

            foreach (QString sname, scriptlist) {
                QAction *action = new QAction(neu);
                action->setText(QFileInfo(sname).baseName());
                qDebug() << "-----------------" << action;

                connect(action, SIGNAL(triggered()), this, SLOT(runPyScript()));
                action->setData(scriptdir.filePath(sname));
                menuPython->addAction(action);
            }
        }
        settings.endGroup();

        return nPanPlug::instantiate(neu);
    }

private:
    neutrino* nparent;

public slots:
    void
    runPyScript() {
        qDebug() << ">><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<>><<";
        QAction *action = qobject_cast<QAction *>(sender());
        if (action) {
            qDebug() << ">>>>>>>>>>>>>>>" << action << ">>>" <<action->data().toString();
            runPyScript(action->data().toString());
        }
    }

    void
    runPyScript(QString fname) {
        QFile t(fname);
        t.open(QIODevice::ReadOnly| QIODevice::Text);
        PythonQt::self()->getMainModule().evalScript(QTextStream(&t).readAll());
        t.close();
    }



};


#endif

