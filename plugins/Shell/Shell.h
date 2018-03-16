#ifndef __Shell
#define __Shell

#include <cmath>
#include <PythonQt.h>

#include <QtGui>
#include <QWidget>
#include <QShortcut>

#include "PythonQt_QtBindings.h"
#include "PythonQt.h"

#include "nGenericPan.h"
#include "ui_Shell.h"
#include "nPluginLoader.h"

#include "neutrino.h"

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
	void loadScript(bool execInline = false);
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
        return QString("Scripts");
    }

    nGenericPan* Python()
    {
        DEBUG("anyway I'm here " << nparent);
        if (nparent)
            return new Shell(nparent);
        else
            return nullptr;
    }

    QIcon icon() {
        return QIcon(":icons/python.png");
    }

    QKeySequence shortcut() {
        return QKeySequence(Qt::CTRL + Qt::ALT + Qt::Key_S);
    }

    int order() {
        return 100;
    }

    bool instantiate(neutrino *neu);

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

