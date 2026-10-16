#include "Python.h"
#include <QFont>
#include <QThread>
#include <QStandardPaths>

#include "nPhysPyWrapper.h"

#include "PyShell.h"

#include "PythonQt_QtAll.h"


bool PyShellPlug::instantiate(neutrino *neu) {
    qDebug() << "here" << neu;
    nparent=neu;

    // PyShell::startup();
    // python is started the first time it is needed (PyShell panel or script), see PyShell::startup()

    QPointer<QMenu> menuPython = nPluginLoader::getMenu(menuEntryPoint(),neu);
    qDebug() << "here";

    QSettings settings("neutrino","");
    settings.beginGroup("PyShell");

    // neu->my_w->menubar->addMenu(menuPython);
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

    qDebug() << "here";
    return nPanPlug::instantiate(neu);
}


PyShell::PyShell(neutrino *nparent) : nGenericPan(nparent)
{
	my_w.setupUi(this);

    if (!startup()) {
        my_w.script->setPlainText(tr("# Python could not be started: its standard library was not found"));
        my_w.script->setEnabled(false);
        show(true);
        return;
    }

    connect(my_w.actionRun_script, SIGNAL(triggered()), this, SLOT(loadScript()));
    connect(my_w.actionCleanup, SIGNAL(triggered()), this, SLOT(cleanup()));

	console = new PythonQtScriptingConsole(this, PythonQt::self()->getMainModule());
    console->setProperty("neutrinoSave",false);
	my_w.console->addWidget(console);
    QFont my_font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
    my_font.setPointSize(10);
    console->setFont(my_font);
	console->show();

    QKeySequence key_seq=QKeySequence(Qt::CTRL | Qt::Key_Return);
    my_w.script->setToolTip("Press "+key_seq.toString(QKeySequence::NativeText)+" to execute "+toolTip());
    QShortcut* my_shortcut = new QShortcut(key_seq, my_w.script);
    connect(my_shortcut, SIGNAL(activated()), this, SLOT(runScript()));

    my_w.script->setFont(my_font);
    my_w.script->setTabStopDistance(my_w.script->fontMetrics().horizontalAdvance(QString(" ").repeated(1000)) / 250.);


    PythonQt::self()->getMainModule().addObject("neu", nparent);

    show(true);
    
    connect(my_w.changeScriptsFolder, SIGNAL(released()), this, SLOT(changeScriptsFolder()));
    connect(my_w.changeSiteFolder, SIGNAL(released()), this, SLOT(changeSiteFolder()));
    
    DEBUG("->\n" << my_w.initScript->toPlainText().toStdString() << "\n<-");
    if (my_w.initScript->toPlainText().isEmpty()) {
        QFile t(":init.py");
        t.open(QIODevice::ReadOnly| QIODevice::Text);
        QString initScript=QTextStream(&t).readAll();
        t.close();    
        my_w.initScript->insertPlainText(initScript);
        runScript(initScript);
    } else {
        runScript(my_w.initScript->toPlainText());
    }

    console->setFocus();
}

// start python only once (PythonQt::self() is null before PythonQt::init and after PythonQt::cleanup)
// returns false if python cannot be started
bool PyShell::startup() {
    if (PythonQt::self()) return true;

    // packaged python: <home>/lib/pythonX.Y (Windows: <home>/Lib) next to the application
    // (<home>/lib64/pythonX.Y for a python with platlibdir lib64, e.g. fedora's in the AppImage)
    const QString pyver=QString("python%1.%2").arg(PY_MAJOR_VERSION).arg(PY_MINOR_VERSION);
    // auto hasStdlib = [&pyver](const QString &home) {
    //     return QFileInfo::exists(QDir(home).filePath("lib/"+pyver+"/os.py")) || QFileInfo::exists(QDir(home).filePath("Lib/os.py"));
    // };
    auto hasStdlib = [&pyver](const QString &home) {
        return QFileInfo::exists(QDir(home).filePath("lib/"+pyver+"/os.py")) || QFileInfo::exists(QDir(home).filePath("lib64/"+pyver+"/os.py")) || QFileInfo::exists(QDir(home).filePath("Lib/os.py"));
    };
    const QDir appDir(qApp->applicationDirPath());
    QString pyhome;
    for (auto &candidate : {appDir.filePath("../Resources/python"), appDir.filePath("python"), appDir.filePath("../lib/neutrino/python")}) {
        if (hasStdlib(candidate)) {
            pyhome=QDir(candidate).absolutePath();
            break;
        }
    }
    if (!pyhome.isEmpty()) {
        qputenv("PYTHONHOME", QFile::encodeName(pyhome));
    } else if (qEnvironmentVariableIsEmpty("PYTHONHOME")) {
#ifdef PYSHELL_PYTHON_HOME
        // not packaged: python finds the installation it was built with, if it is still there
        // (otherwise Py_Initialize would abort the whole application)
        if (!hasStdlib(PYSHELL_PYTHON_HOME)) {
            qWarning() << "PyShell: no python standard library found in" << PYSHELL_PYTHON_HOME;
            return false;
        }
#endif
    }

// #if defined(Q_OS_WIN)
//     qDebug() << "here1";
//         PythonQt::init();
// #else
//     PythonQt::init(PythonQt::RedirectStdOut | PythonQt::IgnoreSiteModule);
// #endif
    // same on every platform: without RedirectStdOut the console does not show the python output.
    // shipped python: its site-packages is added below. System python (linux packages, builds not packaged):
    // the site module adds its site-packages (python3-numpy...) and the user one (~/.local/lib/pythonX.Y)
    // PythonQt::init(PythonQt::RedirectStdOut | PythonQt::IgnoreSiteModule);
    PythonQt::init(pyhome.isEmpty() ? int(PythonQt::RedirectStdOut) : int(PythonQt::RedirectStdOut | PythonQt::IgnoreSiteModule));

//     PythonQt_init_QtBindings();
	PythonQt_QtAll::init();
    PythonQt::self()->addDecorators(new nPhysPyWrapper());
    PythonQt::self()->registerCPPClass("nPhysD",nullptr,"neutrino");

//    foreach (QAction *act, nparent->findChildren<QAction *>()) {
//        if (act->property("neuPlugin").isValid() && act->property("neuPlugin").toBool()) {
//            std::string name=act->text().replace(" ","_").toStdString();
//            qDebug() << act;
//            PythonQt::self()->registerCPPClass(name.c_str(),nullptr,"neutrino");
//        }
//    }

    PythonQt::self()->addDecorators(new nPanPyWrapper());
    PythonQt::self()->registerClass(&nGenericPan::staticMetaObject, "nPan", PythonQtCreateObject<nPanPyWrapper>);

    PythonQt::self()->registerClass(&nCustomPlot::staticMetaObject, "nPlot");
    PythonQt::self()->registerClass(&nLine::staticMetaObject, "nLine");
    PythonQt::self()->registerClass(&nRect::staticMetaObject, "nRect");
    PythonQt::self()->registerClass(&nEllipse::staticMetaObject, "nEllipse");
    PythonQt::self()->registerClass(&nPoint::staticMetaObject, "nPoint");

    PythonQt::self()->addDecorators(new nPyWrapper());
    PythonQt::self()->registerClass(& neutrino::staticMetaObject, "neutrino", PythonQtCreateObject<nPyWrapper>);

    PythonQt::self()->getMainModule().addObject("nApp", qApp);

    QSettings settings("neutrino","");
    settings.beginGroup("PyShell");

    // QStringList sites=settings.value("siteFolder").toString().split(QRegularExpression("\\s*:\\s*"));
    // addSysPath prepends: the last added comes first in sys.path
    // order of precedence: site folders of the settings, user modules, modules shipped with neutrino
    QStringList sites;

    // modules shipped with neutrino (numpy, scipy): site-packages is not added by python (IgnoreSiteModule)
    if (!pyhome.isEmpty()) {
        // sites << QDir(pyhome).filePath("lib/"+pyver+"/site-packages") << QDir(pyhome).filePath("Lib/site-packages");
        sites << QDir(pyhome).filePath("lib/"+pyver+"/site-packages") << QDir(pyhome).filePath("lib64/"+pyver+"/site-packages") << QDir(pyhome).filePath("Lib/site-packages");
    }
    // user modules: python -m pip install --target <folder> <module>, with the same python X.Y
    const QString userModules=QDir(QStandardPaths::writableLocation(QStandardPaths::AppDataLocation)).filePath(pyver);
    QDir().mkpath(userModules);
    sites << userModules;

    // sites << settings.value("siteFolder").toString().split(QRegularExpression("\\s*:\\s*"));
    // ':' on unix, ';' on windows (where ':' is in C:\...)
    sites << settings.value("siteFolder").toString().split(QRegularExpression(QString("\\s*%1\\s*").arg(QDir::listSeparator())));
// //Re-enable below if numpy start to work again:
// #if defined(Q_OS_WIN)
//     QDir base(QDir(qApp->applicationDirPath()).filePath("python2.7"));
//     if (base.exists()) {
//         sites << base.absolutePath();
//         sites << base.filePath("plat-win32");
//         sites << base.filePath("lib-tk");
//         sites << base.filePath("site-packages");
//         sites << base.filePath("lib-dynload");
//     }
// #endif
    // windows: python is shipped in <bin>/python/lib/pythonX.Y (mingw layout), found above

    foreach (QString spath, sites) {
        qDebug() << "Python site folder " << spath;
        if (QFileInfo(spath).isDir()) PythonQt::self()->addSysPath(spath);
    }

    settings.endGroup();
    return true;
}

void PyShell::cleanup() {
    PythonQt::cleanup();
    delete console;
    startup();
    PythonQt::self()->getMainModule().addObject("neu", nparent);
    console = new PythonQtScriptingConsole(this, PythonQt::self()->getMainModule());
    console->setProperty("neutrinoSave",false);
    my_w.console->addWidget(console);
    QFont my_font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
    my_font.setPointSize(10);
    console->setFont(my_font);
    console->show();
}

void PyShell::loadScript(bool execInline) {
	QString fname;
    fname = QFileDialog::getOpenFileName(this,tr("Open python source"),property("NeuSave-fileScript").toString(),tr("Python script")+QString(" (*.py);;")+tr("Any file")+QString(" (*)"));
	if (!fname.isEmpty()) {
        setProperty("NeuSave-fileScript",fname);
		if (execInline) {
			QFile t(fname);
			t.open(QIODevice::ReadOnly| QIODevice::Text);
			QTextStream out(&t);
			QString toRun=out.readAll();
			t.close();
			runScript(toRun);
		} else {
			QString toRun;
			toRun.append("g = globals().copy()\n");
			toRun.append(QString("g['__file__'] = '%1'\n").arg(fname));
			toRun.append(QString("execfile('%1', g)\n").arg(fname));
			runScript(toRun);
		}
	}
}

void PyShell::runScript(QString cmd) {
    saveDefaults();
    DEBUG("\n" << cmd.toStdString());
    QVariant res=PythonQt::self()->getMainModule().evalScript(cmd);
    DEBUG("result " << res.typeId() << "\n" << res.toString().toStdString());
}

void PyShell::runScript() {
    runScript(my_w.script->toPlainText());
}

void PyShell::changeScriptsFolder() {
    QString dirName = QFileDialog::getExistingDirectory(this,tr("Python scripts directory"),my_w.changeScriptsFolder->text());
    if (QFileInfo(dirName).isDir()) {
        my_w.scriptsFolder->setText(dirName);
    }
}

void PyShell::changeSiteFolder() {
    QString dirName = QFileDialog::getExistingDirectory(this,tr("Python site directory"),my_w.changeSiteFolder->text());
    if (QFileInfo(dirName).isDir()) {
        my_w.siteFolder->setText(dirName);
    }
}


