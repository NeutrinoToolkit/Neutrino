#include "Python.h"
#include <QFont>
#include <QThread>

#include "nPhysPyWrapper.h"

#include "Shell.h"

//#include "PythonQt_QtAll.h"


bool ShellPlug::instantiate(neutrino *neu) {
    qDebug() << "here" << neu;
    nparent=neu;

    Shell::startup();

    QPointer<QMenu> menuPython = nPluginLoader::getMenu(menuEntryPoint(),neu);
    qDebug() << "here";

    QSettings settings("neutrino","");
    settings.beginGroup("Shell");

    neu->my_w->menubar->addMenu(menuPython);
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


Shell::Shell(neutrino *nparent) : nGenericPan(nparent)
{
	my_w.setupUi(this);

    connect(my_w.actionRun_script, SIGNAL(triggered()), this, SLOT(loadScript()));
    connect(my_w.actionCleanup, SIGNAL(triggered()), this, SLOT(cleanup()));

	console = new PythonQtScriptingConsole(this, PythonQt::self()->getMainModule());
    console->setProperty("neutrinoSave",false);
	my_w.console->addWidget(console);
    QFont my_font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
    my_font.setPointSize(10);
    console->setFont(my_font);
	console->show();

    QKeySequence key_seq=QKeySequence(Qt::CTRL + Qt::Key_Return);
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

void Shell::startup() {
#if defined(Q_OS_WIN)
    qDebug() << "here1";
        PythonQt::init();
#else
    PythonQt::init(PythonQt::RedirectStdOut | PythonQt::IgnoreSiteModule);
#endif

    PythonQt_init_QtBindings();
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
    settings.beginGroup("Shell");

    QStringList sites=settings.value("siteFolder").toString().split(QRegExp("\\s*:\\s*"));
//Re-enable below if numpy start to work again:
#if defined(Q_OS_WIN)
    QDir base(QDir(qApp->applicationDirPath()).filePath("python2.7"));
    if (base.exists()) {
        sites << base.absolutePath();
        sites << base.filePath("plat-win32");
        sites << base.filePath("lib-tk");
        sites << base.filePath("site-packages");
        sites << base.filePath("lib-dynload");
    }
#endif

    foreach (QString spath, sites) {
        qDebug() << "Python site folder " << spath;
        if (QFileInfo(spath).isDir()) PythonQt::self()->addSysPath(spath);
    }

    settings.endGroup();
}

void Shell::cleanup() {
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

void Shell::loadScript(bool execInline) {
	QString fname;
    fname = QFileDialog::getOpenFileName(this,tr("Open python source"),property("NeuSave-fileScript").toString(),tr("Python script")+QString(" (*.py);;")+tr("Any files")+QString(" (*)"));
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

void Shell::runScript(QString cmd) {
    saveDefaults();
    DEBUG("\n" << cmd.toStdString());
    QVariant res=PythonQt::self()->getMainModule().evalScript(cmd);
    DEBUG("result " << res.type() << "\n" << res.toString().toStdString());
}

void Shell::runScript() {
    runScript(my_w.script->toPlainText());
}

void Shell::changeScriptsFolder() {
    QString dirName = QFileDialog::getExistingDirectory(this,tr("Python scripts directory"),my_w.changeScriptsFolder->text());
    if (QFileInfo(dirName).isDir()) {
        my_w.scriptsFolder->setText(dirName);
    }
}

void Shell::changeSiteFolder() {
    QString dirName = QFileDialog::getExistingDirectory(this,tr("Python site directory"),my_w.changeSiteFolder->text());
    if (QFileInfo(dirName).isDir()) {
        my_w.siteFolder->setText(dirName);
    }
}


