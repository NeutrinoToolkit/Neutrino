#include "gui/PythonQtScriptingConsole.h"
#include <QThread>
#include <QFont>

#include "nPhysPyWrapper.h"

#include "Shell.h"



bool ShellPlug::instantiate(neutrino *neu) {
    nparent=neu;

    PythonQt::init(PythonQt::IgnoreSiteModule|PythonQt::RedirectStdOut);

    PythonQt_init_QtBindings();
    PythonQt::self()->addDecorators(new nPhysPyWrapper());
    PythonQt::self()->registerCPPClass("nPhysD",NULL,"neutrino");

    PythonQt::self()->addDecorators(new nPanPyWrapper());
    PythonQt::self()->registerClass(& nGenericPan::staticMetaObject, "nPan", PythonQtCreateObject<nPanPyWrapper>);

    PythonQt::self()->registerClass(&nCustomPlot::staticMetaObject, "nPlot");
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

//        neu->my_w->toolBar->addAction(QIcon(":/icons/python.png"),"Python");

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


Shell::Shell(neutrino *nparent) : nGenericPan(nparent)
{
	my_w.setupUi(this);

    QFont my_font = QFontDatabase::systemFont(QFontDatabase::FixedFont);

    my_font.setPointSize(10);

    connect(my_w.actionRun_script, SIGNAL(triggered()), this, SLOT(loadScript()));

	PythonQtScriptingConsole *console;
	console = new PythonQtScriptingConsole(this, PythonQt::self()->getMainModule());
    console->setProperty("neutrinoSave",false);
	my_w.console->addWidget(console);
    console->setFont(my_font);
	console->show();

    QKeySequence key_seq=QKeySequence(Qt::CTRL + Qt::Key_Return);
    my_w.script->setToolTip("Press "+key_seq.toString(QKeySequence::NativeText)+" to execute "+toolTip());
    QShortcut* my_shortcut = new QShortcut(key_seq, my_w.script);
    connect(my_shortcut, SIGNAL(activated()), this, SLOT(runScript()));

    my_w.script->setFont(my_font);
    my_w.splitter->setSizes(QList<int>({2,1,0}));

    PythonQt::self()->getMainModule().addObject("neu", nparent);

    show();
    
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

void Shell::loadScript(bool execInline) {
	QString fname;
	fname = QFileDialog::getOpenFileName(this,tr("Open python source"),property("fileTxt").toString(),tr("Python script")+QString(" (*.py);;")+tr("Any files")+QString(" (*)"));
	if (!fname.isEmpty()) {
		setProperty("fileTxt",fname);
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

void Shell::runScript(void) {
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


