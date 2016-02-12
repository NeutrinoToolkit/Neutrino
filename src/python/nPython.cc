#include "nPython.h"
#include "gui/PythonQtScriptingConsole.h"
#include "neutrino.h"
#include <QThread>

nPython::nPython(neutrino *nparent, QString winname) : nGenericPan(nparent, winname) {

	my_w.setupUi(this);

	connect(my_w.loadScript, SIGNAL(released()), this, SLOT(loadScript()));
	connect(my_w.runScript, SIGNAL(released()), this, SLOT(runScript()));

	PythonQtScriptingConsole *console;
	console = new PythonQtScriptingConsole(this, PythonQt::self()->getMainModule());
	console->setFont(my_w.scriptFile->font());
	my_w.console->addWidget(console);
	console->show();

	QCompleter *completer = new QCompleter(this);
	completer->setModel(new QDirModel(completer));
	completer->setCompletionMode(QCompleter::PopupCompletion);
	my_w.scriptFile->setCompleter(completer);

	decorate();
    
    connect(my_w.changeScriptsFolder, SIGNAL(released()), this, SLOT(changeScriptsFolder()));
    connect(my_w.changeSiteFolder, SIGNAL(released()), this, SLOT(changeSiteFolder()));
    
    DEBUG("->" << my_w.initScript->toPlainText().toStdString() << "<-");
    if (my_w.initScript->toPlainText().isEmpty()) {
        QFile t(":neutrinoInit.py");
        t.open(QIODevice::ReadOnly| QIODevice::Text);
        QString initScript=QTextStream(&t).readAll();
        DEBUG(initScript.toStdString());
        PythonQt::self()->getMainModule().evalScript(initScript);
        t.close();    
        my_w.initScript->insertPlainText(initScript);
    }       
    console->setFocus();
}

void nPython::loadScript(void) {
	QString fname;
	fname = QFileDialog::getOpenFileName(this,tr("Open python source"),my_w.scriptFile->text(),tr("Python script")+QString(" (*.py);;")+tr("Any files")+QString(" (*)"));
	if (!fname.isEmpty()) {
		my_w.scriptFile->setText(fname);
		my_w.scriptFile->setCursorPosition(0);
	}
}

void nPython::runScript(void) {
	QFile t(my_w.scriptFile->text());
	t.open(QIODevice::ReadOnly| QIODevice::Text);
	QTextStream out(&t);
	PythonQt::self()->getMainModule().evalScript(out.readAll());
	t.close();
}

void nPython::changeScriptsFolder() {
    QString dirName = QFileDialog::getExistingDirectory(this,tr("Python scripts directory"),my_w.changeScriptsFolder->text());
    if (QFileInfo(dirName).isDir()) {
        my_w.scriptsFolder->setText(my_w.scriptsFolder->text()+":"+dirName);
    }
}

void nPython::changeSiteFolder() {
    QString dirName = QFileDialog::getExistingDirectory(this,tr("Python site directory"),my_w.changeSiteFolder->text());
    if (QFileInfo(dirName).isDir()) {
        my_w.siteFolder->setText(dirName);
    }
}


