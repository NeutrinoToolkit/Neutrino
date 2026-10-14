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

#include "JavaScript.h"
#include "nPhysImageF.h"
#include <QScopedValueRollback>

Q_DECLARE_METATYPE(QList<nGenericPan*>);
Q_DECLARE_METATYPE(QList<neutrino*>);

JavaScript::JavaScript(neutrino *my_nparent) : nGenericPan(my_nparent)
// my_eng(my_nparent) // the engine is a member: no QObject parent
{
    setupUi(this);
    my_eng.globalObject().setProperty("neu", my_eng.newQObject(nparent));
    my_eng.globalObject().setProperty("nApp", my_eng.newQObject(qApp));
    my_eng.setObjectOwnership(nparent, QJSEngine::CppOwnership);
    my_eng.setObjectOwnership(qApp, QJSEngine::CppOwnership);

    my_eng.installExtensions(QJSEngine::ConsoleExtension);

    // print() and console.log write in the output box; processEvents() keeps the GUI (and Stop) alive in long loops
    my_eng.globalObject().setProperty("jsPan", my_eng.newQObject(this));
    my_eng.setObjectOwnership(this, QJSEngine::CppOwnership);
    my_eng.evaluate("function print() { jsPan.appendOutput(Array.prototype.map.call(arguments, function(a) { return String(a); }).join(' ')); }\n"
                    "function processEvents() { jsPan.processScriptEvents(); }\n"
                    "console.log = print;\n");

    qRegisterMetaType<nGenericPan*>("nGenericPan*");
    qRegisterMetaType<nPhysD*>("nPhysD*");

    splitter->setStretchFactor(0, 10);
    splitter->setStretchFactor(1, 1);

    QKeySequence key_seq=QKeySequence(Qt::CTRL | Qt::Key_Return);
    command->setToolTip("Press "+key_seq.toString(QKeySequence::NativeText)+" to execute"+command->toolTip());
    command->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    output->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));

    QShortcut* my_shortcut = new QShortcut(key_seq, command);
    connect(my_shortcut, SIGNAL(activated()), this, SLOT(runIt()));
    connect(actionRun, SIGNAL(triggered()), this, SLOT(runIt()));
    connect(actionStop, SIGNAL(triggered()), this, SLOT(stopIt()));
    actionRun->setToolTip(actionRun->toolTip()+" ("+key_seq.toString(QKeySequence::NativeText)+")");
    show();
}

// void JavaScript::runIt() {
//     saveDefaults();
//     QJSValue retval;
//     output->clear();
//     QString mytext=command->toPlainText();
//     if(QFileInfo::exists(mytext)) {
//         QFile t(mytext);
//         t.open(QIODevice::ReadOnly| QIODevice::Text);
//         QTextStream out(&t);
//         QString toRun=out.readAll();
//         t.close();
//         retval = my_eng.evaluate(toRun, mytext);
//     } else {
//         qDebug() << mytext;
//         retval = my_eng.evaluate(mytext);
//     }
//     output->setPlainText(retval.toString());
// }

void JavaScript::runIt() {
    if (running) return; // processEvents() in a script can bring us here again
    saveDefaults();
    output->clear();
    QString mytext=command->toPlainText();
    QString toRun=mytext;
    QString source;
    // a single line with the name of a file: run the file
    QString fname=mytext.trimmed();
    if (!fname.contains('\n') && QFileInfo(fname).isFile()) {
        QFile t(fname);
        if (!t.open(QIODevice::ReadOnly| QIODevice::Text)) {
            output->setPlainText(tr("Cannot open ")+fname+": "+t.errorString());
            return;
        }
        QTextStream in(&t);
        toRun=in.readAll();
        source=fname;
    }
    // qDebug() << mytext;

    QJSValue retval;
    {
        QScopedValueRollback<bool> guard(running,true);
        actionRun->setEnabled(false);
        actionStop->setEnabled(true);
        my_eng.setInterrupted(false);
        retval = my_eng.evaluate(toRun, source);
        actionRun->setEnabled(true);
        actionStop->setEnabled(false);
    }

    if (my_eng.isInterrupted()) {
        my_eng.setInterrupted(false);
        output->appendPlainText(tr("Stopped"));
    } else if (retval.isError()) {
        output->appendPlainText(tr("line %1: %2").arg(retval.property("lineNumber").toInt()).arg(retval.toString()));
        QString stack=retval.property("stack").toString();
        if (stack.contains('\n')) output->appendPlainText(stack); // only for nested calls
    } else if (!retval.isUndefined()) {
        output->appendPlainText(retval.toString());
    }
}

void JavaScript::stopIt() {
    my_eng.setInterrupted(true);
}

void JavaScript::appendOutput(QString text) {
    output->appendPlainText(text);
    processScriptEvents();
}

void JavaScript::processScriptEvents() {
    QCoreApplication::processEvents();
}

void JavaScript::on_actionOpen_File_triggered() {
    QString fname=QFileDialog::getOpenFileName(this,tr("Open JS file"),property("NeuSave-fileJS").toString(),tr("JS file")+QString(" (*.js);;")+tr("Any file")+QString(" (*)"));
    if (!fname.isEmpty()) {
        setProperty("NeuSave-fileJS", fname);
        QFile f(fname);
        if (f.open(QFile::ReadOnly | QFile::Text)) {
            QTextStream in(&f);
            command->setPlainText(in.readAll());
        } else {
            statusbar->showMessage(tr("Cannot open ")+fname+": "+f.errorString(), 5000);
        }
    }
}

void JavaScript::on_actionSave_File_triggered() {
    QString fname=QFileDialog::getSaveFileName(this,tr("Save JS file"),property("NeuSave-fileJS").toString(),tr("JS file")+QString(" (*.js);;")+tr("Any file")+QString(" (*)"));
    if (!fname.isEmpty()) {
        setProperty("NeuSave-fileJS", fname);
        QFile f(fname);
        // if (f.open(QFile::ReadWrite | QFile::Text)) {
        //     QTextStream in(&f);
        //     in << command->toPlainText();
        // }
        // command->setPlainText(fname);
        if (f.open(QFile::WriteOnly | QFile::Truncate | QFile::Text)) {
            QTextStream out(&f);
            out << command->toPlainText();
            statusbar->showMessage(tr("Saved ")+fname, 5000);
        } else {
            statusbar->showMessage(tr("Cannot write ")+fname+": "+f.errorString(), 5000);
        }
    }
}
